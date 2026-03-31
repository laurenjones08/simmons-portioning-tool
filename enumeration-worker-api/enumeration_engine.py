"""
Staged enumeration engine.

Fetches SKU + bucket data from enumeration_db, enumerates all valid
SKU combinations (size 1..maxCombinationSize), calculates metrics per
bucket, and writes results to the enumeration_results collection.

The engine writes progress updates back to job_status via the
provided JobRepository so the API can reflect live stage progress.
"""

from __future__ import annotations

import itertools
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import UpdateOne
from pymongo.database import Database

from repositories.job_repository import JobRepository

logger = logging.getLogger(__name__)

# Part codes that are illegal to use together in the same combination.
ILLEGAL_PAIRS: Dict[str, List[str]] = {
    "D": ["T"],
    "T": ["D"],
    "R": ["V"],
    "V": ["R"],
    "M": ["K"],
    "K": ["M"],
    "S": ["U"],
    "U": ["S"],
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _flush_batch(collection, writes: List[UpdateOne]) -> None:
    """Write a batch of upserts; falls back to per-doc upserts for mongomock compat."""
    if not writes:
        return
    try:
        collection.bulk_write(writes, ordered=False)
    except TypeError:
        for write in writes:
            collection.update_one(write._filter, write._doc, upsert=write._upsert)


def _compute_metrics(
    skus: List[Dict[str, Any]],
    bucket: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate upgrade/trim/weight metrics for a SKU combination against a bucket.

    Upgrade % = (total_target_weight / bucket_midpoint) * 100
    Trim %    = 100 - upgrade %
    """
    weights = [float(s.get("targetWeight") or 0.0) for s in skus]
    total_weight = sum(weights)

    bmin = float(bucket.get("minWeight") or 0.0)
    bmax = float(bucket.get("maxWeight") or 0.0)
    midpoint = (bmin + bmax) / 2.0 if bmax > bmin else bmax

    upgrade_pct = (total_weight / midpoint * 100.0) if midpoint > 0 else 0.0
    trim_pct = max(0.0, 100.0 - upgrade_pct)

    return {
        "totalTargetWeight": total_weight,
        "averageTargetWeight": total_weight / len(weights) if weights else 0.0,
        "bucketMin": bmin,
        "bucketMax": bmax,
        "bucketMidpoint": midpoint,
        "upgradePercentage": round(upgrade_pct, 4),
        "trimPercentage": round(trim_pct, 4),
        "withinBucket": bmin <= total_weight <= bmax,
    }


def _assign_cuts(skus: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """
    Try to assign a unique cut part code to each SKU from its allowedParts list.
    Also checks ILLEGAL_PAIRS. Returns None if assignment is impossible.
    """
    used: set = set()
    assignment: Dict[str, str] = {}

    for sku in skus:
        trade = str(sku.get("tradeNumber", ""))
        allowed = [p.strip() for p in (sku.get("allowedParts") or []) if p.strip()]
        chosen = None
        for part in allowed:
            if part not in used:
                chosen = part
                break
        if chosen is None:
            return None
        used.add(chosen)
        assignment[trade] = chosen

    # Check illegal pairs
    for part in used:
        bad = set(ILLEGAL_PAIRS.get(part, []))
        if bad & used:
            return None

    return assignment


def run_enumeration(
    db: Database,
    job_id: str,
    run_id: str,
    job_repo: JobRepository,
    max_combination_size: int = 4,
    batch_size: int = 1000,
    sku_filter: Optional[List[str]] = None,
) -> None:
    """
    Main enumeration entry point called from the background thread.

    Steps:
      1. Load candidate SKUs from `skus` collection. (
      2. Load buckets from `buckets` collection.
      3. For each stage k=1..maxCombinationSize, iterate all C(n,k) combos.
      4.
      4. For each combo, calculate metrics against every bucket.
      5. Upsert results into `enumeration_results`.
      6. Checkpoint progress in job_status after each batch.

    Raises:
        ValueError: when no SKUs or buckets are found.
    """
    results_col = db["enumeration_results"]
    skus_col = db["skus"]
    buckets_col = db["buckets"]

    # ------------------------------------------------------------------ #
    # 1. Load candidate SKUs
    # ------------------------------------------------------------------ #
    sku_query: Dict[str, Any] = {}
    if sku_filter:
        sku_query["tradeNumber"] = {"$in": sku_filter}

    candidates: List[Dict[str, Any]] = list(
        skus_col.find(
            sku_query,
            {
                "_id": 0,
                "tradeNumber": 1,
                "targetWeight": 1,
                "minWeight": 1,
                "maxWeight": 1,
                "customerType": 1,
                "productType": 1,
                "allowedParts": 1,
            },
        )
    )
    candidates.sort(key=lambda s: str(s.get("tradeNumber", "")))

    if not candidates:
        raise ValueError("No SKUs found. Seed the skus collection before running enumeration.")

    # ------------------------------------------------------------------ #
    # 2. Load buckets
    # ------------------------------------------------------------------ #
    buckets: List[Dict[str, Any]] = list(
        buckets_col.find({}, {"_id": 1, "minWeight": 1, "maxWeight": 1})
    )
    if not buckets:
        raise ValueError("No buckets found. Create at least one bucket before running enumeration.")

    # ------------------------------------------------------------------ #
    # 3. Initialise stage tracking in the job document
    # ------------------------------------------------------------------ #
    stage_docs = [
        {
            "stage": k,
            "status": "pending",
            "totalCombinations": math.comb(len(candidates), k) if len(candidates) >= k else 0,
            "processedCombinations": 0,
            "startedAt": None,
            "finishedAt": None,
        }
        for k in range(1, max_combination_size + 1)
    ]
    job_repo.mark_running(job_id, sku_count=len(candidates), stages=stage_docs)

    # ------------------------------------------------------------------ #
    # 4. Ensure result indexes exist
    # ------------------------------------------------------------------ #
    results_col.create_index(
        [("runId", 1), ("comboKey", 1), ("bucketId", 1)],
        unique=True,
        name="uniq_run_combo_bucket",
    )
    results_col.create_index([("runId", 1), ("stage", 1)], name="idx_run_stage")
    results_col.create_index([("runId", 1), ("skuTradeNumbers", 1)], name="idx_run_skus")
    results_col.create_index([("bucketId", 1)], name="idx_bucket_id")

    # ------------------------------------------------------------------ #
    # 5. Enumerate per stage
    # ------------------------------------------------------------------ #
    for stage_index, k in enumerate(range(1, max_combination_size + 1)):
        total = stage_docs[stage_index]["totalCombinations"]

        # Check cancellation before starting a stage
        if job_repo.is_cancelled(job_id):
            logger.info("Job %s cancelled before stage %s", job_id, k)
            return

        job_repo.mark_stage_running(job_id, stage_index, total)
        logger.info("Job %s stage k=%s started (total=%s combos × %s buckets)", job_id, k, total, len(buckets))

        writes: List[UpdateOne] = []
        processed = 0

        for combo in itertools.combinations(candidates, k):
            # Check cancellation periodically (every batch boundary)
            if processed % batch_size == 0 and job_repo.is_cancelled(job_id):
                logger.info("Job %s cancelled mid-stage k=%s at combo %s", job_id, k, processed)
                return

            trade_numbers = [str(s.get("tradeNumber", "")) for s in combo]
            combo_key = "|".join(trade_numbers)

            # Cut assignment – skip combo if no valid cut exists
            cut_assignment = _assign_cuts(list(combo))

            for bucket in buckets:
                bucket_id = str(bucket.get("_id", ""))
                metrics = _compute_metrics(list(combo), bucket)

                doc = {
                    "runId": run_id,
                    "stage": k,
                    "combinationSize": k,
                    "comboKey": combo_key,
                    "bucketId": bucket_id,
                    "skuTradeNumbers": trade_numbers,
                    "cutAssignment": cut_assignment,  # None means no valid cut
                    "cutFeasible": cut_assignment is not None,
                    "metrics": metrics,
                    "updatedAt": _now(),
                }

                writes.append(
                    UpdateOne(
                        {"runId": run_id, "comboKey": combo_key, "bucketId": bucket_id},
                        {"$set": doc},
                        upsert=True,
                    )
                )

            processed += 1

            if len(writes) >= batch_size:
                _flush_batch(results_col, writes)
                writes = []
                job_repo.checkpoint_stage(job_id, stage_index, processed)

        # Flush remaining writes
        _flush_batch(results_col, writes)
        job_repo.mark_stage_complete(job_id, stage_index, processed)
        logger.info("Job %s stage k=%s complete (%s combos processed)", job_id, k, processed)

