"""Staged long-running enumeration runner for SKU combination metrics."""

from __future__ import annotations

import itertools
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import UpdateOne
from pymongo.database import Database

logger = logging.getLogger(__name__)


class StagedEnumerationRunner:
    """Enumerate SKU combinations in stages (size 1..N) with resumable checkpoints."""

    def __init__(
        self,
        database: Database,
        run_id: str,
        batch_size: int = 1000,
        max_combination_size: int = 4,
        sku_trade_numbers: Optional[List[str]] = None,
    ):
        self.db = database
        self.run_id = run_id
        self.batch_size = max(1, batch_size)
        self.max_combination_size = max(1, min(max_combination_size, 4))
        self.sku_trade_numbers = sku_trade_numbers or []

        self.sku_collection = self.db["skus"]
        self.runs_collection = self.db["enumeration_runs"]
        self.results_collection = self.db["enumeration_results"]

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def ensure_indexes(self) -> None:
        self.runs_collection.create_index([("status", 1)], name="idx_status")
        self.results_collection.create_index(
            [("runId", 1), ("comboKey", 1)],
            unique=True,
            name="uniq_run_combo",
        )
        self.results_collection.create_index([("runId", 1), ("stage", 1)], name="idx_run_stage")
        self.results_collection.create_index([("runId", 1), ("skuTradeNumbers", 1)], name="idx_run_skus")

    def _load_candidate_skus(self) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if self.sku_trade_numbers:
            query["tradeNumber"] = {"$in": self.sku_trade_numbers}

        projection = {
            "_id": 0,
            "tradeNumber": 1,
            "targetWeight": 1,
            "minWeight": 1,
            "maxWeight": 1,
            "customerType": 1,
            "productType": 1,
            "allowedParts": 1,
        }
        skus = list(self.sku_collection.find(query, projection))
        skus.sort(key=lambda item: str(item.get("tradeNumber", "")))
        return skus

    def _default_stage_state(self) -> Dict[str, Dict[str, Any]]:
        return {
            str(stage): {
                "status": "pending",
                "lastProcessedIndex": -1,
                "processedCombinations": 0,
                "totalCombinations": 0,
            }
            for stage in range(1, self.max_combination_size + 1)
        }

    def _get_or_initialize_run(self, sku_count: int) -> Dict[str, Any]:
        existing = self.runs_collection.find_one({"_id": self.run_id})
        now = self._now()

        if existing:
            # Preserve existing stages/checkpoints to support resuming.
            stages = existing.get("stages") or {}
            for key, value in self._default_stage_state().items():
                stages.setdefault(key, value)

            self.runs_collection.update_one(
                {"_id": self.run_id},
                {
                    "$set": {
                        "status": "running",
                        "updatedAt": now,
                        "skuCount": sku_count,
                        "stages": stages,
                    }
                },
            )
            return self.runs_collection.find_one({"_id": self.run_id})

        run_doc = {
            "_id": self.run_id,
            "status": "running",
            "startedAt": now,
            "updatedAt": now,
            "completedAt": None,
            "skuCount": sku_count,
            "stages": self._default_stage_state(),
        }
        self.runs_collection.insert_one(run_doc)
        return run_doc

    def _mark_stage_running(self, stage: int, total_combinations: int) -> None:
        stage_key = str(stage)
        now = self._now()
        self.runs_collection.update_one(
            {"_id": self.run_id},
            {
                "$set": {
                    f"stages.{stage_key}.status": "running",
                    f"stages.{stage_key}.totalCombinations": total_combinations,
                    f"stages.{stage_key}.startedAt": now,
                    "updatedAt": now,
                }
            },
        )

    def _checkpoint_stage(self, stage: int, last_processed_index: int, processed_count: int) -> None:
        stage_key = str(stage)
        self.runs_collection.update_one(
            {"_id": self.run_id},
            {
                "$set": {
                    f"stages.{stage_key}.lastProcessedIndex": last_processed_index,
                    f"stages.{stage_key}.processedCombinations": processed_count,
                    "updatedAt": self._now(),
                }
            },
        )

    def _mark_stage_complete(self, stage: int, processed_count: int) -> None:
        stage_key = str(stage)
        now = self._now()
        self.runs_collection.update_one(
            {"_id": self.run_id},
            {
                "$set": {
                    f"stages.{stage_key}.status": "completed",
                    f"stages.{stage_key}.finishedAt": now,
                    f"stages.{stage_key}.processedCombinations": processed_count,
                    "updatedAt": now,
                }
            },
        )

    def _finalize_run(self, status: str) -> None:
        now = self._now()
        self.runs_collection.update_one(
            {"_id": self.run_id},
            {
                "$set": {
                    "status": status,
                    "completedAt": now,
                    "updatedAt": now,
                }
            },
        )

    def _build_result_document(
        self,
        stage: int,
        combination_index: int,
        skus: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trade_numbers = [str(item.get("tradeNumber", "")).strip() for item in skus]
        target_weights = [float(item.get("targetWeight", 0.0) or 0.0) for item in skus]
        min_weights = [float(item.get("minWeight", 0.0) or 0.0) for item in skus]
        max_weights = [float(item.get("maxWeight", 0.0) or 0.0) for item in skus]

        total_target = sum(target_weights)
        total_min = sum(min_weights)
        total_max = sum(max_weights)
        combo_key = "|".join(trade_numbers)

        return {
            "runId": self.run_id,
            "stage": stage,
            "combinationSize": len(trade_numbers),
            "combinationIndex": combination_index,
            "comboKey": combo_key,
            "skuTradeNumbers": trade_numbers,
            "metrics": {
                "totalTargetWeight": total_target,
                "averageTargetWeight": (total_target / len(target_weights)) if target_weights else 0.0,
                "minTargetWeight": min(target_weights) if target_weights else 0.0,
                "maxTargetWeight": max(target_weights) if target_weights else 0.0,
                "sumMinWeight": total_min,
                "sumMaxWeight": total_max,
                "targetToMaxRatio": (total_target / total_max) if total_max > 0 else 0.0,
            },
            "createdAt": self._now(),
        }

    def _flush_batch(self, writes: List[UpdateOne]) -> None:
        if not writes:
            return

        try:
            self.results_collection.bulk_write(writes, ordered=False)
            return
        except TypeError:
            # mongomock currently does not support the latest UpdateOne bulk signature.
            pass

        for write in writes:
            self.results_collection.update_one(write._filter, write._doc, upsert=write._upsert)

    def _run_stage(self, run_doc: Dict[str, Any], stage: int, candidates: List[Dict[str, Any]]) -> None:
        stage_key = str(stage)
        stage_state = (run_doc.get("stages") or {}).get(stage_key, {})
        if stage_state.get("status") == "completed":
            logger.info("Run %s stage %s already completed; skipping", self.run_id, stage)
            return

        total = math.comb(len(candidates), stage) if len(candidates) >= stage else 0
        self._mark_stage_running(stage, total)

        start_index = int(stage_state.get("lastProcessedIndex", -1)) + 1
        if total == 0:
            self._mark_stage_complete(stage, 0)
            return

        writes: List[UpdateOne] = []
        processed_count = int(stage_state.get("processedCombinations", 0))
        last_index = start_index - 1

        for index, combo in enumerate(itertools.combinations(candidates, stage)):
            if index < start_index:
                continue

            document = self._build_result_document(stage, index, list(combo))
            writes.append(
                UpdateOne(
                    {"runId": self.run_id, "comboKey": document["comboKey"]},
                    {"$set": document},
                    upsert=True,
                )
            )
            processed_count += 1
            last_index = index

            if len(writes) >= self.batch_size:
                self._flush_batch(writes)
                writes = []
                self._checkpoint_stage(stage, last_index, processed_count)

        self._flush_batch(writes)
        self._checkpoint_stage(stage, last_index, processed_count)
        self._mark_stage_complete(stage, processed_count)
        logger.info(
            "Run %s stage %s complete. Processed %s/%s combinations",
            self.run_id,
            stage,
            processed_count,
            total,
        )

    def run(self) -> Dict[str, Any]:
        self.ensure_indexes()
        candidates = self._load_candidate_skus()
        if not candidates:
            raise ValueError("No SKU candidates found. Seed SKUs before running enumeration.")

        run_doc = self._get_or_initialize_run(sku_count=len(candidates))

        for stage in range(1, self.max_combination_size + 1):
            self._run_stage(run_doc, stage, candidates)
            run_doc = self.runs_collection.find_one({"_id": self.run_id}) or run_doc

        self._finalize_run("completed")
        return self.runs_collection.find_one({"_id": self.run_id}) or {"_id": self.run_id, "status": "completed"}

