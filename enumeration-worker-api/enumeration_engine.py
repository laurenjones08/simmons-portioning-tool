"""
Enumeration Engine — core background worker for the Enumeration Worker API.

This module implements `run_enumeration`, a synchronous multi-stage pipeline
that generates all valid SKU combinations, assigns cut strategies, computes
mix and bucket metrics, persists results, and reports job progress.

All data access goes through the repository classes in enumeration-api/repositories/.
The Global Config API is called over HTTP (via requests) to fetch runtime config values.
"""

import itertools
import logging
from functools import lru_cache
from math import floor
import os
import random
from statistics import NormalDist
import sys
from typing import Any, Dict, List, Optional

import requests
from pymongo.database import Database

# Ensure shared package is importable in Docker dev mode where it is mounted at /shared.
shared_path = "/shared"
if os.path.isdir(shared_path) and shared_path not in sys.path:
    sys.path.insert(0, shared_path)

from enumeration_shared.repositories import (
    SKURepository,
    CutStrategyRepository,
    BucketRepository,
    MixRepository,
    MixMetricRepository,
)

logger = logging.getLogger(__name__)
_STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


def _is_nugget_product(product_type: Optional[str]) -> bool:
    """Return True for any supported nugget product-type alias."""
    return product_type in {"NUGGET", NUGGET_SKU_KEY}


def _bucket_min_weight(bucket: Dict[str, Any]) -> float:
    """Return the bucket's minimum weight, accepting legacy field aliases."""
    if "minWeight" in bucket:
        return bucket["minWeight"]
    if "targetWeight" in bucket:
        return bucket["targetWeight"]
    raise KeyError("bucket must include 'minWeight' or 'targetWeight'")


def _bucket_target_weight(bucket: Dict[str, Any]) -> float:
    """Return the bucket's target weight, accepting legacy field aliases."""
    if "targetWeight" in bucket:
        return bucket["targetWeight"]
    if "minWeight" in bucket:
        return bucket["minWeight"]
    raise KeyError("bucket must include 'targetWeight' or 'minWeight'")


def _bucket_max_weight(bucket: Dict[str, Any]) -> float:
    """Return the bucket's maximum weight."""
    if "maxWeight" in bucket:
        return bucket["maxWeight"]
    raise KeyError("bucket must include 'maxWeight'")


def _combo_sku_keys(combo: List[Dict[str, Any]]) -> List[str]:
    """Return combo trade numbers in first-appearance order without duplicates."""
    return list(dict.fromkeys(str(s["tradeNumber"]) for s in combo))


def _combo_signature_key(combo: List[Dict[str, Any]]) -> str:
    """Return an occurrence-preserving key for a SKU combo."""
    return "|".join(str(s["tradeNumber"]) for s in combo)


def _combo_has_fillet(combo: List[Dict[str, Any]]) -> bool:
    """Return True when the combo contains at least one non-nugget SKU."""
    return any(not _is_nugget_product(s.get("productType")) for s in combo)


def _assign_combo_part_codes(
    combo: List[Dict[str, Any]],
    strategy_parts: List[str],
) -> Optional[List[str]]:
    """
    Assign one part code to each combo occurrence.

    Non-nugget / non-strip SKUs must receive distinct part codes. Nugget and
    strip SKUs can reuse part codes, but every assigned part must still be
    allowed by the SKU and the final assignment must cover every strategy part.
    """
    assignments: List[Optional[str]] = [None] * len(combo)
    used_parts: set[str] = set()

    candidate_parts_by_index: List[List[str]] = []
    for sku in combo:
        allowed = sku.get("allowedParts", [])
        candidate_parts_by_index.append([part for part in strategy_parts if part in allowed])

    def backtrack(index: int) -> bool:
        if index >= len(combo):
            assigned_parts = {part for part in assignments if part is not None}
            return all(part in assigned_parts for part in strategy_parts)

        sku = combo[index]
        candidate_parts = candidate_parts_by_index[index]
        if not candidate_parts:
            return False

        is_nugget = _is_nugget_product(sku.get("productType"))
        for part in candidate_parts:
            if not is_nugget and part in used_parts:
                continue

            assignments[index] = part
            added_part = False
            if not is_nugget:
                used_parts.add(part)
                added_part = True

            if backtrack(index + 1):
                return True

            assignments[index] = None
            if added_part:
                used_parts.remove(part)

        return False

    if not backtrack(0):
        return None

    return [str(part) for part in assignments if part is not None]


def _build_combo_sku_map(combo: List[Dict[str, Any]], part_assignments: List[str]) -> Dict[str, str]:
    """Build a trade-number -> part-code map from per-occurrence assignments."""
    skus: Dict[str, str] = {}
    for sku, part_code in zip(combo, part_assignments):
        trade_number = str(sku["tradeNumber"])
        if trade_number not in skus:
            skus[trade_number] = part_code
    return skus


def _doc_without_id(document: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of a Mongo document without its `_id` field."""
    return {key: value for key, value in document.items() if key != "_id"}

# ---------------------------------------------------------------------------
# Global Config API key constants
# ---------------------------------------------------------------------------
BUCKET_TOLERANCE_CONFIG_KEY = "enumeration.bucketWeightTolerancePct"
FDS_VALUE_CONFIG_KEY        = "enumeration.fdsValueCoefficient"
RTL_VALUE_CONFIG_KEY        = "enumeration.rtlValueCoefficient"
TRIM_VALUE_CONFIG_KEY       = "enumeration.trimValueCoefficient"
UPGRADE_MU_CONFIG_KEY       = "enumeration.upgradeDistributionMu"
UPGRADE_SIGMA_CONFIG_KEY    = "enumeration.upgradeDistributionSigma"

UPGRADE_MONTE_CARLO_SAMPLES = 10
UPGRADE_CONFIDENCE_ALPHA    = 0.05

NUGGET_SKU_KEY = "NUGGET|STRIP"


# ---------------------------------------------------------------------------
# Phase 1 — Reference data loaders
# ---------------------------------------------------------------------------

def _load_candidate_skus(
    db: Database,
    plant_filter: Optional[str],
    bird_size_filter: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Load candidate SKUs from the database, applying optional filters.

    Builds a query dict from the provided filters and delegates to
    ``SKURepository.find_by_criteria``.  Raises ``ValueError`` when the
    result set is empty so the caller can mark the job as failed.

    Args:
        db: PyMongo Database handle (shared with enumeration-api).
        plant_filter: Optional value to match against ``prodPlant``.
        bird_size_filter: Optional value to match against ``birdSize``.

    Returns:
        Non-empty list of SKU documents matching the filters.

    Raises:
        ValueError: When no SKUs match the given filters.
    """
    query: Dict[str, Any] = {}
    if plant_filter is not None:
        query["prodPlant"] = plant_filter
    if bird_size_filter is not None:
        query["birdSize"] = bird_size_filter

    skus = SKURepository(db).find_by_criteria(query)

    if not skus:
        filter_desc = []
        if plant_filter is not None:
            filter_desc.append(f"prodPlant={plant_filter!r}")
        if bird_size_filter is not None:
            filter_desc.append(f"birdSize={bird_size_filter!r}")
        filters_str = ", ".join(filter_desc) if filter_desc else "no filters"
        raise ValueError(
            f"No candidate SKUs found for filters: {filters_str}. "
            "Cannot proceed with enumeration."
        )

    return skus


def _load_cut_strategies(db: Database) -> List[Dict[str, Any]]:
    """
    Load all cut strategies from the database.

    Args:
        db: PyMongo Database handle.

    Returns:
        List of cut strategy documents (may be empty).
    """
    return CutStrategyRepository(db).search({})


def _load_buckets(db: Database) -> List[Dict[str, Any]]:
    """
    Load all bucket documents from the database.

    Args:
        db: PyMongo Database handle.

    Returns:
        List of bucket documents (may be empty).
    """
    return BucketRepository(db).search({})


@lru_cache(maxsize=1024)
def calculate_bucket_mean(lb, ub, mu, sigma):
    alpha = (lb - mu) / sigma
    beta = (ub - mu) / sigma
    denominator = _STANDARD_NORMAL.cdf(beta) - _STANDARD_NORMAL.cdf(alpha)
    if denominator <= 1e-12:
        return min(max(mu, lb), ub)

    numerator = _STANDARD_NORMAL.pdf(alpha) - _STANDARD_NORMAL.pdf(beta)
    return mu + sigma * (numerator / denominator)


def _fetch_config_values(global_config_url: str) -> Dict[str, float]:
    """
    Fetch all runtime config values from the Global Config API.

    Makes one HTTP GET request per config key to ``{global_config_url}/config/{key}``.
    Each individual fetch falls back to ``0.0`` on any error (connection error,
    non-200 status, missing/unparseable value) and logs a warning.

    Config keys fetched:

    +-------------------+------------------------------------------+
    | Dict key          | Config API key                           |
    +===================+==========================================+
    | ``tolerance_pct`` | enumeration.bucketWeightTolerancePct     |
    +-------------------+------------------------------------------+
    | ``fds_value``     | enumeration.fdsValueCoefficient          |
    +-------------------+------------------------------------------+
    | ``rtl_value``     | enumeration.rtlValueCoefficient          |
    +-------------------+------------------------------------------+
    | ``trim_value``    | enumeration.trimValueCoefficient         |
    +-------------------+------------------------------------------+
    | ``upgrade_mu``    | enumeration.upgradeDistributionMu        |
    +-------------------+------------------------------------------+
    | ``upgrade_sigma`` | enumeration.upgradeDistributionSigma     |
    +-------------------+------------------------------------------+

    Args:
        global_config_url: Base URL of the Global Config API
            (e.g. ``"http://global-config-api:8001"``).

    Returns:
        Dict with keys ``tolerance_pct``, ``fds_value``, ``rtl_value``,
        ``trim_value``, ``upgrade_mu``, and ``upgrade_sigma``; each is a
        ``float`` defaulting to ``0.0`` on error.
    """
    key_map = [
        ("tolerance_pct", BUCKET_TOLERANCE_CONFIG_KEY),
        ("fds_value",     FDS_VALUE_CONFIG_KEY),
        ("rtl_value",     RTL_VALUE_CONFIG_KEY),
        ("trim_value",    TRIM_VALUE_CONFIG_KEY),
        ("upgrade_mu",    UPGRADE_MU_CONFIG_KEY),
        ("upgrade_sigma", UPGRADE_SIGMA_CONFIG_KEY),
    ]

    result: Dict[str, float] = {}

    for dict_key, config_key in key_map:
        value = _fetch_single_config_value(global_config_url, config_key, dict_key)
        result[dict_key] = value

    return result


def _fetch_single_config_value(
    global_config_url: str,
    config_key: str,
    dict_key: str,
) -> float:
    """
    Fetch a single float config value from the Global Config API.

    Falls back to ``0.0`` and logs a warning on any error.

    Args:
        global_config_url: Base URL of the Global Config API.
        config_key: The config key to fetch (e.g. ``"enumeration.fdsValueCoefficient"``).
        dict_key: Human-readable label used in warning messages.

    Returns:
        The config value as a float, or ``0.0`` on any error.
    """
    url = f"{global_config_url.rstrip('/')}/config/{config_key}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            logger.warning(
                "Config key %r returned HTTP %d from %s; defaulting %s to 0.0",
                config_key, response.status_code, url, dict_key,
            )
            return 0.0

        data = response.json()
        raw_value = data.get("value")
        if raw_value is None:
            logger.warning(
                "Config key %r has no 'value' field in response; defaulting %s to 0.0",
                config_key, dict_key,
            )
            return 0.0

        return float(raw_value)

    except requests.exceptions.RequestException as exc:
        logger.warning(
            "Failed to fetch config key %r from %s (%s); defaulting %s to 0.0",
            config_key, url, exc, dict_key,
        )
        return 0.0
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Could not parse config key %r value as float (%s); defaulting %s to 0.0",
            config_key, exc, dict_key,
        )
        return 0.0


# ---------------------------------------------------------------------------
# Phase 3 — Cut strategy validation
# ---------------------------------------------------------------------------

def _get_valid_cut_strategies(
    combo: List[Dict],
    cut_strategies: List[Dict],
) -> List[Dict]:
    """
    Return the subset of cut strategies that are valid for the given combo.

    A strategy is valid when:
    1. Every part code in ``strategy["parts"]`` is present in the
       ``allowedParts`` of at least one SKU in the combo.
    2. ``strategy["hasNugget"]`` matches whether the combo contains a nugget
       SKU (``productType == "NUGGET"``).

    This is a pure function with no I/O.

    Args:
        combo: List of SKU documents in the combination.
        cut_strategies: All available cut strategy documents.

    Returns:
        List of cut strategy documents that are valid for this combo.
    """
    # Build the union of all allowedParts across the combo
    combo_allowed_parts: set = set()
    for sku in combo:
        combo_allowed_parts.update(sku.get("allowedParts", []))

    # Determine whether the combo contains a nugget SKU
    combo_has_nugget = any(_is_nugget_product(s.get("productType")) for s in combo)

    valid = []
    for strategy in cut_strategies:
        # hasNugget must match combo's nugget presence
        if strategy.get("hasNugget", False) != combo_has_nugget:
            continue

        # Every part code in the strategy must be covered by at least one SKU
        strategy_parts = strategy.get("parts", [])
        if all(part in combo_allowed_parts for part in strategy_parts):
            # Every SKU in the combo also needs a concrete assignment from the strategy.
            if _assign_combo_part_codes(combo, strategy_parts) is not None:
                valid.append(strategy)

    return valid


# ---------------------------------------------------------------------------
# Phase 4 — Mix construction
# ---------------------------------------------------------------------------

def _build_mix(
    combo: List[Dict],
    strategy: Dict,
    plant_filter: Optional[str],
    bird_size_filter: Optional[str],
) -> Dict:
    """
    Construct a Mix document from a SKU combination and a cut strategy.

    This is a pure function with no I/O.

    **SKU → Part Code assignment**: For each SKU in the combo, iterates through
    ``strategy["parts"]`` and assigns the first part code that appears in
    ``sku["allowedParts"]``. Produces the ``skus`` map ``{tradeNumber: partCode}``.

    Args:
        combo: List of SKU documents in the combination.
        strategy: Cut strategy document to apply.
        plant_filter: Optional plant filter; used as ``reqPlant`` when provided.
        bird_size_filter: Optional bird-size filter; used as ``reqBirdSize`` when provided.

    Returns:
        Dict representing the Mix document with all derived fields.
    """
    # Build skus map: tradeNumber -> first matching part code
    part_assignments = _assign_combo_part_codes(combo, strategy.get("parts", []))
    if part_assignments is None:
        raise ValueError(
            "Cannot build mix because at least one SKU in the combo "
            "does not match any part in the cut strategy"
        )

    skus = _build_combo_sku_map(combo, part_assignments)

    # Derived boolean flags
    includes_fds = any(s.get("customerType") == "FDS" for s in combo)
    includes_rtl = any(s.get("customerType") == "RTL" for s in combo)
    combo_has_nugget = any(_is_nugget_product(s.get("productType")) for s in combo)
    includes_nug = combo_has_nugget and bool(strategy.get("hasNugget", False))

    # Nugget target weight
    nugget_target_weight: Optional[float] = None
    if includes_nug:
        for sku in combo:
            if _is_nugget_product(sku.get("productType")):
                nugget_target_weight = sku["targetWeight"]
                break

    # Plant and bird size
    req_plant = plant_filter if plant_filter is not None else combo[0]["prodPlant"]
    req_bird_size = bird_size_filter if bird_size_filter is not None else combo[0]["birdSize"]

    # Fillet counts and weight (non-nugget SKUs)
    non_nugget_skus = [s for s in combo if not _is_nugget_product(s.get("productType"))]
    num_fillets = len(non_nugget_skus)
    fillet_weight = sum(s["targetWeight"] for s in non_nugget_skus)

    return {
        "skus": skus,
        "_partAssignments": part_assignments,
        "cutStrategyID": strategy["_id"],
        "mfgType": strategy["mfgType"],
        "beltSpeed": strategy["beltSpeed"],
        "includesFDS": includes_fds,
        "includesRTL": includes_rtl,
        "includesNug": includes_nug,
        "nuggetTargetWeight": nugget_target_weight,
        "reqPlant": req_plant,
        "reqBirdSize": req_bird_size,
        "numFillets": num_fillets,
        "filletWeight": fillet_weight,
        "skuKeys": _combo_sku_keys(combo),
        "skuSetKey": _combo_signature_key(combo),
    }


# ---------------------------------------------------------------------------
# Phase 5 — Bucket fitting
# ---------------------------------------------------------------------------

def _fits_bucket(mix_weight: float, bucket: Dict, tolerance_pct: float) -> bool:
    """
    Determine whether a mix weight can be produced in a bucket.

    The bucket's ``targetWeight`` is the hard upper bound.

    Args:
        mix_weight: Total weight of the mix (sum of SKU targetWeights).
        bucket: Bucket document with ``minWeight``, ``targetWeight``, and
            ``maxWeight`` fields.
        tolerance_pct: Retained for compatibility with existing callers.

    Returns:
        ``True`` when ``mix_weight <= bucket["maxWeight"]``,
        ``False`` otherwise.
    """
    return mix_weight <= _bucket_target_weight(bucket) * (1.0 - tolerance_pct)

def _planned_bucket_weight(
    combo: List[Dict],
    bucket: Dict,
    includes_nug: bool,
    config_values: Dict[str, float],
    nugget_target_weight: Optional[float],
) -> float:
    """
    Return the total weight implied by the persisted unit plan for a bucket.

    Bucket-fit validation must use the same planned total that will be stored in
    ``unitPlan``. Otherwise a combo can appear to fit by raw target-weight sum
    while the persisted plan materially exceeds the bucket bounds.
    """
    bucket_target_weight = _bucket_target_weight(bucket) * (1.0 - config_values.get("tolerance_pct", 0.0))
    fixed_weight = 0.0
    nugget_weight = 0.0

    for sku in combo:
        target_weight = sku["targetWeight"]
        units_per_cut = sku.get("unitsPerCut", 1)
        is_nugget = (
            includes_nug
            and _is_nugget_product(sku.get("productType"))
            and nugget_target_weight is not None
            and nugget_target_weight > 0
        )

        if is_nugget:
            continue

        fixed_weight += units_per_cut * target_weight

    if includes_nug and nugget_target_weight is not None and nugget_target_weight > 0 and fixed_weight <= 0.0:
        raise ValueError("Nugget-only mixes are not allowed")

    if includes_nug and nugget_target_weight is not None and nugget_target_weight > 0:
        remaining_weight = max(0.0, bucket_target_weight - fixed_weight)
        nugget_weight = floor(remaining_weight / nugget_target_weight) * nugget_target_weight

    return fixed_weight + nugget_weight


# ---------------------------------------------------------------------------
# Phase 6 — Mix metric computation
# ---------------------------------------------------------------------------

def _compute_mix_metric(
    mix_id: Optional[str],
    combo: List[Dict],
    skus_map: Dict[str, str],
    bucket: Dict,
    includes_nug: bool,
    nugget_target_weight: Optional[float],
    config_values: Dict[str, float],
    part_assignments: Optional[List[str]] = None,
) -> Dict:
    """
    Compute a MixMetric document for a given Mix + Bucket pairing.

    This is a pure function with no I/O.

    **Metric formulas**:

    - ``upgradePercentage``: when ``upgrade_mu`` and ``upgrade_sigma`` are
      configured, this is derived from the expected unavoidable trim of a
      truncated normal over the bucket's actual range
      ``[bucket.minWeight, bucket.maxWeight]`` and computed as
      ``(1 - expected_trim) * 100``. Otherwise it falls back to the legacy
      ``(mix_weight / bucket.targetWeight) * 100`` formula, capped at 100.
    - ``trimPercentage``: ``100.0 - upgradePercentage``.
    - ``value``: ``fds_weight * fds_value + rtl_weight * rtl_value + trim_weight * trim_value``
      where ``trim_weight = max(0.0, mix_weight - bucket.minWeight)``.

    **Unit Plan construction**: one item per SKU entry in the combo (including
    repeated SKUs).  When ``includes_nug`` is ``True`` and
    ``nugget_target_weight > 0``, the nugget SKU's ``unitsInPlan`` is overridden
    to consume the remaining bucket capacity after all non-nugget cuts have
    been fixed: ``floor((bucket.targetWeight - fixed_non_nugget_weight) /
    nugget_target_weight)``.  All non-nugget SKUs retain ``unitsInPlan =
    unitsPerCut``.

    ``skuKeys`` is the list of ``tradeNumber`` values in first-appearance order
    matching ``unitPlan``.

    The composite ``_id`` is ``f"{mix_id}:{bucket['_id']}"``.

    Args:
        mix_id: The ``_id`` of the parent Mix document (may be ``None`` before
            the Mix is persisted; the caller updates ``_id`` afterwards).
        combo: List of SKU documents in the combination (may contain repeats).
        skus_map: Mapping of ``tradeNumber`` → assigned ``partCode``.
        bucket: Bucket document with ``_id``, ``minWeight``, and ``maxWeight``.
        includes_nug: Whether the mix includes a nugget SKU.
        nugget_target_weight: ``targetWeight`` of the nugget SKU, or ``None``.
        config_values: Dict with keys ``fds_value``, ``rtl_value``,
            ``trim_value``, ``upgrade_mu``, ``upgrade_sigma`` (and
            ``tolerance_pct``, unused here).

    Returns:
        Dict representing the MixMetric document.
    """
    # --- unitPlan ---
    global trim_weight
    unit_plan: List[Dict] = []
    seen_trade_numbers: List[str] = []
    planned_mix_weight = 0.0
    planned_fds_weight = 0.0
    planned_rtl_weight = 0.0
    fixed_non_nugget_weight = sum(
        sku.get("unitsPerCut", 1) * sku["targetWeight"]
        for sku in combo
        if not (
            includes_nug
            and _is_nugget_product(sku.get("productType"))
            and nugget_target_weight is not None
            and nugget_target_weight > 0
        )
    )
    if includes_nug and nugget_target_weight is not None and nugget_target_weight > 0 and fixed_non_nugget_weight <= 0.0:
        raise ValueError("Nugget-only mixes are not allowed")
    remaining_nugget_weight = max(0.0, _bucket_target_weight(bucket) - fixed_non_nugget_weight)

    if part_assignments is None:
        part_assignments = [skus_map.get(str(sku["tradeNumber"]), "") for sku in combo]

    for sku, part_code in zip(combo, part_assignments):
        trade_number = sku["tradeNumber"]
        units_per_cut = sku.get("unitsPerCut", 1)
        target_weight = sku["targetWeight"]

        if (
            includes_nug
            and _is_nugget_product(sku.get("productType"))
            and nugget_target_weight is not None
            and nugget_target_weight > 0
        ):
            units_in_plan = floor(remaining_nugget_weight / nugget_target_weight)
            total_weight_in_plan = units_in_plan * nugget_target_weight
        else:
            units_in_plan = units_per_cut
            total_weight_in_plan = units_per_cut * target_weight

        unit_plan.append({
            "sku": trade_number,
            "partCode": part_code,
            "unitsInPlan": units_in_plan,
            "totalWeightInPlan": total_weight_in_plan,
        })
        planned_mix_weight += total_weight_in_plan
        if sku.get("customerType") == "FDS":
            planned_fds_weight += total_weight_in_plan
        elif sku.get("customerType") == "RTL":
            planned_rtl_weight += total_weight_in_plan

        if trade_number not in seen_trade_numbers:
            seen_trade_numbers.append(trade_number)

    sku_keys = seen_trade_numbers
    total_product_produced_grams = float(planned_mix_weight)
    if total_product_produced_grams > 0:
        for item in unit_plan:
            item["pctOfTotal"] = round((item["totalWeightInPlan"] / total_product_produced_grams) * 100.0, 2)
    else:
        for item in unit_plan:
            item["pctOfTotal"] = 0.0

    min_weight = _bucket_min_weight(bucket)
    target_weight = _bucket_target_weight(bucket) * (1.0 - config_values.get("tolerance_pct", 0.0))
    upgrade_mu = config_values.get("upgrade_mu", 0.0)
    upgrade_sigma = config_values.get("upgrade_sigma", 0.0)
    max_weight = bucket["maxWeight"]

    # --- upgradePercentage ---
    if upgrade_mu > 0 and upgrade_sigma > 0 and max_weight > min_weight:
        try:
            bucket_mean = calculate_bucket_mean(
                lb=min_weight,
                ub=max_weight,
                mu=upgrade_mu,
                sigma=upgrade_sigma,
            )

            upgrade_percentage = (total_product_produced_grams / bucket_mean) * 100.0
            trim_weight = max(0.0, total_product_produced_grams - bucket_mean)

        except ValueError:
            if target_weight > 0:
                upgrade_percentage = total_product_produced_grams / target_weight
            else:
                upgrade_percentage = 0.0
    else:
        upgrade_percentage = total_product_produced_grams / target_weight
        trim_weight = max(0.0, total_product_produced_grams - target_weight)

    # --- trimPercentage ---
    trim_percentage = 100.0 - upgrade_percentage

    # --- value ---
    value = (
        planned_fds_weight * config_values.get("fds_value", 0.0)
        + planned_rtl_weight * config_values.get("rtl_value", 0.0)
        + trim_weight * config_values.get("trim_value", 0.0)
    )

    return {
        "_id": f"{mix_id}:{bucket['_id']}",
        "mixId": mix_id,
        "bucketId": bucket["_id"],
        "upgradePercentage": upgrade_percentage,
        "value": value,
        "trimPercentage": trim_percentage,
        "unitPlan": unit_plan,
        "totalProductProducedGrams": total_product_produced_grams,
        "skuKeys": sku_keys,
    }


# ---------------------------------------------------------------------------
# Phase 7 — Persistence helpers
# ---------------------------------------------------------------------------

def _upsert_mix(mix_repo: MixRepository, mix_doc: Dict[str, Any]) -> str:
    """
    Upsert a Mix document into the repository.

    Searches for an existing Mix by ``skuSetKey`` + ``mfgType`` (the unique
    compound index on the ``mixes`` collection).  If a matching document is
    found, it is updated in place; otherwise a new document is created.

    The ``skuSetKey`` is derived from ``mix_doc["skus"]`` as
    ``"|".join(sorted(skus_map.keys()))`` if not already present on the doc.

    Args:
        mix_repo: MixRepository instance for database access.
        mix_doc: Mix document dict (as produced by ``_build_mix``).

    Returns:
        The ``_id`` of the upserted Mix document.
    """
    # Ensure skuSetKey is present on the document
    if "skuSetKey" not in mix_doc:
        mix_doc["skuSetKey"] = "|".join(sorted(mix_doc.get("skuKeys", mix_doc["skus"].keys())))

    sku_set_key = mix_doc["skuSetKey"]
    mfg_type = mix_doc["mfgType"]

    existing = mix_repo.search({"skuSetKey": sku_set_key, "mfgType": mfg_type})

    if existing:
        existing_doc = existing[0]
        mix_id = existing_doc["_id"]
        if _doc_without_id(existing_doc) != _doc_without_id(mix_doc):
            mix_doc["_id"] = mix_id
            mix_repo.update(mix_id, mix_doc)
        return mix_id
    else:
        created = mix_repo.create(mix_doc)
        return created["_id"]


def _upsert_mix_metric(metric_repo: MixMetricRepository, metric_doc: Dict[str, Any]) -> None:
    """
    Upsert a MixMetric document into the repository.

    Attempts to create the document.  If a ``DuplicateKeyError`` is raised
    (meaning a metric with the same composite ``_id`` of ``mixId:bucketId``
    already exists), falls back to updating the existing document.

    Args:
        metric_repo: MixMetricRepository instance for database access.
        metric_doc: MixMetric document dict (as produced by
            ``_compute_mix_metric``).
    """
    from pymongo.errors import DuplicateKeyError

    try:
        metric_repo.create(metric_doc)
    except DuplicateKeyError:
        existing_doc = metric_repo.get_by_id(metric_doc["_id"])
        if existing_doc is None or _doc_without_id(existing_doc) != _doc_without_id(metric_doc):
            metric_repo.update(metric_doc["_id"], metric_doc)


# ---------------------------------------------------------------------------
# Remaining phases — stubs (implemented in subsequent tasks)
# ---------------------------------------------------------------------------

def run_enumeration(
    db: Database,
    job_id: str,
    run_id: str,
    job_repo,
    max_combination_size: int = 4,
    batch_size: int = 1000,
    plant_filter: Optional[str] = None,
    bird_size_filter: Optional[str] = None,
) -> None:
    """
    Orchestrate the full enumeration pipeline.

    Phases:
    1. Load reference data (SKUs, cut strategies, buckets, config values)
    2. For each stage (combination size 1..max_combination_size):
       a. Generate all combinations_with_replacement filtered to at-most-one nugget
       b. For each combo: find valid cut strategies, build mix, check bucket fit,
          compute metrics, persist if at least one bucket fits
       c. Checkpoint progress every batch_size combos
       d. Mark stage complete
    3. Cancellation is checked at the start of each batch

    Does NOT call mark_completed or mark_failed — those are the job service's responsibility.

    Raises on unrecoverable errors so the job service thread wrapper can call
    job_repo.mark_failed.

    Args:
        db: PyMongo Database handle.
        job_id: Unique job identifier.
        run_id: Unique run identifier.
        job_repo: JobRepository instance for progress reporting.
        max_combination_size: Maximum SKU combination size (1–4).
        batch_size: Number of combinations per progress checkpoint.
        plant_filter: Optional plant filter applied to SKU loading.
        bird_size_filter: Optional bird-size filter applied to SKU loading.
    """
    from config import get_settings

    global_config_url = get_settings().global_config_api_url

    # Phase 1 — Load reference data
    skus = _load_candidate_skus(db, plant_filter, bird_size_filter)
    cut_strategies = _load_cut_strategies(db)
    buckets = _load_buckets(db)
    config_values = _fetch_config_values(global_config_url)

    if not cut_strategies:
        logger.warning("No cut strategies loaded; all combinations will be skipped.")
    if not buckets:
        logger.warning("No buckets loaded; no metrics will be produced.")

    sku_count = len(skus)
    mix_repo = MixRepository(db)
    metric_repo = MixMetricRepository(db)

    # Build initial stage list for mark_running
    initial_stage_list = [
        {"stage": stage, "status": "pending"}
        for stage in range(1, max_combination_size + 1)
    ]
    job_repo.mark_running(job_id, sku_count, initial_stage_list)

    # Phase 2 — Stage loop
    for stage in range(1, max_combination_size + 1):
        stage_index = stage  # 1-based

        # Generate all combos for this stage, filtered to at-most-one nugget
        all_combos = [
            c for c in itertools.combinations_with_replacement(skus, stage)
            if sum(1 for s in c if s["productType"] == "NUGGET") <= 1
        ]
        total = len(all_combos)

        job_repo.mark_stage_running(job_id, stage_index, total)

        for i, combo in enumerate(all_combos):
            # Cancellation check at start of each batch
            if i % batch_size == 0 and job_repo.is_cancelled(job_id):
                return

            # Require every persisted mix to include at least one fillet SKU.
            if not _combo_has_fillet(combo):
                continue

            valid_strategies = _get_valid_cut_strategies(combo, cut_strategies)

            for strategy in valid_strategies:
                mix_doc = _build_mix(combo, strategy, plant_filter, bird_size_filter)
                part_assignments = mix_doc.pop("_partAssignments", None)
                mix_weight = sum(s["targetWeight"] for s in combo)

                # Check bucket capacity and compute metrics.
                fitting_metrics = []
                for bucket in buckets:
                    if _fits_bucket(mix_weight, bucket, config_values["tolerance_pct"]):
                        metric = _compute_mix_metric(
                            None,
                            combo,
                            mix_doc["skus"],
                            bucket,
                            mix_doc["includesNug"],
                            mix_doc.get("nuggetTargetWeight"),
                            config_values,
                            part_assignments=part_assignments,
                        )
                        fitting_metrics.append(metric)

                # Only persist if at least one bucket can accommodate the mix.
                if fitting_metrics:
                    mix_id = _upsert_mix(mix_repo, mix_doc)
                    for metric_doc in fitting_metrics:
                        metric_doc["mixId"] = mix_id
                        metric_doc["_id"] = f"{mix_id}:{metric_doc['bucketId']}"
                        _upsert_mix_metric(metric_repo, metric_doc)

            # Checkpoint at end of each batch (batch_size - 1 index)
            if i % batch_size == batch_size - 1:
                job_repo.checkpoint_stage(job_id, stage_index, i + 1)

        job_repo.mark_stage_complete(job_id, stage_index, total)
