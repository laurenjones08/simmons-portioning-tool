# Design Document

## Overview

`run_enumeration` is a synchronous, multi-stage pipeline function that runs in a background thread. It orchestrates five logical phases: SKU loading → combination generation → cut strategy validation → mix/metric computation → result persistence. Each phase is implemented as a focused helper function. All data access goes through the existing repository classes in `enumeration-api/repositories/`. Job progress is reported to `JobRepository` throughout.

Key design constraints:
- A Mix is only persisted if it fits in **at least one** bucket. Mixes that fit no bucket are discarded entirely.
- SKU combinations are generated using `itertools.combinations_with_replacement` to allow the same SKU to appear up to 3 times (cut from different part codes). Each combination may contain **at most one nugget SKU**.
- The `value` metric is a weighted sum of FDS weight, RTL weight, and trim weight, each multiplied by a coefficient fetched from the Global Config API.

---

## Architecture

```
job_service.py (background thread)
    └── run_enumeration(db, job_id, run_id, job_repo, ...)
            ├── _load_candidate_skus(db, plant_filter, bird_size_filter)
            ├── _load_cut_strategies(db)
            ├── _load_buckets(db)
            ├── _fetch_config_values(global_config_url)
            │
            └── for stage in 1..max_combination_size:
                    └── for combo in combinations_with_replacement(skus, stage)
                            │   filtered to at-most-one nugget SKU
                            ├── _get_valid_cut_strategies(combo, cut_strategies)
                            └── for strategy in valid_strategies:
                                    ├── _build_mix(combo, strategy, filters)
                                    ├── fitting_metrics = []
                                    └── for bucket in buckets:
                                            ├── _fits_bucket(mix_weight, bucket, tol_pct)
                                            └── _compute_mix_metric(mix, bucket, config_values)
                                    └── if fitting_metrics:
                                            ├── _upsert_mix(mix_repo, mix_doc)
                                            └── _upsert_mix_metric(metric_repo, metric_doc) × N
```

The worker API shares the same MongoDB instance as the Enumeration API (`enumeration_db`). Repositories are instantiated directly from the shared `db` handle — no HTTP calls are made to the Enumeration API service for persistence.

The Global Config API **is** called over HTTP (via `requests`) to fetch runtime config values (tolerance percentage and value coefficients), since it is a separate service with its own connection. All config values are fetched once at startup and cached for the duration of the job.

---

## Module Structure

```
enumeration-worker-api/
    enumeration_engine.py        ← all logic lives here
    config.py                    ← adds global_config_api_url setting
```

No new files are needed. All helper functions are module-level private functions (`_` prefix) inside `enumeration_engine.py`.

---

## Data Flow

### Phase 1 — Load Reference Data

```python
skus: List[Dict]           = _load_candidate_skus(db, plant_filter, bird_size_filter)
cut_strategies: List[Dict] = _load_cut_strategies(db)
buckets: List[Dict]        = _load_buckets(db)
config_values: Dict        = _fetch_config_values(global_config_url)
# config_values keys:
#   "tolerance_pct"  → enumeration.bucketWeightTolerancePct  (float, default 0.0)
#   "fds_value"      → enumeration.fdsValueCoefficient       (float, default 0.0)
#   "rtl_value"      → enumeration.rtlValueCoefficient       (float, default 0.0)
#   "trim_value"     → enumeration.trimValueCoefficient      (float, default 0.0)
```

All collections and config values are loaded once into memory before the combination loop begins.

### Phase 2 — Combination Generation & Stage Tracking

Combinations are generated using `itertools.combinations_with_replacement` so the same SKU can appear up to 3 times in a combo (representing cuts from different part codes). Each combo is filtered to ensure it contains **at most one nugget SKU** before processing.

```python
for stage in range(1, max_combination_size + 1):
    all_combos = [
        c for c in itertools.combinations_with_replacement(skus, stage)
        if sum(1 for s in c if s["productType"] == "NUGGET") <= 1
    ]
    total = len(all_combos)
    job_repo.mark_stage_running(job_id, stage_index, total)

    for i, combo in enumerate(all_combos):
        if i % batch_size == 0 and job_repo.is_cancelled(job_id):
            return   # graceful cancellation

        valid_strategies = _get_valid_cut_strategies(combo, cut_strategies)
        for strategy in valid_strategies:
            mix_doc = _build_mix(combo, strategy, plant_filter, bird_size_filter)
            mix_weight = sum(s["targetWeight"] for s in combo)
            fitting_metrics = []
            for bucket in buckets:
                if _fits_bucket(mix_weight, bucket, config_values["tolerance_pct"]):
                    fitting_metrics.append(
                        _compute_mix_metric(None, combo, mix_doc["skus"], bucket,
                                            mix_doc["includesNug"], mix_doc.get("nuggetTargetWeight"),
                                            config_values)
                    )
            # Only persist if the mix fits at least one bucket
            if fitting_metrics:
                mix_id = _upsert_mix(mix_repo, mix_doc)
                for metric_doc in fitting_metrics:
                    metric_doc["mixId"] = mix_id
                    metric_doc["_id"] = f"{mix_id}:{metric_doc['bucketId']}"
                    _upsert_mix_metric(metric_repo, metric_doc)

        if i % batch_size == batch_size - 1:
            job_repo.checkpoint_stage(job_id, stage_index, i + 1)

    job_repo.mark_stage_complete(job_id, stage_index, total)
```

### Phase 3 — Cut Strategy Validation

```python
def _get_valid_cut_strategies(
    combo: List[Dict],
    cut_strategies: List[Dict],
) -> List[Dict]:
```

A strategy is valid for a combo when **every** part code in `strategy["parts"]` is present in the `allowedParts` of **at least one** SKU in the combo. This is a pure set-intersection check with no I/O.

Additionally, if the combo contains a nugget SKU (`productType == "NUGGET"`), only strategies where `hasNugget == True` are considered valid. If the combo has no nugget SKU, only strategies where `hasNugget == False` are considered valid.

### Phase 4 — Mix Construction

```python
def _build_mix(
    combo: List[Dict],
    strategy: Dict,
    plant_filter: Optional[str],
    bird_size_filter: Optional[str],
) -> Dict:
```

**SKU → Part Code assignment**: For each SKU in the combo, iterate through `strategy["parts"]` and assign the first part code that appears in `sku["allowedParts"]`. This produces the `skus` map `{tradeNumber: partCode}`.

**Derived fields**:

| Field | Logic |
|---|---|
| `cutStrategyID` | `strategy["_id"]` |
| `mfgType` | `strategy["mfgType"]` |
| `beltSpeed` | `strategy["beltSpeed"]` |
| `includesFDS` | `any(s["customerType"] == "FDS" for s in combo)` |
| `includesRTL` | `any(s["customerType"] == "RTL" for s in combo)` |
| `includesNug` | `any(s["productType"] == "NUGGET" for s in combo) and strategy["hasNugget"]` |
| `nuggetTargetWeight` | nugget SKU's `targetWeight` if `includesNug` else `None` |
| `reqPlant` | `plant_filter` if set, else `combo[0]["prodPlant"]` |
| `reqBirdSize` | `bird_size_filter` if set, else `combo[0]["birdSize"]` |
| `numFillets` | count of non-nugget SKUs in combo |
| `filletWeight` | sum of `targetWeight` for non-nugget SKUs |
| `skuKeys` | `[s["tradeNumber"] for s in combo]` |

### Phase 5 — Bucket Fitting & Metric Computation

```python
def _fits_bucket(mix_weight: float, bucket: Dict, tolerance_pct: float) -> bool:
    effective_min = bucket["minWeight"] * (1 - tolerance_pct / 100)
    return effective_min <= mix_weight <= bucket["maxWeight"]
```

```python
def _compute_mix_metric(
    mix_id: Optional[str],
    combo: List[Dict],
    skus_map: Dict[str, str],
    bucket: Dict,
    includes_nug: bool,
    nugget_target_weight: Optional[float],
    config_values: Dict[str, float],
) -> Dict:
```

**Metric formulas**:

| Field | Formula |
|---|---|
| `upgradePercentage` | `(count of SKUs where targetWeight > bucket.minWeight) / len(combo) * 100` |
| `value` | `fds_weight * fds_value + rtl_weight * rtl_value + trim_weight * trim_value` |
| `trimPercentage` | `((mix_weight - bucket.minWeight) / mix_weight) * 100` if `mix_weight > bucket.minWeight` else `0.0` |

**`value` calculation**:
- `fds_weight` = sum of `targetWeight` for all SKUs in the combo where `customerType == "FDS"`
- `rtl_weight` = sum of `targetWeight` for all SKUs in the combo where `customerType == "RTL"`
- `trim_weight` = `max(0.0, mix_weight - bucket.minWeight)`
- `fds_value`, `rtl_value`, `trim_value` come from `config_values`

**Unit Plan construction**:
- One `UnitPlanItem` per SKU entry in the combo (including repeated SKUs): `sku=tradeNumber`, `partCode=skus_map[tradeNumber]`, `unitsInPlan=sku["unitsPerCut"]`, `totalWeightInPlan=unitsPerCut * targetWeight`
- When `includesNug` is `True` and `nuggetTargetWeight > 0`: override the nugget SKU's `unitsInPlan` to `floor(bucket.minWeight / nuggetTargetWeight)` and `totalWeightInPlan = unitsInPlan * nuggetTargetWeight`. All non-nugget SKUs retain `unitsInPlan = unitsPerCut`.

`skuKeys` is set to the list of `tradeNumber` values in first-appearance order (matching `unitPlan`).

The composite `_id` is `f"{mix_id}:{bucket['_id']}"`.

### Phase 6 — Persistence (Upsert)

**Mix upsert**: Search by `skuSetKey` + `mfgType` (the unique index on `MixRepository`). If found, call `update`; if not, call `create`. The `skuSetKey` is built as `"|".join(sorted(skus_map.keys()))`.

**MixMetric upsert**: The `_id` is the composite `mixId:bucketId`. Call `create` and catch `DuplicateKeyError`, then fall back to `update`.

---

## Key Functions

### `run_enumeration`

```python
def run_enumeration(
    db: Database,
    job_id: str,
    run_id: str,
    job_repo: JobRepository,
    max_combination_size: int = 4,
    batch_size: int = 1000,
    plant_filter: Optional[str] = None,
    bird_size_filter: Optional[str] = None,
) -> None:
```

Orchestrates all phases. Raises on unrecoverable errors so the job service thread wrapper can call `job_repo.mark_failed`.

### `_load_candidate_skus`

Uses `SKURepository(db).find_by_criteria(query)`. Raises `ValueError` if result is empty.

### `_load_cut_strategies`

Uses `CutStrategyRepository(db).search({})`.

### `_load_buckets`

Uses `BucketRepository(db).search({})`.

### `_fetch_config_values`

```python
def _fetch_config_values(global_config_url: str) -> Dict[str, float]:
```

Fetches all four runtime config values from the Global Config API in a single pass. Returns a dict with keys `tolerance_pct`, `fds_value`, `rtl_value`, `trim_value`. Each individual fetch falls back to `0.0` on any error (connection error, 404, parse error). Logs a warning for each fallback.

| Dict key | Config API key |
|---|---|
| `tolerance_pct` | `enumeration.bucketWeightTolerancePct` |
| `fds_value` | `enumeration.fdsValueCoefficient` |
| `rtl_value` | `enumeration.rtlValueCoefficient` |
| `trim_value` | `enumeration.trimValueCoefficient` |

### `_get_valid_cut_strategies`

Pure function. No I/O. Returns the subset of `cut_strategies` that are valid for the given combo.

### `_build_mix`

Pure function. Returns a dict ready for `MixRepository`.

### `_fits_bucket`

Pure function. Returns `True` when `effective_min <= mix_weight <= bucket.maxWeight`.

### `_compute_mix_metric`

Pure function. Returns a dict ready for `MixMetricRepository`.

### `_upsert_mix`

```python
def _upsert_mix(mix_repo: MixRepository, mix_doc: Dict[str, Any]) -> str:
```

Returns the `_id` of the upserted mix.

### `_upsert_mix_metric`

```python
def _upsert_mix_metric(metric_repo: MixMetricRepository, metric_doc: Dict[str, Any]) -> None:
```

---

## Configuration

`config.py` gains one new field:

```python
global_config_api_url: str = Field(
    default="http://global-config-api:8001",
    description="Base URL of the Global Config API for fetching runtime config values",
)
```

Constants in `enumeration_engine.py`:

```python
BUCKET_TOLERANCE_CONFIG_KEY = "enumeration.bucketWeightTolerancePct"
FDS_VALUE_CONFIG_KEY        = "enumeration.fdsValueCoefficient"
RTL_VALUE_CONFIG_KEY        = "enumeration.rtlValueCoefficient"
TRIM_VALUE_CONFIG_KEY       = "enumeration.trimValueCoefficient"
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| No SKUs match filters | Raise `ValueError`; job service marks job failed |
| No cut strategies loaded | Log warning; all combos will be skipped (no mixes produced) |
| No buckets loaded | Log warning; no metrics produced |
| Global Config API unreachable | Log warning; default all config values to `0.0` |
| Mix upsert DB error | Raise; job service marks job failed |
| Metric upsert DB error | Raise; job service marks job failed |
| Job cancelled | Return early; no exception raised |

---

## Correctness Properties

The following properties must hold and will be validated by property-based tests:

1. **Cut strategy validity**: For every Mix produced, every part code in the assigned cut strategy's `parts` list appears in the `allowedParts` of at least one SKU in the combination.

2. **SKU-to-part assignment coverage**: For every SKU in a Mix's `skus` map, the assigned part code appears in that SKU's `allowedParts`.

3. **Nugget compatibility**: `includesNug == True` if and only if the combo contains a nugget SKU AND the cut strategy has `hasNugget == True`.

4. **Bucket fit invariant**: Every MixMetric produced satisfies `effective_min <= mix_weight <= bucket.maxWeight` where `effective_min = bucket.minWeight * (1 - tolerance_pct / 100)`.

5. **No bucket-less mixes**: Every persisted Mix has at least one associated MixMetric. No Mix document exists without a corresponding MixMetric.

6. **Unit plan SKU coverage**: The `skuKeys` on every MixMetric exactly matches the `tradeNumber` values in the `unitPlan`, in first-appearance order.

7. **Nugget unit plan logic**: When `includesNug == True`, the nugget SKU's `unitsInPlan` equals `floor(bucket.minWeight / nuggetTargetWeight)`. When `includesNug == False`, all SKUs use `unitsInPlan = unitsPerCut`.

8. **Value metric composition**: `value = fds_weight * fds_value + rtl_weight * rtl_value + trim_weight * trim_value` where weights are derived from the combo and trim is the excess above `bucket.minWeight`.

9. **Metric percentage bounds**: `upgradePercentage` and `trimPercentage` are in `[0.0, 100.0]`.

10. **At-most-one nugget per combo**: No generated combination contains more than one SKU with `productType == "NUGGET"`.

11. **Repeated SKU limit**: No SKU appears more than 3 times in any combination.

12. **No duplicate mixes**: For any given SKU set and `mfgType`, at most one Mix document exists after enumeration completes.

13. **No duplicate metrics**: For any given `mixId` + `bucketId` pair, at most one MixMetric document exists after enumeration completes.

14. **Stage progress monotonicity**: `processedCombinations` reported to `job_repo` is non-decreasing within a stage.

15. **Cancellation safety**: If `is_cancelled` returns `True` at batch boundary `k`, no writes occur for batches `> k`.
