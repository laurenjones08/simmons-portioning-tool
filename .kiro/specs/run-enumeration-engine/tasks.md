# Implementation Plan

- [x] 1. Add `global_config_api_url` setting to worker config
  - Add `global_config_api_url: str` field to `Settings` in `enumeration-worker-api/config.py` with default `"http://global-config-api:8001"`
  - Add `requests` to `enumeration-worker-api/requirements.txt` if not already present
  - _Requirements: 6.4, 7.2_

- [x] 2. Implement reference data loaders
  - Implement `_load_candidate_skus(db, plant_filter, bird_size_filter)` using `SKURepository`; raise `ValueError` when result is empty
  - Implement `_load_cut_strategies(db)` using `CutStrategyRepository`
  - Implement `_load_buckets(db)` using `BucketRepository`
  - Implement `_fetch_config_values(global_config_url)` fetching all four config keys (`enumeration.bucketWeightTolerancePct`, `enumeration.fdsValueCoefficient`, `enumeration.rtlValueCoefficient`, `enumeration.trimValueCoefficient`) via HTTP GET; default each to `0.0` on any error and log a warning
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 6.1, 6.4, 7.2_

- [x] 2.1. Write property tests for reference data loaders
  - **Property 1**: `_load_candidate_skus` with `plant_filter` returns only SKUs where `prodPlant` matches
  - **Property 2**: `_load_candidate_skus` with `bird_size_filter` returns only SKUs where `birdSize` matches
  - **Property 3**: `_fetch_config_values` returns a dict with all four keys; each value is a float
  - **Property 4**: `_fetch_config_values` returns `0.0` for all keys when the config API is unreachable
  - _Requirements: 2.2, 2.3, 6.4_

- [x] 3. Implement cut strategy validation
  - Implement `_get_valid_cut_strategies(combo, cut_strategies)` as a pure function
  - A strategy is valid when every part code in `strategy["parts"]` appears in the `allowedParts` of at least one SKU in the combo
  - Filter by `hasNugget`: strategies with `hasNugget=True` are only valid for combos containing a nugget SKU; strategies with `hasNugget=False` are only valid for combos with no nugget SKU
  - _Requirements: 4.2, 4.3, 4.4_

- [x] 3.1. Write property tests for cut strategy validation
  - **Property 5**: Every strategy returned by `_get_valid_cut_strategies` has all its `parts` covered by the combo's `allowedParts`
  - **Property 6**: No strategy with `hasNugget=True` is returned for a combo with no nugget SKU, and vice versa
  - **Property 7**: When no strategy is valid, the function returns an empty list
  - _Requirements: 4.2, 4.3, 4.4, 4.5_

- [x] 4. Implement mix construction
  - Implement `_build_mix(combo, strategy, plant_filter, bird_size_filter)` as a pure function
  - Build the `skus` map by assigning each SKU the first part code from `strategy["parts"]` that appears in `sku["allowedParts"]`
  - Derive all Mix fields per the design: `cutStrategyID`, `mfgType`, `beltSpeed`, `includesFDS`, `includesRTL`, `includesNug`, `nuggetTargetWeight`, `reqPlant`, `reqBirdSize`, `numFillets`, `filletWeight`, `skuKeys`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 5.13_

- [x] 4.1. Write property tests for mix construction
  - **Property 8**: For every SKU in the returned `skus` map, the assigned part code appears in that SKU's `allowedParts`
  - **Property 9**: `includesNug` is `True` if and only if the combo contains a nugget SKU and `strategy["hasNugget"]` is `True`
  - **Property 10**: `numFillets` equals the count of non-nugget SKUs; `filletWeight` equals the sum of their `targetWeight` values
  - **Property 11**: `reqPlant` equals `plant_filter` when provided, otherwise `combo[0]["prodPlant"]`
  - _Requirements: 5.1, 5.7, 5.8, 5.10, 5.12, 5.13_

- [x] 5. Implement bucket fitting
  - Implement `_fits_bucket(mix_weight, bucket, tolerance_pct)` as a pure function
  - Compute `effective_min = bucket["minWeight"] * (1 - tolerance_pct / 100)`
  - Return `True` when `effective_min <= mix_weight <= bucket["maxWeight"]`
  - _Requirements: 6.5, 6.6_

- [x] 5.1. Write property tests for bucket fitting
  - **Property 12**: A mix weight exactly equal to `effective_min` fits the bucket
  - **Property 13**: A mix weight exactly equal to `bucket.maxWeight` fits the bucket
  - **Property 14**: A mix weight below `effective_min` does not fit the bucket
  - **Property 15**: A mix weight above `bucket.maxWeight` does not fit the bucket
  - **Property 16**: With `tolerance_pct = 0`, `effective_min == bucket.minWeight`
  - _Requirements: 6.5, 6.6_

- [x] 6. Implement mix metric computation
  - Implement `_compute_mix_metric(mix_id, combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values)` as a pure function
  - Compute `upgradePercentage`, `value`, and `trimPercentage` per the design formulas
  - Build `unitPlan`: one item per SKU entry in the combo; when `includesNug` is `True`, override the nugget SKU's `unitsInPlan` to `floor(bucket.minWeight / nuggetTargetWeight)` and update `totalWeightInPlan` accordingly
  - Set `skuKeys` to `tradeNumber` values in first-appearance order matching `unitPlan`
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 6.1. Write property tests for mix metric computation
  - **Property 17**: `trimPercentage` is `0.0` when `mix_weight <= bucket.minWeight`
  - **Property 18**: `upgradePercentage` is in `[0.0, 100.0]` for any valid combo and bucket
  - **Property 19**: `value = fds_weight * fds_value + rtl_weight * rtl_value + trim_weight * trim_value`
  - **Property 20**: When `includesNug` is `True`, the nugget SKU's `unitsInPlan` equals `floor(bucket.minWeight / nuggetTargetWeight)`
  - **Property 21**: When `includesNug` is `False`, all SKUs have `unitsInPlan == unitsPerCut`
  - **Property 22**: `skuKeys` exactly matches the `tradeNumber` values in `unitPlan` in first-appearance order
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 7. Implement persistence helpers
  - Implement `_upsert_mix(mix_repo, mix_doc)`: search by `skuSetKey` + `mfgType`; call `create` if not found, `update` if found; return the `_id`
  - Implement `_upsert_mix_metric(metric_repo, metric_doc)`: call `create`; on `DuplicateKeyError` fall back to `update`
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 7.1. Write property tests for persistence helpers
  - **Property 23**: Calling `_upsert_mix` twice with the same SKU set and `mfgType` results in exactly one document in the collection
  - **Property 24**: Calling `_upsert_mix_metric` twice with the same `mixId` + `bucketId` results in exactly one document in the collection
  - _Requirements: 8.4, 8.5_

- [x] 8. Implement the main `run_enumeration` orchestrator
  - Implement the full `run_enumeration` function wiring all helpers together
  - Use `itertools.combinations_with_replacement` for combo generation; filter out combos with more than one nugget SKU
  - Only persist a Mix + its metrics when at least one bucket fits
  - Call `job_repo.mark_running`, `mark_stage_running`, `checkpoint_stage`, and `mark_stage_complete` at the correct points
  - Check `job_repo.is_cancelled(job_id)` at the start of each batch; return early if cancelled
  - Do NOT call `mark_completed` or `mark_failed` — leave those to the job service wrapper
  - _Requirements: 1.1, 1.2, 1.3, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 4.5, 4.6, 6.7, 8.3, 8.6, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 10.1, 10.2, 10.3_

- [x] 8.1. Write property tests for the orchestrator
  - **Property 25**: Every persisted Mix has at least one associated MixMetric (no bucket-less mixes)
  - **Property 26**: No combination contains more than one nugget SKU
  - **Property 27**: No SKU appears more than 3 times in any combination
  - **Property 28**: For any given SKU set and `mfgType`, at most one Mix document exists after a run
  - **Property 29**: `processedCombinations` checkpointed to `job_repo` is non-decreasing within a stage
  - _Requirements: 3.2, 3.3, 6.7, 8.3, 8.4, 9.3_

- [x] 9. Final checkpoint — ensure all tests pass
  - Run the full property-based test suite for `enumeration_engine.py`
  - Verify the engine integrates correctly with the job service by submitting a test job end-to-end
  - Ensure all tests pass; ask the user if questions arise
