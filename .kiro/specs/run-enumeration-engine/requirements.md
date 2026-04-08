# Requirements Document

## Introduction

This document specifies the requirements for the `run_enumeration` function in the Enumeration Worker API. This function is the core background worker that executes a multi-stage enumeration pipeline. It is invoked in a background thread by the job service and is responsible for generating all valid SKU combinations, assigning cut strategies, computing mix and bucket metrics, persisting results via the Enumeration API, and reporting job progress throughout.

## Glossary

- **SKU**: Stock Keeping Unit — a product with fields including `tradeNumber`, `birdSize`, `prodPlant`, `allowedParts`, `targetWeight`, `minWeight`, `maxWeight`, `productType`, `customerType`, and `unitsPerCut`
- **Combination / Combo**: A multiset of 1–4 SKUs selected from the filtered candidate pool; the same SKU may appear up to 3 times (representing cuts from different part codes); each combo may contain at most one nugget SKU
- **Part Code**: A single-letter code (e.g. `D`, `R`, `M`) representing a cut type; each SKU declares which part codes it can be produced from via its `allowedParts` field
- **Cut Strategy**: A named configuration stored in the `cut_strategies` collection that defines a set of part codes (`parts`), a manufacturing type (`mfgType`), a belt speed (`beltSpeed`), and whether it produces nuggets (`hasNugget`)
- **Valid Cut Strategy**: A cut strategy whose `parts` list is a subset of the union of `allowedParts` across all SKUs in the combination, and where every part code in the strategy is covered by at least one SKU in the combination; additionally, `hasNugget` must match whether the combo contains a nugget SKU
- **Mix**: A document in the `mixes` collection representing a unique pairing of a SKU combination with a cut strategy; fields include `skus` (map of tradeNumber → partCode), `cutStrategyID`, `mfgType`, `beltSpeed`, `includesFDS`, `includesRTL`, `includesNug`, `nuggetTargetWeight`, `numFillets`, `filletWeight`, `reqPlant`, and `reqBirdSize`
- **Bucket**: A weight range document in the `buckets` collection with `minWeight` and `maxWeight` fields
- **Mix Metric**: A document in the `mix_metrics` collection that records computed metrics (`upgradePercentage`, `value`, `trimPercentage`, `unitPlan`) for a specific Mix + Bucket pairing
- **Unit Plan Item**: A per-SKU entry in the `unitPlan` array of a MixMetric, containing `sku` (tradeNumber), `partCode`, `unitsInPlan`, and `totalWeightInPlan`
- **Nugget SKU**: A SKU whose `productType` is `"NUGGET"`
- **FDS SKU**: A SKU whose `customerType` is `"FDS"`
- **RTL SKU**: A SKU whose `customerType` is `"RTL"`
- **Tolerance Percentage**: A float configuration value fetched from the Global Config API under key `enumeration.bucketWeightTolerancePct` that is subtracted as a percentage from a bucket's `minWeight` before checking fit; defaults to `0.0`
- **Value Coefficients**: Float configuration values fetched from the Global Config API — `enumeration.fdsValueCoefficient`, `enumeration.rtlValueCoefficient`, `enumeration.trimValueCoefficient` — used to compute the `value` metric
- **Job Repository**: The `JobRepository` class used to persist and update job status documents in the `job_status` collection
- **Stage**: One iteration of the enumeration loop corresponding to a combination size (1 through `maxCombinationSize`)

---

## Requirements

### Requirement 1

**User Story:** As the job service, I want `run_enumeration` to accept a standard set of parameters, so that it can be invoked consistently from the background thread.

#### Acceptance Criteria

1. THE `run_enumeration` function SHALL accept the following parameters: `db` (PyMongo Database), `job_id` (str), `run_id` (str), `job_repo` (JobRepository), `max_combination_size` (int, 1–4), `batch_size` (int), `plant_filter` (Optional[str]), and `bird_size_filter` (Optional[str])
2. THE function SHALL be callable from a background thread without requiring any async context
3. THE function SHALL propagate unhandled exceptions to the caller so the job service can mark the job as failed

---

### Requirement 2

**User Story:** As an operator, I want the enumeration engine to load only the relevant SKUs based on the job filters, so that the combination space is scoped correctly.

#### Acceptance Criteria

1. THE engine SHALL query the `skus` collection for candidate SKUs
2. WHEN `plant_filter` is provided, THE engine SHALL include only SKUs where `prodPlant` matches the filter value
3. WHEN `bird_size_filter` is provided, THE engine SHALL include only SKUs where `birdSize` matches the filter value
4. WHEN both filters are provided, THE engine SHALL apply both as AND conditions
5. WHEN no SKUs match the filters, THE engine SHALL mark the job as failed with a descriptive error message and halt execution
6. THE engine SHALL record the total SKU count on the job document via `job_repo.mark_running` before beginning stage iteration

---

### Requirement 3

**User Story:** As an operator, I want the engine to generate all SKU combinations up to the configured maximum size, so that every valid mix candidate is considered.

#### Acceptance Criteria

1. THE engine SHALL generate combinations of sizes 1 through `max_combination_size` (inclusive)
2. THE engine SHALL use multiset (combinations with replacement) selection so that the same SKU may appear up to 3 times in a single combination, representing cuts from different part codes
3. THE engine SHALL exclude any combination that contains more than one nugget SKU (`productType == "NUGGET"`)
4. THE engine SHALL process each combination size as a separate named stage
5. THE engine SHALL update the job stage status to `running` with the total combination count before processing each stage, via `job_repo.mark_stage_running`
6. THE engine SHALL checkpoint progress after each batch of combinations via `job_repo.checkpoint_stage`
7. THE engine SHALL mark each stage as `completed` via `job_repo.mark_stage_complete` when all combinations in that stage have been processed

---

### Requirement 4

**User Story:** As an operator, I want each SKU combination to be validated against available cut strategies, so that only feasible mixes are enumerated.

#### Acceptance Criteria

1. THE engine SHALL load all cut strategies from the `cut_strategies` collection at the start of enumeration
2. FOR each combination, THE engine SHALL determine the set of valid cut strategies by checking that every part code in the strategy's `parts` list is present in the `allowedParts` of at least one SKU in the combination
3. A cut strategy SHALL only be valid for a combination that contains a nugget SKU if the strategy's `hasNugget` field is `true`
4. A cut strategy SHALL only be valid for a combination that contains no nugget SKU if the strategy's `hasNugget` field is `false`
5. WHEN a combination has no valid cut strategies, THE engine SHALL skip that combination entirely and not produce any Mix or MixMetric documents for it
6. WHEN a combination has one or more valid cut strategies, THE engine SHALL produce one Mix candidate per valid cut strategy

---

### Requirement 5

**User Story:** As an operator, I want each valid Combo + Cut Strategy pair to produce a well-formed Mix document, so that the Enumeration API can store and serve it.

#### Acceptance Criteria

1. THE engine SHALL construct the `skus` map by assigning each SKU in the combination to the first part code from the cut strategy's `parts` list that appears in that SKU's `allowedParts`
2. THE engine SHALL set `cutStrategyID` to the `_id` of the matched cut strategy
3. THE engine SHALL set `mfgType` from the cut strategy's `mfgType` field
4. THE engine SHALL set `beltSpeed` from the cut strategy's `beltSpeed` field
5. THE engine SHALL set `includesFDS` to `true` if any SKU in the combination has `customerType == "FDS"`
6. THE engine SHALL set `includesRTL` to `true` if any SKU in the combination has `customerType == "RTL"`
7. THE engine SHALL set `includesNug` to `true` if the combination contains a nugget SKU and the cut strategy has `hasNugget == true`
8. WHEN `includesNug` is `true`, THE engine SHALL set `nuggetTargetWeight` to the `targetWeight` of the nugget SKU in the combination
9. WHEN `includesNug` is `false`, THE engine SHALL set `nuggetTargetWeight` to `null`
10. THE engine SHALL set `reqPlant` to the `plant_filter` value if provided, otherwise to the `prodPlant` of the first SKU in the combination
11. THE engine SHALL set `reqBirdSize` to the `bird_size_filter` value if provided, otherwise to the `birdSize` of the first SKU in the combination
12. THE engine SHALL compute `numFillets` as the count of SKUs in the combination that are not nugget SKUs
13. THE engine SHALL compute `filletWeight` as the sum of `targetWeight` values for all non-nugget SKUs in the combination

---

### Requirement 6

**User Story:** As an operator, I want each Mix to be evaluated against all configured buckets, so that I know which weight ranges it fits into.

#### Acceptance Criteria

1. THE engine SHALL load all buckets from the `buckets` collection at the start of enumeration
2. FOR each Mix candidate, THE engine SHALL iterate over every bucket
3. THE engine SHALL compute the total mix weight as the sum of `targetWeight` for all SKUs in the combination
4. THE engine SHALL fetch the tolerance percentage from the Global Config API using the key `enumeration.bucketWeightTolerancePct`; WHEN the key is not found or the fetch fails, THE engine SHALL default to `0.0`
5. THE engine SHALL compute the effective bucket minimum as: `bucket.minWeight * (1 - tolerancePct / 100)`
6. WHEN the total mix weight is greater than or equal to the effective bucket minimum AND less than or equal to `bucket.maxWeight`, THE engine SHALL consider the mix as fitting the bucket and SHALL compute a MixMetric for that bucket
7. WHEN the total mix weight does not fit any bucket, THE engine SHALL discard the Mix entirely — no Mix or MixMetric documents SHALL be persisted for it

---

### Requirement 7

**User Story:** As an operator, I want the engine to compute accurate MixMetric values for each Mix + Bucket pairing, so that downstream portioning decisions are well-informed.

#### Acceptance Criteria

1. THE engine SHALL compute `upgradePercentage` as the percentage of SKUs in the combination whose `targetWeight` exceeds the bucket's `minWeight` divided by the total number of SKUs, multiplied by 100
2. THE engine SHALL compute `value` as: `(fds_weight * fds_value) + (rtl_weight * rtl_value) + (trim_weight * trim_value)` where:
   - `fds_weight` = sum of `targetWeight` for SKUs with `customerType == "FDS"`
   - `rtl_weight` = sum of `targetWeight` for SKUs with `customerType == "RTL"`
   - `trim_weight` = `max(0, mix_weight - bucket.minWeight)`
   - `fds_value`, `rtl_value`, `trim_value` are fetched from the Global Config API (keys: `enumeration.fdsValueCoefficient`, `enumeration.rtlValueCoefficient`, `enumeration.trimValueCoefficient`); each defaults to `0.0` if not found
3. THE engine SHALL compute `trimPercentage` as `((mix_weight - bucket.minWeight) / mix_weight) * 100` when `mix_weight > bucket.minWeight`, otherwise `0.0`
4. THE engine SHALL build the `unitPlan` array with one `UnitPlanItem` per SKU entry in the combination (including repeated SKUs), setting `sku` to the SKU's `tradeNumber`, `partCode` to the assigned part code from the `skus` map, `unitsInPlan` to the SKU's `unitsPerCut`, and `totalWeightInPlan` to `unitsPerCut * targetWeight`
5. WHEN `includesNug` is `true` and a nugget SKU exists in the combination, THE engine SHALL override the nugget SKU's `unitsInPlan` to `floor(bucket.minWeight / nuggetTargetWeight)` and set `totalWeightInPlan` to `unitsInPlan * nuggetTargetWeight`; all non-nugget SKUs SHALL retain `unitsInPlan = unitsPerCut`
6. THE engine SHALL set `skuKeys` on the MixMetric to the list of `tradeNumber` values from the combination, in first-appearance order matching the `unitPlan`

---

### Requirement 8

**User Story:** As an operator, I want the engine to persist Mix and MixMetric documents via the Enumeration API repositories, so that results are stored in a consistent and validated way.

#### Acceptance Criteria

1. THE engine SHALL upsert each Mix document using `MixRepository` (not raw MongoDB writes)
2. THE engine SHALL upsert each MixMetric document using `MixMetricRepository` (not raw MongoDB writes)
3. THE engine SHALL only persist a Mix if it fits in at least one bucket
4. WHEN a Mix with the same SKU set and `mfgType` already exists, THE engine SHALL update it rather than create a duplicate
5. WHEN a MixMetric with the same `mixId` and `bucketId` already exists, THE engine SHALL update it rather than create a duplicate
6. THE engine SHALL write results in batches of `batch_size` to avoid excessive memory usage

---

### Requirement 9

**User Story:** As an operator, I want the job progress to be updated throughout the enumeration, so that I can monitor long-running jobs in real time.

#### Acceptance Criteria

1. THE engine SHALL call `job_repo.mark_running` with the SKU count and initial stage list before beginning stage iteration
2. THE engine SHALL call `job_repo.mark_stage_running` with the stage index and total combination count before processing each stage
3. THE engine SHALL call `job_repo.checkpoint_stage` after every `batch_size` combinations processed within a stage
4. THE engine SHALL call `job_repo.mark_stage_complete` after all combinations in a stage have been processed
5. THE engine SHALL NOT call `job_repo.mark_completed` or `job_repo.mark_failed` — those are the responsibility of the job service thread wrapper
6. THE engine SHALL check `job_repo.is_cancelled(job_id)` at the start of each batch and halt processing if the job has been cancelled

---

### Requirement 10

**User Story:** As an operator, I want the engine to handle cancellation gracefully, so that cancelled jobs stop promptly without corrupting data.

#### Acceptance Criteria

1. THE engine SHALL check for cancellation at the start of processing each batch within a stage
2. WHEN cancellation is detected, THE engine SHALL stop processing immediately and return without raising an exception
3. WHEN cancellation is detected mid-stage, THE engine SHALL leave already-written results in place (partial results are acceptable for cancelled jobs)

---

### Requirement 11

**User Story:** As a developer, I want the engine to be well-structured and testable, so that individual stages can be unit tested in isolation.

#### Acceptance Criteria

1. THE engine SHALL be decomposed into clearly named helper functions for each logical stage: SKU loading, combination generation, cut strategy validation, Mix construction, bucket fitting, MixMetric computation, and result persistence
2. EACH helper function SHALL have a docstring describing its inputs, outputs, and business logic
3. THE engine SHALL not embed raw MongoDB queries outside of the repository layer — all data access SHALL go through the existing repository classes in `enumeration-api/repositories/`
