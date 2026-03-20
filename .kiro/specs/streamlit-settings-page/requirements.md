# Requirements Document

## Introduction

This document specifies the requirements for the Streamlit Settings Page — a multi-page Streamlit frontend application that serves as the management UI for the portioning tool system. The application provides CRUD interfaces for Buckets, SKUs, and Cut Strategies via the enumeration-api; a Mix Visualization page for browsing enumeration results; a Mix Generation page for submitting and monitoring enumeration jobs via the enumeration-worker-api; and a Global Config Management page for viewing and editing system configuration parameters via the global-config-api.

## Glossary

- **App**: The multi-page Streamlit frontend application described in this document.
- **Enumeration_API**: The backend REST service exposing CRUD endpoints for Buckets, SKUs, Cut Strategies, and Mixes (prefix `/buckets`, `/skus`, `/cut-strategies`, `/mixes`).
- **Worker_API**: The enumeration-worker-api REST service exposing job management endpoints (prefix `/jobs`).
- **Config_API**: The global-config-api REST service exposing configuration endpoints (prefix `/config`).
- **Bucket**: A weight-range definition used for enumeration bucketing, with min/max weight boundaries and a unique MongoDB ObjectId.
- **SKU**: A Stock Keeping Unit with fields: tradeNumber, customerName, customerType, productType, unitsPerCut, prodPlant, minWeight, maxWeight, targetWeight, birdSize, allowedParts.
- **CutStrategy**: A manufacturing cut strategy with fields: name, description, mfgType (DSI/DB20), hasNugget, beltSpeed, parts (list of PartCode).
- **Mix**: An enumeration result record with fields: skus map, includesFDS, includesRTL, includesNug, nuggetTargetWeight, numFillets, filletWeight, mfgType, reqPlant, reqBirdSize, cutStrategyID, beltSpeed.
- **Job**: An enumeration job submitted to the Worker_API with fields: runId, maxCombinationSize, batchSize, plantFilter, birdSizeFilter, and status tracking (pending/running/completed/failed/cancelled).
- **Config**: A system configuration parameter with fields: key, value, valueType (int/string/float/bool), description, updatedAt, minValue, maxValue.
- **API_Client**: The HTTP client layer within the App responsible for communicating with backend APIs.

## Requirements

### Requirement 1: Multi-Page Application Structure

**User Story:** As an operator, I want a single Streamlit application with clearly separated pages, so that I can navigate between different management functions without confusion.

#### Acceptance Criteria

1. THE App SHALL provide a sidebar navigation with the following pages: Buckets, SKUs, Cut Strategies, Mix Visualization, Mix Generation, and Global Config.
2. WHEN a user selects a page from the sidebar, THE App SHALL render that page's content in the main area.
3. THE App SHALL display the active page name in the sidebar to indicate the current location.
4. IF an API call fails due to a network or server error, THEN THE App SHALL display a human-readable error message on the affected page without crashing the application.

### Requirement 2: API Client Configuration

**User Story:** As a developer deploying the App, I want the backend API base URLs to be configurable, so that the App can connect to different environments without code changes.

#### Acceptance Criteria

1. THE API_Client SHALL read the Enumeration_API base URL from an environment variable named `ENUMERATION_API_URL`.
2. THE API_Client SHALL read the Worker_API base URL from an environment variable named `WORKER_API_URL`.
3. THE API_Client SHALL read the Config_API base URL from an environment variable named `CONFIG_API_URL`.
4. IF an environment variable is not set, THEN THE API_Client SHALL use a default localhost URL for that service.

### Requirement 3: Bucket Management

**User Story:** As an operator, I want to create, view, update, and delete weight buckets, so that I can manage the enumeration bucketing configuration.

#### Acceptance Criteria

1. WHEN a user opens the Buckets page, THE App SHALL call `POST /buckets/search` with empty criteria and display all existing buckets in a table showing bucket ID, minWeight, and maxWeight.
2. WHEN a user submits the Create Bucket form with valid minWeight and maxWeight values, THE App SHALL call `POST /buckets` and display the newly created bucket in the table.
3. WHEN a user selects a bucket and submits the Edit Bucket form with updated values, THE App SHALL call `PUT /buckets/{id}` and reflect the updated values in the table.
4. WHEN a user clicks Delete on a bucket, THE App SHALL call `DELETE /buckets/{id}` and remove the bucket from the table.
5. IF a bucket deletion returns a warning about enumeration recomputation, THEN THE App SHALL display that warning message to the user.
6. WHEN creating or editing a bucket, THE App SHALL validate that minWeight is less than maxWeight before submitting the API call.
7. IF the Enumeration_API returns a 409 conflict error, THEN THE App SHALL display the conflict message returned by the API.

### Requirement 4: SKU Management

**User Story:** As an operator, I want to create, search, update, and delete SKUs, so that I can manage the product catalog used for enumeration.

#### Acceptance Criteria

1. WHEN a user opens the SKUs page, THE App SHALL display a search form with optional filters for prodPlant, birdSize, customerType, and productType.
2. WHEN a user submits the search form, THE App SHALL call `POST /skus/search` with the provided filters and display matching SKUs in a table.
3. WHEN a user submits the Create/Update SKU form with all required fields, THE App SHALL call `POST /skus` and reflect the result in the table.
4. WHEN a user clicks Delete on a SKU, THE App SHALL call `DELETE /skus/{id}` and remove the SKU from the table.
5. WHEN a user uploads a CSV or JSON file on the Bulk Import section, THE App SHALL parse the file and call `POST /skus/batch` with the parsed SKU records.
6. WHEN a batch import completes, THE App SHALL display a summary showing total, successful, and failed counts along with any error details.
7. WHEN creating or editing a SKU, THE App SHALL validate that minWeight is less than maxWeight and that targetWeight is between minWeight and maxWeight before submitting.
8. IF the Enumeration_API returns a 400 or 409 error, THEN THE App SHALL display the error detail returned by the API.

### Requirement 5: Cut Strategy Management

**User Story:** As an operator, I want to create, view, update, and delete cut strategies, so that I can manage the manufacturing cut configurations used in enumeration.

#### Acceptance Criteria

1. WHEN a user opens the Cut Strategies page, THE App SHALL call `POST /cut-strategies/search` with empty criteria and display all existing cut strategies in a table showing name, mfgType, hasNugget, beltSpeed, and parts.
2. WHEN a user submits the Create Cut Strategy form with valid fields, THE App SHALL call `POST /cut-strategies` and display the new strategy in the table.
3. WHEN a user selects a cut strategy and submits the Edit form with updated values, THE App SHALL call `PUT /cut-strategies/{id}` and reflect the updated values in the table.
4. WHEN a user clicks Delete on a cut strategy, THE App SHALL call `DELETE /cut-strategies/{id}` and remove the strategy from the table.
5. IF a cut strategy deletion cascades to mixes and mix metrics, THEN THE App SHALL display the counts of deleted mixes and metrics to the user.
6. WHEN creating or editing a cut strategy, THE App SHALL provide a multi-select input for the parts field using valid PartCode values.
7. WHEN creating or editing a cut strategy, THE App SHALL validate that the parts list contains no duplicates before submitting.
8. IF the Enumeration_API returns a 409 conflict error, THEN THE App SHALL display the conflict message returned by the API.

### Requirement 6: Mix Visualization

**User Story:** As an analyst, I want to browse and filter enumeration mix results, so that I can inspect what SKU combinations were generated and their properties.

#### Acceptance Criteria

1. WHEN a user opens the Mix Visualization page, THE App SHALL display a filter panel with optional fields: reqPlant, reqBirdSize, mfgType, cutStrategyID, includesFDS, includesRTL, includesNug, and skuTradeNumber.
2. WHEN a user applies filters and clicks Search, THE App SHALL call `POST /mixes/search` with the selected filter criteria and display the results in a table.
3. THE App SHALL display mix results in a table with columns: mix ID, reqPlant, reqBirdSize, mfgType, numFillets, filletWeight, includesFDS, includesRTL, includesNug, and the list of SKU trade numbers.
4. WHEN a user selects a mix row, THE App SHALL display the full mix detail including the complete skus map (trade number → part code), cutStrategyID, beltSpeed, and nuggetTargetWeight.
5. WHEN no filters are applied and the user clicks Search, THE App SHALL call `POST /mixes/search` with an empty criteria object and display all mixes.
6. WHEN the search returns zero results, THE App SHALL display a message indicating no mixes were found for the given filters.

### Requirement 7: Mix Generation — Job Submission

**User Story:** As an operator, I want to submit enumeration jobs with specific filters, so that I can generate mixes for a particular plant and bird size combination.

#### Acceptance Criteria

1. WHEN a user opens the Mix Generation page, THE App SHALL display a job submission form with fields: runId (text), plantFilter (optional text), birdSizeFilter (optional text), maxCombinationSize (integer 1–4, default 4), and batchSize (integer ≥ 1, default 1000).
2. WHEN a user submits the job form, THE App SHALL call `POST /jobs` with the provided parameters and display the returned job status.
3. IF the Worker_API returns a 409 conflict indicating another job is already running, THEN THE App SHALL display a message informing the user that only one job can run at a time.
4. WHEN a user submits a job without providing either plantFilter or birdSizeFilter, THE App SHALL display a warning that at least one filter is recommended to avoid a failed job.
5. THE App SHALL validate that maxCombinationSize is between 1 and 4 inclusive before submitting.
6. THE App SHALL validate that batchSize is a positive integer before submitting.

### Requirement 8: Mix Generation — Job Monitoring

**User Story:** As an operator, I want to monitor the status and progress of enumeration jobs, so that I can track whether a job is running, completed, or failed.

#### Acceptance Criteria

1. WHEN a user opens the Mix Generation page, THE App SHALL call `GET /jobs` and display all jobs in a table showing jobId, runId, status, createdAt, skuCount, plantFilter, and birdSizeFilter.
2. WHEN a user selects a job from the table, THE App SHALL call `GET /jobs/{job_id}` and display the full job detail including stage-level progress (stage number, status, totalCombinations, processedCombinations).
3. WHEN a job has status "running", THE App SHALL display a progress indicator showing the current stage and processed/total combinations.
4. WHEN a user clicks Refresh on the job list, THE App SHALL re-call `GET /jobs` and update the displayed job statuses.
5. WHEN a job has status "failed", THE App SHALL display the errorMessage field from the job response.
6. WHEN a job has status "completed", THE App SHALL display the finishedAt timestamp and total skuCount.

### Requirement 9: Mix Generation — Job Cancellation

**User Story:** As an operator, I want to cancel a running or pending job, so that I can stop an unwanted enumeration run without waiting for it to finish.

#### Acceptance Criteria

1. WHEN a job has status "pending" or "running", THE App SHALL display a Cancel button for that job.
2. WHEN a user clicks Cancel on a job, THE App SHALL call `POST /jobs/{job_id}/cancel` and update the displayed job status to reflect the cancellation.
3. IF the Worker_API returns a 404 for the cancel request, THEN THE App SHALL display a message indicating the job was not found or is already in a terminal state.
4. WHEN a job has status "completed", "failed", or "cancelled", THE App SHALL NOT display a Cancel button for that job.

### Requirement 10: Global Config Management — View

**User Story:** As an operator, I want to view all system configuration parameters in one place, so that I can understand the current system settings.

#### Acceptance Criteria

1. WHEN a user opens the Global Config page, THE App SHALL call `GET /config` and display all configuration parameters in a table showing key, value, valueType, description, minValue, maxValue, and updatedAt.
2. THE App SHALL group configuration parameters by their key prefix (the portion before the first dot) for readability.
3. WHEN the configuration list is empty, THE App SHALL display a message indicating no configuration parameters are defined.
4. WHEN a user clicks Refresh, THE App SHALL re-call `GET /config` and update the displayed values.

### Requirement 11: Global Config Management — Edit

**User Story:** As an operator, I want to edit individual configuration parameters with type-appropriate inputs, so that I can update system settings safely.

#### Acceptance Criteria

1. WHEN a user selects a configuration parameter to edit, THE App SHALL display an input control appropriate for the valueType: a number input for "int", a decimal input for "float", a text input for "string", and a checkbox for "bool".
2. WHEN a configuration parameter has minValue and/or maxValue defined, THE App SHALL enforce those bounds in the input control and display them as hints.
3. WHEN a user submits an edited value, THE App SHALL call `PUT /config/{key}` with the updated value and reflect the change in the table.
4. IF the Config_API returns a 422 validation error, THEN THE App SHALL display the validation error message to the user.
5. WHEN a configuration update succeeds, THE App SHALL display a success confirmation message.

### Requirement 12: Global Config Management — Batch Edit

**User Story:** As an operator, I want to edit multiple configuration parameters at once, so that I can apply a set of related changes efficiently.

#### Acceptance Criteria

1. THE App SHALL provide a batch edit mode where a user can modify multiple configuration values before submitting.
2. WHEN a user submits a batch edit, THE App SHALL call `POST /config/batch` with all modified key-value pairs.
3. WHEN the batch update completes, THE App SHALL display a summary showing total, successful, and failed counts along with any error details.
4. IF any configuration in the batch fails validation, THEN THE App SHALL display the per-key error details returned by the Config_API.
5. THE App SHALL provide a validate-only option that calls `POST /config/batch` with `validateOnly: true` before applying changes.
