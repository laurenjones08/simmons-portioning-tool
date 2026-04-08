# Implementation Plan: Streamlit Settings Page (Multi-Page Frontend)

## Overview

Build a new `streamlit-app/` directory containing a multi-page Streamlit application with 6 pages, a shared API client, and property-based tests. The app communicates with three backend APIs: enumeration-api, enumeration-worker-api, and global-config-api.

## Tasks

- [x] 1. Scaffold project structure and shared API client
  - [x] 1.1 Create `streamlit-app/` directory structure with `pages/` subdirectory and `requirements.txt`
    - Create `streamlit-app/requirements.txt` with `streamlit`, `requests`, `hypothesis`, `pytest`
    - Create `streamlit-app/app.py` as the multipage entry point with `st.set_page_config` and a landing/home page
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.2 Implement `streamlit-app/api_client.py` with environment-variable-based URL configuration
    - Read `ENUMERATION_API_URL`, `WORKER_API_URL`, `CONFIG_API_URL` from environment with localhost defaults
    - Implement `EnumerationApiClient`, `WorkerApiClient`, `ConfigApiClient` classes (or a single `ApiClient` with namespaced methods)
    - Each method raises a descriptive exception on non-2xx responses, including the response body
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 1.3 Write property tests for API client URL configuration
    - **Property: URL defaults are valid localhost URLs when env vars are absent**
    - **Property: URL from env var is used verbatim when set**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 2. Implement Buckets page
  - [x] 2.1 Create `streamlit-app/pages/1_Buckets.py`
    - On load, call `POST /buckets/search` with empty body and render results in `st.dataframe` (columns: id, minWeight, maxWeight)
    - Implement Create form: number inputs for minWeight/maxWeight, validate minWeight < maxWeight before calling `POST /buckets`
    - Implement Edit form: pre-populate selected bucket, validate, call `PUT /buckets/{id}`
    - Implement Delete button: call `DELETE /buckets/{id}`, display any recomputation warning from response
    - Display 409 conflict messages from API using `st.error`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 2.2 Write property test for bucket weight validation logic
    - **Property 4: Bucket Tuple Ordering — validation rejects minWeight >= maxWeight and accepts minWeight < maxWeight**
    - **Validates: Requirements 3.6**

- [x] 3. Implement SKUs page
  - [x] 3.1 Create `streamlit-app/pages/2_SKUs.py`
    - Display search form with optional filters: prodPlant, birdSize, customerType, productType
    - On search submit, call `POST /skus/search` with provided filters and render results in `st.dataframe`
    - Implement Create/Update SKU form with all required fields; validate minWeight < maxWeight and minWeight ≤ targetWeight ≤ maxWeight before calling `POST /skus` or `PUT /skus/{id}`
    - Implement Delete button calling `DELETE /skus/{id}`
    - Implement Bulk Import section: file uploader for CSV/JSON, parse file, call `POST /skus/batch`, display summary (total/successful/failed + errors)
    - Display 400/409 error details from API using `st.error`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [x] 3.2 Write property test for SKU weight validation logic
    - **Property: SKU weight validation — targetWeight is always between minWeight and maxWeight when valid**
    - **Validates: Requirements 4.7**

- [x] 4. Implement Cut Strategies page
  - [x] 4.1 Create `streamlit-app/pages/3_Cut_Strategies.py`
    - On load, call `POST /cut-strategies/search` with empty body and render results in `st.dataframe` (columns: name, mfgType, hasNugget, beltSpeed, parts)
    - Implement Create form: text inputs for name/description, selectbox for mfgType (DSI/DB20), checkbox for hasNugget, number input for beltSpeed, multiselect for parts using valid PartCode values
    - Validate parts list has no duplicates before calling `POST /cut-strategies`
    - Implement Edit form: pre-populate selected strategy, validate, call `PUT /cut-strategies/{id}`
    - Implement Delete button: call `DELETE /cut-strategies/{id}`, display counts of deleted mixes and metrics from response
    - Display 409 conflict messages using `st.error`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [x] 4.2 Write property test for cut strategy parts deduplication validation
    - **Property: Parts list with duplicates always fails validation; list without duplicates always passes**
    - **Validates: Requirements 5.7**

- [x] 5. Checkpoint — core CRUD pages complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Mix Visualization page
  - [x] 6.1 Create `streamlit-app/pages/4_Mix_Visualization.py`
    - Display filter panel with optional fields: reqPlant, reqBirdSize, mfgType, cutStrategyID, includesFDS, includesRTL, includesNug, skuTradeNumber
    - On Search click, call `POST /mixes/search` with selected filters (empty object if no filters applied)
    - Render results in `st.dataframe` with columns: mix ID, reqPlant, reqBirdSize, mfgType, numFillets, filletWeight, includesFDS, includesRTL, includesNug, SKU trade numbers
    - On row selection, display full mix detail: skus map, cutStrategyID, beltSpeed, nuggetTargetWeight
    - Display "no mixes found" message when results are empty
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 7. Implement Mix Generation page (job submission and monitoring)
  - [x] 7.1 Create `streamlit-app/pages/5_Mix_Generation.py` — job submission form
    - Display form with fields: runId (text), plantFilter (optional text), birdSizeFilter (optional text), maxCombinationSize (int 1–4, default 4), batchSize (int ≥ 1, default 1000)
    - Validate maxCombinationSize in [1, 4] and batchSize > 0 before submitting
    - Display warning when neither plantFilter nor birdSizeFilter is provided
    - On submit, call `POST /jobs` and display returned job status
    - Display 409 conflict message ("only one job can run at a time") using `st.warning`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 7.2 Add job monitoring section to `streamlit-app/pages/5_Mix_Generation.py`
    - On page load, call `GET /jobs` and display all jobs in `st.dataframe` (columns: jobId, runId, status, createdAt, skuCount, plantFilter, birdSizeFilter)
    - On job row selection, call `GET /jobs/{job_id}` and display full detail including stage-level progress
    - For "running" jobs, display progress indicator with current stage and processed/total combinations
    - Display Refresh button that re-calls `GET /jobs`
    - For "failed" jobs, display errorMessage; for "completed" jobs, display finishedAt and skuCount
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 7.3 Add job cancellation to `streamlit-app/pages/5_Mix_Generation.py`
    - Show Cancel button only for jobs with status "pending" or "running"
    - On Cancel click, call `POST /jobs/{job_id}/cancel` and refresh displayed status
    - Display 404 message if job not found or already in terminal state
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 7.4 Write property test for job form validation logic
    - **Property: maxCombinationSize outside [1,4] always fails validation; inside always passes**
    - **Property: batchSize ≤ 0 always fails validation; > 0 always passes**
    - **Validates: Requirements 7.5, 7.6**

- [x] 8. Implement Global Config page
  - [x] 8.1 Create `streamlit-app/pages/6_Global_Config.py` — view and single-edit
    - On load, call `GET /config` and display all parameters in `st.dataframe` (columns: key, value, valueType, description, minValue, maxValue, updatedAt)
    - Group parameters by key prefix (portion before first dot) using `st.expander` or tabs
    - Display "no configuration parameters defined" when list is empty
    - Add Refresh button that re-calls `GET /config`
    - On parameter selection, display type-appropriate input: number input for "int", decimal input for "float", text input for "string", checkbox for "bool"
    - Enforce minValue/maxValue bounds in input and display them as hints
    - On submit, call `PUT /config/{key}` and display success confirmation or 422 validation error
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 11.1, 11.2, 11.3, 11.4, 11.5_

  - [x] 8.2 Add batch edit mode to `streamlit-app/pages/6_Global_Config.py`
    - Provide batch edit mode where multiple values can be modified before submitting
    - Add validate-only button that calls `POST /config/batch` with `validateOnly: true`
    - On batch submit, call `POST /config/batch` with all modified key-value pairs
    - Display summary: total/successful/failed counts and per-key error details
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x] 8.3 Write property test for config value type coercion
    - **Property: A value loaded from the API and re-submitted as the same type always passes PUT /config validation (round-trip type consistency)**
    - **Validates: Requirements 11.1, 11.2**

- [x] 9. Implement property-based tests file
  - [x] 9.1 Create `streamlit-app/test_api_client_properties.py` consolidating all property tests
    - Import and run all property tests from tasks 1.3, 2.2, 3.2, 4.2, 7.4, 8.3
    - Use `hypothesis` with `@given` and `st.integers`, `st.floats`, `st.text`, `st.lists` strategies
    - Each property test runs minimum 100 examples
    - _Requirements: 2.1–2.4, 3.6, 4.7, 5.7, 7.5, 7.6, 11.1, 11.2_

- [x] 10. Final checkpoint — wire everything together
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- All API errors (network, 4xx, 5xx) must be caught and displayed via `st.error` without crashing — this applies to every page
- The `api_client.py` module is the single place for all HTTP calls; pages import from it and never call `requests` directly
- Default URLs: `ENUMERATION_API_URL=http://localhost:8080/api/enumeration`, `WORKER_API_URL=http://localhost:8080/api/enumeration-worker`, `CONFIG_API_URL=http://localhost:8080/api/config`
- Property tests validate pure logic functions (validation helpers) extracted from page modules, not Streamlit UI rendering
