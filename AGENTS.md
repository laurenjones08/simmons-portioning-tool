# AGENTS.md

## Mission-Critical Repo Map
- This is a Docker-first microservice stack behind one gateway (`docker-compose.yml`, `gateway/nginx.conf`). Treat `http://localhost:8080` as the public API surface.
- Core services: `enumeration-api` (CRUD/search for skus/mixes/metrics/buckets/cut strategies), `global-config-api` (runtime config + lines), `scheduling-api` (demand/decision/output APIs + artifact proxy), `enumeration-worker-api` and `scheduling-worker-api` (long-running background jobs).
- Data stores: MongoDB (`enumeration_db`, `scheduling_db`, `config_db`) plus MinIO for scheduling CSV artifacts (`docker-compose.yml`).
- Shared model libs are local packages mounted into services (`enumeration-shared`, `scheduling-shared`).

## Big-Picture Flows You Need To Preserve
- Gateway path routing is contract-critical: `/api/enumeration`, `/api/config`, `/api/scheduling`, `/api/scheduling-worker`, `/api/enumeration-worker` (`gateway/nginx.conf`).
- Enumeration flow: submit job to `enumeration-worker-api /jobs`, worker writes `job_status` + `enumeration_results`, and uses config values from `global-config-api` during scoring (`enumeration-worker-api/enumeration_engine.py`).
- Scheduling flow: submit job to `scheduling-worker-api /jobs`, worker runs `scheduling/pipeline.py`, stores results in `scheduling_results`, optionally uploads CSV to MinIO; `scheduling-api` proxies artifact listing/download (`scheduling-api/routers/job_artifacts_router.py`).
- Mongo bootstrap defines required collections/indexes/default config; schema/index assumptions in code rely on `mongodb-init/init-mongo.js`.

## Dev Workflows (Observed In-Repo)
- Start full stack: `docker compose up -d --build` (root `README.md`).
- DB reset path used by team: `./scripts/reinit-mongodb.ps1` on Windows (see `QUICK_START_DB.md`).
- Preferred local integration checks hit gateway endpoints, not service internal ports (`MICROSERVICE_API_ONBOARDING.md`).
- Service-local test style is pytest with mongomock fixtures; run from service directory (example files: `enumeration-api/conftest.py`, `global-config-api/conftest.py`).
- Specialized property test entrypoint exists: `enumeration-api/run_search_tests.sh`.

## Project-Specific Coding Patterns
- FastAPI apps consistently use `lifespan` startup/shutdown + module-level Mongo client singleton (`*/main.py`, `*/database.py`).
- Request/response payloads are camelCase externally, snake_case internally via Pydantic aliases with `populate_by_name=True` (example: `scheduling-shared/scheduling_shared/models/sku_demand.py`, `enumeration-worker-api/models/job.py`).
- Worker services enforce one active job per process using in-memory lock + `_active_job_id`; do not introduce concurrent execution without redesign (`enumeration-worker-api/job_service.py`, `scheduling-worker-api/job_service.py`).
- `ROOT_PATH` env wiring is important for gateway-prefixed docs/routes in containers (`docker-compose.yml`, each `main.py`).
- Tests often isolate router behavior via FastAPI dependency overrides and monkeypatching (`scheduling-api/test_scheduling_api.py`).

## Integration Gotchas
- If you add a microservice route, update both `docker-compose.yml` and `gateway/nginx.conf`; missing either causes 404/502 via gateway.
- `proxy_pass` trailing slash behavior in Nginx matters for path rewriting (documented in `MICROSERVICE_API_ONBOARDING.md`).
- Changes to `init-mongo.js` do not apply unless Mongo volume is reset (`mongodb-init/README.md`).
- Scheduling artifact download URLs are intentionally rewritten by API proxy, not passed through from worker (`scheduling-api/routers/job_artifacts_router.py`).

## Read-First Files Before Any Non-Trivial Change
- `docker-compose.yml`
- `gateway/nginx.conf`
- `mongodb-init/init-mongo.js`
- `enumeration-worker-api/enumeration_engine.py`
- `scheduling-worker-api/job_service.py`
- `scheduling-api/routers/job_artifacts_router.py`
- `MICROSERVICE_API_ONBOARDING.md`

