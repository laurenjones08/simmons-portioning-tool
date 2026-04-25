# API Runtime Diagram

This view is focused on the current Docker Compose runtime, with emphasis on the frontend, API layer, persistence, and observability.

## Diagram

```mermaid
flowchart TB
    user["User / Planner"]

    subgraph presentation["Frontend and Entry Points"]
        streamlit["Streamlit Frontend<br/>:8501"]
        gateway["Nginx API Gateway<br/>localhost:8080"]
        docs["Docs Site<br/>:3000"]
    end

    subgraph api["API Layer"]
        enumApi["enumeration-api<br/>/api/enumeration"]
        configApi["global-config-api<br/>/api/config"]
        schedApi["scheduling-api<br/>/api/scheduling"]
        enumWorker["enumeration-worker-api<br/>/api/enumeration-worker"]
        schedWorker["scheduling-worker-api<br/>/api/scheduling-worker"]
    end

    subgraph data["Databases and Storage"]
        mongo["MongoDB<br/>mongo:7.0"]
        enumDb["enumeration_db<br/>skus, mixes, metrics, buckets,<br/>cut strategies, job_status, enumeration_results"]
        configDb["config_db<br/>global_config, lines"]
        schedDb["scheduling_db<br/>monthly_contracts, scheduling_decisions,<br/>scheduling_outputs, bucket_usage, available_wip"]
        minio["MinIO Object Store<br/>scheduling CSV artifacts"]
    end

    subgraph obs["Observability"]
        jaeger["Jaeger UI and Collector<br/>localhost:16686"]
        logs["Service Logs and Request Timing"]
        health["Health Endpoints<br/>gateway + FastAPI services"]
    end

    user --> streamlit
    user --> gateway
    user --> docs

    streamlit --> gateway

    gateway --> enumApi
    gateway --> configApi
    gateway --> schedApi
    gateway --> enumWorker
    gateway --> schedWorker

    schedWorker --> gateway
    schedApi --> schedWorker

    enumApi --> enumDb
    enumWorker --> enumDb
    configApi --> configDb
    schedApi --> schedDb
    schedWorker --> schedDb
    schedWorker --> minio
    schedApi --> minio
    mongo --> enumDb
    mongo --> configDb
    mongo --> schedDb

    enumApi -. traces .-> jaeger
    configApi -. traces .-> jaeger
    schedApi -. traces/config .-> jaeger
    enumApi -. logs .-> logs
    configApi -. logs .-> logs
    schedApi -. logs .-> logs
    enumWorker -. health/job status .-> health
    schedWorker -. health/job status .-> health
    gateway -. health .-> health
```

## What This Diagram Shows

- `streamlit-app` is the main interactive frontend and talks to the stack through the gateway rather than calling service ports directly.
- `api-gateway` is the single public API surface on `http://localhost:8080` and routes traffic to all five backend services.
- The API layer is split between CRUD-style domain APIs and worker APIs for long-running jobs.
- MongoDB is the system of record, separated into `enumeration_db`, `config_db`, and `scheduling_db`.
- MinIO is a sidecar persistence layer used by the scheduling workflow for CSV artifacts.
- Observability in the current Compose stack is primarily Jaeger tracing, FastAPI/gateway health endpoints, and application logs with request timing.

## Runtime Notes

- `scheduling-worker-api` depends on other APIs through the gateway for upstream reads and downstream persistence.
- `scheduling-api` does not expose raw object-store URLs directly; it proxies artifact listing and downloads for clients.
- `enumeration-worker-api` and `scheduling-worker-api` each allow only one active job per process today.
- Prometheus and Grafana are mentioned in design material, but they are not part of the running Compose stack shown here.
