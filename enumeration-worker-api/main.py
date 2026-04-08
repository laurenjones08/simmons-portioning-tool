"""
Enumeration Worker API

A dedicated FastAPI service that manages long-running enumeration jobs.
Submit a job via POST /jobs, poll progress via GET /jobs/{job_id},
and cancel via POST /jobs/{job_id}/cancel.

The job runs in a background thread, enumerating all SKU combinations
of size 1..N and writing results + metrics to MongoDB.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import get_settings
from database import close_mongo_connection
from routers import job_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _sanitize_mongodb_url(mongodb_url: str) -> str:
    """
    Return a MongoDB URL safe for logs by stripping any embedded credentials.
    """
    parts = urlsplit(mongodb_url)
    if parts.username is None and parts.password is None:
        return mongodb_url

    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"

    return urlunsplit((parts.scheme, hostname, parts.path, parts.query, parts.fragment))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting %s ...", settings.service_name)
    logger.info("MongoDB: %s / %s", _sanitize_mongodb_url(settings.mongodb_url), settings.mongodb_database)
    yield
    logger.info("Shutting down %s ...", settings.service_name)
    close_mongo_connection()


app = FastAPI(
    title="Enumeration Worker API",
    description="""
## Enumeration Worker API

Manages long-running staged enumeration jobs.

### How it works

1. **Submit** a job via `POST /jobs` – the job is immediately accepted and a
   background thread starts enumerating SKU combinations (size 1 through
   `maxCombinationSize`) against all configured buckets.
2. **Poll** progress via `GET /jobs/{jobId}` – returns real-time stage progress
   (combinations processed vs. total per stage).
3. **Cancel** a running job via `POST /jobs/{jobId}/cancel`.

### Results

Results are written to the `enumeration_results` collection in `enumeration_db`
and can be queried directly from MongoDB or via the Enumeration API.

### Collections written

| Collection | Purpose |
|---|---|
| `job_status` | Job lifecycle & stage progress |
| `enumeration_results` | Per-combination metrics per bucket |
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    root_path=os.getenv("ROOT_PATH", ""),
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info("%s %s - %s - %.3fs", request.method, request.url.path, response.status_code, duration)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "path": str(request.url.path)})


@app.get("/health", tags=["Health"], summary="Health check")
async def health_check():
    settings = get_settings()
    return {"status": "healthy", "service": settings.service_name}


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Enumeration Worker API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


app.include_router(job_router.router, prefix="/jobs", tags=["Jobs"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True, log_level="info")

