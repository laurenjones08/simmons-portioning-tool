from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit

from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from config import get_settings
from database import close_mongo_connection
from routers import job_router
from storage import upload_short_term_demand_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _sanitize_mongodb_url(mongodb_url: str) -> str:
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
    title="Scheduling Worker API",
    description="""
## Scheduling Worker API

Runs the scheduling optimization pipeline in a background job and stores serialized output summaries.
Each job must include a `plantId` and a `skuIds` list. The worker validates that every SKU belongs
to the same plant before scheduling. If a `shortTermFile` is provided, the worker uses it only for that
run of the model and does not persist the rows. When `saveCsv` is enabled, the worker also uploads CSV
artifacts to the S3-compatible object store.

### Collections written

| Collection | Purpose |
|---|---|
| `scheduling_jobs` | Job lifecycle and progress |
| `scheduling_results` | Serialized scheduling outputs and artifact metadata |
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
        "message": "Scheduling Worker API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


app.include_router(job_router.router, prefix="/jobs", tags=["Jobs"])


@app.post("/uploads/short-term-file", tags=["Uploads"], summary="Upload a short-term demand file to object store")
async def upload_short_term_file_endpoint(file: UploadFile = File(...)):
    file_bytes = await file.read()
    suffix = Path(file.filename).suffix or ".csv"
    key = upload_short_term_demand_file(file_bytes, suffix)
    return {"objectKey": key}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True, log_level="info")
