from contextlib import asynccontextmanager
import logging
import os
import time
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

SHARED_PATH = Path("/shared")
if SHARED_PATH.exists() and str(SHARED_PATH) not in sys.path:
    sys.path.insert(0, str(SHARED_PATH))

from config import get_settings
from database import close_mongo_connection
from routers import available_wip_router, bucket_usage_router, job_artifacts_router, monthly_contract_demand_router, scheduling_decision_router, scheduling_output_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Scheduling API...")
    settings = get_settings()
    logger.info(f"Service: {settings.service_name}")
    logger.info(f"MongoDB: {settings.mongodb_database}")
    yield
    logger.info("Shutting down Scheduling API...")
    close_mongo_connection()
    logger.info("MongoDB connection closed")


app = FastAPI(
    title="Scheduling API",
    description="""
    The Scheduling API stores monthly contract demand, scheduling decisions, produced outputs, bucket usage records, and available WIP rows.

    ## Collections

    * **scheduling_decisions**: Decision records for a mix, line, and date, including duration, produced pounds, and upgrade percent
    * **scheduling_outputs**: Produced output records by decision and SKU, including batch upgrade percent plus short-term and long-term contract pounds
    * **monthly_contracts**: Monthly contract demand by SKU and year-month
    * **bucket_usage**: Daily bucket availability and utilization
    * **available_wip**: Plant and bucket level WIP availability
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
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.3f}s"
    )
    return response


@app.get("/health", tags=["Health"])
async def health_check():
    settings = get_settings()
    return {"status": "healthy", "service": settings.service_name}


@app.get("/")
async def root():
    return {
        "message": "Scheduling API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception for {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url.path)},
    )


app.include_router(scheduling_decision_router.router, prefix="/scheduling-decisions", tags=["Scheduling Decisions"])
app.include_router(scheduling_output_router.router, prefix="/scheduling-outputs", tags=["Scheduling Outputs"])
app.include_router(monthly_contract_demand_router.router, prefix="/monthly-contracts", tags=["Monthly Contract Demands"])
app.include_router(bucket_usage_router.router, prefix="/bucket-usage", tags=["Bucket Usage"])
app.include_router(available_wip_router.router, prefix="/available-wip", tags=["Available WIP"])
app.include_router(job_artifacts_router.router)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info",
    )
