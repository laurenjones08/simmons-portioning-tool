from contextlib import asynccontextmanager
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import get_settings
from database import close_mongo_connection

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
    The Scheduling API stores demand, scheduling decisions, produced outputs, and bucket usage records.

    ## Collections

    * **sku_demands**: Incoming SKU demand records
    * **scheduling_decisions**: Decision records for a mix, line, and date
    * **scheduling_outputs**: Produced output records by decision and SKU
    * **bucket_usage**: Daily bucket availability and utilization
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


from routers import bucket_usage_router, scheduling_decision_router, scheduling_output_router, sku_demand_router

app.include_router(sku_demand_router.router, prefix="/sku-demands", tags=["SKU Demands"])
app.include_router(scheduling_decision_router.router, prefix="/scheduling-decisions", tags=["Scheduling Decisions"])
app.include_router(scheduling_output_router.router, prefix="/scheduling-outputs", tags=["Scheduling Outputs"])
app.include_router(bucket_usage_router.router, prefix="/bucket-usage", tags=["Bucket Usage"])


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
