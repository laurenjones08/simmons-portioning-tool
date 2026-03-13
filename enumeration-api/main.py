"""
Main application module for the Enumeration API.

This module initializes and configures the FastAPI application with all necessary
middleware, routers, and event handlers. It serves as the entry point for the
Enumeration API service.

FastAPI Application Structure:
1. Application initialization with metadata
2. Event handlers (startup/shutdown)
3. Middleware configuration
4. Router registration
5. Health check endpoints

The application follows a layered architecture:
- Router Layer: HTTP endpoints and request/response handling
- Service Layer: Business logic and orchestration
- Repository Layer: Data access and MongoDB operations

To run the application:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging
import os

from config import get_settings
from database import close_mongo_connection

# Configure logging
# This sets up structured logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    This async context manager handles startup and shutdown events for the FastAPI
    application. It replaces the older @app.on_event("startup") and
    @app.on_event("shutdown") decorators with a more modern approach.
    
    The lifespan pattern ensures:
    1. Resources are initialized before the app starts serving requests
    2. Resources are properly cleaned up when the app shuts down
    3. Exceptions during startup/shutdown are handled gracefully
    
    Args:
        app: The FastAPI application instance
        
    Yields:
        None: Control is yielded to the application to handle requests
        
    Example flow:
        1. Code before yield runs at startup
        2. Application serves requests
        3. Code after yield runs at shutdown
    """
    # Startup: Initialize resources
    logger.info("Starting Enumeration API...")
    settings = get_settings()
    logger.info(f"Service: {settings.service_name}")
    logger.info(f"MongoDB: {settings.mongodb_database}")
    
    # Yield control to the application
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down Enumeration API...")
    close_mongo_connection()
    logger.info("MongoDB connection closed")


# Initialize FastAPI application
# FastAPI automatically generates OpenAPI documentation and provides
# interactive API docs at /docs (Swagger UI) and /redoc (ReDoc)
app = FastAPI(
    title="Enumeration API",
    description="""
    The Enumeration API manages SKU (Stock Keeping Unit) data for the enumeration engine.
    
    ## Features
    
    * **SKU Management**: Retrieve SKU data by trade number
    * **Search**: Search SKUs using filter criteria
    * **Batch Operations**: Import and export SKUs in bulk
    * **Health Monitoring**: Health check endpoint for service monitoring
    
    ## Architecture
    
    The API follows a three-layer architecture:
    - **Router Layer**: HTTP endpoints (this API)
    - **Service Layer**: Business logic
    - **Repository Layer**: Database operations
    
    ## Collections
    
    * **skus**: Stores SKU documents with trade number as primary key
    """,
    version="1.0.0",
    lifespan=lifespan,  # Register lifespan context manager
    docs_url="/docs",  # Swagger UI endpoint
    redoc_url="/redoc",  # ReDoc endpoint
    openapi_url="/openapi.json",  # OpenAPI schema endpoint
    root_path=os.getenv("ROOT_PATH", "")
)


# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all HTTP requests with timing information.
    
    Middleware in FastAPI:
    - Runs before and after each request
    - Can modify requests and responses
    - Useful for logging, authentication, CORS, etc.
    
    This middleware:
    1. Records the start time
    2. Processes the request
    3. Calculates duration
    4. Logs request details
    
    Args:
        request: The incoming HTTP request
        call_next: Function to call the next middleware or route handler
        
    Returns:
        Response: The HTTP response from the route handler
        
    Example log output:
        2024-03-08 10:30:00 - GET /skus/50624 - 200 - 0.045s
    """
    start_time = time.time()
    
    # Process the request and get the response
    response = await call_next(request)
    
    # Calculate request duration
    duration = time.time() - start_time
    
    # Log request details
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    
    return response


# Health check endpoint
@app.get(
    "/health",
    tags=["Health"],
    summary="Health check endpoint",
    description="Returns the health status of the Enumeration API service",
    response_description="Service health status"
)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    This endpoint is used by:
    - Docker health checks
    - Kubernetes liveness/readiness probes
    - Load balancers to determine service availability
    - Monitoring systems to track uptime
    
    Returns:
        dict: Health status with service name and status
        
    Example response:
        {
            "status": "healthy",
            "service": "enumeration-api"
        }
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "service": settings.service_name
    }


# Root endpoint
@app.get(
    "/",
    tags=["Root"],
    summary="API root endpoint",
    description="Returns basic information about the Enumeration API"
)
async def root():
    """
    Root endpoint providing API information.
    
    Returns basic information about the API and links to documentation.
    
    Returns:
        dict: API information and documentation links
        
    Example response:
        {
            "message": "Enumeration API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    """
    return {
        "message": "Enumeration API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }


# Exception handler for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    
    This handler catches any exceptions that aren't handled by route handlers
    and returns a consistent error response. It also logs the error for debugging.
    
    Args:
        request: The HTTP request that caused the error
        exc: The exception that was raised
        
    Returns:
        JSONResponse: Standardized error response
        
    Example response:
        {
            "detail": "Internal server error",
            "path": "/skus/50624"
        }
    """
    logger.error(
        f"Unhandled exception for {request.method} {request.url.path}: {exc}",
        exc_info=True  # Include stack trace in logs
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "path": str(request.url.path)
        }
    )


# Router registration
# Import and register all routers with appropriate prefixes and tags
from routers import (
    bucket_router,
    cut_strategy_router,
    mix_metric_router,
    mix_router,
    sku_router,
)

# Register SKU router with /skus prefix
# All SKU endpoints will be available under /skus (e.g., /skus/50624, /skus/search)
app.include_router(
    sku_router.router,
    prefix="/skus",
    tags=["SKUs"]
)

app.include_router(
    mix_router.router,
    prefix="/mixes",
    tags=["MIXes"]
)


app.include_router(
    mix_metric_router.router,
    prefix="/metrics",
    tags=["MixMetrics"]
)

app.include_router(
    bucket_router.router,
    prefix="/buckets",
    tags=["Buckets"]
)

app.include_router(
    cut_strategy_router.router,
    prefix="/cut-strategies",
    tags=["CutStrategies"]
)


if __name__ == "__main__":
    """
    Entry point for running the application directly.
    
    This allows running the app with: python main.py
    However, in production, use uvicorn directly:
        uvicorn main:app --host 0.0.0.0 --port 8000
    
    The reload=True option enables auto-reload during development,
    which restarts the server when code changes are detected.
    """
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
