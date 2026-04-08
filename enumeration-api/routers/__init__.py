"""
Routers package for the Enumeration API.

This package contains all FastAPI routers (HTTP endpoint handlers) for the application.
Each router module defines endpoints for a specific domain or resource type.

Available Routers:
- sku_router: Endpoints for SKU management (GET, POST, search, batch operations)
- mix_router: Endpoints for MIX management (GET, POST, search, batch operations)

Usage:
    from routers import sku_router, mix_router

    app.include_router(sku_router.router, prefix="/skus", tags=["SKUs"])
    app.include_router(mix_router.router, prefix="/mixes", tags=["MIXes"])
"""

from . import (
    bucket_router,
    cut_strategy_router,
    mix_metric_router,
    mix_router,
    sku_router,
)

__all__ = [
    "sku_router",
    "mix_router",
    "mix_metric_router",
    "bucket_router",
    "cut_strategy_router",
]
