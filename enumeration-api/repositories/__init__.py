"""Repository layer for database operations."""

import os
import sys

# Prefer mounted shared source in Docker dev mode.
shared_path = "/shared"
if os.path.isdir(shared_path) and shared_path not in sys.path:
    sys.path.insert(0, shared_path)

try:
    from enumeration_shared.repositories import (
        BucketRepository,
        CutStrategyRepository,
        MixMetricRepository,
        MixRepository,
        SKURepository,
    )
except ModuleNotFoundError:
    # Final fallback keeps API functional with local repository modules.
    from .bucket_repository import BucketRepository
    from .cut_strategy_repository import CutStrategyRepository
    from .mix_metric_repository import MixMetricRepository
    from .mix_repository import MixRepository
    from .sku_repository import SKURepository

__all__ = [
    "SKURepository",
    "MixRepository",
    "MixMetricRepository",
    "BucketRepository",
    "CutStrategyRepository",
]
