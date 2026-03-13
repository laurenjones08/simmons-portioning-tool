"""HTTP endpoints for CutStrategy CRUD and search operations."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from models.cut_strategy import (
    CutStrategy,
    CutStrategyCreate,
    CutStrategySearchCriteria,
    CutStrategyUpdate,
)
from repositories.cut_strategy_repository import CutStrategyRepository
from repositories.mix_metric_repository import MixMetricRepository
from repositories.mix_repository import MixRepository
from services.cut_strategy_service import CutStrategyService

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Cut strategy service health check",
)
async def health_check():
    return {"status": "healthy"}


def get_cut_strategy_service(db: Database = Depends(get_database)) -> CutStrategyService:
    return CutStrategyService(
        CutStrategyRepository(db),
        MixRepository(db),
        MixMetricRepository(db),
    )


@router.post(
    "",
    response_model=CutStrategy,
    status_code=status.HTTP_201_CREATED,
    summary="Create cut strategy",
    description="Create a new cut strategy used by mixes.",
)
async def create_cut_strategy(
    payload: CutStrategyCreate,
    service: CutStrategyService = Depends(get_cut_strategy_service),
):
    try:
        return service.create_cut_strategy(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating cut strategy: {exc}")


@router.get(
    "/{strategy_id}",
    response_model=CutStrategy,
    summary="Get cut strategy by id",
)
async def get_cut_strategy(strategy_id: str, service: CutStrategyService = Depends(get_cut_strategy_service)):
    strategy = service.get_cut_strategy_by_id(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail=f"Cut strategy with id {strategy_id} not found")
    return strategy


@router.post(
    "/search",
    response_model=List[CutStrategy],
    summary="Search cut strategies",
    description="Search cut strategies by mfg type, nugget support, and included part.",
)
async def search_cut_strategies(
    criteria: CutStrategySearchCriteria,
    service: CutStrategyService = Depends(get_cut_strategy_service),
):
    try:
        return service.search_cut_strategies(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching cut strategies: {exc}")


@router.put(
    "/{strategy_id}",
    response_model=CutStrategy,
    summary="Update cut strategy",
)
async def update_cut_strategy(
    strategy_id: str,
    payload: CutStrategyUpdate,
    service: CutStrategyService = Depends(get_cut_strategy_service),
):
    try:
        strategy = service.update_cut_strategy(strategy_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating cut strategy: {exc}")

    if strategy is None:
        raise HTTPException(status_code=404, detail=f"Cut strategy with id {strategy_id} not found")
    return strategy


@router.delete(
    "/{strategy_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete cut strategy (cascade)",
    description=(
        "Delete a cut strategy and cascade-delete dependent mixes and mix metrics "
        "that reference those mixes."
    ),
    responses={
        200: {
            "description": "Cut strategy deleted and cascade operation completed",
            "content": {
                "application/json": {
                    "example": {
                        "deleted": True,
                        "cutStrategyId": "65f0c8fd6fb6bd463e25d4b7",
                        "mixesDeleted": 4,
                        "metricsDeleted": 19,
                    }
                }
            },
        }
    },
)
async def delete_cut_strategy(strategy_id: str, service: CutStrategyService = Depends(get_cut_strategy_service)):
    try:
        result = service.delete_cut_strategy(strategy_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting cut strategy: {exc}")

    if not result["deleted"]:
        raise HTTPException(status_code=404, detail=f"Cut strategy with id {strategy_id} not found")

    return {
        "deleted": True,
        "cutStrategyId": strategy_id,
        "mixesDeleted": result["mixes_deleted"],
        "metricsDeleted": result["metrics_deleted"],
    }
