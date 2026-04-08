"""HTTP endpoints for MixMetric CRUD and search operations."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from models.mix_metric import MixMetric, MixMetricSearchCriteria
from repositories.mix_metric_repository import MixMetricRepository
from repositories.mix_repository import MixRepository
from services.mix_metric_service import MixMetricService

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


def get_mix_metric_service(db: Database = Depends(get_database)) -> MixMetricService:
    return MixMetricService(MixMetricRepository(db), MixRepository(db))


@router.post("", response_model=MixMetric, status_code=status.HTTP_201_CREATED)
async def create_metric(payload: MixMetric, service: MixMetricService = Depends(get_mix_metric_service)):
    try:
        return service.create_metric(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating metric: {exc}")


@router.get("/{metric_id}", response_model=MixMetric)
async def get_metric(metric_id: str, service: MixMetricService = Depends(get_mix_metric_service)):
    metric = service.get_metric_by_id(metric_id)
    if metric is None:
        raise HTTPException(status_code=404, detail=f"Metric with id {metric_id} not found")
    return metric


@router.post("/search", response_model=List[MixMetric])
async def search_metrics(criteria: MixMetricSearchCriteria, service: MixMetricService = Depends(get_mix_metric_service)):
    try:
        return service.search_metrics(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching metrics: {exc}")


@router.put("/{metric_id}", response_model=MixMetric)
async def update_metric(metric_id: str, payload: MixMetric, service: MixMetricService = Depends(get_mix_metric_service)):
    try:
        metric = service.update_metric(metric_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating metric: {exc}")

    if metric is None:
        raise HTTPException(status_code=404, detail=f"Metric with id {metric_id} not found")
    return metric


@router.delete("/{metric_id}", status_code=status.HTTP_200_OK)
async def delete_metric(metric_id: str, service: MixMetricService = Depends(get_mix_metric_service)):
    try:
        deleted = service.delete_metric(metric_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting metric: {exc}")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Metric with id {metric_id} not found")

    return {"deleted": True, "metricId": metric_id}
