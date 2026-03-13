"""HTTP endpoints for Bucket CRUD and search operations."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from models.bucket import Bucket, BucketCreate, BucketSearchCriteria, BucketUpdate
from repositories.bucket_repository import BucketRepository
from repositories.mix_metric_repository import MixMetricRepository
from services.bucket_service import BucketService

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Bucket service health check",
)
async def health_check():
    return {"status": "healthy"}


def get_bucket_service(db: Database = Depends(get_database)) -> BucketService:
    return BucketService(BucketRepository(db), MixMetricRepository(db))


@router.post(
    "",
    response_model=Bucket,
    status_code=status.HTTP_201_CREATED,
    summary="Create bucket",
    description="Create a new bucket definition used for enumeration bucketing.",
)
async def create_bucket(payload: BucketCreate, service: BucketService = Depends(get_bucket_service)):
    try:
        return service.create_bucket(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating bucket: {exc}")


@router.get(
    "/{bucket_id}",
    response_model=Bucket,
    summary="Get bucket by id",
)
async def get_bucket(bucket_id: str, service: BucketService = Depends(get_bucket_service)):
    bucket = service.get_bucket_by_id(bucket_id)
    if bucket is None:
        raise HTTPException(status_code=404, detail=f"Bucket with id {bucket_id} not found")
    return bucket


@router.post(
    "/search",
    response_model=List[Bucket],
    summary="Search buckets",
    description="Search buckets by optional min/max boundary constraints.",
)
async def search_buckets(criteria: BucketSearchCriteria, service: BucketService = Depends(get_bucket_service)):
    try:
        return service.search_buckets(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching buckets: {exc}")


@router.put(
    "/{bucket_id}",
    response_model=Bucket,
    summary="Update bucket",
)
async def update_bucket(bucket_id: str, payload: BucketUpdate, service: BucketService = Depends(get_bucket_service)):
    try:
        bucket = service.update_bucket(bucket_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating bucket: {exc}")

    if bucket is None:
        raise HTTPException(status_code=404, detail=f"Bucket with id {bucket_id} not found")
    return bucket


@router.delete(
    "/{bucket_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete bucket (cascade)",
    description=(
        "Delete a bucket and cascade-delete dependent mix metrics for that bucket. "
        "The response warns that enumeration recomputation can take time."
    ),
    responses={
        200: {
            "description": "Bucket deleted and cascade operation completed",
            "content": {
                "application/json": {
                    "example": {
                        "deleted": True,
                        "bucketId": "65f0c8fd6fb6bd463e25d4b7",
                        "metricsDeleted": 12,
                        "warning": "Deleting a bucket requires recomputing the enumeration model, which can take a while.",
                    }
                }
            },
        }
    },
)
async def delete_bucket(bucket_id: str, service: BucketService = Depends(get_bucket_service)):
    try:
        result = service.delete_bucket(bucket_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting bucket: {exc}")

    if not result["deleted"]:
        raise HTTPException(status_code=404, detail=f"Bucket with id {bucket_id} not found")

    return {
        "deleted": True,
        "bucketId": bucket_id,
        "metricsDeleted": result["metrics_deleted"],
        "warning": "Deleting a bucket requires recomputing the enumeration model, which can take a while.",
    }
