from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from repositories.bucket_usage_repository import BucketUsageRepository
from scheduling_shared.models.bucket_usage import BucketUsage, BucketUsageCreate, BucketUsageSearchCriteria, BucketUsageUpdate
from services.bucket_usage_service import BucketUsageService

router = APIRouter()


def get_service(db: Database = Depends(get_database)) -> BucketUsageService:
    return BucketUsageService(BucketUsageRepository(db))


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


@router.post("", response_model=BucketUsage, status_code=status.HTTP_201_CREATED)
async def create(payload: BucketUsageCreate, service: BucketUsageService = Depends(get_service)):
    try:
        return service.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating bucket usage: {exc}")


@router.get("/{document_id}", response_model=BucketUsage)
async def get_by_id(document_id: str, service: BucketUsageService = Depends(get_service)):
    document = service.get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Bucket usage with id {document_id} not found")
    return document


@router.post("/search", response_model=List[BucketUsage])
async def search(criteria: BucketUsageSearchCriteria, service: BucketUsageService = Depends(get_service)):
    try:
        return service.search(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching bucket usage: {exc}")


@router.put("/{document_id}", response_model=BucketUsage)
async def update(document_id: str, payload: BucketUsageUpdate, service: BucketUsageService = Depends(get_service)):
    try:
        document = service.update(document_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating bucket usage: {exc}")
    if document is None:
        raise HTTPException(status_code=404, detail=f"Bucket usage with id {document_id} not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete(document_id: str, service: BucketUsageService = Depends(get_service)):
    try:
        deleted = service.delete(document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting bucket usage: {exc}")
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Bucket usage with id {document_id} not found")
    return {"deleted": True, "bucketUsageId": document_id}

