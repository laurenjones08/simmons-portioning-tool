from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from repositories.sku_demand_repository import SKUDemandRepository
from scheduling_shared.models.sku_demand import (
    SKUDemand,
    SKUDemandBulkImportRequest,
    SKUDemandBulkImportResponse,
    SKUDemandCreate,
    SKUDemandSearchCriteria,
    SKUDemandUpdate,
)
from services.sku_demand_service import SKUDemandService

router = APIRouter()


def get_service(db: Database = Depends(get_database)) -> SKUDemandService:
    return SKUDemandService(SKUDemandRepository(db))


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


@router.post("", response_model=SKUDemand, status_code=status.HTTP_201_CREATED)
async def create(payload: SKUDemandCreate, service: SKUDemandService = Depends(get_service)):
    try:
        return service.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating SKU demand: {exc}")


@router.post("/bulk", response_model=SKUDemandBulkImportResponse, status_code=status.HTTP_201_CREATED)
async def bulk_create(
    payload: SKUDemandBulkImportRequest,
    service: SKUDemandService = Depends(get_service),
):
    try:
        return service.bulk_create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error bulk creating SKU demands: {exc}")


@router.get("/{document_id}", response_model=SKUDemand)
async def get_by_id(document_id: str, service: SKUDemandService = Depends(get_service)):
    document = service.get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"SKU demand with id {document_id} not found")
    return document


@router.post("/search", response_model=List[SKUDemand])
async def search(criteria: SKUDemandSearchCriteria, service: SKUDemandService = Depends(get_service)):
    try:
        return service.search(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching SKU demands: {exc}")


@router.put("/{document_id}", response_model=SKUDemand)
async def update(document_id: str, payload: SKUDemandUpdate, service: SKUDemandService = Depends(get_service)):
    try:
        document = service.update(document_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating SKU demand: {exc}")
    if document is None:
        raise HTTPException(status_code=404, detail=f"SKU demand with id {document_id} not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete(document_id: str, service: SKUDemandService = Depends(get_service)):
    try:
        deleted = service.delete(document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting SKU demand: {exc}")
    if not deleted:
        raise HTTPException(status_code=404, detail=f"SKU demand with id {document_id} not found")
    return {"deleted": True, "skuDemandId": document_id}
