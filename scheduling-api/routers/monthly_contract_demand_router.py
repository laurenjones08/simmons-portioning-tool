from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from repositories.monthly_contract_demand_repository import MonthlyContractDemandRepository
from scheduling_shared.models.monthly_contract_demand import (
    MonthlyContractDemand,
    MonthlyContractDemandBulkImportRequest,
    MonthlyContractDemandBulkImportResponse,
    MonthlyContractDemandBulkSearchRequest,
    MonthlyContractDemandCreate,
    MonthlyContractDemandSearchCriteria,
    MonthlyContractDemandUpdate,
)
from services.monthly_contract_demand_service import MonthlyContractDemandService

router = APIRouter()


def get_service(db: Database = Depends(get_database)) -> MonthlyContractDemandService:
    return MonthlyContractDemandService(MonthlyContractDemandRepository(db))


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


@router.post("", response_model=MonthlyContractDemand, status_code=status.HTTP_201_CREATED)
async def create(payload: MonthlyContractDemandCreate, service: MonthlyContractDemandService = Depends(get_service)):
    try:
        return service.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating monthly contract demand: {exc}")


@router.post("/bulk", response_model=MonthlyContractDemandBulkImportResponse, status_code=status.HTTP_201_CREATED)
async def bulk_create(
    payload: MonthlyContractDemandBulkImportRequest,
    service: MonthlyContractDemandService = Depends(get_service),
):
    try:
        return service.bulk_create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error bulk creating monthly contract demands: {exc}")


@router.get("/{document_id}", response_model=MonthlyContractDemand)
async def get_by_id(document_id: str, service: MonthlyContractDemandService = Depends(get_service)):
    document = service.get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Monthly contract demand with id {document_id} not found")
    return document


@router.post("/search", response_model=List[MonthlyContractDemand])
async def search(criteria: MonthlyContractDemandSearchCriteria, service: MonthlyContractDemandService = Depends(get_service)):
    try:
        return service.search(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching monthly contract demands: {exc}")


@router.post("/bulk-search", response_model=List[MonthlyContractDemand])
async def bulk_search(
    criteria: MonthlyContractDemandBulkSearchRequest,
    service: MonthlyContractDemandService = Depends(get_service),
):
    try:
        return service.bulk_search(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error bulk searching monthly contract demands: {exc}")


@router.put("/{document_id}", response_model=MonthlyContractDemand)
async def update(document_id: str, payload: MonthlyContractDemandUpdate, service: MonthlyContractDemandService = Depends(get_service)):
    try:
        document = service.update(document_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating monthly contract demand: {exc}")
    if document is None:
        raise HTTPException(status_code=404, detail=f"Monthly contract demand with id {document_id} not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete(document_id: str, service: MonthlyContractDemandService = Depends(get_service)):
    try:
        deleted = service.delete(document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting monthly contract demand: {exc}")
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Monthly contract demand with id {document_id} not found")
    return {"deleted": True, "monthlyContractDemandId": document_id}
