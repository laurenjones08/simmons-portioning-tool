from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from repositories.available_wip_repository import AvailableWIPRepository
from scheduling_shared.models.available_wip import (
    AvailableWIP,
    AvailableWIPCreate,
    AvailableWIPSearchCriteria,
    AvailableWIPUpdate,
)
from services.available_wip_service import AvailableWIPService

router = APIRouter()


def get_service(db: Database = Depends(get_database)) -> AvailableWIPService:
    return AvailableWIPService(AvailableWIPRepository(db))


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


@router.post("", response_model=AvailableWIP, status_code=status.HTTP_201_CREATED)
async def create(payload: AvailableWIPCreate, service: AvailableWIPService = Depends(get_service)):
    try:
        return service.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating available WIP: {exc}")


@router.get("/{document_id}", response_model=AvailableWIP)
async def get_by_id(document_id: str, service: AvailableWIPService = Depends(get_service)):
    document = service.get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Available WIP with id {document_id} not found")
    return document


@router.post("/search", response_model=List[AvailableWIP])
async def search(criteria: AvailableWIPSearchCriteria, service: AvailableWIPService = Depends(get_service)):
    try:
        return service.search(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching available WIP: {exc}")


@router.put("/{document_id}", response_model=AvailableWIP)
async def update(document_id: str, payload: AvailableWIPUpdate, service: AvailableWIPService = Depends(get_service)):
    try:
        document = service.update(document_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating available WIP: {exc}")
    if document is None:
        raise HTTPException(status_code=404, detail=f"Available WIP with id {document_id} not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete(document_id: str, service: AvailableWIPService = Depends(get_service)):
    try:
        deleted = service.delete(document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting available WIP: {exc}")
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Available WIP with id {document_id} not found")
    return {"deleted": True, "availableWipId": document_id}

