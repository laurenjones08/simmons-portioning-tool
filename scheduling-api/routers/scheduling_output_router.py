from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from repositories.scheduling_output_repository import SchedulingOutputRepository
from scheduling_shared.models.scheduling_output import (
    SchedulingOutput,
    SchedulingOutputCreate,
    SchedulingOutputSearchCriteria,
    SchedulingOutputUpdate,
)
from services.scheduling_output_service import SchedulingOutputService

router = APIRouter()


def get_service(db: Database = Depends(get_database)) -> SchedulingOutputService:
    return SchedulingOutputService(SchedulingOutputRepository(db))


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


@router.post("", response_model=SchedulingOutput, status_code=status.HTTP_201_CREATED)
async def create(payload: SchedulingOutputCreate, service: SchedulingOutputService = Depends(get_service)):
    try:
        return service.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating scheduling output: {exc}")


@router.get("/{document_id}", response_model=SchedulingOutput)
async def get_by_id(document_id: str, service: SchedulingOutputService = Depends(get_service)):
    document = service.get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Scheduling output with id {document_id} not found")
    return document


@router.post("/search", response_model=List[SchedulingOutput])
async def search(criteria: SchedulingOutputSearchCriteria, service: SchedulingOutputService = Depends(get_service)):
    try:
        return service.search(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching scheduling outputs: {exc}")


@router.put("/{document_id}", response_model=SchedulingOutput)
async def update(document_id: str, payload: SchedulingOutputUpdate, service: SchedulingOutputService = Depends(get_service)):
    try:
        document = service.update(document_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating scheduling output: {exc}")
    if document is None:
        raise HTTPException(status_code=404, detail=f"Scheduling output with id {document_id} not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete(document_id: str, service: SchedulingOutputService = Depends(get_service)):
    try:
        deleted = service.delete(document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting scheduling output: {exc}")
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Scheduling output with id {document_id} not found")
    return {"deleted": True, "schedulingOutputId": document_id}

