from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from repositories.scheduling_decision_repository import SchedulingDecisionRepository
from scheduling_shared.models.scheduling_decision import (
    SchedulingDecision,
    SchedulingDecisionCreate,
    SchedulingDecisionSearchCriteria,
    SchedulingDecisionUpdate,
)
from services.scheduling_decision_service import SchedulingDecisionService

router = APIRouter()


def get_service(db: Database = Depends(get_database)) -> SchedulingDecisionService:
    return SchedulingDecisionService(SchedulingDecisionRepository(db))


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


@router.post("", response_model=SchedulingDecision, status_code=status.HTTP_201_CREATED)
async def create(payload: SchedulingDecisionCreate, service: SchedulingDecisionService = Depends(get_service)):
    try:
        return service.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating scheduling decision: {exc}")


@router.get("/{document_id}", response_model=SchedulingDecision)
async def get_by_id(document_id: str, service: SchedulingDecisionService = Depends(get_service)):
    document = service.get_by_id(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Scheduling decision with id {document_id} not found")
    return document


@router.post("/search", response_model=List[SchedulingDecision])
async def search(criteria: SchedulingDecisionSearchCriteria, service: SchedulingDecisionService = Depends(get_service)):
    try:
        return service.search(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching scheduling decisions: {exc}")


@router.put("/{document_id}", response_model=SchedulingDecision)
async def update(document_id: str, payload: SchedulingDecisionUpdate, service: SchedulingDecisionService = Depends(get_service)):
    try:
        document = service.update(document_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating scheduling decision: {exc}")
    if document is None:
        raise HTTPException(status_code=404, detail=f"Scheduling decision with id {document_id} not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete(document_id: str, service: SchedulingDecisionService = Depends(get_service)):
    try:
        deleted = service.delete(document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting scheduling decision: {exc}")
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Scheduling decision with id {document_id} not found")
    return {"deleted": True, "schedulingDecisionId": document_id}

