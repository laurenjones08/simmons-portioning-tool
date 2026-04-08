"""HTTP endpoints for MIX CRUD and search operations."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from database import get_database
from repositories.cut_strategy_repository import CutStrategyRepository
from models.mix import MIX, MixCreate, MixSearchCriteria, MixUpdate
from repositories.mix_repository import MixRepository
from services.mix_service import MixService

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy"}


def get_mix_service(db: Database = Depends(get_database)) -> MixService:
    return MixService(MixRepository(db), CutStrategyRepository(db))


@router.post("", response_model=MIX, status_code=status.HTTP_201_CREATED)
async def create_mix(payload: MixCreate, service: MixService = Depends(get_mix_service)):
    try:
        return service.create_mix(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating mix: {exc}")


@router.get("/{mix_id}", response_model=MIX)
async def get_mix(mix_id: str, service: MixService = Depends(get_mix_service)):
    mix = service.get_mix_by_id(mix_id)
    if mix is None:
        raise HTTPException(status_code=404, detail=f"Mix with id {mix_id} not found")
    return mix


@router.post("/search", response_model=List[MIX])
async def search_mixes(criteria: MixSearchCriteria, service: MixService = Depends(get_mix_service)):
    try:
        return service.search_mixes(criteria)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error searching mixes: {exc}")


@router.put("/{mix_id}", response_model=MIX)
async def update_mix(mix_id: str, payload: MixUpdate, service: MixService = Depends(get_mix_service)):
    try:
        mix = service.update_mix(mix_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating mix: {exc}")

    if mix is None:
        raise HTTPException(status_code=404, detail=f"Mix with id {mix_id} not found")
    return mix


@router.delete("/{mix_id}", status_code=status.HTTP_200_OK)
async def delete_mix(mix_id: str, service: MixService = Depends(get_mix_service)):
    try:
        deleted = service.delete_mix(mix_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting mix: {exc}")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Mix with id {mix_id} not found")

    return {"deleted": True, "mixId": mix_id}
