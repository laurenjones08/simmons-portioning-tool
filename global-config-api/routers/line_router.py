"""HTTP endpoints for production line management."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from config import get_settings
from database import get_database
from models.line import Line, LineCreate, LineUpdate
from repositories.line_repository import LineRepository
from services.cut_strategy_catalog import CutStrategyCatalogClient
from services.line_service import LineService

router = APIRouter()


def get_line_service(db: Database = Depends(get_database)) -> LineService:
    settings = get_settings()
    return LineService(
        LineRepository(db),
        CutStrategyCatalogClient(settings.enumeration_api_url),
    )


@router.get("/health", tags=["Lines"])
async def health_check():
    return {"status": "healthy"}


@router.get("", response_model=List[Line], tags=["Lines"])
async def list_lines(service: LineService = Depends(get_line_service)):
    return service.list_lines()


@router.get("/active", response_model=List[Line], tags=["Lines"])
async def list_active_lines(service: LineService = Depends(get_line_service)):
    return service.list_active_lines()


@router.get("/{line_id}", response_model=Line, tags=["Lines"])
async def get_line(line_id: str, service: LineService = Depends(get_line_service)):
    line = service.get_line(line_id)
    if line is None:
        raise HTTPException(status_code=404, detail=f"Line with id '{line_id}' not found")
    return line


@router.post("", response_model=Line, status_code=status.HTTP_201_CREATED, tags=["Lines"])
async def create_line(payload: LineCreate, service: LineService = Depends(get_line_service)):
    try:
        return service.create_line(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error creating line: {exc}")


@router.put("/{line_id}", response_model=Line, tags=["Lines"])
async def update_line(
    line_id: str,
    payload: LineUpdate,
    service: LineService = Depends(get_line_service),
):
    try:
        line = service.update_line(line_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error updating line: {exc}")

    if line is None:
        raise HTTPException(status_code=404, detail=f"Line with id '{line_id}' not found")
    return line


@router.delete("/{line_id}", tags=["Lines"])
async def delete_line(line_id: str, service: LineService = Depends(get_line_service)):
    try:
        deleted = service.delete_line(line_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error deleting line: {exc}")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Line with id '{line_id}' not found")
    return {"deleted": True, "lineId": line_id}
