from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pymongo.database import Database

from database import get_database
from job_service import JobService
from models.job import CreateJobRequest, JobStatusResponse

router = APIRouter()


def _get_service(db: Database = Depends(get_database)) -> JobService:
    return JobService(db)


@router.post(
    "",
    response_model=JobStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new scheduling job",
    description="Submits a background scheduling job that runs the optimization pipeline and stores result summaries.",
)
async def submit_job(payload: CreateJobRequest, service: JobService = Depends(_get_service)) -> JobStatusResponse:
    try:
        return service.submit_job(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error submitting scheduling job: {exc}")


@router.get(
    "",
    response_model=List[JobStatusResponse],
    summary="List all scheduling jobs",
)
async def list_jobs(
    status: Optional[str] = Query(default=None, description="Filter by job status"),
    service: JobService = Depends(_get_service),
) -> List[JobStatusResponse]:
    return service.list_jobs(status_filter=status)


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get scheduling job status",
)
async def get_job(job_id: str, service: JobService = Depends(_get_service)) -> JobStatusResponse:
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.post(
    "/{job_id}/cancel",
    response_model=JobStatusResponse,
    summary="Cancel a scheduling job",
)
async def cancel_job(job_id: str, service: JobService = Depends(_get_service)) -> JobStatusResponse:
    result = service.cancel_job(job_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found or is already in a terminal state (completed/failed/cancelled)",
        )
    return result

