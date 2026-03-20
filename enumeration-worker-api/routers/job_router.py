"""HTTP endpoints for enumeration job management."""
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
    summary="Submit a new enumeration job",
    description=(
        "Submits a long-running enumeration job that loads candidate SKUs using "
        "`plantFilter` and/or `birdSizeFilter`, then enumerates all SKU combinations "
        "of size 1 through 4 and calculates metrics against every bucket. "
        "If neither filter is provided, the submitted job is marked `failed`. "
        "Only one job can run at a time. Results are written to the "
        "`enumeration_results` collection."
    ),
    responses={
        202: {"description": "Job accepted and queued"},
        409: {"description": "Another job is already running"},
    },
)
async def submit_job(
    payload: CreateJobRequest,
    service: JobService = Depends(_get_service),
) -> JobStatusResponse:
    try:
        return service.submit_job(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error submitting job: {exc}")
@router.get(
    "",
    response_model=List[JobStatusResponse],
    summary="List all enumeration jobs",
    description="Returns all jobs, optionally filtered by status.",
)
async def list_jobs(
    status: Optional[str] = Query(
        default=None,
        description="Filter by job status: pending | running | completed | failed | cancelled",
    ),
    service: JobService = Depends(_get_service),
) -> List[JobStatusResponse]:
    return service.list_jobs(status_filter=status)
@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Returns the current status and stage-level progress for a single job.",
    responses={404: {"description": "Job not found"}},
)
async def get_job(
    job_id: str,
    service: JobService = Depends(_get_service),
) -> JobStatusResponse:
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job
@router.post(
    "/{job_id}/cancel",
    response_model=JobStatusResponse,
    summary="Cancel a running or pending job",
    description=(
        "Signals the background worker to stop at the next batch boundary. "
        "The job will transition to `cancelled` status. "
        "Has no effect on already-completed or failed jobs."
    ),
    responses={
        200: {"description": "Job cancelled"},
        404: {"description": "Job not found or already in a terminal state"},
    },
)
async def cancel_job(
    job_id: str,
    service: JobService = Depends(_get_service),
) -> JobStatusResponse:
    result = service.cancel_job(job_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found or is already in a terminal state (completed/failed/cancelled)",
        )
    return result
