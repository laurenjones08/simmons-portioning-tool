from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pymongo.database import Database

from database import get_database
from job_service import JobService
from models.job import ArtifactFile, CreateJobRequest, JobStatusResponse

router = APIRouter()


def _get_service(db: Database = Depends(get_database)) -> JobService:
    return JobService(db)


@router.post(
    "",
    response_model=JobStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new scheduling job",
    description="Submits a background scheduling job for a single plant and explicit skuIds list, then stores result summaries.",
)
def submit_job(payload: CreateJobRequest, service: JobService = Depends(_get_service)) -> JobStatusResponse:
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
def list_jobs(
    status: Optional[str] = Query(default=None, description="Filter by job status"),
    service: JobService = Depends(_get_service),
) -> List[JobStatusResponse]:
    return service.list_jobs(status_filter=status)


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get scheduling job status",
)
def get_job(job_id: str, service: JobService = Depends(_get_service)) -> JobStatusResponse:
    job = service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.post(
    "/{job_id}/cancel",
    response_model=JobStatusResponse,
    summary="Cancel a scheduling job",
)
def cancel_job(job_id: str, service: JobService = Depends(_get_service)) -> JobStatusResponse:
    result = service.cancel_job(job_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found or is already in a terminal state (completed/failed/cancelled)",
        )
    return result


@router.get(
    "/{job_id}/artifacts",
    response_model=List[ArtifactFile],
    summary="Get downloadable scheduling artifacts",
)
def get_job_artifacts(job_id: str, service: JobService = Depends(_get_service)) -> List[ArtifactFile]:
    artifacts = service.get_artifacts(job_id)
    if artifacts is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return artifacts


@router.get(
    "/{job_id}/results",
    summary="Get stored scheduling results payload",
)
def get_job_results(job_id: str, service: JobService = Depends(_get_service)):
    results = service.get_results(job_id)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Results for job {job_id} not found")
    return results


@router.get(
    "/{job_id}/debug-data-prep",
    summary="Get short-lived dataprep debug dump",
)
def get_job_debug_data_prep(job_id: str, service: JobService = Depends(_get_service)):
    debug_dump = service.get_debug_dump(job_id)
    if debug_dump is None:
        raise HTTPException(status_code=404, detail=f"Debug dataprep dump for job {job_id} not found")
    return debug_dump
