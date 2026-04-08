"""Job status models for the enumeration worker."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from bson import ObjectId
from pydantic import BaseModel, Field
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
class StageProgress(BaseModel):
    stage: int = Field(..., description="Combination size (1-4)")
    status: StageStatus = Field(default=StageStatus.PENDING)
    total_combinations: int = Field(default=0, alias="totalCombinations")
    processed_combinations: int = Field(default=0, alias="processedCombinations")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    finished_at: Optional[datetime] = Field(default=None, alias="finishedAt")
    model_config = {"populate_by_name": True}
class EnumerationJob(BaseModel):
    """A single enumeration job document stored in the job_status collection."""
    job_id: str = Field(
        default_factory=lambda: str(ObjectId()),
        alias="_id",
        description="Unique job ID (MongoDB _id)",
    )
    status: JobStatus = Field(default=JobStatus.PENDING)
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=_utcnow, alias="updatedAt")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    finished_at: Optional[datetime] = Field(default=None, alias="finishedAt")
    # Input parameters
    run_id: str = Field(..., alias="runId", description="Human-readable run label used as key in enumeration_runs")
    max_combination_size: int = Field(default=4, alias="maxCombinationSize", ge=1, le=4)
    batch_size: int = Field(default=1000, alias="batchSize", ge=1)
    plant_filter: Optional[str] = Field(
        default=None,
        alias="plantFilter",
        description="Optional plant filter used when loading candidate SKUs (matches skus.prodPlant).",
    )
    bird_size_filter: Optional[str] = Field(
        default=None,
        alias="birdSizeFilter",
        description="Optional bird-size filter used when loading candidate SKUs (matches skus.birdSize).",
    )
    # Progress tracking
    sku_count: int = Field(default=0, alias="skuCount")
    stages: List[StageProgress] = Field(default_factory=list)
    # Outcome
    error_message: Optional[str] = Field(default=None, alias="errorMessage")
    results_collection: str = Field(default="enumeration_results", alias="resultsCollection")
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "_id": "65f0c8fd6fb6bd463e25d4b7",
                "status": "pending",
                "runId": "run-2026-03-13",
                "maxCombinationSize": 4,
                "batchSize": 1000,
                "plantFilter": "P1",
                "birdSizeFilter": "L",
                "skuCount": 0,
                "stages": [],
                "createdAt": "2026-03-13T00:00:00Z",
                "updatedAt": "2026-03-13T00:00:00Z",
            }
        },
    }
class CreateJobRequest(BaseModel):
    """Request body for submitting a new enumeration job."""
    run_id: str = Field(
        ...,
        alias="runId",
        min_length=1,
        description="Unique label for this enumeration run. Re-using a runId resumes from last checkpoint.",
    )
    max_combination_size: int = Field(default=4, alias="maxCombinationSize", ge=1, le=4)
    batch_size: int = Field(default=1000, alias="batchSize", ge=1)
    plant_filter: Optional[str] = Field(
        default=None,
        alias="plantFilter",
        description="Plant filter for candidate SKU selection (matches skus.prodPlant).",
    )
    bird_size_filter: Optional[str] = Field(
        default=None,
        alias="birdSizeFilter",
        description="Bird-size filter for candidate SKU selection (matches skus.birdSize).",
    )
    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "runId": "run-2026-03-13",
                "maxCombinationSize": 4,
                "batchSize": 1000,
                "plantFilter": "P1",
                "birdSizeFilter": "L",
            }
        },
    }
class JobStatusResponse(BaseModel):
    """Public-facing job status returned by the API."""
    job_id: str = Field(..., alias="jobId")
    status: JobStatus
    run_id: str = Field(..., alias="runId")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    finished_at: Optional[datetime] = Field(default=None, alias="finishedAt")
    sku_count: int = Field(default=0, alias="skuCount")
    max_combination_size: int = Field(default=4, alias="maxCombinationSize")
    plant_filter: Optional[str] = Field(default=None, alias="plantFilter")
    bird_size_filter: Optional[str] = Field(default=None, alias="birdSizeFilter")
    stages: List[StageProgress] = Field(default_factory=list)
    error_message: Optional[str] = Field(default=None, alias="errorMessage")
    results_collection: str = Field(default="enumeration_results", alias="resultsCollection")
    model_config = {"populate_by_name": True}
    @classmethod
    def from_job(cls, job: EnumerationJob) -> "JobStatusResponse":
        return cls(
            jobId=job.job_id,
            status=job.status,
            runId=job.run_id,
            createdAt=job.created_at,
            updatedAt=job.updated_at,
            startedAt=job.started_at,
            finishedAt=job.finished_at,
            skuCount=job.sku_count,
            maxCombinationSize=job.max_combination_size,
            plantFilter=job.plant_filter,
            birdSizeFilter=job.bird_size_filter,
            stages=job.stages,
            errorMessage=job.error_message,
            resultsCollection=job.results_collection,
        )
