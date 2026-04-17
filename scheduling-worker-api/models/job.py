from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

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


class ArtifactFile(BaseModel):
    artifact_name: str = Field(..., alias="artifactName")
    file_name: str = Field(..., alias="fileName")
    bucket: str = Field(..., alias="bucket")
    key: str = Field(..., alias="key")
    download_url: Optional[str] = Field(default=None, alias="downloadUrl")

    model_config = {"populate_by_name": True}


class CreateJobRequest(BaseModel):
    run_id: str = Field(..., alias="runId", min_length=1)
    plant_id: str = Field(..., alias="plantId", min_length=1)
    sku_ids: List[str] = Field(..., alias="skuIds", min_length=1)
    short_term_file: Optional[str] = Field(None, alias="shortTermFile")
    save_csv: bool = Field(default=False, alias="saveCsv")
    output_dir: str = Field(default="outputs", alias="outputDir")
    tee: bool = Field(default=False)
    plan_start_date: str = Field(default="2026-01-05", alias="planStartDate")
    horizon_days: int = Field(default=12, alias="horizonDays", ge=1)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "runId": "schedule-2026-04-10",
                "plantId": "FSP",
                "skuIds": ["50624", "50625"],
                "shortTermFile": None,
                "saveCsv": False,
                "outputDir": "outputs",
                "tee": False,
                "planStartDate": "2026-01-05",
                "horizonDays": 12,
            }
        },
    }


class SchedulingJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    status: JobStatus = Field(default=JobStatus.PENDING)
    created_at: datetime = Field(default_factory=_utcnow, alias="createdAt")
    updated_at: datetime = Field(default_factory=_utcnow, alias="updatedAt")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    finished_at: Optional[datetime] = Field(default=None, alias="finishedAt")
    run_id: str = Field(..., alias="runId")
    short_term_file: Optional[str] = Field(default=None, alias="shortTermFile")
    save_csv: bool = Field(default=False, alias="saveCsv")
    output_dir: str = Field(default="outputs", alias="outputDir")
    tee: bool = Field(default=False)
    plan_start_date: str = Field(default="2026-01-05", alias="planStartDate")
    horizon_days: int = Field(default=12, alias="horizonDays")
    current_stage: Optional[str] = Field(default=None, alias="currentStage")
    stage_message: Optional[str] = Field(default=None, alias="stageMessage")
    stage_details: Dict[str, Any] = Field(default_factory=dict, alias="stageDetails")
    stage_updated_at: Optional[datetime] = Field(default=None, alias="stageUpdatedAt")
    timings: Dict[str, float] = Field(default_factory=dict, alias="timings")
    error_message: Optional[str] = Field(default=None, alias="errorMessage")
    error_traceback: Optional[str] = Field(default=None, alias="errorTraceback")
    results_collection: str = Field(default="scheduling_results", alias="resultsCollection")
    plant_id: str = Field(..., alias="plantId")
    sku_ids: List[str] = Field(default_factory=list, alias="skuIds")
    artifact_bucket: Optional[str] = Field(default=None, alias="artifactBucket")
    artifact_prefix: Optional[str] = Field(default=None, alias="artifactPrefix")
    artifact_keys: List[str] = Field(default_factory=list, alias="artifactKeys")
    artifact_files: List[ArtifactFile] = Field(default_factory=list, alias="artifactFiles")

    model_config = {
        "populate_by_name": True,
    }


class JobStatusResponse(BaseModel):
    job_id: str = Field(..., alias="jobId")
    status: JobStatus
    run_id: str = Field(..., alias="runId")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    finished_at: Optional[datetime] = Field(default=None, alias="finishedAt")
    short_term_file: Optional[str] = Field(default=None, alias="shortTermFile")
    save_csv: bool = Field(default=False, alias="saveCsv")
    output_dir: str = Field(default="outputs", alias="outputDir")
    tee: bool = Field(default=False)
    plan_start_date: str = Field(default="2026-01-05", alias="planStartDate")
    horizon_days: int = Field(default=12, alias="horizonDays")
    current_stage: Optional[str] = Field(default=None, alias="currentStage")
    stage_message: Optional[str] = Field(default=None, alias="stageMessage")
    stage_details: Dict[str, Any] = Field(default_factory=dict, alias="stageDetails")
    stage_updated_at: Optional[datetime] = Field(default=None, alias="stageUpdatedAt")
    timings: Dict[str, float] = Field(default_factory=dict, alias="timings")
    error_message: Optional[str] = Field(default=None, alias="errorMessage")
    error_traceback: Optional[str] = Field(default=None, alias="errorTraceback")
    results_collection: str = Field(default="scheduling_results", alias="resultsCollection")
    plant_id: str = Field(..., alias="plantId")
    sku_ids: List[str] = Field(default_factory=list, alias="skuIds")
    artifact_bucket: Optional[str] = Field(default=None, alias="artifactBucket")
    artifact_prefix: Optional[str] = Field(default=None, alias="artifactPrefix")
    artifact_keys: List[str] = Field(default_factory=list, alias="artifactKeys")
    artifact_files: List[ArtifactFile] = Field(default_factory=list, alias="artifactFiles")

    model_config = {"populate_by_name": True}

    @classmethod
    def from_job(cls, job: SchedulingJob) -> "JobStatusResponse":
        return cls(
            jobId=job.job_id,
            status=job.status,
            runId=job.run_id,
            createdAt=job.created_at,
            updatedAt=job.updated_at,
            startedAt=job.started_at,
            finishedAt=job.finished_at,
            shortTermFile=job.short_term_file,
            saveCsv=job.save_csv,
            outputDir=job.output_dir,
            tee=job.tee,
            planStartDate=job.plan_start_date,
            horizonDays=job.horizon_days,
            currentStage=job.current_stage,
            stageMessage=job.stage_message,
            stageDetails=job.stage_details,
            stageUpdatedAt=job.stage_updated_at,
            timings=job.timings,
            errorMessage=job.error_message,
            errorTraceback=job.error_traceback,
            resultsCollection=job.results_collection,
            plantId=job.plant_id,
            skuIds=job.sku_ids,
            artifactBucket=job.artifact_bucket,
            artifactPrefix=job.artifact_prefix,
            artifactKeys=job.artifact_keys,
            artifactFiles=job.artifact_files,
        )
