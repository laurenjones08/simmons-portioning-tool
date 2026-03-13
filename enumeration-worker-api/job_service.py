"""
Job service: submits enumeration jobs to a background thread and exposes
status/cancel operations. One job runs at a time per process.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.database import Database

from enumeration_engine import run_enumeration
from models.job import (
    CreateJobRequest,
    JobStatus,
    JobStatusResponse,
)
from repositories.job_repository import JobRepository

logger = logging.getLogger(__name__)

# Simple in-process lock to prevent concurrent jobs on the same instance.
_job_lock = threading.Lock()
_active_job_id: Optional[str] = None


def _run_job_thread(
    db: Database,
    job_id: str,
    run_id: str,
    max_combination_size: int,
    batch_size: int,
    sku_filter: List[str],
) -> None:
    """Thread target: run enumeration and update job status."""
    global _active_job_id
    repo = JobRepository(db)
    try:
        run_enumeration(
            db=db,
            job_id=job_id,
            run_id=run_id,
            job_repo=repo,
            max_combination_size=max_combination_size,
            batch_size=batch_size,
            sku_filter=sku_filter or None,
        )
        # Only mark completed if not already cancelled
        doc = repo.get_by_id(job_id)
        if doc and doc.get("status") not in ("cancelled", "failed"):
            repo.mark_completed(job_id)
            logger.info("Job %s completed successfully", job_id)
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        repo.mark_failed(job_id, str(exc))
    finally:
        _active_job_id = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobService:
    def __init__(self, db: Database):
        self.db = db
        self.repo = JobRepository(db)

    def submit_job(self, request: CreateJobRequest) -> JobStatusResponse:
        global _active_job_id

        with _job_lock:
            if _active_job_id is not None:
                # Check if truly still running
                doc = self.repo.get_by_id(_active_job_id)
                if doc and doc.get("status") in ("pending", "running"):
                    raise ValueError(
                        f"Another job is already running ({_active_job_id}). "
                        "Cancel it or wait for it to finish before submitting a new one."
                    )

            job_id = str(ObjectId())
            now = _now()
            job_doc = {
                "_id": job_id,
                "status": JobStatus.PENDING.value,
                "createdAt": now,
                "updatedAt": now,
                "startedAt": None,
                "finishedAt": None,
                "runId": request.run_id,
                "maxCombinationSize": request.max_combination_size,
                "batchSize": request.batch_size,
                "skuFilter": request.sku_filter,
                "skuCount": 0,
                "stages": [],
                "errorMessage": None,
                "resultsCollection": "enumeration_results",
            }
            self.repo.insert(job_doc)
            _active_job_id = job_id

        # Start background thread
        thread = threading.Thread(
            target=_run_job_thread,
            args=(
                self.db,
                job_id,
                request.run_id,
                request.max_combination_size,
                request.batch_size,
                request.sku_filter,
            ),
            daemon=True,
            name=f"enumeration-job-{job_id}",
        )
        thread.start()
        logger.info("Job %s submitted (runId=%s), thread started", job_id, request.run_id)

        doc = self.repo.get_by_id(job_id)
        return self._doc_to_response(doc)

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        doc = self.repo.get_by_id(job_id)
        if doc is None:
            return None
        return self._doc_to_response(doc)

    def list_jobs(self, status_filter: Optional[str] = None) -> List[JobStatusResponse]:
        docs = self.repo.list_all(status_filter)
        return [self._doc_to_response(d) for d in docs]

    def cancel_job(self, job_id: str) -> Optional[JobStatusResponse]:
        cancelled = self.repo.mark_cancelled(job_id)
        if not cancelled:
            return None  # job not found or already terminal
        doc = self.repo.get_by_id(job_id)
        return self._doc_to_response(doc) if doc else None

    @staticmethod
    def _doc_to_response(doc: Dict[str, Any]) -> JobStatusResponse:
        now = _now()
        return JobStatusResponse(
            jobId=str(doc["_id"]),
            status=doc.get("status", "pending"),
            runId=doc.get("runId", ""),
            createdAt=doc.get("createdAt", now),
            updatedAt=doc.get("updatedAt", now),
            startedAt=doc.get("startedAt"),
            finishedAt=doc.get("finishedAt"),
            skuCount=doc.get("skuCount", 0),
            maxCombinationSize=doc.get("maxCombinationSize", 4),
            skuFilter=doc.get("skuFilter", []),
            stages=doc.get("stages", []),
            errorMessage=doc.get("errorMessage"),
            resultsCollection=doc.get("resultsCollection", "enumeration_results"),
        )

