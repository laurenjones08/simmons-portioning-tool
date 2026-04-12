from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.database import Database

from models.job import CreateJobRequest, JobStatus, JobStatusResponse
from repositories.job_repository import JobRepository

logger = logging.getLogger(__name__)

_job_lock = threading.Lock()
_active_job_id: Optional[str] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_scheduling_path() -> None:
    root = Path(__file__).resolve().parents[1]
    scheduling_path = root.parent / "scheduling"
    if str(scheduling_path) not in sys.path:
        sys.path.insert(0, str(scheduling_path))


def _serialize_dataframe(df) -> List[Dict[str, Any]]:
    try:
        return json.loads(df.to_json(orient="records", date_format="iso"))
    except Exception:
        return []


def _run_job_thread(db: Database, job_id: str, request: CreateJobRequest) -> None:
    global _active_job_id
    repo = JobRepository(db)
    try:
        repo.mark_running(job_id)
        _ensure_scheduling_path()
        from pipeline import run_pipeline  # type: ignore

        results = run_pipeline(
            short_term_file=request.short_term_file,
            save_csv=request.save_csv,
            output_dir=request.output_dir,
            tee=request.tee,
        )

        output_payload = {
            name: _serialize_dataframe(df)
            for name, df in results.get("outputs", {}).items()
        }
        output_payload["inputs"] = {
            "planStartDate": request.plan_start_date,
            "horizonDays": request.horizon_days,
            "shortTermFile": request.short_term_file,
        }
        repo.store_results(job_id, request.run_id, output_payload)

        doc = repo.get_by_id(job_id)
        if doc and doc.get("status") not in ("cancelled", "failed"):
            repo.mark_completed(job_id)
            logger.info("Scheduling job %s completed successfully", job_id)
    except Exception as exc:
        logger.exception("Scheduling job %s failed: %s", job_id, exc)
        repo.mark_failed(job_id, str(exc))
    finally:
        _active_job_id = None


class JobService:
    def __init__(self, db: Database):
        self.db = db
        self.repo = JobRepository(db)

    def submit_job(self, request: CreateJobRequest) -> JobStatusResponse:
        global _active_job_id

        with _job_lock:
            if _active_job_id is not None:
                doc = self.repo.get_by_id(_active_job_id)
                if doc and doc.get("status") in ("pending", "running"):
                    raise ValueError(
                        f"Another scheduling job is already running ({_active_job_id}). "
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
                "shortTermFile": request.short_term_file,
                "saveCsv": request.save_csv,
                "outputDir": request.output_dir,
                "tee": request.tee,
                "planStartDate": request.plan_start_date,
                "horizonDays": request.horizon_days,
                "errorMessage": None,
                "resultsCollection": "scheduling_results",
            }
            self.repo.insert(job_doc)
            _active_job_id = job_id

        thread = threading.Thread(
            target=_run_job_thread,
            args=(self.db, job_id, request),
            daemon=True,
            name=f"scheduling-job-{job_id}",
        )
        thread.start()

        doc = self.repo.get_by_id(job_id)
        return self._doc_to_response(doc)

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        doc = self.repo.get_by_id(job_id)
        if doc is None:
            return None
        return self._doc_to_response(doc)

    def list_jobs(self, status_filter: Optional[str] = None) -> List[JobStatusResponse]:
        docs = self.repo.list_all(status_filter)
        return [self._doc_to_response(doc) for doc in docs]

    def cancel_job(self, job_id: str) -> Optional[JobStatusResponse]:
        cancelled = self.repo.mark_cancelled(job_id)
        if not cancelled:
            return None
        doc = self.repo.get_by_id(job_id)
        return self._doc_to_response(doc) if doc else None

    @staticmethod
    def _doc_to_response(doc: Dict[str, Any]) -> JobStatusResponse:
        now = _now()
        return JobStatusResponse(
            jobId=str(doc["_id"]),
            status=doc.get("status", JobStatus.PENDING.value),
            runId=doc.get("runId", ""),
            createdAt=doc.get("createdAt", now),
            updatedAt=doc.get("updatedAt", now),
            startedAt=doc.get("startedAt"),
            finishedAt=doc.get("finishedAt"),
            shortTermFile=doc.get("shortTermFile"),
            saveCsv=doc.get("saveCsv", False),
            outputDir=doc.get("outputDir", "outputs"),
            tee=doc.get("tee", False),
            planStartDate=doc.get("planStartDate", "2026-01-05"),
            horizonDays=doc.get("horizonDays", 12),
            errorMessage=doc.get("errorMessage"),
            resultsCollection=doc.get("resultsCollection", "scheduling_results"),
        )

