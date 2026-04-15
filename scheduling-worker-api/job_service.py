from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import httpx
from bson import ObjectId
from pymongo.database import Database

from config import get_settings
from models.job import ArtifactFile, CreateJobRequest, JobStatus, JobStatusResponse
from repositories.job_repository import JobRepository
from storage import build_download_url, dataframe_to_csv_bytes, upload_csv_artifacts

logger = logging.getLogger(__name__)

_job_lock = threading.Lock()
_active_job_id: Optional[str] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_scheduling_path() -> None:
    root = Path(__file__).resolve().parents[1]
    worker_path = root / "scheduling-worker-api"
    scheduling_path = root / "scheduling"
    if str(worker_path) not in sys.path:
        sys.path.insert(0, str(worker_path))
    if str(scheduling_path) not in sys.path:
        sys.path.insert(1, str(scheduling_path))


def _serialize_dataframe(df) -> List[Dict[str, Any]]:
    try:
        return json.loads(df.to_json(orient="records", date_format="iso"))
    except Exception:
        return []


def _artifact_file_name(artifact_name: str) -> str:
    return f"{artifact_name}.csv"


def _build_artifact_prefix(request: CreateJobRequest) -> str:
    prefix = request.output_dir.strip().strip("/")
    if prefix:
        return prefix
    return "outputs"


def _normalize_sku_ids(sku_ids: List[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for sku_id in sku_ids:
        cleaned = str(sku_id).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def _fetch_sku_document(sku_id: str) -> Dict[str, Any]:
    settings = get_settings()
    url = f"{settings.enumeration_api_url.rstrip('/')}/skus/{sku_id}"
    response = httpx.get(url, timeout=15.0)
    if response.status_code == 404:
        raise ValueError(f"SKU {sku_id} was not found in the Enumeration API")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"SKU {sku_id} returned an invalid payload from the Enumeration API")
    return payload


def _validate_job_skus(plant_id: str, sku_ids: List[str]) -> List[str]:
    normalized_ids = _normalize_sku_ids(sku_ids)
    if not normalized_ids:
        raise ValueError("skuIds must contain at least one SKU")

    mismatches: List[str] = []
    for sku_id in normalized_ids:
        sku = _fetch_sku_document(sku_id)
        sku_plant = str(sku.get("prodPlant", "")).strip()
        if not sku_plant:
            raise ValueError(f"SKU {sku_id} does not have a prodPlant value in the Enumeration API")
        if sku_plant != plant_id:
            mismatches.append(f"{sku_id} ({sku_plant})")

    if mismatches:
        raise ValueError(
            "All skuIds on a scheduling job must belong to the same plant as plantId. "
            f"Plant {plant_id} did not match: {', '.join(mismatches)}"
        )

    return normalized_ids


def _run_job_thread(db: Database, job_id: str, request: CreateJobRequest) -> None:
    global _active_job_id
    repo = JobRepository(db)
    try:
        repo.mark_running(job_id)
        _ensure_scheduling_path()
        from run_model import run_for_job  # type: ignore

        results = run_for_job(
            job=request,
            short_term_file=request.short_term_file,
            output_dir=request.output_dir,
            tee=request.tee,
            plan_start_date=request.plan_start_date,
            horizon_days=request.horizon_days,
            plant_id=request.plant_id,
            sku_ids=request.sku_ids,
        )

        artifact_bucket = None
        artifact_prefix = None
        artifact_keys: List[str] = []
        artifact_files: List[Dict[str, Any]] = []
        if request.save_csv:
            csv_payloads = {
                name: dataframe_to_csv_bytes(df)
                for name, df in results.get("outputs", {}).items()
            }
            artifact_prefix = _build_artifact_prefix(request)
            artifact_bucket, artifact_prefix, artifact_keys = upload_csv_artifacts(
                run_id=request.run_id,
                prefix=artifact_prefix,
                csv_payloads=csv_payloads,
            )
            artifact_files = [
                {
                    "artifactName": name,
                    "fileName": _artifact_file_name(name),
                    "bucket": artifact_bucket,
                    "key": key,
                    "downloadUrl": build_download_url(artifact_bucket, key),
                }
                for name, key in zip(results.get("outputs", {}).keys(), artifact_keys)
            ]
            logger.info("Uploaded scheduling CSV artifacts to bucket=%s prefix=%s", artifact_bucket, artifact_prefix)

        output_payload = cast(Dict[str, Any], {})
        for name, df in results.get("outputs", {}).items():
            output_payload[name] = _serialize_dataframe(df)

        inputs_payload: Dict[str, Any] = {}
        inputs_payload["planStartDate"] = request.plan_start_date
        inputs_payload["horizonDays"] = request.horizon_days
        inputs_payload["shortTermFile"] = request.short_term_file
        inputs_payload["plantId"] = request.plant_id
        inputs_payload["skuIds"] = request.sku_ids

        artifact_payload: Dict[str, Any] = {}
        artifact_payload["bucket"] = artifact_bucket
        artifact_payload["prefix"] = artifact_prefix
        artifact_payload["keys"] = artifact_keys
        artifact_payload["files"] = artifact_files
        artifact_payload["savedCsv"] = request.save_csv
        output_payload["inputs"] = inputs_payload  # type: ignore[assignment]
        output_payload["artifacts"] = artifact_payload  # type: ignore[assignment]
        repo.store_results(job_id, request.run_id, output_payload)

        doc = repo.get_by_id(job_id)
        if doc and doc.get("status") not in ("cancelled", "failed"):
            repo.collection.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "artifactBucket": artifact_bucket,
                        "artifactPrefix": artifact_prefix,
                        "artifactKeys": artifact_keys,
                        "artifactFiles": artifact_files,
                        "updatedAt": _now(),
                    }
                },
            )
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
        sku_ids = _validate_job_skus(request.plant_id, request.sku_ids)

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
                "plantId": request.plant_id,
                "skuIds": sku_ids,
                "errorMessage": None,
                "resultsCollection": "scheduling_results",
                "artifactBucket": None,
                "artifactPrefix": None,
                "artifactKeys": [],
                "artifactFiles": [],
            }
            self.repo.insert(job_doc)
            _active_job_id = job_id

        thread = threading.Thread(
            target=_run_job_thread,
            args=(self.db, job_id, request.model_copy(update={"sku_ids": sku_ids})),
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

    def get_artifacts(self, job_id: str) -> Optional[List[ArtifactFile]]:
        doc = self.repo.get_by_id(job_id)
        if doc is None:
            return None
        return self._doc_to_response(doc).artifact_files

    @staticmethod
    def _doc_to_response(doc: Dict[str, Any]) -> JobStatusResponse:
        now = _now()
        artifact_files = []
        for artifact in doc.get("artifactFiles", []):
            artifact_data = dict(artifact)
            bucket = artifact_data.get("bucket")
            key = artifact_data.get("key")
            if bucket and key:
                artifact_data["downloadUrl"] = build_download_url(bucket, key)
            artifact_files.append(ArtifactFile(**artifact_data))
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
            plantId=doc.get("plantId"),
            skuIds=doc.get("skuIds", []),
            errorMessage=doc.get("errorMessage"),
            resultsCollection=doc.get("resultsCollection", "scheduling_results"),
            artifactBucket=doc.get("artifactBucket"),
            artifactPrefix=doc.get("artifactPrefix"),
            artifactKeys=doc.get("artifactKeys", []),
            artifactFiles=artifact_files,
        )
