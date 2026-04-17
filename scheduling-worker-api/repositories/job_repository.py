from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError


class JobRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["scheduling_jobs"]
        self.results_collection: Collection = database["scheduling_results"]
        self.collection.create_index([("status", 1)], name="idx_scheduling_job_status")
        self.collection.create_index([("runId", 1)], name="idx_scheduling_job_run_id")
        self.collection.create_index([("createdAt", -1)], name="idx_scheduling_job_created_at")
        self.results_collection.create_index([("runId", 1)], name="idx_scheduling_result_run_id")
        self.results_collection.create_index([("jobId", 1)], name="idx_scheduling_result_job_id")

    def insert(self, job_document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(job_document)
            return job_document
        except PyMongoError as exc:
            raise Exception(f"Database error creating scheduling job: {exc}")

    def get_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": job_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving scheduling job: {exc}")

    def list_all(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            query: Dict[str, Any] = {}
            if status_filter:
                query["status"] = status_filter
            return list(self.collection.find(query).sort("createdAt", -1))
        except PyMongoError as exc:
            raise Exception(f"Database error listing scheduling jobs: {exc}")

    def mark_running(self, job_id: str) -> None:
        try:
            self.collection.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "running",
                        "startedAt": datetime.utcnow(),
                        "updatedAt": datetime.utcnow(),
                        "currentStage": "starting",
                        "stageMessage": "Scheduling worker accepted the job.",
                        "stageUpdatedAt": datetime.utcnow(),
                    }
                },
            )
        except PyMongoError as exc:
            raise Exception(f"Database error marking scheduling job running: {exc}")

    def update_progress(
        self,
        job_id: str,
        current_stage: str,
        stage_message: Optional[str] = None,
        stage_details: Optional[Dict[str, Any]] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> None:
        try:
            updates: Dict[str, Any] = {
                "currentStage": current_stage,
                "updatedAt": datetime.utcnow(),
                "stageUpdatedAt": datetime.utcnow(),
            }
            if stage_message is not None:
                updates["stageMessage"] = stage_message
            if stage_details is not None:
                updates["stageDetails"] = stage_details
            if timings is not None:
                updates["timings"] = timings

            self.collection.update_one({"_id": job_id}, {"$set": updates})
        except PyMongoError as exc:
            raise Exception(f"Database error updating scheduling job progress: {exc}")

    def mark_completed(self, job_id: str) -> None:
        try:
            self.collection.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "completed",
                        "finishedAt": datetime.utcnow(),
                        "updatedAt": datetime.utcnow(),
                        "currentStage": "completed",
                        "stageMessage": "Scheduling job completed successfully.",
                        "stageUpdatedAt": datetime.utcnow(),
                    }
                },
            )
        except PyMongoError as exc:
            raise Exception(f"Database error marking scheduling job completed: {exc}")

    def mark_failed(self, job_id: str, error_message: str, error_traceback: Optional[str] = None) -> None:
        try:
            self.collection.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "finishedAt": datetime.utcnow(),
                        "currentStage": "failed",
                        "stageMessage": error_message,
                        "stageUpdatedAt": datetime.utcnow(),
                        "errorMessage": error_message,
                        "errorTraceback": error_traceback,
                        "updatedAt": datetime.utcnow(),
                    }
                },
            )
        except PyMongoError as exc:
            raise Exception(f"Database error marking scheduling job failed: {exc}")

    def mark_cancelled(self, job_id: str) -> bool:
        try:
            result = self.collection.update_one(
                {"_id": job_id, "status": {"$in": ["pending", "running"]}},
                {
                    "$set": {
                        "status": "cancelled",
                        "finishedAt": datetime.utcnow(),
                        "updatedAt": datetime.utcnow(),
                        "currentStage": "cancelled",
                        "stageMessage": "Scheduling job was cancelled.",
                        "stageUpdatedAt": datetime.utcnow(),
                    }
                },
            )
            return result.modified_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error cancelling scheduling job: {exc}")

    def store_results(self, job_id: str, run_id: str, outputs: Dict[str, Any]) -> None:
        try:
            self.results_collection.insert_one(
                {
                    "_id": job_id,
                    "jobId": job_id,
                    "runId": run_id,
                    "createdAt": datetime.utcnow(),
                    "outputs": outputs,
                }
            )
        except PyMongoError as exc:
            raise Exception(f"Database error storing scheduling results: {exc}")
