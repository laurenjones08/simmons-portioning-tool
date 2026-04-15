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
                {"$set": {"status": "running", "startedAt": datetime.utcnow(), "updatedAt": datetime.utcnow()}},
            )
        except PyMongoError as exc:
            raise Exception(f"Database error marking scheduling job running: {exc}")

    def mark_completed(self, job_id: str) -> None:
        try:
            self.collection.update_one(
                {"_id": job_id},
                {"$set": {"status": "completed", "finishedAt": datetime.utcnow(), "updatedAt": datetime.utcnow()}},
            )
        except PyMongoError as exc:
            raise Exception(f"Database error marking scheduling job completed: {exc}")

    def mark_failed(self, job_id: str, error_message: str) -> None:
        try:
            self.collection.update_one(
                {"_id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "finishedAt": datetime.utcnow(),
                        "errorMessage": error_message,
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
                {"$set": {"status": "cancelled", "finishedAt": datetime.utcnow(), "updatedAt": datetime.utcnow()}},
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

