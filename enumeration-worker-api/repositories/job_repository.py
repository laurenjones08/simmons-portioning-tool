"""Repository layer for the job_status collection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobRepository:
    COLLECTION = "job_status"

    def __init__(self, database: Database):
        self.collection: Collection = database[self.COLLECTION]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        existing_indexes = self.collection.index_information()

        def index_exists(keys):
            for idx in existing_indexes.values():
                if idx.get("key") == keys:
                    return True
            return False

        if not index_exists([("status", 1)]):
            self.collection.create_index([("status", 1)], name="idx_job_status")

        if not index_exists([("runId", 1)]):
            self.collection.create_index([("runId", 1)], name="idx_run_id")

        if not index_exists([("createdAt", -1)]):
            self.collection.create_index([("createdAt", -1)], name="idx_created_at")

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def insert(self, job_doc: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(job_doc)
            return job_doc
        except PyMongoError as exc:
            raise RuntimeError(f"DB error inserting job: {exc}") from exc

    def update_field(self, job_id: str, fields: Dict[str, Any]) -> None:
        try:
            fields["updatedAt"] = _now()
            self.collection.update_one({"_id": job_id}, {"$set": fields})
        except PyMongoError as exc:
            raise RuntimeError(f"DB error updating job {job_id}: {exc}") from exc

    @staticmethod
    def _stage_array_index(stage_index: int) -> int:
        """Translate 1-based stage numbers to 0-based Mongo array indexes."""
        return max(stage_index - 1, 0)

    def mark_running(self, job_id: str, sku_count: int, stages: List[Dict]) -> None:
        now = _now()
        self.update_field(
            job_id,
            {
                "status": "running",
                "startedAt": now,
                "skuCount": sku_count,
                "stages": stages,
            },
        )

    def mark_stage_running(self, job_id: str, stage_index: int, total: int) -> None:
        now = _now()
        array_index = self._stage_array_index(stage_index)
        self.collection.update_one(
            {"_id": job_id},
            {
                "$set": {
                    f"stages.{array_index}.status": "running",
                    f"stages.{array_index}.totalCombinations": total,
                    f"stages.{array_index}.startedAt": now,
                    "updatedAt": now,
                }
            },
        )

    def checkpoint_stage(self, job_id: str, stage_index: int, processed: int) -> None:
        now = _now()
        array_index = self._stage_array_index(stage_index)
        self.collection.update_one(
            {"_id": job_id},
            {
                "$set": {
                    f"stages.{array_index}.processedCombinations": processed,
                    "updatedAt": now,
                }
            },
        )

    def mark_stage_complete(self, job_id: str, stage_index: int, processed: int) -> None:
        now = _now()
        array_index = self._stage_array_index(stage_index)
        self.collection.update_one(
            {"_id": job_id},
            {
                "$set": {
                    f"stages.{array_index}.status": "completed",
                    f"stages.{array_index}.processedCombinations": processed,
                    f"stages.{array_index}.finishedAt": now,
                    "updatedAt": now,
                }
            },
        )

    def mark_completed(self, job_id: str) -> None:
        now = _now()
        self.update_field(job_id, {"status": "completed", "finishedAt": now})

    def mark_failed(self, job_id: str, error: str) -> None:
        now = _now()
        self.update_field(job_id, {"status": "failed", "finishedAt": now, "errorMessage": error})

    def mark_cancelled(self, job_id: str) -> bool:
        try:
            now = _now()
            result = self.collection.update_one(
                {"_id": job_id, "status": {"$in": ["pending", "running"]}},
                {"$set": {"status": "cancelled", "finishedAt": now, "updatedAt": now}},
            )
            return result.modified_count > 0
        except PyMongoError as exc:
            raise RuntimeError(f"DB error cancelling job {job_id}: {exc}") from exc

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": job_id})
        except PyMongoError as exc:
            raise RuntimeError(f"DB error fetching job {job_id}: {exc}") from exc

    def list_all(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            query: Dict[str, Any] = {}
            if status_filter:
                query["status"] = status_filter
            return list(self.collection.find(query).sort("createdAt", -1))
        except PyMongoError as exc:
            raise RuntimeError(f"DB error listing jobs: {exc}") from exc

    def is_cancelled(self, job_id: str) -> bool:
        doc = self.get_by_id(job_id)
        return doc is not None and doc.get("status") == "cancelled"

