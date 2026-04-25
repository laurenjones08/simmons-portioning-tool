from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

_DEBUG_DUMP_CHUNK_SIZE_BYTES = 1_000_000
_DEBUG_DUMP_INLINE_LIMIT_BYTES = 4_000_000


class JobRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["scheduling_jobs"]
        self.debug_collection: Collection = database["scheduling_debug_payloads"]
        self.debug_chunk_collection: Collection = database["scheduling_debug_payload_chunks"]
        self.collection.create_index([("status", 1)], name="idx_scheduling_job_status")
        self.collection.create_index([("runId", 1)], name="idx_scheduling_job_run_id")
        self.collection.create_index([("createdAt", -1)], name="idx_scheduling_job_created_at")
        self.debug_collection.create_index([("jobId", 1)], name="idx_scheduling_debug_job_id")
        self.debug_collection.create_index([("expiresAt", 1)], name="ttl_scheduling_debug_expires", expireAfterSeconds=0)
        self.debug_chunk_collection.create_index(
            [("jobId", 1), ("chunkIndex", 1)],
            name="idx_scheduling_debug_chunk_job_idx",
            unique=True,
        )
        self.debug_chunk_collection.create_index(
            [("expiresAt", 1)],
            name="ttl_scheduling_debug_chunk_expires",
            expireAfterSeconds=0,
        )

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

    @staticmethod
    def _serialize_debug_payload(payload: Dict[str, Any]) -> bytes:
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def _replace_debug_metadata(
        self,
        job_id: str,
        run_id: str,
        created_at: datetime,
        expires_at: datetime,
        payload: Optional[Dict[str, Any]],
        storage_kind: str,
        chunk_count: int = 0,
    ) -> None:
        self.debug_collection.replace_one(
            {"jobId": job_id},
            {
                "_id": job_id,
                "jobId": job_id,
                "runId": run_id,
                "createdAt": created_at,
                "expiresAt": expires_at,
                "payload": payload,
                "storageKind": storage_kind,
                "chunkCount": chunk_count,
            },
            upsert=True,
        )

    def store_debug_dump(self, job_id: str, run_id: str, payload: Dict[str, Any], ttl_minutes: int = 5) -> None:
        try:
            created_at = datetime.utcnow()
            expires_at = created_at + timedelta(minutes=ttl_minutes)
            self.debug_chunk_collection.delete_many({"jobId": job_id})

            serialized_payload = self._serialize_debug_payload(payload)
            if len(serialized_payload) <= _DEBUG_DUMP_INLINE_LIMIT_BYTES:
                self._replace_debug_metadata(
                    job_id=job_id,
                    run_id=run_id,
                    created_at=created_at,
                    expires_at=expires_at,
                    payload=payload,
                    storage_kind="inline",
                )
                return

            chunks = [
                {
                    "_id": f"{job_id}:{chunk_index}",
                    "jobId": job_id,
                    "runId": run_id,
                    "chunkIndex": chunk_index,
                    "createdAt": created_at,
                    "expiresAt": expires_at,
                    "data": serialized_payload[start : start + _DEBUG_DUMP_CHUNK_SIZE_BYTES].decode("utf-8"),
                }
                for chunk_index, start in enumerate(range(0, len(serialized_payload), _DEBUG_DUMP_CHUNK_SIZE_BYTES))
            ]
            if chunks:
                self.debug_chunk_collection.insert_many(chunks, ordered=True)

            self._replace_debug_metadata(
                job_id=job_id,
                run_id=run_id,
                created_at=created_at,
                expires_at=expires_at,
                payload=None,
                storage_kind="chunked",
                chunk_count=len(chunks),
            )
        except PyMongoError as exc:
            raise Exception(f"Database error storing scheduling debug dump: {exc}")

    def get_debug_dump(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = self.debug_collection.find_one({"jobId": job_id})
            if doc is None:
                return None

            if doc.get("storageKind") != "chunked":
                return doc

            chunk_docs = list(
                self.debug_chunk_collection.find({"jobId": job_id}).sort("chunkIndex", 1)
            )
            if not chunk_docs:
                return None

            payload_json = "".join(str(chunk.get("data", "")) for chunk in chunk_docs)
            hydrated = dict(doc)
            hydrated["payload"] = json.loads(payload_json)
            return hydrated
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving scheduling debug dump: {exc}")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise Exception(f"Database error retrieving scheduling debug dump: {exc}")
