from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError


class SchedulingDecisionRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["scheduling_decisions"]
        self.collection.create_index(
            [("mixId", 1), ("lineId", 1), ("date", 1)],
            unique=True,
            name="uniq_mix_line_date",
        )
        self.collection.create_index([("mixId", 1)], name="idx_scheduling_decision_mix_id")
        self.collection.create_index([("lineId", 1)], name="idx_scheduling_decision_line_id")
        self.collection.create_index([("date", 1)], name="idx_scheduling_decision_date")

    def create(self, document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(document)
            return document
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating scheduling decision: {exc}")

    def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": document_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving scheduling decision: {exc}")

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find(criteria))
        except PyMongoError as exc:
            raise Exception(f"Database error searching scheduling decisions: {exc}")

    def update(self, document_id: str, document: Dict[str, Any]) -> bool:
        try:
            result = self.collection.replace_one({"_id": document_id}, document)
            return result.matched_count > 0
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating scheduling decision: {exc}")

    def delete(self, document_id: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": document_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting scheduling decision: {exc}")

    def delete_by_dates(self, dates: List[str]) -> int:
        try:
            result = self.collection.delete_many({"date": {"$in": dates}})
            return result.deleted_count
        except PyMongoError as exc:
            raise Exception(f"Database error deleting scheduling decisions by date: {exc}")

    def bulk_create(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []
        for doc in documents:
            if "_id" not in doc:
                doc["_id"] = str(ObjectId())
        try:
            self.collection.insert_many(documents, ordered=False)
        except Exception:
            pass
        return documents

