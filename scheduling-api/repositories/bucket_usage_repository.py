from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError, DuplicateKeyError, PyMongoError


class BucketUsageRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["bucket_usage"]
        self.collection.create_index(
            [("bucketId", 1), ("date", 1)],
            unique=True,
            name="uniq_bucket_date",
        )
        self.collection.create_index([("bucketId", 1)], name="idx_bucket_usage_bucket_id")
        self.collection.create_index([("date", 1)], name="idx_bucket_usage_date")

    def create(self, document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(document)
            return document
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating bucket usage: {exc}")

    def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": document_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving bucket usage: {exc}")

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find(criteria))
        except PyMongoError as exc:
            raise Exception(f"Database error searching bucket usage: {exc}")

    def update(self, document_id: str, document: Dict[str, Any]) -> bool:
        try:
            result = self.collection.replace_one({"_id": document_id}, document)
            return result.matched_count > 0
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating bucket usage: {exc}")

    def delete(self, document_id: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": document_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting bucket usage: {exc}")

    def delete_by_dates(self, dates: List[str]) -> int:
        try:
            result = self.collection.delete_many({"date": {"$in": dates}})
            return result.deleted_count
        except PyMongoError as exc:
            raise Exception(f"Database error deleting bucket usage by date: {exc}")

    def bulk_create(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []
        for doc in documents:
            if "_id" not in doc:
                doc["_id"] = str(ObjectId())
        try:
            self.collection.insert_many(documents, ordered=False)
        except BulkWriteError as exc:
            raise Exception(f"Database error bulk creating bucket usage rows: {exc.details}")
        except PyMongoError as exc:
            raise Exception(f"Database error bulk creating bucket usage rows: {exc}")
        return documents

