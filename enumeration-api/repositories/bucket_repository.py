"""Repository layer for Bucket collection operations."""

from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError


class BucketRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["buckets"]

    def create(self, bucket_document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(bucket_document)
            return bucket_document
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating bucket: {exc}")

    def get_by_id(self, bucket_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": bucket_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving bucket: {exc}")

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find(criteria))
        except PyMongoError as exc:
            raise Exception(f"Database error searching buckets: {exc}")

    def update(self, bucket_id: str, bucket_document: Dict[str, Any]) -> bool:
        try:
            result = self.collection.replace_one({"_id": bucket_id}, bucket_document)
            return result.matched_count > 0
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating bucket: {exc}")

    def delete(self, bucket_id: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": bucket_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting bucket: {exc}")

