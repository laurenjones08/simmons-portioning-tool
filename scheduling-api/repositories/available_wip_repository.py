from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError


class AvailableWIPRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["available_wip"]
        self.collection.create_index(
            [("plantName", 1), ("bucketId", 1)],
            unique=True,
            name="uniq_plant_bucket",
        )
        self.collection.create_index([("plantName", 1)], name="idx_available_wip_plant_name")
        self.collection.create_index([("bucketId", 1)], name="idx_available_wip_bucket_id")

    def create(self, document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(document)
            return document
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating available WIP: {exc}")

    def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": document_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving available WIP: {exc}")

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find(criteria))
        except PyMongoError as exc:
            raise Exception(f"Database error searching available WIP: {exc}")

    def update(self, document_id: str, document: Dict[str, Any]) -> bool:
        try:
            result = self.collection.replace_one({"_id": document_id}, document)
            return result.matched_count > 0
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating available WIP: {exc}")

    def delete(self, document_id: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": document_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting available WIP: {exc}")

