"""Repository layer for MixMetric collection operations."""

from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError, DuplicateKeyError


class MixMetricRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["mix_metrics"]
        # Ensure unique composite key enforced by using _id field as the composite
        self.collection.create_index([("_id", 1)], unique=True, name="uniq_metric_id")
        # Index skuKeys for fast lookup when denormalized
        self.collection.create_index([("skuKeys", 1)], name="idx_sku_keys")

    def create(self, metric_document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(metric_document)
            return metric_document
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating metric: {exc}")

    def get_by_id(self, metric_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": metric_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving metric: {exc}")

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find(criteria))
        except PyMongoError as exc:
            raise Exception(f"Database error searching metrics: {exc}")

    def update(self, metric_id: str, metric_document: Dict[str, Any]) -> bool:
        try:
            result = self.collection.replace_one({"_id": metric_id}, metric_document)
            return result.matched_count > 0
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating metric: {exc}")

    def delete(self, metric_id: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": metric_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting metric: {exc}")

    def delete_by_mix_id(self, mix_id: str) -> int:
        """Delete all metrics associated with a mixId.

        Returns the count of deleted documents.
        """
        try:
            result = self.collection.delete_many({"mixId": mix_id})
            return result.deleted_count
        except PyMongoError as exc:
            raise Exception(f"Database error deleting metrics by mixId: {exc}")

    def delete_by_bucket_id(self, bucket_id: str) -> int:
        """Delete all metrics associated with a bucketId."""
        try:
            result = self.collection.delete_many({"bucketId": bucket_id})
            return result.deleted_count
        except PyMongoError as exc:
            raise Exception(f"Database error deleting metrics by bucketId: {exc}")

    def delete_by_mix_ids(self, mix_ids: List[str]) -> int:
        """Delete all metrics whose mixId is in the provided list."""
        if not mix_ids:
            return 0
        try:
            result = self.collection.delete_many({"mixId": {"$in": mix_ids}})
            return result.deleted_count
        except PyMongoError as exc:
            raise Exception(f"Database error deleting metrics by mixIds: {exc}")
