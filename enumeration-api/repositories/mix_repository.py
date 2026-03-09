"""Repository layer for MIX collection operations."""

from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError, DuplicateKeyError


class MixRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["mixes"]
        self.collection.create_index(
            [("mfgType", 1), ("skuSetKey", 1)],
            unique=True,
            name="uniq_mfg_type_sku_set_key",
        )

    def create(self, mix_document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(mix_document)
            return mix_document
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating mix: {exc}")

    def get_by_id(self, mix_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": mix_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving mix: {exc}")

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find(criteria))
        except PyMongoError as exc:
            raise Exception(f"Database error searching mixes: {exc}")

    def update(self, mix_id: str, mix_document: Dict[str, Any]) -> bool:
        try:
            result = self.collection.replace_one({"_id": mix_id}, mix_document)
            return result.matched_count > 0
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating mix: {exc}")

    def delete(self, mix_id: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": mix_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting mix: {exc}")

    def delete_by_sku_trade_number(self, sku_trade_number: str) -> int:
        """
        Delete all mixes that contain the specified SKU trade number.

        Args:
            sku_trade_number: The SKU trade number to search for in mixes

        Returns:
            Count of mixes deleted
        """
        try:
            # Find all mixes where the SKU trade number exists as a key in the skus map
            result = self.collection.delete_many(
                {f"skus.{sku_trade_number}": {"$exists": True}}
            )
            return result.deleted_count
        except PyMongoError as exc:
            raise Exception(f"Database error deleting mixes by SKU: {exc}")
