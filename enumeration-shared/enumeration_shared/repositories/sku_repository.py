"""
SKU Repository - Data Access Layer for SKU Collection.
"""

from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError


class SKURepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["skus"]

    def find_by_trade_number(self, trade_number: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": trade_number})
        except PyMongoError as e:
            raise Exception(f"Database error finding SKU by trade number: {str(e)}")

    def find_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find(criteria))
        except PyMongoError as e:
            raise Exception(f"Database error searching SKUs: {str(e)}")

    def insert(self, sku_document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(sku_document)
            return sku_document
        except DuplicateKeyError:
            raise
        except PyMongoError as e:
            raise Exception(f"Database error inserting SKU: {str(e)}")

    def update(self, trade_number: str, sku_document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            result = self.collection.replace_one({"_id": trade_number}, sku_document)
            return sku_document if result.matched_count > 0 else None
        except PyMongoError as e:
            raise Exception(f"Database error updating SKU: {str(e)}")

    def insert_many(self, sku_documents: List[Dict[str, Any]]) -> int:
        try:
            if not sku_documents:
                return 0
            result = self.collection.insert_many(sku_documents, ordered=True)
            return len(result.inserted_ids)
        except DuplicateKeyError:
            raise
        except PyMongoError as e:
            raise Exception(f"Database error during bulk insert: {str(e)}")

    def find_all(self) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find())
        except PyMongoError as e:
            raise Exception(f"Database error retrieving all SKUs: {str(e)}")

    def delete_by_trade_number(self, trade_number: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": trade_number})
            return result.deleted_count > 0
        except PyMongoError as e:
            raise Exception(f"Database error deleting SKU: {str(e)}")
