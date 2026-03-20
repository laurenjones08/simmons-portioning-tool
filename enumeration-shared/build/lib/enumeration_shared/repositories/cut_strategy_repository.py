"""Repository layer for CutStrategy collection operations."""

from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError


class CutStrategyRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["cut_strategies"]
        self.collection.create_index(
            [("mfgType", 1), ("name", 1)],
            unique=True,
            name="uniq_mfg_type_name",
        )

    def create(self, strategy_document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(strategy_document)
            return strategy_document
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating cut strategy: {exc}")

    def get_by_id(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": strategy_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving cut strategy: {exc}")

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find(criteria))
        except PyMongoError as exc:
            raise Exception(f"Database error searching cut strategies: {exc}")

    def update(self, strategy_id: str, strategy_document: Dict[str, Any]) -> bool:
        try:
            result = self.collection.replace_one({"_id": strategy_id}, strategy_document)
            return result.matched_count > 0
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating cut strategy: {exc}")

    def delete(self, strategy_id: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": strategy_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting cut strategy: {exc}")
