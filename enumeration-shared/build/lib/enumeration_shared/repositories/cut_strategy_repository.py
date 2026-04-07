"""Repository layer for CutStrategy collection operations."""

import hashlib
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError


class CutStrategyRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["cut_strategies"]

        # Keep legacy documents compatible before index checks.
        self._ensure_compat_fields()
        self._ensure_indexes()

    @staticmethod
    def generate_parts_key(parts: List[str]) -> str:
        normalized = sorted(set(parts))
        joined = "|".join(normalized)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_parts(parts: Any) -> List[str]:
        if not isinstance(parts, list):
            raise ValueError("'parts' must be a list")

        normalized_parts: List[str] = []
        for part in parts:
            if not isinstance(part, str):
                part = str(part)
            normalized_parts.append(part.strip().upper())
        return normalized_parts

    def _ensure_compat_fields(self) -> None:
        """Backfill partsKey/lineType so modern indexing works with legacy records."""
        try:
            cursor = self.collection.find({}, {"_id": 1, "parts": 1, "partsKey": 1, "lineType": 1, "mfgType": 1})
            for doc in cursor:
                raw_parts = doc.get("parts", [])
                if not isinstance(raw_parts, list):
                    continue

                normalized_parts = self._normalize_parts(raw_parts)
                parts_key = self.generate_parts_key(normalized_parts)

                updates: Dict[str, Any] = {}
                if raw_parts != normalized_parts:
                    updates["parts"] = normalized_parts
                if doc.get("partsKey") != parts_key:
                    updates["partsKey"] = parts_key
                if doc.get("lineType") is None and doc.get("mfgType") is not None:
                    updates["lineType"] = doc["mfgType"]

                if updates:
                    self.collection.update_one({"_id": doc["_id"]}, {"$set": updates})
        except PyMongoError as exc:
            raise Exception(f"Failed to backfill cut strategy compatibility fields: {exc}") from exc

    def _ensure_indexes(self) -> None:
        existing_indexes = self.collection.index_information()

        # Legacy index from older worker code path; remove to avoid false duplicates by name.
        if "uniq_mfg_type_name" in existing_indexes:
            self.collection.drop_index("uniq_mfg_type_name")

        if "uniq_part_lineType_nugget" not in existing_indexes:
            self.collection.create_index(
                [("partsKey", 1), ("lineType", 1), ("hasNugget", 1)],
                unique=True,
                name="uniq_part_lineType_nugget",
                background=True,
            )

    def _prepare_strategy_document(self, strategy_document: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(strategy_document)
        normalized_parts = self._normalize_parts(prepared.get("parts", []))
        prepared["parts"] = normalized_parts
        prepared["partsKey"] = self.generate_parts_key(normalized_parts)
        if prepared.get("lineType") is None and prepared.get("mfgType") is not None:
            prepared["lineType"] = prepared["mfgType"]
        prepared.pop("mfgType", None)
        return prepared

    def create(self, strategy_document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prepared = self._prepare_strategy_document(strategy_document)
            self.collection.insert_one(prepared)
            return prepared
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
            prepared = self._prepare_strategy_document(strategy_document)
            prepared["_id"] = strategy_id
            result = self.collection.replace_one({"_id": strategy_id}, prepared)
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
