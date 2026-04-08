"""Repository layer for CutStrategy collection operations."""

from __future__ import annotations

import hashlib
from bson import ObjectId
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError


class CutStrategyRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["cut_strategies"]

        # Backfill first so existing documents have partsKey before unique index creation.
        self.backfill_parts_key()

        # Then ensure indexes.
        self._ensure_indexes()

    @staticmethod
    def generate_parts_key(parts: List[str]) -> str:
        """
        Generate a deterministic key for parts so uniqueness is enforced
        regardless of array order.

        Example:
            ["A", "B"] and ["B", "A"] -> same key
        """
        normalized = sorted(set(parts))
        joined = "|".join(normalized)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_parts(parts: Any) -> List[str]:
        """Normalize part codes to trimmed uppercase strings."""
        if not isinstance(parts, list):
            raise ValueError("'parts' must be a list")

        normalized_parts: List[str] = []
        for part in parts:
            if not isinstance(part, str):
                part = str(part)
            normalized_parts.append(part.strip().upper())
        return normalized_parts

    def _prepare_strategy_document(self, strategy_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a copy of the document with partsKey populated from parts.
        """
        prepared = dict(strategy_document)

        normalized_parts = self._normalize_parts(prepared.get("parts", []))
        prepared["parts"] = normalized_parts
        prepared["partsKey"] = self.generate_parts_key(normalized_parts)
        if prepared.get("lineType") is None and prepared.get("mfgType") is not None:
            prepared["lineType"] = prepared["mfgType"]
        prepared.pop("mfgType", None)
        return prepared

    def backfill_parts_key(self) -> None:
        """
        Normalize parts and populate partsKey for legacy documents.
        Safe to run on every container startup.
        """
        try:
            cursor = self.collection.find(
                {},
                {"_id": 1, "parts": 1, "partsKey": 1, "lineType": 1, "mfgType": 1},
            )

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
                    self.collection.update_one(
                        {"_id": doc["_id"]},
                        {"$set": updates},
                    )
        except PyMongoError as exc:
            raise Exception(f"Failed to backfill partsKey: {exc}") from exc

    def _ensure_indexes(self) -> None:
        existing_indexes = self.collection.index_information()

        if "uniq_part_lineType_nugget" not in existing_indexes:
            self.collection.create_index(
                [("partsKey", 1), ("lineType", 1), ("hasNugget", 1)],
                unique=True,
                name="uniq_part_lineType_nugget",
                background=True,
            )

    @staticmethod
    def _normalize_document_id(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return a copy with `_id` coerced to string for API/model compatibility."""
        if document is None:
            return None
        normalized = dict(document)
        if "_id" in normalized and normalized["_id"] is not None:
            normalized["_id"] = str(normalized["_id"])

        raw_parts = normalized.get("parts")
        if isinstance(raw_parts, list):
            normalized["parts"] = [str(part).strip().upper() for part in raw_parts]
        if normalized.get("lineType") is None and normalized.get("mfgType") is not None:
            normalized["lineType"] = normalized["mfgType"]

        return normalized

    @staticmethod
    def _id_filter(strategy_id: str) -> Dict[str, Any]:
        """Match by exact string id, and by ObjectId when strategy_id is a valid hex id."""
        filters: List[Dict[str, Any]] = [{"_id": strategy_id}]
        if ObjectId.is_valid(strategy_id):
            filters.append({"_id": ObjectId(strategy_id)})
        if len(filters) == 1:
            return filters[0]
        return {"$or": filters}

    def create(self, strategy_document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prepared = self._prepare_strategy_document(strategy_document)
            self.collection.insert_one(prepared)
            return self._normalize_document_id(prepared)
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating cut strategy: {exc}") from exc

    def save(self, strategy_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert or replace a cut strategy.
        partsKey is always recalculated before saving.

        Requires strategy_document to contain '_id'.
        """
        try:
            if "_id" not in strategy_document:
                raise ValueError("strategy_document must contain '_id' for save()")

            prepared = self._prepare_strategy_document(strategy_document)

            self.collection.replace_one(
                {"_id": prepared["_id"]},
                prepared,
                upsert=True,
            )
            return self._normalize_document_id(prepared)

        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error saving cut strategy: {exc}") from exc

    def get_by_id(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        try:
            found = self.collection.find_one(self._id_filter(strategy_id))
            return self._normalize_document_id(found)
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving cut strategy: {exc}") from exc

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return [self._normalize_document_id(doc) for doc in self.collection.find(criteria)]
        except PyMongoError as exc:
            raise Exception(f"Database error searching cut strategies: {exc}") from exc

    def update(self, strategy_id: str, strategy_document: Dict[str, Any]) -> bool:
        """
        Fully replace an existing strategy.
        partsKey is recalculated so changes to parts stay consistent.
        """
        try:
            prepared = self._prepare_strategy_document(strategy_document)
            prepared["_id"] = strategy_id

            result = self.collection.replace_one(
                self._id_filter(strategy_id),
                prepared,
            )
            return result.matched_count > 0

        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating cut strategy: {exc}") from exc

    def delete(self, strategy_id: str) -> bool:
        try:
            result = self.collection.delete_one(self._id_filter(strategy_id))
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting cut strategy: {exc}") from exc
