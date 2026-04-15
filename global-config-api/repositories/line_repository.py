"""Repository for production line documents."""

from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError, PyMongoError


class LineRepository:
    """Data access layer for the lines collection."""

    def __init__(self, database: Database):
        self.collection: Collection = database["lines"]
        self.collection.create_index("lineId", name="uniq_line_id", unique=True)
        self.collection.create_index("isActive", name="idx_lines_active")
        self.backfill_line_type()
        self.backfill_units_available()

    @staticmethod
    def _infer_line_type(document: Dict[str, Any]) -> Optional[str]:
        line_type = document.get("lineType") or document.get("mfgType")
        if isinstance(line_type, str) and line_type.strip():
            return line_type.strip()

        line_id = document.get("lineId")
        if not isinstance(line_id, str):
            return None

        normalized_line_id = line_id.strip().upper()
        if normalized_line_id in {"DB20", "DSI884", "DSI888"}:
            return normalized_line_id
        return None

    def _normalize_document(self, document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if document is None:
            return None

        normalized = dict(document)
        inferred_line_type = self._infer_line_type(normalized)
        if inferred_line_type:
            normalized["lineType"] = inferred_line_type
        if "unitsAvailable" not in normalized:
            normalized["unitsAvailable"] = 0
        return normalized

    def backfill_line_type(self) -> None:
        try:
            cursor = self.collection.find({}, {"_id": 1, "lineId": 1, "lineType": 1, "mfgType": 1})
            for document in cursor:
                inferred_line_type = self._infer_line_type(document)
                if inferred_line_type and document.get("lineType") != inferred_line_type:
                    self.collection.update_one(
                        {"_id": document["_id"]},
                        {"$set": {"lineType": inferred_line_type}},
                    )
        except PyMongoError as exc:
            raise Exception(f"Database error backfilling lineType: {exc}") from exc

    def backfill_units_available(self) -> None:
        try:
            self.collection.update_many(
                {"unitsAvailable": {"$exists": False}},
                {"$set": {"unitsAvailable": 0}},
            )
        except PyMongoError as exc:
            raise Exception(f"Database error backfilling unitsAvailable: {exc}") from exc

    def create(self, document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(document)
            return document
        except DuplicateKeyError as exc:
            raise ValueError(f"Line with id '{document['lineId']}' already exists") from exc
        except PyMongoError as exc:
            raise Exception(f"Database error creating line: {exc}") from exc

    def find_by_id(self, line_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._normalize_document(self.collection.find_one({"lineId": line_id}))
        except PyMongoError as exc:
            raise Exception(f"Database error finding line '{line_id}': {exc}") from exc

    def find_all(self) -> List[Dict[str, Any]]:
        try:
            return [self._normalize_document(document) for document in self.collection.find().sort("lineId", 1)]
        except PyMongoError as exc:
            raise Exception(f"Database error listing lines: {exc}") from exc

    def find_active(self) -> List[Dict[str, Any]]:
        try:
            return [
                self._normalize_document(document)
                for document in self.collection.find({"isActive": True}).sort("lineId", 1)
            ]
        except PyMongoError as exc:
            raise Exception(f"Database error listing active lines: {exc}") from exc

    def update(self, line_id: str, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            result = self.collection.replace_one({"lineId": line_id}, document)
            if result.matched_count == 0:
                return None
            return document
        except PyMongoError as exc:
            raise Exception(f"Database error updating line '{line_id}': {exc}") from exc

    def delete(self, line_id: str) -> bool:
        try:
            result = self.collection.delete_one({"lineId": line_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting line '{line_id}': {exc}") from exc
