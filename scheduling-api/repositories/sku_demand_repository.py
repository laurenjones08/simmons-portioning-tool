from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError, DuplicateKeyError, PyMongoError


class SKUDemandRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["sku_demands"]
        self.collection.create_index(
            [("skuId", 1), ("demandType", 1), ("dueDate", 1)],
            unique=True,
            name="uniq_sku_demand_key",
        )
        self.collection.create_index(
            [("demandType", 1), ("dueDate", 1), ("skuId", 1)],
            name="idx_sku_demand_type_due_date_sku_id",
        )
        self.collection.create_index([("skuId", 1)], name="idx_sku_demand_sku_id")
        self.collection.create_index([("demandType", 1)], name="idx_sku_demand_type")
        self.collection.create_index([("dueDate", 1)], name="idx_sku_demand_due_date")

    def create(self, document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(document)
            return document
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating SKU demand: {exc}")

    def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": document_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving SKU demand: {exc}")

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        sku_ids = criteria.get("skuIds")
        if sku_ids:
            query["skuId"] = {"$in": sku_ids}
        elif criteria.get("skuId"):
            query["skuId"] = criteria["skuId"]

        demand_type = criteria.get("demandType")
        if demand_type:
            query["demandType"] = demand_type

        due_dates = criteria.get("dueDates")
        if due_dates:
            query["dueDate"] = {"$in": due_dates}
        elif criteria.get("dueDate"):
            query["dueDate"] = criteria["dueDate"]

        try:
            return list(self.collection.find(query))
        except PyMongoError as exc:
            raise Exception(f"Database error searching SKU demands: {exc}")

    def bulk_create(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            result = self.collection.insert_many(documents, ordered=False)
            return {"inserted_count": len(result.inserted_ids), "write_errors": []}
        except BulkWriteError as exc:
            details = exc.details or {}
            return {
                "inserted_count": int(details.get("nInserted", 0)),
                "write_errors": details.get("writeErrors", []),
            }
        except PyMongoError as exc:
            raise Exception(f"Database error bulk creating SKU demands: {exc}")

    def update(self, document_id: str, document: Dict[str, Any]) -> bool:
        try:
            result = self.collection.replace_one({"_id": document_id}, document)
            return result.matched_count > 0
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating SKU demand: {exc}")

    def delete(self, document_id: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": document_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting SKU demand: {exc}")
