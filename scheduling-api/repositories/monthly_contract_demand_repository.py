from typing import Any, Dict, List, Optional

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError, DuplicateKeyError, PyMongoError


class MonthlyContractDemandRepository:
    def __init__(self, database: Database):
        self.collection: Collection = database["monthly_contracts"]
        self.collection.create_index(
            [("skuId", 1), ("yearMonth", 1)],
            unique=True,
            name="uniq_sku_year_month",
        )
        self.collection.create_index([("skuId", 1)], name="idx_monthly_contract_sku_id")
        self.collection.create_index([("yearMonth", 1)], name="idx_monthly_contract_year_month")

    def create(self, document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.collection.insert_one(document)
            return document
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error creating monthly contract demand: {exc}")

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
            raise Exception(f"Database error bulk creating monthly contract demands: {exc}")

    def get_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self.collection.find_one({"_id": document_id})
        except PyMongoError as exc:
            raise Exception(f"Database error retrieving monthly contract demand: {exc}")

    def search(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            return list(self.collection.find(criteria))
        except PyMongoError as exc:
            raise Exception(f"Database error searching monthly contract demands: {exc}")

    def bulk_search(self, sku_ids: List[str], year_months: List[str]) -> List[Dict[str, Any]]:
        query: Dict[str, Any] = {}
        if sku_ids:
            query["skuId"] = {"$in": sku_ids}
        if year_months:
            query["yearMonth"] = {"$in": year_months}
        return self.search(query)

    def update(self, document_id: str, document: Dict[str, Any]) -> bool:
        try:
            result = self.collection.replace_one({"_id": document_id}, document)
            return result.matched_count > 0
        except DuplicateKeyError:
            raise
        except PyMongoError as exc:
            raise Exception(f"Database error updating monthly contract demand: {exc}")

    def delete(self, document_id: str) -> bool:
        try:
            result = self.collection.delete_one({"_id": document_id})
            return result.deleted_count > 0
        except PyMongoError as exc:
            raise Exception(f"Database error deleting monthly contract demand: {exc}")
