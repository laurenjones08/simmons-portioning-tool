from typing import Any, Dict, List, Optional

from pymongo.errors import DuplicateKeyError

from repositories.sku_demand_repository import SKUDemandRepository
from scheduling_shared.models.sku_demand import (
    SKUDemand,
    SKUDemandBulkImportError,
    SKUDemandBulkImportRequest,
    SKUDemandBulkImportResponse,
    SKUDemandCreate,
    SKUDemandSearchCriteria,
    SKUDemandUpdate,
)


class SKUDemandService:
    def __init__(self, repository: SKUDemandRepository):
        self.repository = repository

    def create(self, payload: SKUDemandCreate) -> SKUDemand:
        document = SKUDemand(**payload.model_dump(by_alias=True)).model_dump(by_alias=True)
        try:
            inserted = self.repository.create(document)
        except DuplicateKeyError:
            raise ValueError("A SKU demand for this SKU, type, and due date already exists")
        return SKUDemand(**inserted)

    def get_by_id(self, document_id: str) -> Optional[SKUDemand]:
        document = self.repository.get_by_id(document_id)
        if not document:
            return None
        return SKUDemand(**document)

    def search(self, criteria: SKUDemandSearchCriteria) -> List[SKUDemand]:
        mongo_criteria: Dict[str, Any] = criteria.model_dump(by_alias=True, exclude_none=True, mode="json")
        return [SKUDemand(**doc) for doc in self.repository.search(mongo_criteria)]

    def bulk_create(self, payload: SKUDemandBulkImportRequest) -> SKUDemandBulkImportResponse:
        documents: List[Dict[str, Any]] = []
        sku_ids: List[str] = []
        for demand in payload.demands:
            document = SKUDemand(**demand.model_dump(by_alias=True)).model_dump(by_alias=True)
            documents.append(document)
            sku_ids.append(document.get("skuId", ""))

        result = self.repository.bulk_create(documents)
        write_errors = result.get("write_errors", [])
        errors = []
        for error in write_errors:
            index = int(error.get("index", 0))
            errors.append(
                SKUDemandBulkImportError(
                    rowIndex=index + 1,
                    skuId=sku_ids[index] if 0 <= index < len(sku_ids) else None,
                    error=error.get("errmsg") or error.get("errMsg") or "Bulk insert failed",
                )
            )

        successful = int(result.get("inserted_count", 0))
        failed = max(0, len(documents) - successful)
        if errors:
            failed = max(failed, len(errors))

        return SKUDemandBulkImportResponse(
            total=len(documents),
            successful=successful,
            failed=failed,
            errors=errors,
        )

    def update(self, document_id: str, payload: SKUDemandUpdate) -> Optional[SKUDemand]:
        document = SKUDemand(_id=document_id, **payload.model_dump(by_alias=True)).model_dump(by_alias=True)
        try:
            updated = self.repository.update(document_id, document)
        except DuplicateKeyError:
            raise ValueError("A conflicting SKU demand already exists")
        if not updated:
            return None
        return SKUDemand(**document)

    def delete(self, document_id: str) -> bool:
        return self.repository.delete(document_id)
