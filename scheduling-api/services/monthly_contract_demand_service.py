from typing import Any, Dict, List, Optional

from pymongo.errors import DuplicateKeyError

from repositories.monthly_contract_demand_repository import MonthlyContractDemandRepository
from scheduling_shared.models.monthly_contract_demand import (
    MonthlyContractDemand,
    MonthlyContractDemandBulkImportError,
    MonthlyContractDemandBulkImportRequest,
    MonthlyContractDemandBulkImportResponse,
    MonthlyContractDemandBulkSearchRequest,
    MonthlyContractDemandCreate,
    MonthlyContractDemandSearchCriteria,
    MonthlyContractDemandUpdate,
)


class MonthlyContractDemandService:
    def __init__(self, repository: MonthlyContractDemandRepository):
        self.repository = repository

    def create(self, payload: MonthlyContractDemandCreate) -> MonthlyContractDemand:
        document = MonthlyContractDemand(**payload.model_dump(by_alias=True)).model_dump(by_alias=True)
        try:
            inserted = self.repository.create(document)
        except DuplicateKeyError:
            raise ValueError("A monthly contract demand row for this SKU and month already exists")
        return MonthlyContractDemand(**inserted)

    def bulk_create(self, payload: MonthlyContractDemandBulkImportRequest) -> MonthlyContractDemandBulkImportResponse:
        documents: List[Dict[str, Any]] = []
        sku_ids: List[str] = []
        year_months: List[str] = []
        for demand in payload.demands:
            document = MonthlyContractDemand(**demand.model_dump(by_alias=True)).model_dump(by_alias=True)
            documents.append(document)
            sku_ids.append(document.get("skuId", ""))
            year_months.append(document.get("yearMonth", ""))

        result = self.repository.bulk_create(documents)
        errors = []
        for error in result.get("write_errors", []):
            index = int(error.get("index", 0))
            errors.append(
                MonthlyContractDemandBulkImportError(
                    rowIndex=index + 1,
                    skuId=sku_ids[index] if 0 <= index < len(sku_ids) else None,
                    yearMonth=year_months[index] if 0 <= index < len(year_months) else None,
                    error=error.get("errmsg") or error.get("errMsg") or "Bulk insert failed",
                )
            )

        successful = int(result.get("inserted_count", 0))
        failed = max(0, len(documents) - successful)
        if errors:
            failed = max(failed, len(errors))

        return MonthlyContractDemandBulkImportResponse(
            total=len(documents),
            successful=successful,
            failed=failed,
            errors=errors,
        )

    def get_by_id(self, document_id: str) -> Optional[MonthlyContractDemand]:
        document = self.repository.get_by_id(document_id)
        if not document:
            return None
        return MonthlyContractDemand(**document)

    def search(self, criteria: MonthlyContractDemandSearchCriteria) -> List[MonthlyContractDemand]:
        mongo_criteria: Dict[str, Any] = criteria.model_dump(by_alias=True, exclude_none=True)
        return [MonthlyContractDemand(**doc) for doc in self.repository.search(mongo_criteria)]

    def bulk_search(self, criteria: MonthlyContractDemandBulkSearchRequest) -> List[MonthlyContractDemand]:
        return [
            MonthlyContractDemand(**doc)
            for doc in self.repository.bulk_search(criteria.sku_ids, criteria.year_months)
        ]

    def update(self, document_id: str, payload: MonthlyContractDemandUpdate) -> Optional[MonthlyContractDemand]:
        document = MonthlyContractDemand(_id=document_id, **payload.model_dump(by_alias=True)).model_dump(by_alias=True)
        try:
            updated = self.repository.update(document_id, document)
        except DuplicateKeyError:
            raise ValueError("A conflicting monthly contract demand row already exists")
        if not updated:
            return None
        return MonthlyContractDemand(**document)

    def delete(self, document_id: str) -> bool:
        return self.repository.delete(document_id)
