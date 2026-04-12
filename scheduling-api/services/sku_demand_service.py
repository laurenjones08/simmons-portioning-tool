from typing import Any, Dict, List, Optional

from pymongo.errors import DuplicateKeyError

from repositories.sku_demand_repository import SKUDemandRepository
from scheduling_shared.models.sku_demand import SKUDemand, SKUDemandCreate, SKUDemandSearchCriteria, SKUDemandUpdate


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
        mongo_criteria: Dict[str, Any] = criteria.model_dump(by_alias=True, exclude_none=True)
        return [SKUDemand(**doc) for doc in self.repository.search(mongo_criteria)]

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

