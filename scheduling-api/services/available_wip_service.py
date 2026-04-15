from typing import Any, Dict, List, Optional

from pymongo.errors import DuplicateKeyError

from repositories.available_wip_repository import AvailableWIPRepository
from scheduling_shared.models.available_wip import (
    AvailableWIP,
    AvailableWIPCreate,
    AvailableWIPSearchCriteria,
    AvailableWIPUpdate,
)


class AvailableWIPService:
    def __init__(self, repository: AvailableWIPRepository):
        self.repository = repository

    def create(self, payload: AvailableWIPCreate) -> AvailableWIP:
        document = AvailableWIP(**payload.model_dump(by_alias=True)).model_dump(by_alias=True)
        try:
            inserted = self.repository.create(document)
        except DuplicateKeyError:
            raise ValueError("A WIP row for this plant and bucket already exists")
        return AvailableWIP(**inserted)

    def get_by_id(self, document_id: str) -> Optional[AvailableWIP]:
        document = self.repository.get_by_id(document_id)
        if not document:
            return None
        return AvailableWIP(**document)

    def search(self, criteria: AvailableWIPSearchCriteria) -> List[AvailableWIP]:
        mongo_criteria: Dict[str, Any] = criteria.model_dump(by_alias=True, exclude_none=True)
        return [AvailableWIP(**doc) for doc in self.repository.search(mongo_criteria)]

    def update(self, document_id: str, payload: AvailableWIPUpdate) -> Optional[AvailableWIP]:
        document = AvailableWIP(_id=document_id, **payload.model_dump(by_alias=True)).model_dump(by_alias=True)
        try:
            updated = self.repository.update(document_id, document)
        except DuplicateKeyError:
            raise ValueError("A conflicting WIP row already exists")
        if not updated:
            return None
        return AvailableWIP(**document)

    def delete(self, document_id: str) -> bool:
        return self.repository.delete(document_id)

