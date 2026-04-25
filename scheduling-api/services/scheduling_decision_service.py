from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from repositories.scheduling_decision_repository import SchedulingDecisionRepository
from scheduling_shared.models.scheduling_decision import (
    SchedulingDecision,
    SchedulingDecisionCreate,
    SchedulingDecisionSearchCriteria,
    SchedulingDecisionUpdate,
)


class SchedulingDecisionService:
    def __init__(self, repository: SchedulingDecisionRepository):
        self.repository = repository

    @staticmethod
    def _to_mongo_document(model) -> Dict[str, Any]:
        return model.model_dump(by_alias=True, mode="json")

    def create(self, payload: SchedulingDecisionCreate) -> SchedulingDecision:
        document = self._to_mongo_document(SchedulingDecision(**payload.model_dump(by_alias=True)))
        try:
            inserted = self.repository.create(document)
        except DuplicateKeyError:
            raise ValueError("A scheduling decision for this mix, line, and date already exists")
        return SchedulingDecision(**inserted)

    def get_by_id(self, document_id: str) -> Optional[SchedulingDecision]:
        document = self.repository.get_by_id(document_id)
        if not document:
            return None
        return SchedulingDecision(**document)

    def search(self, criteria: SchedulingDecisionSearchCriteria) -> List[SchedulingDecision]:
        mongo_criteria: Dict[str, Any] = criteria.model_dump(by_alias=True, exclude_none=True, mode="json")
        return [SchedulingDecision(**doc) for doc in self.repository.search(mongo_criteria)]

    def update(self, document_id: str, payload: SchedulingDecisionUpdate) -> Optional[SchedulingDecision]:
        document = self._to_mongo_document(SchedulingDecision(_id=document_id, **payload.model_dump(by_alias=True)))
        try:
            updated = self.repository.update(document_id, document)
        except DuplicateKeyError:
            raise ValueError("A conflicting scheduling decision already exists")
        if not updated:
            return None
        return SchedulingDecision(**document)

    def delete(self, document_id: str) -> bool:
        return self.repository.delete(document_id)

    def bulk_create(self, items: List[SchedulingDecisionCreate], clear_dates: List[str]) -> Dict[str, Any]:
        if clear_dates:
            self.repository.delete_by_dates(clear_dates)
        documents = []
        for item in items:
            doc = self._to_mongo_document(SchedulingDecision(**item.model_dump(by_alias=True)))
            doc["_id"] = str(ObjectId())
            documents.append(doc)
        inserted = self.repository.bulk_create(documents)
        return {
            "total": len(items),
            "successful": len(inserted),
            "failed": len(items) - len(inserted),
            "items": [SchedulingDecision(**doc) for doc in inserted],
        }

