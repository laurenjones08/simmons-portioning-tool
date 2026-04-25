from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from repositories.scheduling_output_repository import SchedulingOutputRepository
from scheduling_shared.models.scheduling_output import (
    SchedulingOutput,
    SchedulingOutputCreate,
    SchedulingOutputSearchCriteria,
    SchedulingOutputUpdate,
)


class SchedulingOutputService:
    def __init__(self, repository: SchedulingOutputRepository):
        self.repository = repository

    @staticmethod
    def _to_mongo_document(model) -> Dict[str, Any]:
        return model.model_dump(by_alias=True, mode="json")

    def create(self, payload: SchedulingOutputCreate) -> SchedulingOutput:
        document = self._to_mongo_document(SchedulingOutput(**payload.model_dump(by_alias=True)))
        try:
            inserted = self.repository.create(document)
        except DuplicateKeyError:
            raise ValueError("A scheduling output for this decision and SKU already exists")
        return SchedulingOutput(**inserted)

    def get_by_id(self, document_id: str) -> Optional[SchedulingOutput]:
        document = self.repository.get_by_id(document_id)
        if not document:
            return None
        return SchedulingOutput(**document)

    def search(self, criteria: SchedulingOutputSearchCriteria) -> List[SchedulingOutput]:
        mongo_criteria: Dict[str, Any] = criteria.model_dump(by_alias=True, exclude_none=True, mode="json")
        return [SchedulingOutput(**doc) for doc in self.repository.search(mongo_criteria)]

    def update(self, document_id: str, payload: SchedulingOutputUpdate) -> Optional[SchedulingOutput]:
        document = self._to_mongo_document(SchedulingOutput(_id=document_id, **payload.model_dump(by_alias=True)))
        try:
            updated = self.repository.update(document_id, document)
        except DuplicateKeyError:
            raise ValueError("A conflicting scheduling output already exists")
        if not updated:
            return None
        return SchedulingOutput(**document)

    def delete(self, document_id: str) -> bool:
        return self.repository.delete(document_id)

    def bulk_create(self, items: List[SchedulingOutputCreate], clear_dates: List[str]) -> Dict[str, Any]:
        if clear_dates:
            self.repository.delete_by_dates(clear_dates)
        documents = []
        for item in items:
            doc = self._to_mongo_document(SchedulingOutput(**item.model_dump(by_alias=True)))
            doc["_id"] = str(ObjectId())
            documents.append(doc)
        inserted = self.repository.bulk_create(documents)
        return {
            "total": len(items),
            "successful": len(inserted),
            "failed": len(items) - len(inserted),
            "items": [SchedulingOutput(**doc) for doc in inserted],
        }

