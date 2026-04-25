from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from repositories.bucket_usage_repository import BucketUsageRepository
from scheduling_shared.models.bucket_usage import BucketUsage, BucketUsageCreate, BucketUsageSearchCriteria, BucketUsageUpdate


class BucketUsageService:
    def __init__(self, repository: BucketUsageRepository):
        self.repository = repository

    @staticmethod
    def _to_mongo_document(model) -> Dict[str, Any]:
        return model.model_dump(by_alias=True, mode="json")

    def create(self, payload: BucketUsageCreate) -> BucketUsage:
        document = self._to_mongo_document(BucketUsage(**payload.model_dump(by_alias=True)))
        try:
            inserted = self.repository.create(document)
        except DuplicateKeyError:
            raise ValueError("A bucket usage row for this bucket and date already exists")
        return BucketUsage(**inserted)

    def get_by_id(self, document_id: str) -> Optional[BucketUsage]:
        document = self.repository.get_by_id(document_id)
        if not document:
            return None
        return BucketUsage(**document)

    def search(self, criteria: BucketUsageSearchCriteria) -> List[BucketUsage]:
        mongo_criteria: Dict[str, Any] = criteria.model_dump(by_alias=True, exclude_none=True, mode="json")
        return [BucketUsage(**doc) for doc in self.repository.search(mongo_criteria)]

    def update(self, document_id: str, payload: BucketUsageUpdate) -> Optional[BucketUsage]:
        document = self._to_mongo_document(BucketUsage(_id=document_id, **payload.model_dump(by_alias=True)))
        try:
            updated = self.repository.update(document_id, document)
        except DuplicateKeyError:
            raise ValueError("A conflicting bucket usage row already exists")
        if not updated:
            return None
        return BucketUsage(**document)

    def delete(self, document_id: str) -> bool:
        return self.repository.delete(document_id)

    def bulk_create(self, items: List[BucketUsageCreate], clear_dates: List[str]) -> Dict[str, Any]:
        if clear_dates:
            self.repository.delete_by_dates(clear_dates)
        documents = []
        for item in items:
            doc = self._to_mongo_document(BucketUsage(**item.model_dump(by_alias=True)))
            doc["_id"] = str(ObjectId())
            documents.append(doc)
        inserted = self.repository.bulk_create(documents)
        return {
            "total": len(items),
            "successful": len(inserted),
            "failed": len(items) - len(inserted),
            "items": [BucketUsage(**doc) for doc in inserted],
        }

