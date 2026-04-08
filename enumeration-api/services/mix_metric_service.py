"""Business logic layer for MixMetric operations."""

from typing import List, Optional, Dict, Any
from pymongo.errors import DuplicateKeyError

from models.mix_metric import MixMetric, MixMetricSearchCriteria
from repositories.mix_metric_repository import MixMetricRepository
from repositories.mix_repository import MixRepository


class MixMetricService:
    def __init__(self, repository: MixMetricRepository, mix_repository: MixRepository):
        self.repository = repository
        self.mix_repository = mix_repository

    def create_metric(self, payload: MixMetric) -> MixMetric:
        # Populate model and document
        metric = MixMetric(**payload.model_dump(by_alias=True))
        doc = metric.model_dump(by_alias=True)

        # If skuKeys not set on metric, attempt to denormalize from mix document
        if not doc.get("skuKeys"):
            mix_doc = self.mix_repository.get_by_id(metric.mix_id)
            if mix_doc and "skuKeys" in mix_doc:
                doc["skuKeys"] = mix_doc.get("skuKeys")

        try:
            inserted = self.repository.create(doc)
        except DuplicateKeyError:
            raise ValueError("A metric already exists for this mixId and bucketId")

        return MixMetric(**inserted)

    def get_metric_by_id(self, metric_id: str) -> Optional[MixMetric]:
        doc = self.repository.get_by_id(metric_id)
        return MixMetric(**doc) if doc else None

    def search_metrics(self, criteria: MixMetricSearchCriteria) -> List[MixMetric]:
        raw = criteria.model_dump(by_alias=True, exclude_none=True)
        mongo_criteria: Dict[str, Any] = {}
        if raw.get("mixId"):
            mongo_criteria["mixId"] = raw["mixId"]
        if raw.get("bucketId"):
            mongo_criteria["bucketId"] = raw["bucketId"]
        if raw.get("skuTradeNumber"):
            # Search denormalized skuKeys array
            mongo_criteria["skuKeys"] = raw["skuTradeNumber"]

        docs = self.repository.search(mongo_criteria)
        return [MixMetric(**d) for d in docs]

    def update_metric(self, metric_id: str, payload: MixMetric) -> Optional[MixMetric]:
        metric = MixMetric(_id=metric_id, **payload.model_dump(by_alias=True))
        doc = metric.model_dump(by_alias=True)

        # Ensure denormalized skuKeys are preserved/updated
        if not doc.get("skuKeys"):
            mix_doc = self.mix_repository.get_by_id(metric.mix_id)
            if mix_doc and "skuKeys" in mix_doc:
                doc["skuKeys"] = mix_doc.get("skuKeys")

        try:
            updated = self.repository.update(metric_id, doc)
        except DuplicateKeyError:
            raise ValueError("A conflicting metric exists for this mixId and bucketId")

        if not updated:
            return None
        return metric

    def delete_metric(self, metric_id: str) -> bool:
        return self.repository.delete(metric_id)

    def delete_metrics_by_mix_id(self, mix_id: str) -> int:
        return self.repository.delete_by_mix_id(mix_id)
