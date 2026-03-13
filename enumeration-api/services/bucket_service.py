"""Business logic layer for Bucket operations."""

from typing import Any, Dict, List, Optional

from pymongo.errors import DuplicateKeyError

from models.bucket import Bucket, BucketCreate, BucketSearchCriteria, BucketUpdate
from repositories.bucket_repository import BucketRepository
from repositories.mix_metric_repository import MixMetricRepository


class BucketService:
    def __init__(self, repository: BucketRepository, mix_metric_repository: MixMetricRepository):
        self.repository = repository
        self.mix_metric_repository = mix_metric_repository

    def create_bucket(self, payload: BucketCreate) -> Bucket:
        bucket = Bucket(**payload.model_dump(by_alias=True))
        doc = bucket.model_dump(by_alias=True)

        try:
            inserted = self.repository.create(doc)
        except DuplicateKeyError:
            raise ValueError("A bucket with this id already exists")

        return Bucket(**inserted)

    def get_bucket_by_id(self, bucket_id: str) -> Optional[Bucket]:
        doc = self.repository.get_by_id(bucket_id)
        return Bucket(**doc) if doc else None

    def search_buckets(self, criteria: BucketSearchCriteria) -> List[Bucket]:
        raw = criteria.model_dump(by_alias=True, exclude_none=True)
        mongo_criteria: Dict[str, Any] = {}

        min_weight_gte = raw.get("minWeightGte")
        max_weight_lte = raw.get("maxWeightLte")

        if min_weight_gte is not None:
            mongo_criteria.setdefault("minWeight", {})["$gte"] = min_weight_gte
        if max_weight_lte is not None:
            mongo_criteria.setdefault("maxWeight", {})["$lte"] = max_weight_lte

        docs = self.repository.search(mongo_criteria)
        return [Bucket(**doc) for doc in docs]

    def update_bucket(self, bucket_id: str, payload: BucketUpdate) -> Optional[Bucket]:
        bucket = Bucket(_id=bucket_id, **payload.model_dump(by_alias=True))
        doc = bucket.model_dump(by_alias=True)

        try:
            updated = self.repository.update(bucket_id, doc)
        except DuplicateKeyError:
            raise ValueError("A conflicting bucket already exists")

        if not updated:
            return None
        return bucket

    def delete_bucket(self, bucket_id: str) -> Dict[str, Any]:
        # First remove dependent metrics for this bucket.
        metrics_deleted = self.mix_metric_repository.delete_by_bucket_id(bucket_id)
        deleted = self.repository.delete(bucket_id)
        return {
            "deleted": deleted,
            "metrics_deleted": metrics_deleted,
        }
