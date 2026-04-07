import mongomock

from models.bucket import BucketCreate, BucketSearchCriteria, BucketUpdate
from repositories.bucket_repository import BucketRepository
from repositories.mix_metric_repository import MixMetricRepository
from .bucket_service import BucketService


def build_service():
    db = mongomock.MongoClient()["test_db"]
    return BucketService(BucketRepository(db), MixMetricRepository(db)), db


def test_bucket_service_crud_and_search():
    service, db = build_service()

    created = service.create_bucket(BucketCreate(minWeight=380, targetWeight=420, maxWeight=480))
    fetched = service.get_bucket_by_id(created.bucket_id)
    assert fetched is not None
    assert fetched.bucket_id == created.bucket_id

    search_result = service.search_buckets(BucketSearchCriteria(minWeightGte=300, maxWeightLte=500))
    assert len(search_result) == 1
    assert search_result[0].bucket_id == created.bucket_id

    updated = service.update_bucket(
        created.bucket_id,
        BucketUpdate(minWeight=390, targetWeight=430, maxWeight=490),
    )
    assert updated is not None
    assert updated.min_weight == 390
    assert updated.target_weight == 430

    # Seed a dependent metric that should be cascade-deleted with the bucket.
    db["mix_metrics"].insert_one(
        {
            "_id": f"mix-1:{created.bucket_id}",
            "mixId": "mix-1",
            "bucketId": created.bucket_id,
            "upgradePercentage": 10.0,
            "value": 20.0,
            "trimPercentage": 2.0,
            "unitPlan": [],
        }
    )

    result = service.delete_bucket(created.bucket_id)
    assert result["deleted"] is True
    assert result["metrics_deleted"] == 1
    assert service.get_bucket_by_id(created.bucket_id) is None
