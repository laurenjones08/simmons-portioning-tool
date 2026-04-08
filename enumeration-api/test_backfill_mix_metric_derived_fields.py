import os
import sys

import mongomock

sys.path.insert(0, os.path.dirname(__file__))

from scripts.backfill_mix_metric_derived_fields import backfill_mix_metrics


def test_backfill_mix_metrics_updates_derived_fields():
    client = mongomock.MongoClient()
    db = client["enumeration_db"]
    collection = db["mix_metrics"]

    collection.insert_one(
        {
            "_id": "mix-1:bucket-1",
            "mixId": "mix-1",
            "bucketId": "bucket-1",
            "upgradePercentage": 1.5,
            "value": 2.0,
            "trimPercentage": 0.5,
            "unitPlan": [
                {
                    "sku": "45065",
                    "partCode": "D",
                    "unitsInPlan": 1,
                    "totalWeightInPlan": 109.0,
                },
                {
                    "sku": "45065",
                    "partCode": "R",
                    "unitsInPlan": 1,
                    "totalWeightInPlan": 109.0,
                },
                {
                    "sku": "45065",
                    "partCode": "M",
                    "unitsInPlan": 1,
                    "totalWeightInPlan": 109.0,
                },
            ],
            "skuKeys": ["45065"],
        }
    )

    scanned, updated, skipped = backfill_mix_metrics(dry_run=False, collection=collection)

    assert scanned == 1
    assert updated == 1
    assert skipped == 0

    stored = collection.find_one({"_id": "mix-1:bucket-1"})
    assert stored["totalProductProducedGrams"] == 327.0
    assert stored["unitPlan"][0]["pctOfTotal"] == 33.33
