import os
import sys

import mongomock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))

from database import get_database
from main import app


def _make_test_client():
    client = mongomock.MongoClient()
    db = client["test_enumeration_db"]

    def override_get_database():
        yield db

    app.dependency_overrides[get_database] = override_get_database
    return TestClient(app), db


def test_mix_metric_crud_persists_derived_fields():
    client, db = _make_test_client()

    payload = {
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

    response = client.post("/metrics", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["totalProductProducedGrams"] == 327.0
    assert body["unitPlan"][0]["pctOfTotal"] == 33.33
    assert body["unitPlan"][1]["pctOfTotal"] == 33.33
    assert body["unitPlan"][2]["pctOfTotal"] == 33.33

    stored = db["mix_metrics"].find_one({"_id": "mix-1:bucket-1"})
    assert stored is not None
    assert stored["totalProductProducedGrams"] == 327.0
    assert stored["unitPlan"][0]["pctOfTotal"] == 33.33


def test_mix_metric_search_filters_by_max_trim_percentage():
    client, _db = _make_test_client()

    low_trim_payload = {
        "mixId": "mix-1",
        "bucketId": "bucket-1",
        "upgradePercentage": 1.5,
        "value": 2.0,
        "trimPercentage": 10.0,
        "unitPlan": [
            {
                "sku": "45065",
                "partCode": "D",
                "unitsInPlan": 1,
                "totalWeightInPlan": 109.0,
            }
        ],
        "skuKeys": ["45065"],
    }
    high_trim_payload = {
        "mixId": "mix-2",
        "bucketId": "bucket-2",
        "upgradePercentage": 1.5,
        "value": 2.0,
        "trimPercentage": 30.0,
        "unitPlan": [
            {
                "sku": "45065",
                "partCode": "D",
                "unitsInPlan": 1,
                "totalWeightInPlan": 109.0,
            }
        ],
        "skuKeys": ["45065"],
    }

    assert client.post("/metrics", json=low_trim_payload).status_code == 201
    assert client.post("/metrics", json=high_trim_payload).status_code == 201

    response = client.post(
        "/metrics/search",
        json={"skuTradeNumber": "45065", "maxTrimPercentage": 20.0},
    )

    assert response.status_code == 200
    assert [metric["mixId"] for metric in response.json()] == ["mix-1"]
