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
