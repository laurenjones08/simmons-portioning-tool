"""Integration tests for bucket and cut strategy CRUD/search endpoints."""

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


def test_bucket_crud_flow():
    client, db = _make_test_client()

    # Create
    create_resp = client.post("/buckets", json={"minWeight": 380, "targetWeight": 430, "maxWeight": 480})
    assert create_resp.status_code == 201
    created = create_resp.json()
    bucket_id = created["_id"]

    # Get
    get_resp = client.get(f"/buckets/{bucket_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["_id"] == bucket_id

    # Search
    search_resp = client.post("/buckets/search", json={"minWeightGte": 300, "maxWeightLte": 500})
    assert search_resp.status_code == 200
    assert any(doc["_id"] == bucket_id for doc in search_resp.json())

    # Update
    update_resp = client.put(
        f"/buckets/{bucket_id}",
        json={"minWeight": 390, "targetWeight": 440, "maxWeight": 490},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["minWeight"] == 390.0

    # Seed one dependent metric to verify cascade behavior.
    db["mix_metrics"].insert_one(
        {
            "_id": f"mix-1:{bucket_id}",
            "mixId": "mix-1",
            "bucketId": bucket_id,
            "upgradePercentage": 10.0,
            "value": 20.0,
            "trimPercentage": 2.0,
            "unitPlan": [],
        }
    )

    # Delete + warning + cascade count
    delete_resp = client.delete(f"/buckets/{bucket_id}")
    assert delete_resp.status_code == 200
    payload = delete_resp.json()
    assert payload["deleted"] is True
    assert payload["metricsDeleted"] == 1
    assert "recomputing the enumeration model" in payload["warning"]


def test_bucket_search_backfills_missing_legacy_target_weight():
    client, db = _make_test_client()

    db["buckets"].insert_one(
        {
            "_id": "legacy-bucket",
            "minWeight": 380.0,
            "maxWeight": 390.0,
        }
    )

    response = client.post("/buckets/search", json={})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["_id"] == "legacy-bucket"
    assert payload[0]["targetWeight"] == 385.0


def test_cut_strategy_crud_flow():
    client, db = _make_test_client()

    create_resp = client.post(
        "/cut-strategies",
        json={
            "name": "DSI Strategy",
            "description": "Default DSI strategy",
            "mfgType": "DSI",
            "hasNugget": True,
            "beltSpeed": 1.2,
            "parts": ["D", "R", "M"],
        },
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    strategy_id = created["_id"]

    get_resp = client.get(f"/cut-strategies/{strategy_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["_id"] == strategy_id

    search_resp = client.post("/cut-strategies/search", json={"mfgType": "DSI", "includesPart": "D"})
    assert search_resp.status_code == 200
    assert any(doc["_id"] == strategy_id for doc in search_resp.json())

    update_resp = client.put(
        f"/cut-strategies/{strategy_id}",
        json={
            "name": "DSI Strategy",
            "description": "Updated strategy",
            "mfgType": "DSI",
            "hasNugget": False,
            "beltSpeed": 1.1,
            "parts": ["D", "R"],
        },
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["hasNugget"] is False

    # Seed dependent mixes + metrics for cascade behavior.
    db["mixes"].insert_many(
        [
            {
                "_id": "mix-1",
                "skus": {"123": "D"},
                "skuKeys": ["123"],
                "includesFDS": True,
                "includesRTL": False,
                "includesNug": False,
                "nuggetTargetWeight": None,
                "numFillets": 2,
                "filletWeight": 12.0,
                "mfgType": "DSI",
                "reqPlant": "FSP",
                "reqBirdSize": "SB",
                "cutStrategyID": strategy_id,
                "beltSpeed": 1.2,
                "skuSetKey": "123",
            },
            {
                "_id": "mix-2",
                "skus": {"456": "R"},
                "skuKeys": ["456"],
                "includesFDS": True,
                "includesRTL": False,
                "includesNug": False,
                "nuggetTargetWeight": None,
                "numFillets": 2,
                "filletWeight": 11.0,
                "mfgType": "DSI",
                "reqPlant": "FSP",
                "reqBirdSize": "SB",
                "cutStrategyID": strategy_id,
                "beltSpeed": 1.1,
                "skuSetKey": "456",
            },
        ]
    )
    db["mix_metrics"].insert_many(
        [
            {
                "_id": "mix-1:bucket-a",
                "mixId": "mix-1",
                "bucketId": "bucket-a",
                "upgradePercentage": 10.0,
                "value": 20.0,
                "trimPercentage": 1.0,
                "unitPlan": [],
            },
            {
                "_id": "mix-2:bucket-b",
                "mixId": "mix-2",
                "bucketId": "bucket-b",
                "upgradePercentage": 8.0,
                "value": 15.0,
                "trimPercentage": 2.0,
                "unitPlan": [],
            },
        ]
    )

    delete_resp = client.delete(f"/cut-strategies/{strategy_id}")
    assert delete_resp.status_code == 200
    payload = delete_resp.json()
    assert payload["deleted"] is True
    assert payload["mixesDeleted"] == 2
    assert payload["metricsDeleted"] == 2


def test_cut_strategy_search_skips_invalid_legacy_documents():
    client, db = _make_test_client()

    # "invalid-1" uses a lineType value that is not in the LineType enum, so it
    # will be rejected by Pydantic validation and silently skipped.
    # Different parts are used to avoid a unique-index conflict.
    db["cut_strategies"].insert_many(
        [
            {
                "_id": "valid-1",
                "name": "Valid Strategy",
                "description": "ok",
                "lineType": "DSI",
                "hasNugget": False,
                "beltSpeed": 1.0,
                "parts": ["D", "R"],
                "partsKey": "d-r",
            },
            {
                "_id": "invalid-1",
                "name": "Legacy Bad Strategy",
                "description": "bad",
                "lineType": "LEGACY_ONLY",
                "hasNugget": False,
                "beltSpeed": 1.0,
                "parts": ["K"],
                "partsKey": "k",
            },
        ]
    )

    resp = client.post("/cut-strategies/search", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["_id"] == "valid-1"


def test_sku_search_bird_size_includes_all():
    client, db = _make_test_client()

    db["skus"].insert_many(
        [
            {
                "_id": "sku-sb",
                "tradeNumber": "sku-sb",
                "customerName": "Customer A",
                "customerType": "FDS",
                "productType": "NUGGET",
                "unitsPerCut": 1,
                "prodPlant": "FSP",
                "minWeight": 10.0,
                "maxWeight": 19.0,
                "targetWeight": 15.0,
                "birdSize": "SB",
                "allowedParts": ["D"],
            },
            {
                "_id": "sku-all",
                "tradeNumber": "sku-all",
                "customerName": "Customer B",
                "customerType": "FDS",
                "productType": "NUGGET",
                "unitsPerCut": 1,
                "prodPlant": "FSP",
                "minWeight": 10.0,
                "maxWeight": 19.0,
                "targetWeight": 15.0,
                "birdSize": "ALL",
                "allowedParts": ["D"],
            },
            {
                "_id": "sku-bb",
                "tradeNumber": "sku-bb",
                "customerName": "Customer C",
                "customerType": "FDS",
                "productType": "NUGGET",
                "unitsPerCut": 1,
                "prodPlant": "FSP",
                "minWeight": 10.0,
                "maxWeight": 19.0,
                "targetWeight": 15.0,
                "birdSize": "BB",
                "allowedParts": ["D"],
            },
        ]
    )

    response = client.post("/skus/search", json={"birdSize": "SB"})

    assert response.status_code == 200
    payload = response.json()
    returned_trade_numbers = {row["tradeNumber"] for row in payload}

    assert returned_trade_numbers == {"sku-sb", "sku-all"}
