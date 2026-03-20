"""
Integration tests for the Enumeration Worker API job lifecycle.
Uses mongomock so no real MongoDB is needed.
"""

import sys
import time
from pathlib import Path

import mongomock
from fastapi.testclient import TestClient

# Ensure the service root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Patch database.get_mongo_client to return a mongomock client before importing app
import database as db_module

_mock_client = mongomock.MongoClient()


def _mock_get_mongo_client():
    return _mock_client


db_module.get_mongo_client = _mock_get_mongo_client

from main import app  # noqa: E402 – must come after patching

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_db():
    """Insert the minimum data the engine needs: scoped skus + buckets."""
    db = _mock_client["enumeration_db"]
    db["skus"].drop()
    db["buckets"].drop()
    db["job_status"].drop()
    db["enumeration_results"].drop()

    db["skus"].insert_many([
        {
            "_id": "100", "tradeNumber": "100", "targetWeight": 100.0,
            "minWeight": 80.0, "maxWeight": 120.0,
            "customerType": "FDS", "productType": "NUGGET", "allowedParts": ["D"],
            "prodPlant": "P1", "birdSize": "L",
        },
        {
            "_id": "200", "tradeNumber": "200", "targetWeight": 200.0,
            "minWeight": 170.0, "maxWeight": 240.0,
            "customerType": "RTL", "productType": "FILET", "allowedParts": ["R"],
            "prodPlant": "P1", "birdSize": "L",
        },
        {
            "_id": "300", "tradeNumber": "300", "targetWeight": 140.0,
            "minWeight": 120.0, "maxWeight": 170.0,
            "customerType": "RTL", "productType": "FILET", "allowedParts": ["M"],
            "prodPlant": "P1", "birdSize": "L",
        },
        {
            "_id": "400", "tradeNumber": "400", "targetWeight": 110.0,
            "minWeight": 90.0, "maxWeight": 130.0,
            "customerType": "RTL", "productType": "FILET", "allowedParts": ["T"],
            "prodPlant": "P1", "birdSize": "L",
        },
    ])
    db["buckets"].insert_many([
        {"_id": "b1", "minWeight": 50.0, "maxWeight": 500.0},
    ])


def _wait_for_terminal_status(job_id: str, timeout_seconds: float = 4.0) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not reach a terminal status within timeout")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_submit_and_poll_job_with_filters():
    _seed_db()
    resp = client.post(
        "/jobs",
        json={
            "runId": "test-run",
            "maxCombinationSize": 4,
            "batchSize": 10,
            "plantFilter": "P1",
            "birdSizeFilter": "L",
        },
    )
    assert resp.status_code == 202
    job_id = resp.json()["jobId"]

    terminal = _wait_for_terminal_status(job_id)
    assert terminal["status"] == "completed"
    assert terminal["skuCount"] == 4
    assert [s["totalCombinations"] for s in terminal["stages"]] == [4, 6, 4, 1]

    results = list(_mock_client["enumeration_db"]["enumeration_results"].find({"runId": "test-run"}))
    assert len(results) == 15


def test_submit_job_without_filters_is_failed():
    _seed_db()
    resp = client.post(
        "/jobs",
        json={"runId": "no-filters", "maxCombinationSize": 4, "batchSize": 10},
    )
    assert resp.status_code == 202
    job_id = resp.json()["jobId"]

    terminal = _wait_for_terminal_status(job_id)
    assert terminal["status"] == "failed"
    assert "at least one filter" in (terminal.get("errorMessage") or "")


def test_list_jobs():
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_nonexistent_job_returns_404():
    resp = client.get("/jobs/000000000000000000000000")
    assert resp.status_code == 404


def test_cancel_nonexistent_job_returns_404():
    resp = client.post("/jobs/000000000000000000000000/cancel")
    assert resp.status_code == 404


def test_openapi_schema():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/jobs" in paths
    assert "/jobs/{job_id}" in paths
    assert "/jobs/{job_id}/cancel" in paths

