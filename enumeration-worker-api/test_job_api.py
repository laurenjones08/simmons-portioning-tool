"""
Integration tests for the Enumeration Worker API job lifecycle.
Uses mongomock so no real MongoDB is needed.
"""

import sys
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
    """Insert the minimum data the engine needs: skus + buckets."""
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
        },
        {
            "_id": "200", "tradeNumber": "200", "targetWeight": 200.0,
            "minWeight": 170.0, "maxWeight": 240.0,
            "customerType": "RTL", "productType": "FILET", "allowedParts": ["R"],
        },
    ])
    db["buckets"].insert_many([
        {"_id": "b1", "minWeight": 50.0, "maxWeight": 350.0},
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_submit_and_poll_job():
    _seed_db()
    # Submit
    resp = client.post("/jobs", json={"runId": "test-run", "maxCombinationSize": 2, "batchSize": 10})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] in ("pending", "running", "completed")
    job_id = body["jobId"]

    # Poll until done (TestClient runs synchronously in the same thread, so we may get completed)
    resp2 = client.get(f"/jobs/{job_id}")
    assert resp2.status_code == 200
    assert resp2.json()["jobId"] == job_id


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


