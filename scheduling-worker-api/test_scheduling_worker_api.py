from pathlib import Path
import sys

import mongomock
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for module_name in [
    "main",
    "routers",
    "config",
    "database",
    "job_service",
    "repositories",
    "services",
    "models",
    "storage",
    "models.job",
    "routers.job_router",
    "pipeline",
    "data_prep",
    "model_builder",
    "results",
    "solver",
    "run_model",
]:
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(ROOT / "scheduling-shared"))
sys.path.insert(0, str(ROOT / "scheduling"))
sys.path.insert(0, str(ROOT / "scheduling-worker-api"))

from main import app  # noqa: E402
import job_service as job_service_module  # noqa: E402
import run_model as run_model_module  # noqa: E402
from models.job import CreateJobRequest  # noqa: E402
from repositories.job_repository import JobRepository  # noqa: E402


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Scheduling Worker API"


def test_short_term_upload_endpoint_returns_object_key(monkeypatch):
    captured = {}

    def fake_upload_short_term_demand_file(file_bytes: bytes, suffix: str) -> str:
        captured["file_bytes"] = file_bytes
        captured["suffix"] = suffix
        return "short-term-uploads/test-file.csv"

    monkeypatch.setattr("main.upload_short_term_demand_file", fake_upload_short_term_demand_file)

    response = client.post(
        "/uploads/short-term-file",
        files={"file": ("demand.csv", b"skuId,date,lbs\n50624,2026-04-20,100\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == {"objectKey": "short-term-uploads/test-file.csv"}
    assert captured["suffix"] == ".csv"
    assert captured["file_bytes"].startswith(b"skuId,date,lbs")


def test_job_request_model_validation():
    request = CreateJobRequest(
        runId="schedule-2026-04-10",
        plantId="FSP",
        skuIds=["50624", "50625"],
        shortTermFile=None,
        debugMode=True,
        maxTrimPercentage=35.0,
    )
    assert request.run_id == "schedule-2026-04-10"
    assert request.plant_id == "FSP"
    assert request.sku_ids == ["50624", "50625"]
    assert request.debug_mode is True
    assert request.horizon_days == 12
    assert request.max_trim_percentage == 35.0


def test_make_json_safe_serializes_tuple_keyed_maps():
    payload = {
        "WIP": {
            ("bucket-1", pd.Timestamp("2026-01-05")): 123.0,
        }
    }

    safe = job_service_module._make_json_safe(payload)

    assert safe == {
        "WIP": [
            {
                "key": ["bucket-1", "2026-01-05T00:00:00"],
                "value": 123.0,
            }
        ]
    }


def test_job_sku_validation_requires_single_plant(monkeypatch):
    def fake_fetch_sku_document(sku_id: str):
        mapping = {
            "50624": {"tradeNumber": "50624", "prodPlant": "FSP"},
            "50625": {"tradeNumber": "50625", "prodPlant": "FSP"},
            "99999": {"tradeNumber": "99999", "prodPlant": "SS2"},
        }
        return mapping[sku_id]

    monkeypatch.setattr(job_service_module, "_fetch_sku_document", fake_fetch_sku_document)

    assert job_service_module._validate_job_skus("FSP", ["50624", "50625"]) == ["50624", "50625"]

    try:
        job_service_module._validate_job_skus("FSP", ["50624", "99999"])
        raise AssertionError("Expected a ValueError for mixed plants")
    except ValueError as exc:
        assert "same plant" in str(exc)


def test_job_repository_chunks_large_debug_dump_and_rehydrates_payload():
    database = mongomock.MongoClient().db
    repo = JobRepository(database)

    large_payload = {
        "debugDataPrep": {
            "K": [f"metric-{i}" for i in range(500000)],
            "counts": {"metricCount": 500000},
        }
    }

    repo.store_debug_dump("job-1", "run-1", large_payload, ttl_minutes=5)
    stored = repo.get_debug_dump("job-1")

    assert stored is not None
    assert stored["storageKind"] == "chunked"
    assert stored["chunkCount"] > 1
    assert stored["payload"] == large_payload


def test_persist_results_to_scheduling_api_uses_new_decision_and_output_fields(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, base_url, timeout_seconds):
            captured["base_url"] = base_url
            captured["timeout_seconds"] = timeout_seconds

        def bulk_create_decisions(self, payload):
            captured["decisions_payload"] = payload
            return {
                "total": len(payload["items"]),
                "successful": len(payload["items"]),
                "failed": 0,
                "items": [
                    {
                        "_id": "decision-1",
                        "mixId": payload["items"][0]["mixId"],
                        "lineId": payload["items"][0]["lineId"],
                        "date": payload["items"][0]["date"],
                    }
                ],
            }

        def bulk_create_outputs(self, payload):
            captured["outputs_payload"] = payload
            return {
                "total": len(payload["items"]),
                "successful": len(payload["items"]),
                "failed": 0,
            }

        def bulk_create_bucket_usage(self, payload):
            captured["bucket_payload"] = payload
            return {
                "total": len(payload["items"]),
                "successful": len(payload["items"]),
                "failed": 0,
            }

    class FakeEndpoints:
        scheduling_api_url = "http://fake-scheduling-api"
        timeout_seconds = 12.0

    monkeypatch.setattr(job_service_module, "SchedulingApiClient", FakeClient)
    monkeypatch.setattr(job_service_module, "ApiEndpoints", FakeEndpoints)

    results = {
        "inputs": {
            "P": ["50624"],
            "Y": {("50624", "mix-1"): 0.5},
            "decisionOutputShares": {
                "mix-1": [
                    {"skuId": "50624", "yieldFraction": 0.5},
                ]
            },
            "R": {"mix-1": 100.0},
            "D_week1": {("50624", "2026-04-15"): 1200.0},
            "month_of_day": {"2026-04-15": "2026-04"},
            "monthly_contract": {("50624", "2026-04"): 2600.0},
        },
        "outputs": {
            "x_long_nonzero": pd.DataFrame(
                [
                    {
                        "decision": "mix-1",
                        "line": "DSI884",
                        "date": "2026-04-15",
                        "assigned_lbs": 2400.0,
                        "upgrade_pct": 0.0875,
                    }
                ]
            ),
            "bucket_usage_by_date": pd.DataFrame(
                [
                    {
                        "bucket": "bucket-1",
                        "date": "2026-04-15",
                        "available_lbs": 3000.0,
                        "used_lbs": 2400.0,
                    }
                ]
            ),
        },
    }

    summary = job_service_module._persist_results_to_scheduling_api(
        results,
        CreateJobRequest(
            runId="schedule-2026-04-15",
            plantId="FSP",
            skuIds=["50624"],
        ),
    )

    assert summary["decisions"]["successful"] == 1
    assert captured["decisions_payload"]["items"] == [
        {
            "mixId": "mix-1",
            "lineId": "DSI884",
            "date": "2026-04-15",
            "lbsProduced": 2400.0,
            "duration": 24.0,
            "upgradePct": 0.0875,
        }
    ]
    assert captured["outputs_payload"]["items"] == [
        {
            "decisionId": "decision-1",
            "skuId": "50624",
            "date": "2026-04-15",
            "batchUpgradePct": 0.0875,
            "lbsProduced": 1200.0,
            "shortTermContractLbs": 1200.0,
            "longTermContractLbs": 2600.0,
        }
    ]
    assert captured["bucket_payload"]["items"] == [
        {
            "bucketId": "bucket-1",
            "date": "2026-04-15",
            "availableLbs": 3000.0,
            "utilizedLbs": 2400.0,
        }
    ]


def test_job_results_endpoint_returns_gone_when_results_are_not_persisted(monkeypatch):
    class FakeService:
        def get_job(self, job_id):
            assert job_id == "job-123"
            return {"jobId": job_id}

    from routers import job_router  # noqa: E402

    app.dependency_overrides[job_router._get_service] = lambda: FakeService()
    try:
        response = client.get("/jobs/job-123/results")
        assert response.status_code == 410
        assert "no longer persist raw scheduling_results payloads" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(job_router._get_service, None)


def test_run_for_job_disables_local_csv_writes(monkeypatch):
    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return {"outputs": {}, "timings": {}}

    monkeypatch.setattr(run_model_module, "run_pipeline", fake_run_pipeline)

    run_model_module.run_for_job(
        short_term_file=None,
        output_dir="minio-prefix",
        tee=True,
        plan_start_date="2026-01-05",
        horizon_days=12,
        plant_id="FSP",
        sku_ids=["50624"],
    )

    assert captured["save_csv"] is False
    assert captured["output_dir"] == "minio-prefix"


