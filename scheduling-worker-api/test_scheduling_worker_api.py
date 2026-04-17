from pathlib import Path
import sys

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
from models.job import CreateJobRequest  # noqa: E402


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Scheduling Worker API"


def test_job_request_model_validation():
    request = CreateJobRequest(
        runId="schedule-2026-04-10",
        plantId="FSP",
        skuIds=["50624", "50625"],
        shortTermFile=None,
    )
    assert request.run_id == "schedule-2026-04-10"
    assert request.plant_id == "FSP"
    assert request.sku_ids == ["50624", "50625"]
    assert request.horizon_days == 12


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


def test_short_term_demand_save_posts_to_scheduling_api(monkeypatch, tmp_path):
    short_term_file = tmp_path / "short_term_demand.csv"
    pd.DataFrame(
        [
            {"sku": "50624", "demand": 10, "type": "Short", "dueDate": "2026-01-05"},
            {"sku": "50624", "demand": 5, "type": "short", "dueDate": "2026-01-06"},
            {"sku": "50624", "demand": 99, "type": "Long", "dueDate": "2026-01-05"},
        ]
    ).to_csv(short_term_file, index=False)

    captured = {}

    class FakeClient:
        def __init__(self, base_url, timeout_seconds):
            captured["base_url"] = base_url
            captured["timeout_seconds"] = timeout_seconds

        def bulk_create_sku_demands(self, demands):
            captured["demands"] = demands
            return {"total": 1, "successful": 1, "failed": 0, "errors": []}

    class FakeEndpoints:
        scheduling_api_url = "http://example.test"
        timeout_seconds = 3.0

    monkeypatch.setattr(job_service_module, "ApiEndpoints", lambda: FakeEndpoints())
    monkeypatch.setattr(job_service_module, "SchedulingApiClient", FakeClient)

    result = job_service_module._save_short_term_demands(
        CreateJobRequest(
            runId="schedule-2026-04-10",
            plantId="FSP",
            skuIds=["50624"],
            shortTermFile=str(short_term_file),
        )
    )

    assert captured["base_url"] == "http://example.test"
    assert captured["timeout_seconds"] == 3.0
    assert captured["demands"] == [
        {
            "skuId": "50624",
            "demandValue": 10.0,
            "demandType": "Short",
            "dueDate": "2026-01-05",
        },
        {
            "skuId": "50624",
            "demandValue": 5.0,
            "demandType": "Short",
            "dueDate": "2026-01-06",
        },
    ]
    assert result == {"total": 1, "successful": 1, "failed": 0, "errors": []}
