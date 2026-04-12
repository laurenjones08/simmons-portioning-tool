from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for module_name in ["main", "routers", "config", "database", "job_service", "repositories", "services", "models", "pipeline", "data_prep", "model_builder", "results", "solver", "run_model"]:
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(ROOT / "scheduling-worker-api"))
sys.path.insert(0, str(ROOT / "scheduling"))
sys.path.insert(0, str(ROOT / "scheduling-shared"))

from main import app  # noqa: E402
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
    request = CreateJobRequest(runId="schedule-2026-04-10", shortTermFile=None)
    assert request.run_id == "schedule-2026-04-10"
    assert request.horizon_days == 12
