from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for module_name in ["main", "routers", "config", "database", "job_service", "repositories", "services", "models"]:
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(ROOT / "scheduling-api"))
sys.path.insert(0, str(ROOT / "scheduling-shared"))

from main import app  # noqa: E402
from scheduling_shared.models.sku_demand import SKUDemandCreate  # noqa: E402
from scheduling_shared.models.scheduling_decision import SchedulingDecisionCreate  # noqa: E402


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Scheduling API"


def test_sku_demand_model_validation():
    model = SKUDemandCreate(
        skuId="50624",
        demandValue=1500.0,
        demandType="Short",
        dueDate="2026-04-15",
    )
    assert model.sku_id == "50624"
    assert model.demand_value == 1500.0


def test_scheduling_decision_model_validation():
    model = SchedulingDecisionCreate(
        mixId="mix-001",
        lineId="DSI884",
        date="2026-04-15",
        duration=6.5,
        lbsProduced=2400.0,
    )
    assert model.line_id == "DSI884"
    assert model.lbs_produced == 2400.0
