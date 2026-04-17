from pathlib import Path
import sys

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
    "models.artifacts",
    "models.job",
    "models.available_wip",
    "models.bucket_usage",
    "models.monthly_contract_demand",
    "models.scheduling_decision",
    "models.scheduling_output",
    "models.sku_demand",
    "routers.available_wip_router",
    "routers.bucket_usage_router",
    "routers.job_artifacts_router",
    "routers.monthly_contract_demand_router",
    "routers.scheduling_decision_router",
    "routers.scheduling_output_router",
    "routers.sku_demand_router",
]:
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(ROOT / "scheduling-api"))
sys.path.insert(0, str(ROOT / "scheduling-shared"))

from main import app  # noqa: E402
from models.artifacts import ArtifactFile  # noqa: E402
from scheduling_shared.models.available_wip import AvailableWIPCreate  # noqa: E402
from scheduling_shared.models.monthly_contract_demand import MonthlyContractDemandBulkImportRequest  # noqa: E402
from scheduling_shared.models.monthly_contract_demand import MonthlyContractDemandBulkSearchRequest  # noqa: E402
from scheduling_shared.models.monthly_contract_demand import MonthlyContractDemandCreate  # noqa: E402
from scheduling_shared.models.sku_demand import SKUDemandBulkImportRequest  # noqa: E402
from scheduling_shared.models.sku_demand import SKUDemandCreate  # noqa: E402
from scheduling_shared.models.sku_demand import SKUDemandSearchCriteria  # noqa: E402
from scheduling_shared.models.scheduling_decision import SchedulingDecisionCreate  # noqa: E402
from routers import available_wip_router, job_artifacts_router, monthly_contract_demand_router, sku_demand_router  # noqa: E402


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


def test_sku_demand_bulk_import_model_validation():
    model = SKUDemandBulkImportRequest(
        demands=[
            {
                "skuId": "50624",
                "demandValue": 1500.0,
                "demandType": "Short",
                "dueDate": "2026-04-15",
            }
        ]
    )
    assert len(model.demands) == 1
    assert model.demands[0].sku_id == "50624"


def test_sku_demand_search_model_validation():
    model = SKUDemandSearchCriteria(
        skuIds=["50624", "50625"],
        demandType="Short",
        dueDates=["2026-04-15", "2026-04-16"],
    )
    assert model.sku_ids == ["50624", "50625"]
    assert model.demand_type.value == "Short"
    assert [str(date) for date in model.due_dates] == ["2026-04-15", "2026-04-16"]


def test_available_wip_model_validation():
    model = AvailableWIPCreate(
        plantName="FSP",
        bucketId="B 0-390",
        availableLbs=30468.11,
    )
    assert model.plant_name == "FSP"
    assert model.bucket_id == "B 0-390"
    assert model.available_lbs == 30468.11


def test_monthly_contract_model_validation():
    model = MonthlyContractDemandCreate(
        skuId="50624",
        yearMonth="2026-01",
        demandLbs=3000.0,
    )
    assert model.sku_id == "50624"
    assert model.year_month == "2026-01"
    assert model.demand_lbs == 3000.0


def test_monthly_contract_bulk_search_model_validation():
    model = MonthlyContractDemandBulkSearchRequest(
        skuIds=["50624", "50625"],
        yearMonths=["2026-01", "2026-02"],
    )
    assert model.sku_ids == ["50624", "50625"]
    assert model.year_months == ["2026-01", "2026-02"]


def test_available_wip_search_route_uses_service(monkeypatch):
    class FakeService:
        def search(self, criteria):
            assert criteria.plant_name == "FSP"
            return [
                {
                    "_id": "wip-1",
                    "plantName": "FSP",
                    "bucketId": "B 0-390",
                    "availableLbs": 30468.11,
                }
            ]

    app.dependency_overrides[available_wip_router.get_service] = lambda: FakeService()
    try:
        response = client.post("/available-wip/search", json={"plantName": "FSP"})
        assert response.status_code == 200
        body = response.json()
        assert body[0]["plantName"] == "FSP"
        assert body[0]["bucketId"] == "B 0-390"
    finally:
        app.dependency_overrides.pop(available_wip_router.get_service, None)


def test_monthly_contract_search_route_uses_service():
    class FakeService:
        def search(self, criteria):
            assert criteria.sku_id == "50624"
            return [
                {
                    "_id": "mc-1",
                    "skuId": "50624",
                    "yearMonth": "2026-01",
                    "demandLbs": 3000.0,
                }
            ]

    app.dependency_overrides[monthly_contract_demand_router.get_service] = lambda: FakeService()
    try:
        response = client.post("/monthly-contracts/search", json={"skuId": "50624"})
        assert response.status_code == 200
        body = response.json()
        assert body[0]["skuId"] == "50624"
        assert body[0]["yearMonth"] == "2026-01"
    finally:
        app.dependency_overrides.pop(monthly_contract_demand_router.get_service, None)


def test_monthly_contract_bulk_search_route_uses_service():
    class FakeService:
        def bulk_search(self, criteria):
            assert criteria.sku_ids == ["50624", "50625"]
            assert criteria.year_months == ["2026-01", "2026-02"]
            return [
                {
                    "_id": "mc-1",
                    "skuId": "50624",
                    "yearMonth": "2026-01",
                    "demandLbs": 3000.0,
                }
            ]

    app.dependency_overrides[monthly_contract_demand_router.get_service] = lambda: FakeService()
    try:
        response = client.post(
            "/monthly-contracts/bulk-search",
            json={"skuIds": ["50624", "50625"], "yearMonths": ["2026-01", "2026-02"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body[0]["skuId"] == "50624"
        assert body[0]["yearMonth"] == "2026-01"
    finally:
        app.dependency_overrides.pop(monthly_contract_demand_router.get_service, None)


def test_sku_demand_bulk_import_route_uses_service():
    class FakeService:
        def bulk_create(self, payload):
            assert len(payload.demands) == 2
            return {
                "total": 2,
                "successful": 2,
                "failed": 0,
                "errors": [],
            }

    app.dependency_overrides[sku_demand_router.get_service] = lambda: FakeService()
    try:
        response = client.post(
            "/sku-demands/bulk",
            json={
                "demands": [
                    {
                        "skuId": "50624",
                        "demandValue": 1500.0,
                        "demandType": "Short",
                        "dueDate": "2026-04-15",
                    },
                    {
                        "skuId": "50625",
                        "demandValue": 2500.0,
                        "demandType": "Long",
                        "dueDate": "2026-04-16",
                    },
                ]
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["total"] == 2
        assert body["successful"] == 2
    finally:
        app.dependency_overrides.pop(sku_demand_router.get_service, None)


def test_sku_demand_search_route_uses_service():
    class FakeService:
        def search(self, criteria):
            assert criteria.sku_ids == ["50624", "50625"]
            assert criteria.demand_type.value == "Short"
            assert [str(date) for date in criteria.due_dates] == ["2026-04-15", "2026-04-16"]
            return [
                {
                    "_id": "sd-1",
                    "skuId": "50624",
                    "demandValue": 1500.0,
                    "demandType": "Short",
                    "dueDate": "2026-04-15",
                }
            ]

    app.dependency_overrides[sku_demand_router.get_service] = lambda: FakeService()
    try:
        response = client.post(
            "/sku-demands/search",
            json={
                "skuIds": ["50624", "50625"],
                "demandType": "Short",
                "dueDates": ["2026-04-15", "2026-04-16"],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body[0]["skuId"] == "50624"
        assert body[0]["demandType"] == "Short"
    finally:
        app.dependency_overrides.pop(sku_demand_router.get_service, None)


def test_sku_demand_service_serializes_search_dates():
    from services.sku_demand_service import SKUDemandService

    captured = {}

    class FakeRepository:
        def search(self, criteria):
            captured["criteria"] = criteria
            return [
                {
                    "_id": "sd-1",
                    "skuId": "50624",
                    "demandValue": 1500.0,
                    "demandType": "Short",
                    "dueDate": "2026-04-15",
                }
            ]

    service = SKUDemandService(FakeRepository())
    result = service.search(
        SKUDemandSearchCriteria(
            skuIds=["50624", "50625"],
            demandType="Short",
            dueDates=["2026-04-15", "2026-04-16"],
        )
    )

    assert captured["criteria"] == {
        "skuIds": ["50624", "50625"],
        "demandType": "Short",
        "dueDates": ["2026-04-15", "2026-04-16"],
    }
    assert result[0].sku_id == "50624"


def test_job_artifact_proxy_rewrites_download_urls(monkeypatch):
    async def fake_fetch_worker_artifacts(job_id: str):
        return [
            ArtifactFile(
                artifactName="line_schedule",
                fileName="line_schedule.csv",
                bucket="scheduling-artifacts",
                key=f"runs/{job_id}/outputs/line_schedule.csv",
                downloadUrl="http://minio:9000/presigned",
            )
        ]

    monkeypatch.setattr(job_artifacts_router, "_fetch_worker_artifacts", fake_fetch_worker_artifacts)

    response = client.get("/jobs/job-123/artifacts")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["artifactName"] == "line_schedule"
    assert body[0]["downloadUrl"] == "http://testserver/jobs/job-123/artifacts/line_schedule"


def test_job_artifact_download_streams_csv(monkeypatch):
    async def fake_fetch_worker_artifacts(job_id: str):
        return [
            ArtifactFile(
                artifactName="line_schedule",
                fileName="line_schedule.csv",
                bucket="scheduling-artifacts",
                key=f"runs/{job_id}/outputs/line_schedule.csv",
                downloadUrl=None,
            )
        ]

    def fake_read_object_bytes(bucket: str, key: str) -> bytes:
        assert bucket == "scheduling-artifacts"
        assert key == "runs/job-123/outputs/line_schedule.csv"
        return b"date,line\n2026-04-15,DSI884\n"

    monkeypatch.setattr(job_artifacts_router, "_fetch_worker_artifacts", fake_fetch_worker_artifacts)
    monkeypatch.setattr(job_artifacts_router, "read_object_bytes", fake_read_object_bytes)

    response = client.get("/jobs/job-123/artifacts/line_schedule")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="line_schedule.csv"'
    assert response.text == "date,line\n2026-04-15,DSI884\n"
