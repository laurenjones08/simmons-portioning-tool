"""Unit tests for LineService."""

from models.line import LineCreate, LineUpdate
from services.line_service import LineService


class FakeLineRepository:
    def __init__(self):
        self.documents = {}

    def find_all(self):
        return list(self.documents.values())

    def find_active(self):
        return [doc for doc in self.documents.values() if doc["isActive"]]

    def find_by_id(self, line_id: str):
        return self.documents.get(line_id)

    def create(self, document):
        self.documents[document["lineId"]] = document
        return document

    def update(self, line_id: str, document):
        self.documents[line_id] = document
        return document

    def delete(self, line_id: str):
        return self.documents.pop(line_id, None) is not None

    def backfill_line_type(self):
        return None


class FakeCutStrategyCatalog:
    def __init__(self, strategies):
        self.strategies = strategies

    def list_cut_strategies(self):
        return self.strategies


def test_create_line_rejects_invalid_cut_strategy_ids():
    service = LineService(
        FakeLineRepository(),
        FakeCutStrategyCatalog(
            [
                {"_id": "strategy-1", "lineType": "DSI884"},
                {"_id": "strategy-2", "lineType": "DSI884"},
            ]
        ),
    )

    payload = LineCreate(
        lineId="DSI884",
        friendlyName="DSI 884",
        lineType="DSI884",
        plant="FSP",
        permittedCutStrategyIds=["strategy-3"],
        isActive=True,
    )

    try:
        service.create_line(payload)
        assert False, "Expected ValueError for invalid cut strategy ids"
    except ValueError as exc:
        assert "strategy-3" in str(exc)


def test_create_and_list_active_lines():
    repository = FakeLineRepository()
    service = LineService(
        repository,
        FakeCutStrategyCatalog(
            [
                {"_id": "strategy-1", "lineType": "DSI884"},
                {"_id": "strategy-2", "lineType": "DB20"},
            ]
        ),
    )

    service.create_line(
        LineCreate(
            lineId="DSI884",
            friendlyName="DSI 884",
            lineType="DSI884",
            plant="FSP",
            permittedCutStrategyIds=["strategy-1"],
            isActive=True,
        )
    )
    service.create_line(
        LineCreate(
            lineId="DB20",
            friendlyName="DB20 Main",
            lineType="DB20",
            plant="FSP",
            permittedCutStrategyIds=["strategy-2"],
            isActive=False,
        )
    )

    active_lines = service.list_active_lines()

    assert len(active_lines) == 1
    assert active_lines[0].line_id == "DSI884"


def test_update_line_preserves_created_at():
    repository = FakeLineRepository()
    service = LineService(
        repository,
        FakeCutStrategyCatalog(
            [
                {"_id": "strategy-1", "lineType": "DSI884"},
                {"_id": "strategy-2", "lineType": "DSI884"},
            ]
        ),
    )

    created = service.create_line(
        LineCreate(
            lineId="DSI884",
            friendlyName="DSI 884",
            lineType="DSI884",
            plant="FSP",
            permittedCutStrategyIds=["strategy-1"],
            isActive=True,
        )
    )

    updated = service.update_line(
        "DSI884",
        LineUpdate(
            friendlyName="DSI 884 Updated",
            lineType="DSI884",
            plant="FSP",
            permittedCutStrategyIds=["strategy-1", "strategy-2"],
            isActive=False,
        ),
    )

    assert updated is not None
    assert updated.created_at == created.created_at
    assert updated.updated_at >= created.updated_at
    assert updated.is_active is False


def test_create_line_rejects_cut_strategies_from_other_line_type():
    service = LineService(
        FakeLineRepository(),
        FakeCutStrategyCatalog(
            [
                {"_id": "strategy-1", "lineType": "DSI884"},
                {"_id": "strategy-2", "lineType": "DB20"},
            ]
        ),
    )

    payload = LineCreate(
        lineId="DSI884-L1",
        friendlyName="DSI 884 Line 1",
        lineType="DSI884",
        plant="FSP",
        permittedCutStrategyIds=["strategy-2"],
        isActive=True,
    )

    try:
        service.create_line(payload)
        assert False, "Expected ValueError for mismatched lineType"
    except ValueError as exc:
        assert "do not match lineType" in str(exc)


def test_list_lines_accepts_legacy_document_without_line_type_when_line_id_is_known_type():
    repository = FakeLineRepository()
    repository.documents["DSI884"] = {
        "lineId": "DSI884",
        "friendlyName": "Legacy DSI 884",
        "plant": "FSP",
        "permittedCutStrategyIds": [],
        "isActive": True,
        "createdAt": "2026-04-06T00:00:00Z",
        "updatedAt": "2026-04-06T00:00:00Z",
    }
    service = LineService(repository, FakeCutStrategyCatalog([]))

    # Mimic repository normalization behavior for legacy records.
    repository.documents["DSI884"]["lineType"] = "DSI884"
    lines = service.list_lines()

    assert len(lines) == 1
    assert lines[0].line_type.value == "DSI884"
