"""Tests for LineRepository legacy document normalization."""

import mongomock

from repositories.line_repository import LineRepository


def test_find_all_backfills_line_type_from_line_id():
    database = mongomock.MongoClient()["test_config_db"]
    database["lines"].insert_one(
        {
            "lineId": "DSI884",
            "friendlyName": "Legacy DSI 884",
            "plant": "FSP",
            "permittedCutStrategyIds": [],
            "isActive": True,
            "createdAt": "2026-04-06T00:00:00Z",
            "updatedAt": "2026-04-06T00:00:00Z",
        }
    )

    repository = LineRepository(database)
    documents = repository.find_all()

    assert len(documents) == 1
    assert documents[0]["lineType"] == "DSI884"


def test_find_by_id_backfills_line_type_from_legacy_mfg_type():
    database = mongomock.MongoClient()["test_config_db"]
    database["lines"].insert_one(
        {
            "lineId": "LINE-001",
            "friendlyName": "Legacy DB20",
            "mfgType": "DB20",
            "plant": "FSP",
            "permittedCutStrategyIds": [],
            "isActive": True,
            "createdAt": "2026-04-06T00:00:00Z",
            "updatedAt": "2026-04-06T00:00:00Z",
        }
    )

    repository = LineRepository(database)
    document = repository.find_by_id("LINE-001")

    assert document is not None
    assert document["lineType"] == "DB20"
