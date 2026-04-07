import mongomock
from bson import ObjectId

from models.cut_strategy import (
    CutStrategyCreate,
    CutStrategySearchCriteria,
    CutStrategyUpdate,
)
from repositories.cut_strategy_repository import CutStrategyRepository
from repositories.mix_metric_repository import MixMetricRepository
from repositories.mix_repository import MixRepository
from .cut_strategy_service import CutStrategyService


def build_service():
    db = mongomock.MongoClient()["test_db"]
    return CutStrategyService(
        CutStrategyRepository(db),
        MixRepository(db),
        MixMetricRepository(db),
    ), db


def test_cut_strategy_service_crud_and_search():
    service, db = build_service()

    created = service.create_cut_strategy(
        CutStrategyCreate(
            name="DSI Strategy",
            description="Default DSI strategy",
            mfgType="DSI",
            hasNugget=True,
            beltSpeed=1.2,
            parts=["D", "R", "M"],
        )
    )

    fetched = service.get_cut_strategy_by_id(created.strategy_id)
    assert fetched is not None
    assert fetched.strategy_id == created.strategy_id

    search_result = service.search_cut_strategies(
        CutStrategySearchCriteria(mfgType="DSI", includesPart="D")
    )
    assert len(search_result) == 1
    assert search_result[0].strategy_id == created.strategy_id

    updated = service.update_cut_strategy(
        created.strategy_id,
        CutStrategyUpdate(
            name="DSI Strategy",
            description="Updated",
            mfgType="DSI",
            hasNugget=False,
            beltSpeed=1.1,
            parts=["D", "R"],
        ),
    )
    assert updated is not None
    assert updated.has_nugget is False

    # Seed dependent mixes + metrics that should be cascade-deleted.
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
                "cutStrategyID": created.strategy_id,
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
                "cutStrategyID": created.strategy_id,
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

    result = service.delete_cut_strategy(created.strategy_id)
    assert result["deleted"] is True
    assert result["mixes_deleted"] == 2
    assert result["metrics_deleted"] == 2
    assert service.get_cut_strategy_by_id(created.strategy_id) is None


def test_cut_strategy_search_handles_objectid_documents():
    service, db = build_service()

    raw_id = ObjectId()
    db["cut_strategies"].insert_one(
        {
            "_id": raw_id,
            "name": "Legacy Strategy",
            "description": "legacy",
            "mfgType": "DB20",
            "hasNugget": False,
            "beltSpeed": 1.0,
            "parts": ["K", " V"],
            "partsKey": CutStrategyRepository.generate_parts_key(["K", "V"]),
        }
    )

    result = service.search_cut_strategies(CutStrategySearchCriteria(mfgType="DB20"))

    assert len(result) == 1
    assert result[0].strategy_id == str(raw_id)
    assert result[0].parts == ["K", "V"]


def test_cut_strategy_search_skips_invalid_mfgtype_documents():
    service, db = build_service()

    db["cut_strategies"].insert_many(
        [
            {
                "_id": "valid-1",
                "name": "Valid Strategy",
                "description": "ok",
                "mfgType": "DSI",
                "hasNugget": False,
                "beltSpeed": 1.0,
                "parts": ["D", "R"],
                "partsKey": CutStrategyRepository.generate_parts_key(["D", "R"]),
            },
            {
                "_id": "invalid-1",
                "name": "Bad Legacy Strategy",
                "description": "bad",
                "mfgType": "DSI888",
                "hasNugget": False,
                "beltSpeed": 1.0,
                "parts": ["D", "R"],
                "partsKey": CutStrategyRepository.generate_parts_key(["D", "R"]),
            },
        ]
    )

    result = service.search_cut_strategies(CutStrategySearchCriteria())

    assert len(result) == 1
    assert result[0].strategy_id == "valid-1"


def test_cut_strategy_get_by_id_returns_none_for_invalid_legacy_document():
    service, db = build_service()

    db["cut_strategies"].insert_one(
        {
            "_id": "invalid-by-id",
            "name": "Bad Legacy Strategy",
            "description": "bad",
            "mfgType": "DSI888",
            "hasNugget": False,
            "beltSpeed": 1.0,
            "parts": ["D", "R"],
            "partsKey": CutStrategyRepository.generate_parts_key(["D", "R"]),
        }
    )

    result = service.get_cut_strategy_by_id("invalid-by-id")

    assert result is None
