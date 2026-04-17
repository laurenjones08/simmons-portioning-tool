import mongomock
from bson import ObjectId

from repositories.mix_metric_repository import MixMetricRepository
from repositories.mix_repository import MixRepository
from repositories.cut_strategy_repository import CutStrategyRepository
from .mix_service import MixService
from .mix_metric_service import MixMetricService
from models.mix import MixSearchCriteria
from models.mix_metric import MixMetricSearchCriteria


def build_services():
    db = mongomock.MongoClient()["test_db"]
    return (
        MixService(MixRepository(db), CutStrategyRepository(db)),
        MixMetricService(MixMetricRepository(db), MixRepository(db)),
        db,
    )


def test_mix_search_parses_legacy_dsi_documents():
    mix_service, _, db = build_services()

    db["mixes"].insert_many(
        [
            {
                "_id": "legacy-mix-1",
                "skus": {"123": "D", "456": "R"},
                "skuKeys": ["123", "456"],
                "includesFDS": True,
                "includesRTL": False,
                "includesNug": False,
                "nuggetTargetWeight": None,
                "numFillets": 2,
                "filletWeight": 24.0,
                "mfgType": "DSI",
                "reqPlant": "FSP",
                "reqBirdSize": "SB",
                "cutStrategyID": "strategy-1",
                "beltSpeed": 1.2,
                "skuSetKey": "123|456",
            },
            {
                "_id": "legacy-mix-bad",
                "skus": {"789": "T"},
                "includesFDS": True,
                "includesRTL": False,
                "includesNug": False,
                "nuggetTargetWeight": None,
                "numFillets": 1,
                "filletWeight": 10.0,
                "mfgType": "INVALID",
                "reqPlant": "FSP",
                "reqBirdSize": "SB",
                "cutStrategyID": "strategy-1",
                "beltSpeed": 1.0,
                "skuSetKey": "789",
            },
        ]
    )

    result = mix_service.search_mixes(MixSearchCriteria())

    assert [mix.mix_id for mix in result] == ["legacy-mix-1"]
    assert result[0].mfg_type.value == "DSI"


def test_mix_search_normalizes_objectid_fields():
    mix_service, _, db = build_services()
    cut_strategy_id = ObjectId()
    mix_id = ObjectId()

    db["mixes"].insert_one(
        {
            "_id": mix_id,
            "skus": {"123": "D"},
            "skuKeys": ["123"],
            "includesFDS": True,
            "includesRTL": False,
            "includesNug": False,
            "nuggetTargetWeight": None,
            "numFillets": 1,
            "filletWeight": 12.0,
            "mfgType": "DSI",
            "reqPlant": "FSP",
            "reqBirdSize": "SB",
            "cutStrategyID": cut_strategy_id,
            "beltSpeed": 1.2,
            "skuSetKey": "123",
        }
    )

    result = mix_service.search_mixes(MixSearchCriteria())

    assert [mix.mix_id for mix in result] == [str(mix_id)]
    assert result[0].cut_strategy_id == str(cut_strategy_id)


def test_mix_metric_search_populates_missing_sku_keys():
    _, metric_service, db = build_services()

    db["mixes"].insert_one(
        {
            "_id": "mix-1",
            "skus": {"123": "D"},
            "skuKeys": ["123"],
            "includesFDS": True,
            "includesRTL": False,
            "includesNug": False,
            "nuggetTargetWeight": None,
            "numFillets": 1,
            "filletWeight": 12.0,
            "mfgType": "DSI",
            "reqPlant": "FSP",
            "reqBirdSize": "SB",
            "cutStrategyID": "strategy-1",
            "beltSpeed": 1.2,
            "skuSetKey": "123",
        }
    )
    db["mix_metrics"].insert_many(
        [
            {
                "_id": "mix-1:bucket-1",
                "mixId": "mix-1",
                "bucketId": "bucket-1",
                "upgradePercentage": 20.0,
                "value": 4.0,
                "trimPercentage": 5.0,
                "unitPlan": [
                    {
                        "sku": "123",
                        "partCode": "D",
                        "unitsInPlan": 1,
                        "totalWeightInPlan": 12.0,
                    }
                ],
            },
            {
                "_id": "mix-1:bucket-bad",
                "mixId": "mix-1",
                "bucketId": "bucket-bad",
                "upgradePercentage": 10.0,
                "value": 2.0,
                "trimPercentage": 3.0,
                "unitPlan": [
                    {
                        "sku": "",
                        "partCode": "D",
                        "unitsInPlan": 1,
                        "totalWeightInPlan": 12.0,
                    }
                ],
            },
        ]
    )

    result = metric_service.search_metrics(MixMetricSearchCriteria())

    assert [metric.metric_id for metric in result] == ["mix-1:bucket-1"]
    assert result[0].sku_keys == ["123"]


def test_mix_metric_search_normalizes_objectid_fields():
    _, metric_service, db = build_services()
    mix_id = ObjectId()
    bucket_id = ObjectId()

    db["mix_metrics"].insert_one(
        {
            "_id": f"{mix_id}:{bucket_id}",
            "mixId": mix_id,
            "bucketId": bucket_id,
            "upgradePercentage": 20.0,
            "value": 4.0,
            "trimPercentage": 5.0,
            "unitPlan": [
                {
                    "sku": "123",
                    "partCode": "D",
                    "unitsInPlan": 1,
                    "totalWeightInPlan": 12.0,
                }
            ],
        }
    )

    result = metric_service.search_metrics(MixMetricSearchCriteria())

    assert len(result) == 1
    assert result[0].mix_id == str(mix_id)
    assert result[0].bucket_id == str(bucket_id)
