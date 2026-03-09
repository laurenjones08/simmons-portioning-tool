import mongomock
import pytest

from models.mix import MixCreate
from repositories.mix_repository import MixRepository
from .mix_service import MixService


def base_payload(mfg_type: str):
    return MixCreate(
        skus={"123": "A", "456": "B", "789": "C"},
        includesFDS=True,
        includesRTL=False,
        includesNug=False,
        nuggetTargetWeight=None,
        numFillets=2,
        filletWeight=12.0,
        mfgType=mfg_type,
        cutStrategyID="strategy-1",
        beltSpeed=1.1,
    )


def build_service():
    db = mongomock.MongoClient()["test_db"]
    return MixService(MixRepository(db))


def test_allows_same_sku_set_for_different_mfg_types():
    service = build_service()

    mix_dsi = service.create_mix(base_payload("DSI"))
    mix_db20 = service.create_mix(base_payload("DB20"))

    assert mix_dsi.mix_id != mix_db20.mix_id
    assert mix_dsi.mfg_type.value == "DSI"
    assert mix_db20.mfg_type.value == "DB20"


def test_rejects_duplicate_sku_set_for_same_mfg_type_even_if_part_ids_differ():
    service = build_service()

    service.create_mix(base_payload("DSI"))

    duplicate = MixCreate(
        skus={"123": "X", "456": "Y", "789": "Z"},
        includesFDS=False,
        includesRTL=True,
        includesNug=False,
        nuggetTargetWeight=None,
        numFillets=3,
        filletWeight=20.0,
        mfgType="DSI",
        cutStrategyID="strategy-2",
        beltSpeed=1.4,
        reqPlant="FSP",
        reqBirdSize="SB",
    )

    with pytest.raises(ValueError, match="already exists"):
        service.create_mix(duplicate)
