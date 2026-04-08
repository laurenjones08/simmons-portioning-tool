import mongomock
import pytest

from repositories.cut_strategy_repository import CutStrategyRepository
from models.mix import MixCreate
from repositories.mix_repository import MixRepository
from .mix_service import MixService


def seed_cut_strategy(db, strategy_id: str, line_type: str, parts: list[str]):
    db["cut_strategies"].insert_one(
        {
            "_id": strategy_id,
            "name": f"{line_type} Strategy",
            "description": f"{line_type} strategy",
            "mfgType": line_type,
            "hasNugget": False,
            "beltSpeed": 1.1,
            "parts": parts,
            "partsKey": CutStrategyRepository.generate_parts_key(parts),
        }
    )


def base_payload(mfg_type: str, strategy_id: str, skus: dict[str, str], num_fillets: int):
    return MixCreate(
        skus=skus,
        includesFDS=True,
        includesRTL=False,
        includesNug=False,
        nuggetTargetWeight=None,
        numFillets=num_fillets,
        filletWeight=12.0,
        mfgType=mfg_type,
        cutStrategyID=strategy_id,
        beltSpeed=1.1,
        reqPlant="FSP",
        reqBirdSize="SB",
    )


def build_service():
    db = mongomock.MongoClient()["test_db"]
    seed_cut_strategy(db, "strategy-1", "DSI888", ["D", "R", "M"])
    seed_cut_strategy(db, "strategy-2", "DB20", ["T", "V", "K"])
    return MixService(MixRepository(db), CutStrategyRepository(db))


def test_allows_same_sku_set_for_different_mfg_types():
    service = build_service()

    mix_dsi = service.create_mix(
        base_payload(
            "DSI888",
            "strategy-1",
            {"123": "D", "456": "R", "789": "M"},
            3,
        )
    )
    mix_db20 = service.create_mix(
        base_payload(
            "DB20",
            "strategy-2",
            {"123": "T", "456": "V", "789": "K"},
            3,
        )
    )

    assert mix_dsi.mix_id != mix_db20.mix_id
    assert mix_dsi.mfg_type.value == "DSI888"
    assert mix_db20.mfg_type.value == "DB20"


def test_rejects_duplicate_sku_set_for_same_mfg_type_even_if_part_ids_differ():
    service = build_service()

    service.create_mix(
        base_payload(
            "DSI888",
            "strategy-1",
            {"123": "D", "456": "R", "789": "M"},
            3,
        )
    )

    duplicate = MixCreate(
        skus={"123": "D", "456": "R", "789": "M"},
        includesFDS=False,
        includesRTL=True,
        includesNug=False,
        nuggetTargetWeight=None,
        numFillets=3,
        filletWeight=20.0,
        mfgType="DSI888",
        cutStrategyID="strategy-1",
        beltSpeed=1.4,
        reqPlant="FSP",
        reqBirdSize="SB",
    )

    with pytest.raises(ValueError, match="already exists"):
        service.create_mix(duplicate)


def test_rejects_mix_missing_required_strategy_parts():
    service = build_service()

    incomplete = MixCreate(
        skus={"123": "D", "456": "R"},
        includesFDS=True,
        includesRTL=False,
        includesNug=False,
        nuggetTargetWeight=None,
        numFillets=2,
        filletWeight=12.0,
        mfgType="DSI888",
        cutStrategyID="strategy-1",
        beltSpeed=1.1,
        reqPlant="FSP",
        reqBirdSize="SB",
    )

    with pytest.raises(ValueError, match="every required part"):
        service.create_mix(incomplete)
