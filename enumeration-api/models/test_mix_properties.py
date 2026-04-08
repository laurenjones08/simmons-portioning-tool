from pydantic import ValidationError

from .mix import MIX, MfgType


def valid_payload():
    return {
        "skus": {"123": "A", "345": "B", "567": "C"},
        "includesFDS": True,
        "includesRTL": False,
        "includesNug": True,
        "nuggetTargetWeight": 15.5,
        "numFillets": 2,
        "filletWeight": 12.75,
        "mfgType": "DSI",
        "cutStrategyID": "strategy-001",
        "beltSpeed": 1.2,
        "reqPlant": "FSP",
        "reqBirdSize": "SB",
    }


def test_mix_accepts_valid_payload():
    mix = MIX(**valid_payload())
    assert mix.mix_id
    assert mix.mfg_type == MfgType.DSI
    assert mix.skus["123"] == "A"


def test_mix_requires_positive_nugget_target_when_includes_nug_true():
    payload = valid_payload()
    payload["nuggetTargetWeight"] = 0

    try:
        MIX(**payload)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        assert "nuggetTargetWeight must be > 0" in str(exc)


def test_mix_requires_null_nugget_target_when_includes_nug_false():
    payload = valid_payload()
    payload["includesNug"] = False
    payload["nuggetTargetWeight"] = 1.0

    try:
        MIX(**payload)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        assert "nuggetTargetWeight must be null" in str(exc)


def test_mix_accepts_null_nugget_target_when_includes_nug_false():
    payload = valid_payload()
    payload["includesNug"] = False
    payload["nuggetTargetWeight"] = None

    mix = MIX(**payload)
    assert mix.nugget_target_weight is None
