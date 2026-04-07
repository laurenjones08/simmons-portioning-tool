from pydantic import ValidationError

from .cut_strategy import CutStrategy


def valid_payload():
    return {
        "name": "DSI Strategy",
        "description": "Default DSI strategy",
        "mfgType": "DSI",
        "hasNugget": True,
        "beltSpeed": 1.2,
        "parts": ["D", "R", "M"],
    }


def test_cut_strategy_accepts_valid_payload():
    strategy = CutStrategy(**valid_payload())
    assert strategy.strategy_id
    assert strategy.name == "DSI Strategy"


def test_cut_strategy_rejects_duplicate_parts():
    payload = valid_payload()
    payload["parts"] = ["D", "D"]

    try:
        CutStrategy(**payload)
        assert False, "Expected ValidationError"
    except ValidationError as exc:
        assert "parts must not contain duplicates" in str(exc)


def test_cut_strategy_normalizes_whitespace_parts():
    payload = valid_payload()
    payload["parts"] = [" d", " V "]

    strategy = CutStrategy(**payload)
    assert strategy.parts == ["D", "V"]
