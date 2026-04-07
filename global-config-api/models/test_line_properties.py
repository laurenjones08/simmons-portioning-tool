"""Property-style tests for line model normalization."""

import pytest

from models.line import LineCreate


def test_line_create_normalizes_cut_strategy_ids():
    line = LineCreate(
        lineId=" DSI884 ",
        friendlyName=" DSI 884 ",
        lineType="DSI884",
        plant=" FSP ",
        permittedCutStrategyIds=[" strategy-1 ", "strategy-2"],
        isActive=True,
    )

    assert line.line_id == "DSI884"
    assert line.friendly_name == "DSI 884"
    assert line.line_type.value == "DSI884"
    assert line.plant == "FSP"
    assert line.permitted_cut_strategy_ids == ["strategy-1", "strategy-2"]


def test_line_create_rejects_duplicate_cut_strategy_ids():
    with pytest.raises(ValueError):
        LineCreate(
            lineId="DSI884",
            friendlyName="DSI 884",
            lineType="DSI884",
            plant="FSP",
            permittedCutStrategyIds=["strategy-1", "strategy-1"],
            isActive=True,
        )
