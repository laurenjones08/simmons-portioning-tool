"""Unit tests for scheduling create helper behavior."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from views.scheduling_create import (  # noqa: E402
    _apply_query_prefill,
    _coerce_max_trim_percentage,
    _parse_query_sku_ids,
)


def test_parse_query_sku_ids_dedupes_and_trims():
    assert _parse_query_sku_ids(" 17642,45066,17642 ,,31742 ") == ["17642", "45066", "31742"]


def test_coerce_max_trim_percentage_clamps_range():
    assert _coerce_max_trim_percentage("150") == 100.0
    assert _coerce_max_trim_percentage("-5") == 0.0


def test_apply_query_prefill_sets_plant_skus_and_trim():
    updates = _apply_query_prefill(
        {"scheduling_job_prefill_signature": ""},
        ["VBS", "FSP"],
        [
            {"tradeNumber": "17642"},
            {"tradeNumber": "31742"},
            {"tradeNumber": "45066"},
        ],
        "VBS",
        "17642,45066,99999",
        "12.5",
    )

    assert updates["scheduling_job_plant_id"] == "VBS"
    assert updates["scheduling_job_plant_selector"] == "VBS"
    assert updates["scheduling_job_selected_skus"] == ["17642", "45066"]
    assert updates["scheduling_job_max_trim_percentage"] == 12.5
