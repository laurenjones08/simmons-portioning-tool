"""Unit tests for scheduling insights helper behavior."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from views.scheduling_insights import (  # noqa: E402
    _derived_cut_schedule,
    _display_cut_name,
    _enrich_decision_frame,
    _global_context,
    _line_config_frame,
    _load_monthly_contracts,
    _mix_metric_context_map,
    _mix_label_map,
)


def test_line_config_frame_accepts_dataframe_input():
    lines = pd.DataFrame(
        [
            {
                "lineId": "DSI884",
                "friendlyName": "Main Line",
                "lineType": "DSI",
                "plant": "FSP",
            }
        ]
    )

    frame = _line_config_frame(lines)

    assert not frame.empty
    assert frame.iloc[0]["lineId"] == "DSI884"


def test_global_context_uses_available_dates():
    decisions = pd.DataFrame([{"date": "2026-04-15"}])
    outputs = pd.DataFrame([{"date": "2026-04-17"}])
    demands = pd.DataFrame([{"dueDate": "2026-04-20"}])

    start, end = _global_context(decisions, outputs, demands)

    assert str(start.date()) == "2026-04-15"
    assert str(end.date()) == "2026-04-20"


def test_load_monthly_contracts_uses_single_search_when_only_months(monkeypatch):
    _load_monthly_contracts.clear()
    single_search = MagicMock(side_effect=[[{"skuId": "50624", "yearMonth": "2026-04"}], [{"skuId": "50624", "yearMonth": "2026-05"}]])
    bulk_search = MagicMock()

    monkeypatch.setattr("views.scheduling_insights.search_monthly_contract_demands", single_search)
    monkeypatch.setattr("views.scheduling_insights.search_monthly_contract_demands_bulk", bulk_search)

    rows = _load_monthly_contracts(("month:2026-04", "month:2026-05"))

    assert rows == [
        {"skuId": "50624", "yearMonth": "2026-04"},
        {"skuId": "50624", "yearMonth": "2026-05"},
    ]
    assert single_search.call_count == 2
    bulk_search.assert_not_called()


def test_mix_label_map_builds_readable_mix_labels():
    labels = _mix_label_map(
        [
            {
                "_id": "mix-1",
                "mfgType": "DSI884",
                "reqPlant": "FSP",
                "reqBirdSize": "SB",
                "skuKeys": ["50624", "50625"],
            }
        ]
    )

    assert labels["mix-1"] == "DSI884 | FSP | SB | 2 SKUs"


def test_display_cut_name_uses_mix_label_for_metric_identifier():
    label = _display_cut_name("mix-1:bucket-1", {"mix-1": "DSI884 | FSP | SB | 2 SKUs"})

    assert label == "DSI884 | FSP | SB | 2 SKUs"


def test_display_cut_name_prefers_metric_context_details():
    label = _display_cut_name(
        "mix-1:bucket-1",
        {"mix-1": "DSI884 | FSP | SB | 2 SKUs"},
        {"mix-1:bucket-1": {"mixLabel": "DSI884 | FSP | SB | 2 SKUs", "bucketLabel": "bucket-1 [0, 390]", "skuIds": "50624, 50625"}},
    )

    assert label == "DSI884 | FSP | SB | 2 SKUs | bucket-1 [0, 390] | 50624, 50625"


def test_derived_cut_schedule_replaces_metric_ids_with_mix_labels():
    frame = pd.DataFrame(
        [
            {
                "date": "2026-04-20",
                "lineId": "DSI884",
                "mixId": "mix-1:bucket-1",
                "lbsProduced": 1200.0,
                "duration": 4.0,
            }
        ]
    )

    schedule = _derived_cut_schedule(frame, {"mix-1": "DSI884 | FSP | SB | 2 SKUs"})

    assert schedule.iloc[0]["cuts"] == "DSI884 | FSP | SB | 2 SKUs (1,200 lbs, 4.0h)"


def test_mix_metric_context_map_includes_bucket_and_unit_plan_details():
    mix_labels = {"mix-1": "DSI884 | FSP | SB | 2 SKUs"}
    bucket_labels = {"bucket-1": "bucket-1 [0, 390]"}

    context = _mix_metric_context_map(
        [
            {
                "_id": "mix-1:bucket-1",
                "mixId": "mix-1",
                "bucketId": "bucket-1",
                "unitPlan": [
                    {"sku": "50624", "partCode": "D", "unitsInPlan": 1, "totalWeightInPlan": 109.0},
                    {"sku": "50625", "partCode": "R", "unitsInPlan": 2, "totalWeightInPlan": 218.0},
                ],
            }
        ],
        mix_labels,
        bucket_labels,
    )

    assert context["mix-1:bucket-1"]["bucketLabel"] == "bucket-1 [0, 390]"
    assert context["mix-1:bucket-1"]["skuIds"] == "50624, 50625"
    assert context["mix-1:bucket-1"]["unitPlan"] == "50624 D x1 (109g); 50625 R x2 (218g)"


def test_enrich_decision_frame_adds_bucket_skus_and_unit_plan():
    decisions = pd.DataFrame(
        [
            {
                "date": "2026-04-20",
                "lineId": "DSI884",
                "mixId": "mix-1:bucket-1",
                "lbsProduced": 1200.0,
                "duration": 4.0,
            }
        ]
    )

    enriched = _enrich_decision_frame(
        decisions,
        {"mix-1": "DSI884 | FSP | SB | 2 SKUs"},
        {
            "mix-1:bucket-1": {
                "mixLabel": "DSI884 | FSP | SB | 2 SKUs",
                "bucketLabel": "bucket-1 [0, 390]",
                "skuIds": "50624, 50625",
                "unitPlan": "50624 D x1 (109g); 50625 R x2 (218g)",
            }
        },
    )

    assert enriched.iloc[0]["mix"] == "DSI884 | FSP | SB | 2 SKUs"
    assert enriched.iloc[0]["bucket"] == "bucket-1 [0, 390]"
    assert enriched.iloc[0]["skuIds"] == "50624, 50625"
    assert enriched.iloc[0]["unitPlan"] == "50624 D x1 (109g); 50625 R x2 (218g)"
