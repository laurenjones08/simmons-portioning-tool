"""Unit tests for the scheduling analytics helpers."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from views.scheduling_shared import (  # noqa: E402
    build_bucket_usage_summary,
    build_demand_progress,
    build_line_utilization,
    build_todays_cut_schedule,
    build_upcoming_cuts,
    focus_window,
)


def test_focus_window_uses_job_plan_start_and_horizon():
    job = {"planStartDate": "2026-04-15", "horizonDays": 4}
    start, end = focus_window(job, [])
    assert str(start.date()) == "2026-04-15"
    assert str(end.date()) == "2026-04-18"


def test_build_todays_and_upcoming_cuts():
    frame = pd.DataFrame(
        [
            {"date": "2026-04-15", "line": "DSI884", "cuts": "mix-a"},
            {"date": "2026-04-15", "line": "DB20", "cuts": "mix-b"},
            {"date": "2026-04-16", "line": "DSI884", "cuts": "mix-c"},
        ]
    )
    focus = pd.Timestamp("2026-04-15")
    today = build_todays_cut_schedule(frame, focus)
    upcoming = build_upcoming_cuts(frame, focus, days=2)
    assert len(today) == 2
    assert len(upcoming) == 1
    assert upcoming.iloc[0]["line"] == "DSI884"


def test_build_demand_progress_and_line_utilization():
    outputs = pd.DataFrame(
        [
            {"date": "2026-04-15", "skuId": "50624", "lbsProduced": 1000.0},
            {"date": "2026-04-16", "skuId": "50624", "lbsProduced": 500.0},
        ]
    )
    demands = pd.DataFrame(
        [
            {"dueDate": "2026-04-15", "skuId": "50624", "demandValue": 1200.0},
            {"dueDate": "2026-04-16", "skuId": "50624", "demandValue": 400.0},
        ]
    )
    progress = build_demand_progress(outputs, demands, pd.Timestamp("2026-04-15"), pd.Timestamp("2026-04-16"))
    assert list(progress["coverage_pct"]) == [83.3, 125.0]

    line_load = pd.DataFrame(
        [
            {"date": "2026-04-15", "line": "DSI884", "hours_used": 6.0, "hours_available": 8.0, "util_pct": 75.0, "throughput_util_pct": 50.0},
            {"date": "2026-04-16", "line": "DSI884", "hours_used": 4.0, "hours_available": 8.0, "util_pct": 50.0, "throughput_util_pct": 25.0},
        ]
    )
    utilization = build_line_utilization(line_load, pd.Timestamp("2026-04-15"), pd.Timestamp("2026-04-16"))
    assert utilization.iloc[0]["line"] == "DSI884"
    assert utilization.iloc[0]["avg_util_pct"] == 62.5


def test_build_bucket_usage_summary():
    frame = pd.DataFrame(
        [
            {"date": "2026-04-15", "bucket": "B 0-390", "used_lbs": 30.0, "available_lbs": 100.0, "util_pct": 30.0},
            {"date": "2026-04-16", "bucket": "B 0-390", "used_lbs": 20.0, "available_lbs": 100.0, "util_pct": 20.0},
            {"date": "2026-04-15", "bucket": "B 390-440", "used_lbs": 10.0, "available_lbs": 50.0, "util_pct": 20.0},
        ]
    )
    summary = build_bucket_usage_summary(frame, pd.Timestamp("2026-04-15"), pd.Timestamp("2026-04-16"))
    assert list(summary["bucket"]) == ["B 0-390", "B 390-440"]
    assert summary.iloc[0]["used_lbs"] == 50.0

