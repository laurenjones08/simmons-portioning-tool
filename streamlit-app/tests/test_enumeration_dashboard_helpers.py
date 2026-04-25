"""Unit tests for dialog helper behavior on the enumeration dashboard."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from views.enumeration_dashboard import (  # noqa: E402
    _activate_dialog_state,
    _consume_job_progress_selection,
    _consume_mix_selection,
    _mix_detail_href,
    _mixes_for_sku,
    _scheduling_create_href,
    _default_mix_detail_bucket_id,
    _job_progress_summary,
)


def test_consume_mix_selection_opens_modal_for_new_selection():
    remembered_selection, mix_to_open = _consume_mix_selection("mix-123", None)

    assert remembered_selection == "mix-123"
    assert mix_to_open == "mix-123"


def test_consume_mix_selection_ignores_same_selection():
    remembered_selection, mix_to_open = _consume_mix_selection("mix-123", "mix-123")

    assert remembered_selection == "mix-123"
    assert mix_to_open is None


def test_consume_mix_selection_clears_when_selection_removed():
    remembered_selection, mix_to_open = _consume_mix_selection(None, "mix-123")

    assert remembered_selection is None
    assert mix_to_open is None


def test_consume_job_progress_selection_opens_for_new_job():
    remembered_selection, job_to_open = _consume_job_progress_selection("job-123", None)

    assert remembered_selection == "job-123"
    assert job_to_open == "job-123"


def test_consume_job_progress_selection_ignores_same_job():
    remembered_selection, job_to_open = _consume_job_progress_selection("job-123", "job-123")

    assert remembered_selection == "job-123"
    assert job_to_open is None


def test_activate_dialog_state_clears_other_dialogs(monkeypatch):
    fake_state = {
        "enum_sku_mix_open_sku": "17642",
        "enum_job_progress_open_id": "job-123",
        "enum_mix_detail_open_id": None,
    }
    monkeypatch.setattr("views.enumeration_dashboard.st.session_state", fake_state)

    _activate_dialog_state("enum_mix_detail_open_id", "mix-123", "enum_sku_mix_open_sku", "enum_job_progress_open_id")

    assert fake_state["enum_mix_detail_open_id"] == "mix-123"
    assert fake_state["enum_sku_mix_open_sku"] is None
    assert fake_state["enum_job_progress_open_id"] is None


def test_job_progress_summary_aggregates_stage_totals():
    summary = _job_progress_summary(
        {
            "stages": [
                {"status": "completed", "processedCombinations": 10, "totalCombinations": 10},
                {"status": "running", "processedCombinations": 3, "totalCombinations": 10},
                {"status": "pending", "processedCombinations": 0, "totalCombinations": 5},
            ]
        }
    )

    assert summary == {
        "processed": 13,
        "total": 25,
        "completed": 1,
        "running": 1,
        "pending": 1,
        "progress": 13 / 25,
    }


def test_default_mix_detail_bucket_id_uses_best_upgrade_bucket():
    bucket_id = _default_mix_detail_bucket_id(
        [
            {"bucketId": "bucket-a", "upgradePercentage": 92.0},
            {"bucketId": "bucket-b", "upgradePercentage": 96.5},
            {"bucketId": "bucket-c", "upgradePercentage": 94.0},
        ]
    )

    assert bucket_id == "bucket-b"


def test_mix_detail_href_encodes_page_and_mix_id():
    href = _mix_detail_href("mix-123")

    assert href == "?page=Enumeration+Dashboard&mixId=mix-123"


def test_scheduling_create_href_encodes_plant_and_skus():
    href = _scheduling_create_href({"reqPlant": "VBS", "skus": {"45066": "D", "17642": "R"}})

    assert href == "?page=Scheduling+Create&plantId=VBS&skuIds=17642%2C45066"


def test_mixes_for_sku_returns_matching_mix_rows():
    df = _mixes_for_sku(
        "17642",
        [
            {
                "_id": "mix-a",
                "skus": {"17642": "R", "45066": "D"},
                "mfgType": "DSI888",
                "reqPlant": "VBS",
                "reqBirdSize": "SB",
                "cutStrategyID": "strategy-a",
            },
            {
                "_id": "mix-b",
                "skus": {"99999": "R"},
                "mfgType": "DSI884",
                "reqPlant": "FSP",
                "reqBirdSize": "BB",
                "cutStrategyID": "strategy-b",
            },
        ],
        {
            "mix-a": [
                {"bucketId": "bucket-1", "upgradePercentage": 96.0, "trimPercentage": 4.0},
                {"bucketId": "bucket-2", "upgradePercentage": 94.0, "trimPercentage": 6.0},
            ],
            "mix-b": [
                {"bucketId": "bucket-3", "upgradePercentage": 90.0, "trimPercentage": 10.0},
            ],
        },
        {
            "strategy-a": {"parts": ["D", "R"]},
            "strategy-b": {"parts": ["R"]},
        },
        {
            "bucket-1": "[640, 690]",
            "bucket-2": "[590, 640]",
            "bucket-3": "[540, 590]",
        },
        None,
    )

    assert df["Mix ID"].tolist() == ["mix-a"]
    assert df.iloc[0]["Usable Buckets"] == 2
    assert df.iloc[0]["Best Bucket"] == "[640, 690]"
    assert df.iloc[0]["Open Mix"] == "?page=Enumeration+Dashboard&mixId=mix-a"
