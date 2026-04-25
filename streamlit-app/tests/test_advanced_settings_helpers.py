"""Unit tests for helper functions used by the Advanced Settings inline editors."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from views.advanced_settings import (  # noqa: E402
    _available_wip_payload,
    _available_wip_rows,
    _format_list_value,
    _format_cut_strategy_label,
    _parse_bool,
    _parse_config_value,
    _parse_list_value,
    _plant_options,
    _strategy_options_for_line,
)


def test_format_list_value_joins_lists():
    assert _format_list_value(["D", "R", ""]) == "D, R"


def test_parse_list_value_splits_and_trims():
    assert _parse_list_value("D, R, , M") == ["D", "R", "M"]


def test_parse_bool_accepts_common_text_values():
    assert _parse_bool("true") is True
    assert _parse_bool("False") is False
    assert _parse_bool(True) is True


def test_parse_config_value_coerces_by_type():
    assert _parse_config_value("42", "int") == 42
    assert _parse_config_value("3.5", "float") == 3.5
    assert _parse_config_value("yes", "bool") is True
    assert _parse_config_value("FSP", "string") == "FSP"


def test_format_cut_strategy_label_is_descriptive():
    label = _format_cut_strategy_label(
        {
            "name": "DSI 2 for 1",
            "lineType": "DSI884",
            "parts": ["D", "R"],
        }
    )
    assert label == "DSI 2 for 1 - DSI884 [D, R]"


def test_plant_options_reads_global_config_value():
    configs = [
        {"key": "mix.availablePlants", "value": "FSP, VBS, SS2"},
        {"key": "other.setting", "value": "ignore"},
    ]
    assert _plant_options(configs) == ["FSP", "VBS", "SS2"]


def test_strategy_options_are_filtered_by_line_type():
    strategies = [
        {"_id": "a", "name": "One", "lineType": "DB20", "parts": ["D"]},
        {"_id": "b", "name": "Two", "lineType": "DSI884", "parts": ["R"]},
    ]
    ids, labels = _strategy_options_for_line(strategies, "DSI884")
    assert ids == ["b"]
    assert labels == {"b": "Two - DSI884 [R]"}


def test_available_wip_rows_normalize_api_documents():
    rows = _available_wip_rows(
        [
            {
                "_id": "wip-1",
                "plantName": "FSP",
                "bucketId": "bucket-1",
                "availableLbs": 1250.0,
            }
        ]
    )
    assert rows == [
        {
            "availableWipId": "wip-1",
            "plantName": "FSP",
            "bucketId": "bucket-1",
            "availableLbs": 1250.0,
        }
    ]


def test_available_wip_payload_validates_and_trims_values():
    payload = _available_wip_payload(
        {
            "plantName": " FSP ",
            "bucketId": " bucket-1 ",
            "availableLbs": "42.5",
        },
        {},
    )
    assert payload == {
        "plantName": "FSP",
        "bucketId": "bucket-1",
        "availableLbs": 42.5,
    }
