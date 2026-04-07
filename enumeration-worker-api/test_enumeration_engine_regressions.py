from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import mongomock


sys.path.insert(0, str(Path(__file__).parent))

from enumeration_engine import (
    _build_mix,
    _compute_mix_metric,
    _get_valid_cut_strategies,
    _planned_bucket_weight,
    run_enumeration,
)


def make_sku(trade_number, allowed_parts, target_weight, customer_type="FDS", product_type="FILET"):
    return {
        "_id": trade_number,
        "tradeNumber": trade_number,
        "prodPlant": "SS2",
        "birdSize": "BB",
        "targetWeight": target_weight,
        "minWeight": target_weight * 0.8,
        "maxWeight": target_weight * 1.2,
        "customerType": customer_type,
        "productType": product_type,
        "allowedParts": allowed_parts,
        "unitsPerCut": 1,
    }


def make_cut_strategy(strategy_id, parts, has_nugget=False):
    return {
        "_id": strategy_id,
        "name": strategy_id,
        "description": strategy_id,
        "mfgType": "DSI888",
        "beltSpeed": 31,
        "hasNugget": has_nugget,
        "parts": parts,
        "lineType": "DSI888",
    }


def make_bucket(bucket_id, min_weight, target_weight, max_weight):
    return {
        "_id": bucket_id,
        "minWeight": min_weight,
        "targetWeight": target_weight,
        "maxWeight": max_weight,
    }


def _mock_requests_get(url, timeout=5):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"value": 0.0}
    return response


def _mock_settings():
    settings = MagicMock()
    settings.global_config_api_url = "http://mock-config:8001"
    return settings


def test_get_valid_cut_strategies_rejects_combo_with_unassignable_sku():
    combo = [
        make_sku("50623", ["D"], 102),
        make_sku("24479", ["R"], 151),
        make_sku("53304", ["S"], 90),
    ]
    strategy = make_cut_strategy("cs-1", ["D"])

    result = _get_valid_cut_strategies(combo, [strategy])

    assert result == []


def test_build_mix_uses_full_combo_sku_keys():
    combo = [
        make_sku("31619", ["D"], 200),
        make_sku("51309", ["R"], 180),
        make_sku("39771", ["M"], 170),
    ]
    strategy = make_cut_strategy("cs-2", ["D", "R", "M"])

    mix = _build_mix(combo, strategy, plant_filter=None, bird_size_filter=None)

    assert mix["skuKeys"] == ["31619", "51309", "39771"]
    assert set(mix["skus"].keys()) == {"31619", "51309", "39771"}


def test_repeated_non_nugget_sku_can_use_distinct_allowed_parts():
    sku = make_sku("12345", ["D", "M"], 100)
    combo = [sku, sku]
    strategy = make_cut_strategy("cs-dup", ["D", "M"])

    mix = _build_mix(combo, strategy, plant_filter=None, bird_size_filter=None)
    metric = _compute_mix_metric(
        mix_id="mix-dup",
        combo=combo,
        skus_map=mix["skus"],
        bucket=make_bucket("bucket-dup", min_weight=150, target_weight=150, max_weight=250),
        includes_nug=False,
        nugget_target_weight=None,
        config_values={"fds_value": 0.0, "rtl_value": 0.0, "trim_value": 0.0},
        part_assignments=mix["_partAssignments"],
    )

    assert [item["partCode"] for item in metric["unitPlan"]] == ["D", "M"]


def test_nugget_strip_skus_can_reuse_a_part_code():
    combo = [
        make_sku("31619", ["D"], 200, product_type="FILET"),
        make_sku("NUG01", ["D"], 25, product_type="NUGGET|STRIP"),
    ]
    strategy = make_cut_strategy("cs-nug", ["D"], has_nugget=True)

    mix = _build_mix(combo, strategy, plant_filter=None, bird_size_filter=None)
    metric = _compute_mix_metric(
        mix_id="mix-nug",
        combo=combo,
        skus_map=mix["skus"],
        bucket=make_bucket("bucket-nug", min_weight=200, target_weight=200, max_weight=260),
        includes_nug=True,
        nugget_target_weight=25,
        config_values={"fds_value": 0.0, "rtl_value": 0.0, "trim_value": 0.0},
        part_assignments=mix["_partAssignments"],
    )

    assert [item["partCode"] for item in metric["unitPlan"]] == ["D", "D"]


def test_compute_mix_metric_upgrade_percentage_uses_bucket_target_weight():
    combo = [
        make_sku("50623", ["D"], 102),
        make_sku("24479", ["R"], 151, customer_type="RTL"),
        make_sku("53304", ["M"], 90),
    ]
    metric = _compute_mix_metric(
        mix_id="mix-1",
        combo=combo,
        skus_map={"50623": "D", "24479": "R", "53304": "M"},
        bucket=make_bucket("bucket-1", min_weight=300, target_weight=350, max_weight=400),
        includes_nug=False,
        nugget_target_weight=None,
        config_values={"fds_value": 0.0, "rtl_value": 0.0, "trim_value": 0.0},
    )

    assert round(metric["upgradePercentage"], 6) == round((343 / 350) * 100, 6)


def test_compute_mix_metric_upgrade_percentage_uses_bucket_range_when_mu_sigma_configured():
    combo = [
        make_sku("50623", ["D"], 102),
        make_sku("24479", ["R"], 151, customer_type="RTL"),
        make_sku("53304", ["M"], 90),
    ]
    metric = _compute_mix_metric(
        mix_id="mix-1",
        combo=combo,
        skus_map={"50623": "D", "24479": "R", "53304": "M"},
        bucket=make_bucket("bucket-1", min_weight=300, target_weight=350, max_weight=400),
        includes_nug=False,
        nugget_target_weight=None,
        config_values={
            "fds_value": 0.0,
            "rtl_value": 0.0,
            "trim_value": 0.0,
            "upgrade_mu": 360.0,
            "upgrade_sigma": 25.0,
        },
    )

    assert metric["upgradePercentage"] != round((343 / 350) * 100, 6)
    assert 0.0 <= metric["upgradePercentage"] <= 100.0


def test_run_enumeration_does_not_persist_partial_multi_sku_mix_documents():
    client = mongomock.MongoClient()
    db = client["enumeration_db"]

    db["skus"].insert_many([
        make_sku("31619", ["D"], 180),
        make_sku("51309", ["R"], 190),
        make_sku("39771", ["M"], 170),
    ])
    db["cut_strategies"].insert_one(make_cut_strategy("cs-3", ["M"]))
    db["buckets"].insert_one(make_bucket("bucket-2", min_weight=300, target_weight=350, max_weight=600))

    job_repo = MagicMock()
    job_repo.is_cancelled.return_value = False

    with patch("config.get_settings", return_value=_mock_settings()), \
         patch("enumeration_engine.requests.get", side_effect=_mock_requests_get):
        run_enumeration(
            db=db,
            job_id="job-1",
            run_id="run-1",
            job_repo=job_repo,
            max_combination_size=3,
            batch_size=1000,
        )

    bad_mixes = list(db["mixes"].find({"skuKeys": ["31619", "51309", "39771"]}))
    bad_metrics = list(db["mix_metrics"].find({"skuKeys": ["31619", "51309", "39771"]}))

    assert bad_mixes == []
    assert bad_metrics == []


def test_planned_bucket_weight_is_used_for_bucket_fit_validation():
    client = mongomock.MongoClient()
    db = client["enumeration_db"]

    combo = [
        make_sku("50624", ["D"], 15),
        make_sku("50623", ["D"], 102),
        make_sku("39771", ["D"], 247),
    ]
    combo[0]["unitsPerCut"] = 20
    db["skus"].insert_many(combo)
    db["cut_strategies"].insert_one(make_cut_strategy("cs-4", ["D"], has_nugget=False))
    db["buckets"].insert_one(make_bucket("bucket-3", min_weight=300, target_weight=300, max_weight=390))

    job_repo = MagicMock()
    job_repo.is_cancelled.return_value = False

    with patch("config.get_settings", return_value=_mock_settings()), \
         patch("enumeration_engine.requests.get", side_effect=_mock_requests_get):
        run_enumeration(
            db=db,
            job_id="job-2",
            run_id="run-2",
            job_repo=job_repo,
            max_combination_size=3,
            batch_size=1000,
        )

    metric = _compute_mix_metric(
        mix_id="mix-2",
        combo=combo,
        skus_map={"50624": "D", "50623": "D", "39771": "D"},
        bucket=make_bucket("bucket-3", min_weight=300, target_weight=300, max_weight=390),
        includes_nug=False,
        nugget_target_weight=None,
        config_values={"fds_value": 0.0, "rtl_value": 0.0, "trim_value": 0.0},
    )

    planned_weight = _planned_bucket_weight(
        combo,
        make_bucket("bucket-3", min_weight=300, target_weight=300, max_weight=390),
        includes_nug=False,
        nugget_target_weight=None,
    )

    assert sum(item["totalWeightInPlan"] for item in metric["unitPlan"]) == planned_weight == 649
    assert list(db["mix_metrics"].find({"bucketId": "bucket-3", "skuKeys": ["50624", "50623", "39771"]})) == []
