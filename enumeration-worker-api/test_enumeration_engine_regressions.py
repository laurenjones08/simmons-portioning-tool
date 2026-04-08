from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import mongomock
import pytest
from pymongo.errors import DuplicateKeyError


sys.path.insert(0, str(Path(__file__).parent))

from enumeration_engine import (
    _build_mix,
    _compute_mix_metric,
    _get_valid_cut_strategies,
    _planned_bucket_weight,
    _upsert_mix,
    _upsert_mix_metric,
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


def test_nugget_only_mix_is_rejected():
    combo = [make_sku("NUG01", ["D"], 20, product_type="NUGGET|STRIP")]
    strategy = make_cut_strategy("cs-nug", ["D"], has_nugget=True)

    mix = _build_mix(combo, strategy, plant_filter=None, bird_size_filter=None)

    with pytest.raises(ValueError, match="Nugget-only mixes are not allowed"):
        _compute_mix_metric(
            mix_id="mix-nug-only",
            combo=combo,
            skus_map=mix["skus"],
            bucket=make_bucket("bucket-nug-only", min_weight=390, target_weight=390, max_weight=440),
            includes_nug=True,
            nugget_target_weight=20,
            config_values={"fds_value": 0.0, "rtl_value": 0.0, "trim_value": 0.0},
            part_assignments=mix["_partAssignments"],
        )


def test_nugget_units_in_plan_scales_from_remaining_bucket_weight():
    combo = [
        make_sku("10001", ["D"], 100, product_type="FILET"),
        make_sku("10002", ["M"], 100, product_type="FILET"),
        make_sku("45066", ["R"], 20, customer_type="FDS", product_type="NUGGET|STRIP"),
    ]
    strategy = make_cut_strategy("cs-target", ["D", "M", "R"], has_nugget=True)

    mix = _build_mix(combo, strategy, plant_filter=None, bird_size_filter=None)
    metric = _compute_mix_metric(
        mix_id="mix-target",
        combo=combo,
        skus_map=mix["skus"],
        bucket=make_bucket("bucket-target", min_weight=390, target_weight=390, max_weight=440),
        includes_nug=True,
        nugget_target_weight=20,
        config_values={"fds_value": 0.0, "rtl_value": 0.0, "trim_value": 0.0},
        part_assignments=mix["_partAssignments"],
    )

    nugget_item = next(item for item in metric["unitPlan"] if item["sku"] == "45066")
    assert nugget_item["unitsInPlan"] == 9
    assert nugget_item["totalWeightInPlan"] == 180


def test_nugget_value_increases_with_larger_bucket_when_more_nuggets_fit():
    combo = [
        make_sku("45066", ["D"], 16, customer_type="FDS", product_type="NUGGET|STRIP"),
        make_sku("38130", ["M"], 166, customer_type="RTL", product_type="FILET"),
    ]
    strategy = make_cut_strategy("cs-value", ["D", "M"], has_nugget=True)

    mix = _build_mix(combo, strategy, plant_filter=None, bird_size_filter=None)
    small_bucket_metric = _compute_mix_metric(
        mix_id="mix-value",
        combo=combo,
        skus_map=mix["skus"],
        bucket=make_bucket("bucket-small", min_weight=0.01, target_weight=300, max_weight=390),
        includes_nug=True,
        nugget_target_weight=16,
        config_values={"fds_value": 1.0, "rtl_value": 0.0, "trim_value": 0.0},
        part_assignments=mix["_partAssignments"],
    )
    large_bucket_metric = _compute_mix_metric(
        mix_id="mix-value",
        combo=combo,
        skus_map=mix["skus"],
        bucket=make_bucket("bucket-large", min_weight=390, target_weight=390, max_weight=440),
        includes_nug=True,
        nugget_target_weight=16,
        config_values={"fds_value": 1.0, "rtl_value": 0.0, "trim_value": 0.0},
        part_assignments=mix["_partAssignments"],
    )

    assert small_bucket_metric["value"] == 128.0
    assert large_bucket_metric["value"] == 224.0
    assert large_bucket_metric["value"] > small_bucket_metric["value"]


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


def test_compute_mix_metric_upgrade_percentage_uses_monte_carlo_when_mu_sigma_configured():
    combo = [
        make_sku("50623", ["D"], 102),
        make_sku("24479", ["R"], 151, customer_type="RTL"),
        make_sku("53304", ["M"], 90),
    ]
    with patch("enumeration_engine._mc_truncated_avg_pdf", return_value=(262.5, 0.0, 0.25)) as mocked_mc:
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

    mocked_mc.assert_called_once()
    assert metric["upgradePercentage"] == 75.0
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


def test_mix_weight_above_bucket_target_does_not_persist_metric():
    client = mongomock.MongoClient()
    db = client["enumeration_db"]

    combo = [
        make_sku("50624", ["D"], 15),
        make_sku("50623", ["R"], 102),
        make_sku("39771", ["M"], 247),
    ]
    combo[0]["unitsPerCut"] = 20
    db["skus"].insert_many(combo)
    db["cut_strategies"].insert_one(make_cut_strategy("cs-4", ["D", "R", "M"], has_nugget=False))
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
    persisted = list(db["mix_metrics"].find({"bucketId": "bucket-3", "skuKeys": ["50624", "50623", "39771"]}))
    assert persisted == []


def test_mix_weight_below_bucket_minimum_still_persists_metric():
    client = mongomock.MongoClient()
    db = client["enumeration_db"]

    combo = [
        make_sku("38130", ["K"], 166, customer_type="RTL"),
        make_sku("55103", ["D"], 95),
    ]
    db["skus"].insert_many(combo)
    db["cut_strategies"].insert_one(make_cut_strategy("cs-5", ["D", "K"], has_nugget=False))
    db["buckets"].insert_one(make_bucket("bucket-7", min_weight=390, target_weight=390, max_weight=440))

    job_repo = MagicMock()
    job_repo.is_cancelled.return_value = False

    with patch("config.get_settings", return_value=_mock_settings()), \
         patch("enumeration_engine.requests.get", side_effect=_mock_requests_get):
        run_enumeration(
            db=db,
            job_id="job-6",
            run_id="run-6",
            job_repo=job_repo,
            max_combination_size=2,
            batch_size=1000,
        )

    persisted = list(db["mix_metrics"].find({"bucketId": "bucket-7", "skuKeys": ["38130", "55103"]}))
    assert len(persisted) == 1
    assert persisted[0]["totalProductProducedGrams"] == 261


def test_repeated_sku_combos_persist_as_distinct_mixes():
    client = mongomock.MongoClient()
    db = client["enumeration_db"]

    db["skus"].insert_one(make_sku("31035", ["D", "M", "R"], 100))
    db["cut_strategies"].insert_many([
        make_cut_strategy("cs-1", ["D"], has_nugget=False),
        make_cut_strategy("cs-2", ["D", "M"], has_nugget=False),
        make_cut_strategy("cs-3", ["D", "M", "R"], has_nugget=False),
    ])
    db["buckets"].insert_one(make_bucket("bucket-4", min_weight=50, target_weight=300, max_weight=390))

    job_repo = MagicMock()
    job_repo.is_cancelled.return_value = False

    with patch("config.get_settings", return_value=_mock_settings()), \
         patch("enumeration_engine.requests.get", side_effect=_mock_requests_get):
        run_enumeration(
            db=db,
            job_id="job-3",
            run_id="run-3",
            job_repo=job_repo,
            max_combination_size=3,
            batch_size=1000,
        )

    sku_set_keys = sorted(mix["skuSetKey"] for mix in db["mixes"].find({}))
    assert sku_set_keys == [
        "31035",
        "31035|31035",
        "31035|31035|31035",
    ]


def test_upsert_mix_skips_rewriting_an_unchanged_existing_document():
    existing_doc = {
        "_id": "mix-1",
        "skus": {"31035": "D"},
        "skuSetKey": "31035",
        "mfgType": "DSI888",
        "cutStrategyID": "cs-1",
        "beltSpeed": 31,
        "includesFDS": True,
        "includesRTL": False,
        "includesNug": False,
        "nuggetTargetWeight": None,
        "reqPlant": "SS2",
        "reqBirdSize": "BB",
        "numFillets": 1,
        "filletWeight": 100.0,
        "skuKeys": ["31035"],
    }
    mix_repo = MagicMock()
    mix_repo.search.return_value = [existing_doc]

    mix_id = _upsert_mix(mix_repo, dict(existing_doc))

    assert mix_id == "mix-1"
    mix_repo.update.assert_not_called()
    mix_repo.create.assert_not_called()


def test_upsert_mix_metric_skips_rewriting_an_unchanged_existing_document():
    existing_doc = {
        "_id": "mix-1:bucket-1",
        "mixId": "mix-1",
        "bucketId": "bucket-1",
        "upgradePercentage": 42.0,
        "value": 1.0,
        "trimPercentage": 58.0,
        "unitPlan": [
            {
                "sku": "31035",
                "partCode": "D",
                "unitsInPlan": 1,
                "totalWeightInPlan": 100.0,
                "pctOfTotal": 100.0,
            }
        ],
        "totalProductProducedGrams": 100.0,
        "skuKeys": ["31035"],
    }
    metric_repo = MagicMock()
    metric_repo.create.side_effect = DuplicateKeyError("duplicate key error")
    metric_repo.get_by_id.return_value = existing_doc

    _upsert_mix_metric(metric_repo, dict(existing_doc))

    metric_repo.update.assert_not_called()


def test_run_enumeration_skips_nugget_only_combos():
    client = mongomock.MongoClient()
    db = client["enumeration_db"]

    db["skus"].insert_many([
        make_sku("NUG01", ["D"], 16, product_type="NUGGET|STRIP"),
        make_sku("FIL01", ["R"], 140, product_type="FILET"),
    ])
    db["cut_strategies"].insert_many([
        make_cut_strategy("cs-nug", ["D"], has_nugget=True),
        make_cut_strategy("cs-fil", ["R"], has_nugget=False),
    ])
    db["buckets"].insert_one(make_bucket("bucket-5", min_weight=10, target_weight=200, max_weight=300))

    job_repo = MagicMock()
    job_repo.is_cancelled.return_value = False

    with patch("config.get_settings", return_value=_mock_settings()), \
         patch("enumeration_engine.requests.get", side_effect=_mock_requests_get):
        run_enumeration(
            db=db,
            job_id="job-4",
            run_id="run-4",
            job_repo=job_repo,
            max_combination_size=1,
            batch_size=1000,
        )

    mix_sku_keys = sorted(mix["skuKeys"] for mix in db["mixes"].find({}))
    metric_sku_keys = sorted(metric["skuKeys"] for metric in db["mix_metrics"].find({}))

    assert ["FIL01"] in mix_sku_keys
    assert ["NUG01"] not in mix_sku_keys
    assert ["FIL01"] in metric_sku_keys
    assert ["NUG01"] not in metric_sku_keys


def test_run_enumeration_persists_metric_derived_fields():
    client = mongomock.MongoClient()
    db = client["enumeration_db"]

    db["skus"].insert_many([
        make_sku("FIL01", ["R"], 140, product_type="FILET"),
        make_sku("FIL02", ["M"], 160, product_type="FILET"),
    ])
    db["cut_strategies"].insert_one(make_cut_strategy("cs-fil", ["R", "M"], has_nugget=False))
    db["buckets"].insert_one(make_bucket("bucket-6", min_weight=100, target_weight=300, max_weight=400))

    job_repo = MagicMock()
    job_repo.is_cancelled.return_value = False

    with patch("config.get_settings", return_value=_mock_settings()), \
         patch("enumeration_engine.requests.get", side_effect=_mock_requests_get):
        run_enumeration(
            db=db,
            job_id="job-5",
            run_id="run-5",
            job_repo=job_repo,
            max_combination_size=2,
            batch_size=1000,
        )

    metric = db["mix_metrics"].find_one({"bucketId": "bucket-6"})
    assert metric is not None
    assert metric["totalProductProducedGrams"] == sum(item["totalWeightInPlan"] for item in metric["unitPlan"])
    assert all("pctOfTotal" in item for item in metric["unitPlan"])
    assert round(sum(item["pctOfTotal"] for item in metric["unitPlan"]), 2) == 100.0
