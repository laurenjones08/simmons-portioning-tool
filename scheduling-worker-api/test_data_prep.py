from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for module_name in ["data_prep"]:
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(ROOT / "scheduling-worker-api"))

from data_prep import (  # noqa: E402
    DataPrepSources,
    SchedulingWorkerDataPrep,
    calculate_bucket_mean,
    extract_short_term_demand_records,
    load_short_term_demand_from_table,
)


class StubPrep(SchedulingWorkerDataPrep):
    def __post_init__(self):
        # Skip network client initialization for this test.
        class FakeSchedulingApi:
            def search_available_wip(self, criteria=None):
                assert criteria == {"plantName": "FSP"}
                return [{"bucketId": "bucket-1", "plantName": "FSP", "availableLbs": 999.0}]

        class FakeConfigApi:
            def list_lines(self):
                return [
                    {
                        "lineId": "line-1",
                        "friendlyName": "Line 1",
                        "lineType": "DSI888",
                        "plant": "FSP",
                        "hoursOfLaborAvailablePerShift": 7.5,
                        "unitsAvailable": 2,
                        "lineThroughput": 9000.0,
                        "isActive": True,
                    }
                ]

            def list_configs(self):
                return [
                    {"key": "scheduling.gamma_value", "value": 0.25},
                    {"key": "enumeration.upgradeDistributionMu", "value": 100.0},
                    {"key": "enumeration.upgradeDistributionSigma", "value": 1.0},
                ]

        class FakeEnumerationApi:
            def list_skus(self):
                return [{"tradeNumber": "SKU-1", "birdSize": "SB"}]

            def list_mixes(self):
                return [{"_id": "mix-1", "mfgType": "DSI888", "reqPlant": "FSP", "reqBirdSize": "SB", "beltSpeed": 11.0}]

            def list_buckets(self):
                return [{"_id": "bucket-1", "minWeight": 95.0, "targetWeight": 100.0, "maxWeight": 105.0}]

            def list_cut_strategies(self):
                return []

            def list_mix_metrics(self, sku_trade_numbers=None):
                return [
                    {
                        "_id": "mix-1:bucket-1",
                        "mixId": "mix-1",
                        "bucketId": "bucket-1",
                        "upgradePercentage": 50.0,
                        "value": 4.2,
                        "unitPlan": [
                            {
                                "sku": "SKU-1",
                                "pctOfTotal": 100.0,
                                "totalWeightInPlan": 1.0,
                            }
                        ],
                        "skuKeys": ["SKU-1"],
                    }
                ]

        self.scheduling_api = FakeSchedulingApi()
        self.config_api = FakeConfigApi()
        self.enumeration_api = FakeEnumerationApi()
        return None

    def _build_monthly_contract(self, sku_ids, months):
        return {(sku_ids[0], months[0]): 500.0}


def test_prepare_builds_api_backed_inputs():
    prep = StubPrep(use_demo_fallbacks=True)
    inputs = prep.prepare(
        job={
            "plantId": "FSP",
            "skuIds": ["SKU-1"],
            "planStartDate": "2026-01-05",
            "horizonDays": 12,
        }
    )

    assert inputs["P"] == ["SKU-1"]
    assert inputs["K"] == ["mix-1:bucket-1"]
    assert inputs["L"] == ["line-1"]
    assert inputs["B"] == ["bucket-1"]
    assert inputs["gamma"] == 0.25

    first_day = pd.Timestamp("2026-01-05")
    assert inputs["D_short"][("SKU-1", first_day)] == 123.0
    assert inputs["D_week1"][("SKU-1", first_day)] == 123.0
    assert inputs["WIP"][("bucket-1", first_day)] == 999.0
    assert inputs["bucket_of_k"]["mix-1:bucket-1"] == "bucket-1"
    assert inputs["line_of_k"]["mix-1:bucket-1"] == "line-1"
    assert inputs["base_wip_by_bucket"]["bucket-1"] == 999.0
    assert inputs["base_hours_by_line"]["line-1"] == 7.5
    assert inputs["line_throughput"]["line-1"] == 9000.0
    assert inputs["Y"][("SKU-1", "mix-1:bucket-1")] == 1.0
    assert inputs["V"]["mix-1:bucket-1"] == 4.2
    assert round(inputs["R"]["mix-1:bucket-1"], 6) == round(11.0 * 60.0 * (100.0 / 453.6) * 0.5 * 2.0, 6)
    assert inputs["monthly_contract"][("SKU-1", pd.Period("2026-01", freq="M"))] == 500.0
    assert inputs["big_allowed"]["SKU-1"] == 0
    assert inputs["small_allowed"]["SKU-1"] == 1
    assert inputs["bird_type"]["SKU-1"] == "small"


def test_prepare_removes_zero_monthly_contract_skus_before_metric_selection():
    class ContractPrunePrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            class FakeSchedulingApi:
                def search_available_wip(self, criteria=None):
                    assert criteria == {"plantName": "FSP"}
                    return [{"bucketId": "bucket-1", "plantName": "FSP", "availableLbs": 999.0}]

            class FakeConfigApi:
                def list_lines(self):
                    return [
                        {
                            "lineId": "line-1",
                            "friendlyName": "Line 1",
                            "lineType": "DSI888",
                            "plant": "FSP",
                            "hoursOfLaborAvailablePerShift": 7.5,
                            "unitsAvailable": 2,
                            "lineThroughput": 9000.0,
                            "isActive": True,
                        }
                    ]

                def list_configs(self):
                    return [
                        {"key": "scheduling.gamma_value", "value": 0.25},
                        {"key": "enumeration.upgradeDistributionMu", "value": 100.0},
                        {"key": "enumeration.upgradeDistributionSigma", "value": 1.0},
                    ]

            class FakeEnumerationApi:
                def list_mixes(self):
                    return [{"_id": "mix-1", "mfgType": "DSI888", "reqPlant": "FSP", "reqBirdSize": "SB", "beltSpeed": 11.0}]

                def list_buckets(self):
                    return [{"_id": "bucket-1", "minWeight": 95.0, "targetWeight": 100.0, "maxWeight": 105.0}]

                def list_mix_metrics(self, sku_trade_numbers=None):
                    assert sku_trade_numbers == ["SKU-1"]
                    return [
                        {
                            "_id": "mix-1:bucket-1",
                            "mixId": "mix-1",
                            "bucketId": "bucket-1",
                            "upgradePercentage": 50.0,
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        },
                        {
                            "_id": "mix-1:bucket-2",
                            "mixId": "mix-1",
                            "bucketId": "bucket-1",
                            "upgradePercentage": 50.0,
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-2", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-2"],
                        },
                    ]

            self.scheduling_api = FakeSchedulingApi()
            self.config_api = FakeConfigApi()
            self.enumeration_api = FakeEnumerationApi()
            return None

        def _build_monthly_contract(self, sku_ids, months):
            return {
                ("SKU-1", months[0]): 500.0,
                ("SKU-2", months[0]): 0.0,
            }

    prep = ContractPrunePrep(use_demo_fallbacks=False)
    inputs = prep.prepare(
        job={
            "plantId": "FSP",
            "skuIds": ["SKU-1", "SKU-2"],
            "planStartDate": "2026-01-05",
            "horizonDays": 12,
        }
    )

    assert inputs["P"] == ["SKU-1"]
    assert inputs["K"] == ["mix-1:bucket-1"]
    assert list(inputs["monthly_contract"].keys()) == [("SKU-1", pd.Period("2026-01", freq="M"))]
    assert ("SKU-2", "mix-1:bucket-2") not in inputs["Y"]


def test_build_rate_map_uses_bucket_mean_upgrade_ratio_and_line_units():
    class RatePrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            return None

    prep = RatePrep(use_demo_fallbacks=False)
    mixes = [{"_id": "mix-1", "beltSpeed": 11.0}]
    metrics = [{"_id": "mix-1:bucket-1", "mixId": "mix-1", "bucketId": "bucket-1", "upgradePercentage": 50.0}]
    buckets = [{"_id": "bucket-1", "minWeight": 95.0, "targetWeight": 100.0, "maxWeight": 105.0}]
    lines = [{"lineId": "line-1", "unitsAvailable": 2}]
    configs = [
        {"key": "enumeration.upgradeDistributionMu", "value": 100.0},
        {"key": "enumeration.upgradeDistributionSigma", "value": 1.0},
    ]
    line_of_k = {"mix-1:bucket-1": "line-1"}

    rates = prep._build_rate_map(mixes, metrics, buckets, lines, configs, line_of_k)

    expected_bucket_mean = calculate_bucket_mean(95.0, 105.0, 100.0, 1.0)
    assert rates == {
        "mix-1:bucket-1": 11.0 * 60.0 * (expected_bucket_mean / 453.6) * 0.5 * 2.0,
    }


def test_filter_runnable_metrics_uses_metric_ids_not_mix_ids():
    class FilterPrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            return None

    prep = FilterPrep(use_demo_fallbacks=False)
    metrics = [
        {"_id": "mix-1:bucket-1", "mixId": "mix-1", "bucketId": "bucket-1"},
        {"_id": "mix-1:bucket-2", "mixId": "mix-1", "bucketId": "bucket-2"},
    ]

    selected = prep._filter_runnable_metrics(metrics, {"mix-1:bucket-1": "line-1"})

    assert [metric["_id"] for metric in selected] == ["mix-1:bucket-1"]


def test_filter_metrics_with_valid_target_buckets_drops_invalid_bucket_rows():
    class FilterPrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            return None

    prep = FilterPrep(use_demo_fallbacks=False)
    metrics = [
        {"_id": "mix-1:bucket-good", "mixId": "mix-1", "bucketId": "bucket-good"},
        {"_id": "mix-1:bucket-bad", "mixId": "mix-1", "bucketId": "bucket-bad"},
    ]
    buckets = [
        {"_id": "bucket-good", "targetWeight": 100.0},
        {"_id": "bucket-bad"},
    ]

    selected = prep._filter_metrics_with_valid_target_buckets(metrics, buckets)

    assert [metric["_id"] for metric in selected] == ["mix-1:bucket-good"]


def test_prepare_assigns_line_by_mix_mfg_type(monkeypatch):
    chosen_lines = []

    def fake_choice(values):
        chosen_lines.append(list(values))
        return values[0]

    monkeypatch.setattr("data_prep.random.choice", fake_choice)

    class LinePrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            class FakeSchedulingApi:
                def search_available_wip(self, criteria=None):
                    return [{"bucketId": "bucket-1", "plantName": "FSP", "availableLbs": 999.0}]

            class FakeConfigApi:
                def list_lines(self):
                    return [
                        {
                            "lineId": "line-dsi-884",
                            "friendlyName": "DSI 884 Line",
                            "lineType": "DSI884",
                            "plant": "FSP",
                            "hoursOfLaborAvailablePerShift": 7.5,
                            "lineThroughput": 9000.0,
                            "isActive": True,
                        },
                        {
                            "lineId": "line-dsi-888",
                            "friendlyName": "DSI 888 Line",
                            "lineType": "DSI888",
                            "plant": "FSP",
                            "hoursOfLaborAvailablePerShift": 7.5,
                            "lineThroughput": 8500.0,
                            "isActive": True,
                        },
                    ]

                def list_configs(self):
                    return [{"key": "scheduling.gamma_value", "value": 0.25}]

            class FakeEnumerationApi:
                def list_skus(self):
                    return [{"tradeNumber": "SKU-1", "birdSize": "SB"}]

                def list_mixes(self):
                    return [{"_id": "mix-1", "mfgType": "DSI888", "reqPlant": "FSP", "reqBirdSize": "SB", "beltSpeed": 11.0}]

                def list_buckets(self):
                    return [{"name": "bucket-1"}]

                def list_cut_strategies(self):
                    return []

                def list_mix_metrics(self, sku_trade_numbers=None):
                    return [
                        {
                            "_id": "mix-1:bucket-1",
                            "mixId": "mix-1",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        }
                    ]

            self.scheduling_api = FakeSchedulingApi()
            self.config_api = FakeConfigApi()
            self.enumeration_api = FakeEnumerationApi()
            return None

        def _build_monthly_contract(self, sku_ids, months):
            return {(sku_ids[0], months[0]): 500.0}

    prep = LinePrep(use_demo_fallbacks=True)
    inputs = prep.prepare(
        job={
            "plantId": "FSP",
            "skuIds": ["SKU-1"],
            "planStartDate": "2026-01-05",
            "horizonDays": 12,
        }
    )

    assert chosen_lines == [["line-dsi-888"]]
    assert inputs["line_of_k"]["mix-1:bucket-1"] == "line-dsi-888"


def test_prepare_uses_cut_strategy_line_type_when_mix_mfg_type_is_generic_dsi(monkeypatch):
    chosen_lines = []

    def fake_choice(values):
        chosen_lines.append(list(values))
        return values[0]

    monkeypatch.setattr("data_prep.random.choice", fake_choice)

    class StrategyLinePrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            class FakeSchedulingApi:
                def search_available_wip(self, criteria=None):
                    return [{"bucketId": "bucket-1", "plantName": "VBS", "availableLbs": 999.0}]

            class FakeConfigApi:
                def list_lines(self):
                    return [
                        {
                            "lineId": "line-dsi-884",
                            "friendlyName": "DSI 884 Line",
                            "lineType": "DSI884",
                            "plant": "VBS",
                            "hoursOfLaborAvailablePerShift": 7.5,
                            "lineThroughput": 9000.0,
                            "isActive": True,
                        },
                        {
                            "lineId": "line-dsi-888",
                            "friendlyName": "DSI 888 Line",
                            "lineType": "DSI888",
                            "plant": "VBS",
                            "hoursOfLaborAvailablePerShift": 7.5,
                            "lineThroughput": 8500.0,
                            "isActive": True,
                        },
                    ]

                def list_configs(self):
                    return [{"key": "scheduling.gamma_value", "value": 0.25}]

            class FakeEnumerationApi:
                def list_skus(self):
                    return [{"tradeNumber": "SKU-1", "birdSize": "SB"}]

                def list_mixes(self):
                    return [
                        {
                            "_id": "mix-1",
                            "mfgType": "DSI",
                            "cutStrategyID": "strategy-884",
                            "reqPlant": "VBS",
                            "reqBirdSize": "SB",
                            "beltSpeed": 11.0,
                        }
                    ]

                def list_buckets(self):
                    return [{"name": "bucket-1"}]

                def list_cut_strategies(self):
                    return [
                        {
                            "_id": "strategy-884",
                            "lineType": "DSI884",
                        }
                    ]

                def list_mix_metrics(self, sku_trade_numbers=None):
                    return [
                        {
                            "_id": "mix-1:bucket-1",
                            "mixId": "mix-1",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        }
                    ]

            self.scheduling_api = FakeSchedulingApi()
            self.config_api = FakeConfigApi()
            self.enumeration_api = FakeEnumerationApi()
            return None

        def _build_monthly_contract(self, sku_ids, months):
            return {(sku_ids[0], months[0]): 500.0}

    prep = StrategyLinePrep(use_demo_fallbacks=True)
    inputs = prep.prepare(
        job={
            "plantId": "VBS",
            "skuIds": ["SKU-1"],
            "planStartDate": "2026-01-05",
            "horizonDays": 12,
        }
    )

    assert chosen_lines == [["line-dsi-884"]]
    assert inputs["line_of_k"]["mix-1:bucket-1"] == "line-dsi-884"


def test_prepare_prefers_line_permitted_cut_strategy_ids(monkeypatch):
    chosen_lines = []

    def fake_choice(values):
        chosen_lines.append(list(values))
        return values[0]

    monkeypatch.setattr("data_prep.random.choice", fake_choice)

    class PermittedStrategyPrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            class FakeSchedulingApi:
                def search_available_wip(self, criteria=None):
                    return [{"bucketId": "bucket-1", "plantName": "VBS", "availableLbs": 999.0}]

            class FakeConfigApi:
                def list_lines(self):
                    return [
                        {
                            "lineId": "line-exact-strategy",
                            "friendlyName": "Strategy Line",
                            "lineType": "DSI884",
                            "plant": "VBS",
                            "hoursOfLaborAvailablePerShift": 7.5,
                            "lineThroughput": 9000.0,
                            "permittedCutStrategyIds": ["strategy-884"],
                            "isActive": True,
                        },
                        {
                            "lineId": "line-other",
                            "friendlyName": "Other Line",
                            "lineType": "DSI884",
                            "plant": "VBS",
                            "hoursOfLaborAvailablePerShift": 7.5,
                            "lineThroughput": 8500.0,
                            "permittedCutStrategyIds": ["strategy-other"],
                            "isActive": True,
                        },
                    ]

                def list_configs(self):
                    return [{"key": "scheduling.gamma_value", "value": 0.25}]

            class FakeEnumerationApi:
                def list_skus(self):
                    return [{"tradeNumber": "SKU-1", "birdSize": "SB"}]

                def list_mixes(self):
                    return [
                        {
                            "_id": "mix-1",
                            "mfgType": "DSI",
                            "cutStrategyID": "strategy-884",
                            "reqPlant": "VBS",
                            "reqBirdSize": "SB",
                            "beltSpeed": 11.0,
                        }
                    ]

                def list_buckets(self):
                    return [{"name": "bucket-1"}]

                def list_cut_strategies(self):
                    return [{"_id": "strategy-884", "lineType": "DSI884"}]

                def list_mix_metrics(self, sku_trade_numbers=None):
                    return [
                        {
                            "_id": "mix-1:bucket-1",
                            "mixId": "mix-1",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        }
                    ]

            self.scheduling_api = FakeSchedulingApi()
            self.config_api = FakeConfigApi()
            self.enumeration_api = FakeEnumerationApi()
            return None

        def _build_monthly_contract(self, sku_ids, months):
            return {(sku_ids[0], months[0]): 500.0}

    prep = PermittedStrategyPrep(use_demo_fallbacks=True)
    inputs = prep.prepare(
        job={
            "plantId": "VBS",
            "skuIds": ["SKU-1"],
            "planStartDate": "2026-01-05",
            "horizonDays": 12,
        }
    )

    assert chosen_lines == [["line-exact-strategy"]]
    assert inputs["line_of_k"]["mix-1:bucket-1"] == "line-exact-strategy"


def test_load_short_term_demand_from_table_uses_row_format(tmp_path):
    short_term_file = tmp_path / "short_term_demand.csv"
    pd.DataFrame(
        [
            {"sku": "SKU-1", "demand": 10, "type": "Short", "dueDate": "2026-01-05"},
            {"sku": "SKU-1", "demand": 5, "type": "short", "dueDate": "2026-01-05"},
            {"sku": "SKU-1", "demand": 99, "type": "Long", "dueDate": "2026-01-05"},
            {"sku": "SKU-2", "demand": 12, "type": "Short", "dueDate": "2026-01-12"},
            {"sku": "SKU-3", "demand": 8, "type": "Short", "dueDate": "2026-01-06"},
        ]
    ).to_csv(short_term_file, index=False)

    week1_dates = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")]
    demand = load_short_term_demand_from_table(short_term_file, ["SKU-1", "SKU-3"], week1_dates)

    assert demand[("SKU-1", pd.Timestamp("2026-01-05"))] == 15.0
    assert demand[("SKU-3", pd.Timestamp("2026-01-06"))] == 8.0
    assert ("SKU-2", pd.Timestamp("2026-01-12")) not in demand


def test_extract_short_term_demand_records_returns_api_payload_shape(tmp_path):
    short_term_file = tmp_path / "short_term_demand.csv"
    pd.DataFrame(
        [
            {"sku": "SKU-1", "demand": 10, "type": "Short", "dueDate": "2026-01-05"},
            {"sku": "SKU-1", "demand": 5, "type": "short", "dueDate": "2026-01-06"},
            {"sku": "SKU-1", "demand": 99, "type": "Long", "dueDate": "2026-01-05"},
            {"sku": "SKU-2", "demand": 12, "type": "Short", "dueDate": "2026-01-12"},
        ]
    ).to_csv(short_term_file, index=False)

    records = extract_short_term_demand_records(short_term_file, ["SKU-1"], [pd.Timestamp("2026-01-05")])

    assert records == [
        {
            "skuId": "SKU-1",
            "demandValue": 10.0,
            "demandType": "Short",
            "dueDate": "2026-01-05",
        }
    ]


def test_prepare_combines_bird_requirements_across_mixes():
    class BirdPrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            class FakeSchedulingApi:
                def search_available_wip(self, criteria=None):
                    return [{"bucketId": "bucket-1", "plantName": "FSP", "availableLbs": 999.0}]

            class FakeConfigApi:
                def list_lines(self):
                    return [{"lineId": "line-1", "friendlyName": "Line 1", "lineType": "DSI888", "plant": "FSP", "hoursOfLaborAvailablePerShift": 7.5, "lineThroughput": 9000.0, "isActive": True}]

                def list_configs(self):
                    return [{"key": "scheduling.gamma_value", "value": 0.25}]

            class FakeEnumerationApi:
                def list_skus(self):
                    return [{"tradeNumber": "SKU-1", "birdSize": "ALL"}]

                def list_mixes(self):
                    return [
                        {"_id": "mix-sb", "mfgType": "DSI888", "reqPlant": "FSP", "reqBirdSize": "SB", "beltSpeed": 11.0},
                        {"_id": "mix-bb", "mfgType": "DSI888", "reqPlant": "FSP", "reqBirdSize": "BB", "beltSpeed": 11.0},
                    ]

                def list_buckets(self):
                    return [{"name": "bucket-1"}]

                def list_cut_strategies(self):
                    return []

                def list_mix_metrics(self, sku_trade_numbers=None):
                    return [
                        {
                            "_id": "mix-sb:bucket-1",
                            "mixId": "mix-sb",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        },
                        {
                            "_id": "mix-bb:bucket-1",
                            "mixId": "mix-bb",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        },
                    ]

            self.scheduling_api = FakeSchedulingApi()
            self.config_api = FakeConfigApi()
            self.enumeration_api = FakeEnumerationApi()
            return None

        def _build_monthly_contract(self, sku_ids, months):
            return {(sku_ids[0], months[0]): 500.0}

    prep = BirdPrep(use_demo_fallbacks=True)
    inputs = prep.prepare(
        job={
            "plantId": "FSP",
            "skuIds": ["SKU-1"],
            "planStartDate": "2026-01-05",
            "horizonDays": 12,
        }
    )

    assert inputs["big_allowed"]["SKU-1"] == 1
    assert inputs["small_allowed"]["SKU-1"] == 1
    assert inputs["bird_type"]["SKU-1"] == "all"


def test_prepare_filters_out_metrics_with_no_matching_active_line():
    class RunnableOnlyPrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            class FakeSchedulingApi:
                def search_available_wip(self, criteria=None):
                    return [{"bucketId": "bucket-1", "plantName": "VBS", "availableLbs": 999.0}]

            class FakeConfigApi:
                def list_lines(self):
                    return [
                        {
                            "lineId": "line-dsi-888",
                            "friendlyName": "DSI 888 Line",
                            "lineType": "DSI888",
                            "plant": "VBS",
                            "hoursOfLaborAvailablePerShift": 7.5,
                            "lineThroughput": 8500.0,
                            "isActive": True,
                        }
                    ]

                def list_configs(self):
                    return [{"key": "scheduling.gamma_value", "value": 0.25}]

            class FakeEnumerationApi:
                def list_skus(self):
                    return [{"tradeNumber": "SKU-1", "birdSize": "SB"}]

                def list_mixes(self):
                    return [
                        {"_id": "mix-dsi", "mfgType": "DSI888", "reqPlant": "VBS", "reqBirdSize": "SB", "beltSpeed": 11.0},
                        {"_id": "mix-db20", "mfgType": "DB20", "reqPlant": "VBS", "reqBirdSize": "SB", "beltSpeed": 11.0},
                    ]

                def list_buckets(self):
                    return [{"name": "bucket-1"}]

                def list_cut_strategies(self):
                    return []

                def list_mix_metrics(self, sku_trade_numbers=None):
                    return [
                        {
                            "_id": "mix-dsi:bucket-1",
                            "mixId": "mix-dsi",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        },
                        {
                            "_id": "mix-db20:bucket-1",
                            "mixId": "mix-db20",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        },
                    ]

            self.scheduling_api = FakeSchedulingApi()
            self.config_api = FakeConfigApi()
            self.enumeration_api = FakeEnumerationApi()
            return None

        def _build_monthly_contract(self, sku_ids, months):
            return {(sku_ids[0], months[0]): 500.0}

    prep = RunnableOnlyPrep(use_demo_fallbacks=False)
    inputs = prep.prepare(
        job={
            "plantId": "VBS",
            "skuIds": ["SKU-1"],
            "planStartDate": "2026-01-05",
            "horizonDays": 12,
        }
    )

    assert inputs["K"] == ["mix-dsi:bucket-1"]
    assert inputs["line_of_k"] == {"mix-dsi:bucket-1": "line-dsi-888"}


def test_prepare_filters_out_metrics_with_missing_mix_documents():
    class OrphanMetricPrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            class FakeSchedulingApi:
                def search_available_wip(self, criteria=None):
                    return [{"bucketId": "bucket-1", "plantName": "VBS", "availableLbs": 999.0}]

            class FakeConfigApi:
                def list_lines(self):
                    return [
                        {
                            "lineId": "VBS-DSI888",
                            "friendlyName": "Van Buren DSI888",
                            "lineType": "DSI888",
                            "plant": "VBS",
                            "hoursOfLaborAvailablePerShift": 8.0,
                            "lineThroughput": 8500.0,
                            "isActive": True,
                        }
                    ]

                def list_configs(self):
                    return [{"key": "scheduling.gamma_value", "value": 0.25}]

            class FakeEnumerationApi:
                def list_skus(self):
                    return [{"tradeNumber": "SKU-1", "birdSize": "SB"}]

                def list_mixes(self):
                    return [
                        {"_id": "mix-live", "mfgType": "DSI888", "reqPlant": "VBS", "reqBirdSize": "SB", "beltSpeed": 11.0},
                    ]

                def list_buckets(self):
                    return [{"name": "bucket-1"}]

                def list_cut_strategies(self):
                    return []

                def list_mix_metrics(self, sku_trade_numbers=None):
                    return [
                        {
                            "_id": "mix-live:bucket-1",
                            "mixId": "mix-live",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        },
                        {
                            "_id": "mix-missing:bucket-1",
                            "mixId": "mix-missing",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        },
                    ]

            self.scheduling_api = FakeSchedulingApi()
            self.config_api = FakeConfigApi()
            self.enumeration_api = FakeEnumerationApi()
            return None

        def _build_monthly_contract(self, sku_ids, months):
            return {(sku_ids[0], months[0]): 500.0}

    prep = OrphanMetricPrep(use_demo_fallbacks=False)
    inputs = prep.prepare(
        job={
            "plantId": "VBS",
            "skuIds": ["SKU-1"],
            "planStartDate": "2026-01-05",
            "horizonDays": 12,
        }
    )

    assert inputs["K"] == ["mix-live:bucket-1"]
    assert inputs["line_of_k"] == {"mix-live:bucket-1": "VBS-DSI888"}


def test_prepare_maps_mix_level_line_assignments_back_to_metric_ids():
    class SharedMixPrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            class FakeSchedulingApi:
                def search_available_wip(self, criteria=None):
                    return [{"bucketId": "bucket-1", "plantName": "VBS", "availableLbs": 999.0}]

            class FakeConfigApi:
                def list_lines(self):
                    return [
                        {
                            "lineId": "line-dsi-888",
                            "friendlyName": "DSI 888 Line",
                            "lineType": "DSI888",
                            "plant": "VBS",
                            "hoursOfLaborAvailablePerShift": 7.5,
                            "lineThroughput": 8500.0,
                            "isActive": True,
                        }
                    ]

                def list_configs(self):
                    return [{"key": "scheduling.gamma_value", "value": 0.25}]

            class FakeEnumerationApi:
                def list_skus(self):
                    return [{"tradeNumber": "SKU-1", "birdSize": "SB"}]

                def list_mixes(self):
                    return [
                        {"_id": "mix-1", "mfgType": "DSI888", "reqPlant": "VBS", "reqBirdSize": "SB", "beltSpeed": 11.0},
                    ]

                def list_buckets(self):
                    return [{"name": "bucket-1"}, {"name": "bucket-2"}]

                def list_cut_strategies(self):
                    return []

                def list_mix_metrics(self, sku_trade_numbers=None):
                    return [
                        {
                            "_id": "mix-1:bucket-1",
                            "mixId": "mix-1",
                            "bucketId": "bucket-1",
                            "value": 4.2,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        },
                        {
                            "_id": "mix-1:bucket-2",
                            "mixId": "mix-1",
                            "bucketId": "bucket-2",
                            "value": 4.4,
                            "unitPlan": [{"sku": "SKU-1", "pctOfTotal": 100.0, "totalWeightInPlan": 1.0}],
                            "skuKeys": ["SKU-1"],
                        },
                    ]

            self.scheduling_api = FakeSchedulingApi()
            self.config_api = FakeConfigApi()
            self.enumeration_api = FakeEnumerationApi()
            return None

        def _build_monthly_contract(self, sku_ids, months):
            return {(sku_ids[0], months[0]): 500.0}

    prep = SharedMixPrep(use_demo_fallbacks=False)
    inputs = prep.prepare(
        job={
            "plantId": "VBS",
            "skuIds": ["SKU-1"],
            "planStartDate": "2026-01-05",
            "horizonDays": 12,
        }
    )

    assert inputs["K"] == ["mix-1:bucket-1", "mix-1:bucket-2"]
    assert inputs["line_of_k"] == {
        "mix-1:bucket-1": "line-dsi-888",
        "mix-1:bucket-2": "line-dsi-888",
    }


def test_select_mix_metrics_requires_mix_skus_to_be_subset_of_job_skus():
    class SubsetMetricPrep(SchedulingWorkerDataPrep):
        def __post_init__(self):
            return None

    prep = SubsetMetricPrep(use_demo_fallbacks=False)
    sources = DataPrepSources(
        mix_metrics=[
            {
                "_id": "mix-single:bucket-1",
                "mixId": "mix-single",
                "bucketId": "bucket-1",
                "skuKeys": ["SKU-1"],
            },
            {
                "_id": "mix-subset:bucket-1",
                "mixId": "mix-subset",
                "bucketId": "bucket-1",
                "skuKeys": ["SKU-1", "SKU-2"],
            },
            {
                "_id": "mix-superset:bucket-1",
                "mixId": "mix-superset",
                "bucketId": "bucket-1",
                "skuKeys": ["SKU-1", "SKU-3"],
            },
            {
                "_id": "mix-unit-plan-fallback:bucket-1",
                "mixId": "mix-unit-plan-fallback",
                "bucketId": "bucket-1",
                "unitPlan": [
                    {"sku": "SKU-1", "pctOfTotal": 50.0, "totalWeightInPlan": 1.0},
                    {"sku": "SKU-2", "pctOfTotal": 50.0, "totalWeightInPlan": 1.0},
                ],
            },
        ]
    )

    selected = prep._select_mix_metrics(sources, ["SKU-1", "SKU-2"])

    assert [metric["_id"] for metric in selected] == [
        "mix-single:bucket-1",
        "mix-subset:bucket-1",
        "mix-unit-plan-fallback:bucket-1",
    ]

