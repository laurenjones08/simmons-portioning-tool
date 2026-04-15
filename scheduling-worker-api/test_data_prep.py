from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for module_name in ["data_prep"]:
    sys.modules.pop(module_name, None)
sys.path.insert(0, str(ROOT / "scheduling-worker-api"))

from data_prep import DataPrepSources, SchedulingWorkerDataPrep  # noqa: E402


class StubPrep(SchedulingWorkerDataPrep):
    def __post_init__(self):
        # Skip network client initialization for this test.
        class FakeSchedulingApi:
            def search_available_wip(self, criteria=None):
                assert criteria == {"plantName": "FSP"}
                return [{"bucketId": "bucket-1", "plantName": "FSP", "availableLbs": 999.0}]

        self.scheduling_api = FakeSchedulingApi()
        return None

    def load_sources(self):
        return DataPrepSources(
            skus=[{"tradeNumber": "SKU-1"}],
            mixes=[{"_id": "mix-1", "mfgType": "DSI888", "reqPlant": "FSP", "beltSpeed": 11.0}],
            buckets=[{"name": "bucket-1"}],
            mix_metrics=[
                {
                    "_id": "mix-1:bucket-1",
                    "mixId": "mix-1",
                    "bucketId": "bucket-1",
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
            ],
            lines=[
                {
                    "lineId": "line-1",
                    "friendlyName": "Line 1",
                    "lineType": "DSI888",
                    "plant": "FSP",
                    "hoursOfLaborAvailablePerShift": 7.5,
                    "lineThroughput": 9000.0,
                    "isActive": True,
                }
            ],
            configs=[
                {"key": "scheduling.gamma_value", "value": 0.25},
            ],
            sku_demands=[
                {"skuId": "SKU-1", "demandValue": 123.0, "demandType": "Short", "dueDate": "2026-01-05"}
            ],
            available_wip=[],
        )

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
    assert inputs["R"]["mix-1:bucket-1"] == 11.0
    assert inputs["monthly_contract"][("SKU-1", pd.Period("2026-01", freq="M"))] == 500.0

