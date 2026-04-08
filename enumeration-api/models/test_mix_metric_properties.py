from .mix_metric import MixMetric


def _valid_payload():
    return {
        "mixId": "mix-1",
        "bucketId": "bucket-1",
        "upgradePercentage": 1.5,
        "value": 2.0,
        "trimPercentage": 0.5,
        "unitPlan": [
            {
                "sku": "45065",
                "partCode": "D",
                "unitsInPlan": 1,
                "totalWeightInPlan": 109.0,
            },
            {
                "sku": "45065",
                "partCode": "R",
                "unitsInPlan": 1,
                "totalWeightInPlan": 109.0,
            },
            {
                "sku": "45065",
                "partCode": "M",
                "unitsInPlan": 1,
                "totalWeightInPlan": 109.0,
            },
        ],
        "skuKeys": ["45065"],
    }


def test_mix_metric_populates_pct_of_total_and_total_product_grams():
    metric = MixMetric(**_valid_payload())

    assert metric.total_product_produced_grams == 327.0
    assert [item.pct_of_total for item in metric.unit_plan] == [33.33, 33.33, 33.33]

    dumped = metric.model_dump(by_alias=True)
    assert dumped["totalProductProducedGrams"] == 327.0
    assert dumped["unitPlan"][0]["pctOfTotal"] == 33.33


def test_mix_metric_backfills_legacy_payload_without_derived_fields():
    payload = _valid_payload()
    payload["unitPlan"][0]["pctOfTotal"] = 12.5
    payload["totalProductProducedGrams"] = None

    metric = MixMetric(**payload)

    assert metric.unit_plan[0].pct_of_total == 33.33
    assert metric.total_product_produced_grams == 327.0
