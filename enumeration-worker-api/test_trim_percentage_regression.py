from enumeration_engine import _compute_mix_metric


def test_trim_percentage_is_remaining_percentage():
    combo = [
        {
            "_id": "T001",
            "tradeNumber": "T001",
            "prodPlant": "P1",
            "birdSize": "L",
            "targetWeight": 100.0,
            "minWeight": 80.0,
            "maxWeight": 120.0,
            "customerType": "FDS",
            "productType": "FILET",
            "allowedParts": ["D"],
            "unitsPerCut": 1,
        }
    ]
    skus_map = {"T001": "D"}
    bucket = {
        "_id": "B1",
        "minWeight": 50.0,
        "targetWeight": 200.0,
        "maxWeight": 250.0,
    }
    config_values = {
        "tolerance_pct": 0.0,
        "fds_value": 0.0,
        "rtl_value": 0.0,
        "trim_value": 0.0,
        "upgrade_mu": 0.0,
        "upgrade_sigma": 0.0,
    }

    result = _compute_mix_metric("mix1", combo, skus_map, bucket, False, None, config_values)

    assert result["upgradePercentage"] == 50.0
    assert result["trimPercentage"] == 50.0
    assert result["trimPercentage"] >= 0.0
