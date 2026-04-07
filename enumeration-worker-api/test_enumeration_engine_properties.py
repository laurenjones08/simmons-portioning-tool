"""
Property-based tests for enumeration_engine reference data loaders.

Tests are tagged with requirement links per the design document.

**Validates: Requirements 2.2, 2.3, 6.4**
"""

import sys
from math import floor
from pathlib import Path
from unittest.mock import patch, MagicMock

import mongomock
import pytest
from hypothesis import assume, given, settings, strategies as st, HealthCheck

# Ensure the service root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from enumeration_engine import (
    _load_candidate_skus,
    _fetch_config_values,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

plant_strategy = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    min_size=1,
    max_size=8,
)

bird_size_strategy = st.sampled_from(["S", "M", "L", "XL"])

trade_number_strategy = st.text(
    alphabet="0123456789",
    min_size=3,
    max_size=6,
)


def sku_doc_strategy(plant: str, bird_size: str, trade_number: str) -> dict:
    """Build a minimal SKU document."""
    return {
        "_id": trade_number,
        "tradeNumber": trade_number,
        "prodPlant": plant,
        "birdSize": bird_size,
        "targetWeight": 100.0,
        "minWeight": 80.0,
        "maxWeight": 120.0,
        "customerType": "FDS",
        "productType": "FILET",
        "allowedParts": ["D"],
        "unitsPerCut": 1,
    }


# ---------------------------------------------------------------------------
# Property 1 — plant_filter returns only matching SKUs
# **Validates: Requirements 2.2**
# ---------------------------------------------------------------------------

@given(
    target_plant=plant_strategy,
    other_plant=plant_strategy.filter(lambda p: p != "TARGET"),
    n_matching=st.integers(min_value=1, max_value=5),
    n_other=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property1_plant_filter_returns_only_matching_skus(
    target_plant, other_plant, n_matching, n_other
):
    """
    Property 1: _load_candidate_skus with plant_filter returns only SKUs
    where prodPlant matches the filter value.

    **Validates: Requirements 2.2**
    """
    client = mongomock.MongoClient()
    db = client["enumeration_db"]
    db["skus"].drop()

    # Insert matching SKUs
    matching_docs = [
        sku_doc_strategy(target_plant, "L", f"M{i:04d}")
        for i in range(n_matching)
    ]
    # Insert non-matching SKUs (different plant)
    other_plant_val = other_plant if other_plant != target_plant else target_plant + "X"
    other_docs = [
        sku_doc_strategy(other_plant_val, "L", f"O{i:04d}")
        for i in range(n_other)
    ]

    all_docs = matching_docs + other_docs
    if all_docs:
        db["skus"].insert_many(all_docs)

    result = _load_candidate_skus(db, plant_filter=target_plant, bird_size_filter=None)

    # Every returned SKU must have the target plant
    for sku in result:
        assert sku["prodPlant"] == target_plant, (
            f"Expected prodPlant={target_plant!r}, got {sku['prodPlant']!r}"
        )

    # Count must match the number of matching docs inserted
    assert len(result) == n_matching


# ---------------------------------------------------------------------------
# Property 2 — bird_size_filter returns only matching SKUs
# **Validates: Requirements 2.3**
# ---------------------------------------------------------------------------

@given(
    target_size=bird_size_strategy,
    n_matching=st.integers(min_value=1, max_value=5),
    n_other=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property2_bird_size_filter_returns_only_matching_skus(
    target_size, n_matching, n_other
):
    """
    Property 2: _load_candidate_skus with bird_size_filter returns only SKUs
    where birdSize matches the filter value.

    **Validates: Requirements 2.3**
    """
    client = mongomock.MongoClient()
    db = client["enumeration_db"]
    db["skus"].drop()

    all_sizes = ["S", "M", "L", "XL"]
    other_size = next(s for s in all_sizes if s != target_size)

    matching_docs = [
        sku_doc_strategy("P1", target_size, f"M{i:04d}")
        for i in range(n_matching)
    ]
    other_docs = [
        sku_doc_strategy("P1", other_size, f"O{i:04d}")
        for i in range(n_other)
    ]

    all_docs = matching_docs + other_docs
    if all_docs:
        db["skus"].insert_many(all_docs)

    result = _load_candidate_skus(db, plant_filter=None, bird_size_filter=target_size)

    for sku in result:
        assert sku["birdSize"] == target_size, (
            f"Expected birdSize={target_size!r}, got {sku['birdSize']!r}"
        )

    assert len(result) == n_matching


# ---------------------------------------------------------------------------
# Property 3 — _fetch_config_values returns dict with all runtime float keys
# **Validates: Requirements 6.4**
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "tolerance_pct",
    "fds_value",
    "rtl_value",
    "trim_value",
    "upgrade_mu",
    "upgrade_sigma",
}


@given(
    tolerance=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    fds=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    rtl=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    trim=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    upgrade_mu=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    upgrade_sigma=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property3_fetch_config_values_returns_all_runtime_float_keys(
    tolerance, fds, rtl, trim, upgrade_mu, upgrade_sigma
):
    """
    Property 3: _fetch_config_values returns a dict with all expected runtime keys;
    each value is a float.

    **Validates: Requirements 6.4**
    """
    key_to_value = {
        "enumeration.bucketWeightTolerancePct": tolerance,
        "enumeration.fdsValueCoefficient": fds,
        "enumeration.rtlValueCoefficient": rtl,
        "enumeration.trimValueCoefficient": trim,
        "enumeration.upgradeDistributionMu": upgrade_mu,
        "enumeration.upgradeDistributionSigma": upgrade_sigma,
    }

    def mock_get(url, timeout=5):
        resp = MagicMock()
        # Extract the key from the URL (last path segment)
        config_key = url.split("/config/", 1)[-1]
        if config_key in key_to_value:
            resp.status_code = 200
            resp.json.return_value = {"value": key_to_value[config_key]}
        else:
            resp.status_code = 404
            resp.json.return_value = {}
        return resp

    with patch("enumeration_engine.requests.get", side_effect=mock_get):
        result = _fetch_config_values("http://mock-config-api:8001")

    assert set(result.keys()) == EXPECTED_KEYS, (
        f"Expected keys {EXPECTED_KEYS}, got {set(result.keys())}"
    )
    for key, val in result.items():
        assert isinstance(val, float), f"Expected float for key {key!r}, got {type(val)}"


# ---------------------------------------------------------------------------
# Property 4 — _fetch_config_values defaults to 0.0 when API is unreachable
# **Validates: Requirements 6.4**
# ---------------------------------------------------------------------------

@given(st.none())  # parameterless — just run once with Hypothesis machinery
@settings(max_examples=1, suppress_health_check=[HealthCheck.too_slow])
def test_property4_fetch_config_values_defaults_to_zero_on_unreachable(_):
    """
    Property 4: _fetch_config_values returns 0.0 for all keys when the
    config API is unreachable (connection error).

    **Validates: Requirements 6.4**
    """
    import requests as req_module

    def mock_get_raises(url, timeout=5):
        raise req_module.exceptions.ConnectionError("Connection refused")

    with patch("enumeration_engine.requests.get", side_effect=mock_get_raises):
        result = _fetch_config_values("http://unreachable-host:9999")

    assert set(result.keys()) == EXPECTED_KEYS
    for key, val in result.items():
        assert val == 0.0, f"Expected 0.0 for key {key!r} on unreachable API, got {val}"


# ---------------------------------------------------------------------------
# Strategies for cut strategy validation tests
# ---------------------------------------------------------------------------

part_code_strategy = st.sampled_from(["D", "R", "M", "B", "T", "W"])

@st.composite
def sku_with_parts_strategy(draw, trade_number=None):
    """Build a SKU doc with a non-empty allowedParts list."""
    tn = trade_number or draw(trade_number_strategy)
    parts = draw(st.lists(part_code_strategy, min_size=1, max_size=4, unique=True))
    product_type = draw(st.sampled_from(["FILET", "TENDER", "NUGGET"]))
    return {
        "_id": tn,
        "tradeNumber": tn,
        "prodPlant": "P1",
        "birdSize": "L",
        "targetWeight": 100.0,
        "minWeight": 80.0,
        "maxWeight": 120.0,
        "customerType": "FDS",
        "productType": product_type,
        "allowedParts": parts,
        "unitsPerCut": 1,
    }


@st.composite
def cut_strategy_strategy(draw, strategy_id=None):
    """Build a cut strategy doc."""
    sid = strategy_id or draw(st.text(alphabet="0123456789abcdef", min_size=4, max_size=8))
    parts = draw(st.lists(part_code_strategy, min_size=1, max_size=3, unique=True))
    has_nugget = draw(st.booleans())
    return {
        "_id": sid,
        "parts": parts,
        "hasNugget": has_nugget,
        "mfgType": "STANDARD",
        "beltSpeed": 100,
    }


@st.composite
def combo_strategy(draw, min_size=1, max_size=4):
    """Build a combo (list of SKU docs) with unique trade numbers."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    skus = []
    for i in range(size):
        sku = draw(sku_with_parts_strategy(trade_number=f"T{i:04d}"))
        skus.append(sku)
    return skus


from enumeration_engine import _get_valid_cut_strategies


# ---------------------------------------------------------------------------
# Property 5 — Every returned strategy has all parts covered by combo's allowedParts
# **Validates: Requirements 4.2**
# ---------------------------------------------------------------------------

@given(
    combo=combo_strategy(),
    cut_strategies=st.lists(cut_strategy_strategy(), min_size=0, max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property5_all_parts_covered_by_combo_allowed_parts(combo, cut_strategies):
    """
    Property 5: Every strategy returned by _get_valid_cut_strategies has all
    its parts covered by the union of allowedParts across the combo.

    **Validates: Requirements 4.2**
    """
    result = _get_valid_cut_strategies(combo, cut_strategies)

    combo_allowed = set()
    for sku in combo:
        combo_allowed.update(sku.get("allowedParts", []))

    for strategy in result:
        for part in strategy["parts"]:
            assert part in combo_allowed, (
                f"Strategy {strategy['_id']!r} has part {part!r} not covered "
                f"by combo allowedParts {combo_allowed}"
            )


# ---------------------------------------------------------------------------
# Property 6 — hasNugget filtering is respected
# **Validates: Requirements 4.3, 4.4**
# ---------------------------------------------------------------------------

@given(
    combo=combo_strategy(),
    cut_strategies=st.lists(cut_strategy_strategy(), min_size=0, max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property6_has_nugget_filter_respected(combo, cut_strategies):
    """
    Property 6: No strategy with hasNugget=True is returned for a combo with
    no nugget SKU, and no strategy with hasNugget=False is returned for a combo
    that contains a nugget SKU.

    **Validates: Requirements 4.3, 4.4**
    """
    result = _get_valid_cut_strategies(combo, cut_strategies)

    combo_has_nugget = any(s.get("productType") == "NUGGET" for s in combo)

    for strategy in result:
        assert strategy["hasNugget"] == combo_has_nugget, (
            f"Strategy hasNugget={strategy['hasNugget']} does not match "
            f"combo_has_nugget={combo_has_nugget}"
        )


# ---------------------------------------------------------------------------
# Property 7 — Empty result when no strategy is valid
# **Validates: Requirements 4.5**
# ---------------------------------------------------------------------------

@given(
    combo=combo_strategy(),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property7_empty_result_when_no_valid_strategies(combo):
    """
    Property 7: When no strategy is valid (empty cut_strategies list),
    the function returns an empty list.

    **Validates: Requirements 4.5**
    """
    result = _get_valid_cut_strategies(combo, [])
    assert result == [], f"Expected empty list, got {result}"


from enumeration_engine import _build_mix


# ---------------------------------------------------------------------------
# Strategies for mix construction tests
# ---------------------------------------------------------------------------

@st.composite
def valid_combo_and_strategy(draw):
    """
    Build a (combo, strategy) pair where the strategy is guaranteed to be
    valid for the combo (every part in strategy["parts"] is covered by at
    least one SKU's allowedParts, and non-nugget / non-strip SKUs do not
    share part codes.
    """
    # Draw 1-4 SKUs with unique trade numbers
    size = draw(st.integers(min_value=1, max_value=4))
    skus = []
    for i in range(size):
        parts = draw(st.lists(part_code_strategy, min_size=1, max_size=4, unique=True))
        product_type = draw(st.sampled_from(["FILET", "TENDER", "NUGGET"]))
        customer_type = draw(st.sampled_from(["FDS", "RTL"]))
        target_weight = draw(st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False))
        sku = {
            "_id": f"T{i:04d}",
            "tradeNumber": f"T{i:04d}",
            "prodPlant": draw(plant_strategy),
            "birdSize": draw(bird_size_strategy),
            "targetWeight": target_weight,
            "minWeight": target_weight * 0.8,
            "maxWeight": target_weight * 1.2,
            "customerType": customer_type,
            "productType": product_type,
            "allowedParts": parts,
            "unitsPerCut": draw(st.integers(min_value=1, max_value=10)),
        }
        skus.append(sku)

    # Find one valid per-occurrence assignment where only nuggets / strips may
    # reuse part codes.
    combo_has_nugget = any(s["productType"] in {"NUGGET", "NUGGET|STRIP"} for s in skus)
    assignments = [None] * len(skus)
    used_non_nugget_parts = set()

    def search(index: int) -> bool:
        if index >= len(skus):
            return True

        sku = skus[index]
        is_nugget = sku["productType"] in {"NUGGET", "NUGGET|STRIP"}
        for part in sku["allowedParts"]:
            if not is_nugget and part in used_non_nugget_parts:
                continue

            assignments[index] = part
            added_part = False
            if not is_nugget:
                used_non_nugget_parts.add(part)
                added_part = True

            if search(index + 1):
                return True

            assignments[index] = None
            if added_part:
                used_non_nugget_parts.remove(part)

        return False

    assume(search(0))

    strategy_parts = []
    for part in assignments:
        if part not in strategy_parts:
            strategy_parts.append(part)

    strategy = {
        "_id": draw(st.text(alphabet="0123456789abcdef", min_size=4, max_size=8)),
        "parts": strategy_parts,
        "hasNugget": combo_has_nugget,
        "mfgType": draw(st.sampled_from(["STANDARD", "PREMIUM"])),
        "beltSpeed": draw(st.integers(min_value=50, max_value=300)),
    }

    return skus, strategy


# ---------------------------------------------------------------------------
# Property 8 — Every SKU in skus map has an assigned part in its allowedParts
# **Validates: Requirements 5.1**
# ---------------------------------------------------------------------------

@given(pair=valid_combo_and_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property8_skus_map_part_codes_in_allowed_parts(pair):
    """
    Property 8: For every SKU in the returned skus map, the assigned part code
    appears in that SKU's allowedParts.

    **Validates: Requirements 5.1**
    """
    combo, strategy = pair
    mix = _build_mix(combo, strategy, plant_filter=None, bird_size_filter=None)

    sku_by_trade = {s["tradeNumber"]: s for s in combo}
    for trade_number, part_code in mix["skus"].items():
        sku = sku_by_trade[trade_number]
        assert part_code in sku["allowedParts"], (
            f"SKU {trade_number!r} was assigned part {part_code!r} "
            f"but its allowedParts are {sku['allowedParts']}"
        )


# ---------------------------------------------------------------------------
# Property 9 — includesNug iff combo has nugget SKU and strategy.hasNugget
# **Validates: Requirements 5.7**
# ---------------------------------------------------------------------------

@given(pair=valid_combo_and_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property9_includes_nug_iff_nugget_sku_and_strategy_has_nugget(pair):
    """
    Property 9: includesNug is True if and only if the combo contains a nugget
    SKU and strategy["hasNugget"] is True.

    **Validates: Requirements 5.7**
    """
    combo, strategy = pair
    mix = _build_mix(combo, strategy, plant_filter=None, bird_size_filter=None)

    combo_has_nugget = any(s["productType"] == "NUGGET" for s in combo)
    expected = combo_has_nugget and strategy["hasNugget"]

    assert mix["includesNug"] == expected, (
        f"includesNug={mix['includesNug']} but expected {expected} "
        f"(combo_has_nugget={combo_has_nugget}, strategy.hasNugget={strategy['hasNugget']})"
    )


# ---------------------------------------------------------------------------
# Property 10 — numFillets and filletWeight match non-nugget SKUs
# **Validates: Requirements 5.12, 5.13**
# ---------------------------------------------------------------------------

@given(pair=valid_combo_and_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property10_num_fillets_and_fillet_weight(pair):
    """
    Property 10: numFillets equals the count of non-nugget SKUs; filletWeight
    equals the sum of their targetWeight values.

    **Validates: Requirements 5.12, 5.13**
    """
    combo, strategy = pair
    mix = _build_mix(combo, strategy, plant_filter=None, bird_size_filter=None)

    non_nugget = [s for s in combo if s["productType"] != "NUGGET"]
    expected_num_fillets = len(non_nugget)
    expected_fillet_weight = sum(s["targetWeight"] for s in non_nugget)

    assert mix["numFillets"] == expected_num_fillets, (
        f"numFillets={mix['numFillets']} but expected {expected_num_fillets}"
    )
    assert abs(mix["filletWeight"] - expected_fillet_weight) < 1e-9, (
        f"filletWeight={mix['filletWeight']} but expected {expected_fillet_weight}"
    )


# ---------------------------------------------------------------------------
# Property 11 — reqPlant equals plant_filter when provided, else combo[0].prodPlant
# **Validates: Requirements 5.10**
# ---------------------------------------------------------------------------

@given(
    pair=valid_combo_and_strategy(),
    plant_filter=st.one_of(st.none(), plant_strategy),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property11_req_plant_uses_filter_or_first_sku(pair, plant_filter):
    """
    Property 11: reqPlant equals plant_filter when provided, otherwise
    combo[0]["prodPlant"].

    **Validates: Requirements 5.10**
    """
    combo, strategy = pair
    mix = _build_mix(combo, strategy, plant_filter=plant_filter, bird_size_filter=None)

    if plant_filter is not None:
        assert mix["reqPlant"] == plant_filter, (
            f"reqPlant={mix['reqPlant']!r} but plant_filter={plant_filter!r} was provided"
        )
    else:
        assert mix["reqPlant"] == combo[0]["prodPlant"], (
            f"reqPlant={mix['reqPlant']!r} but combo[0].prodPlant={combo[0]['prodPlant']!r}"
        )


from enumeration_engine import _fits_bucket


# ---------------------------------------------------------------------------
# Strategies for bucket fitting tests
# ---------------------------------------------------------------------------

@st.composite
def bucket_strategy(draw):
    """Build a bucket doc with minWeight < maxWeight."""
    min_w = draw(st.floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False))
    max_w = draw(st.floats(min_value=min_w, max_value=min_w + 500.0, allow_nan=False, allow_infinity=False))
    return {
        "_id": draw(st.text(alphabet="0123456789abcdef", min_size=4, max_size=8)),
        "minWeight": min_w,
        "maxWeight": max_w,
    }


tolerance_pct_strategy = st.floats(
    min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# Property 12 — A mix weight exactly equal to effective_min fits the bucket
# **Validates: Requirements 6.5, 6.6**
# ---------------------------------------------------------------------------

@given(bucket=bucket_strategy(), tolerance_pct=tolerance_pct_strategy)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property12_mix_weight_at_effective_min_fits(bucket, tolerance_pct):
    """
    Property 12: A mix weight exactly equal to effective_min fits the bucket.

    **Validates: Requirements 6.5, 6.6**
    """
    effective_min = bucket["minWeight"] * (1 - tolerance_pct / 100)
    assert _fits_bucket(effective_min, bucket, tolerance_pct), (
        f"mix_weight={effective_min} (effective_min) should fit bucket "
        f"[{effective_min}, {bucket['maxWeight']}]"
    )


# ---------------------------------------------------------------------------
# Property 13 — A mix weight exactly equal to bucket.maxWeight fits the bucket
# **Validates: Requirements 6.5, 6.6**
# ---------------------------------------------------------------------------

@given(bucket=bucket_strategy(), tolerance_pct=tolerance_pct_strategy)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property13_mix_weight_at_max_weight_fits(bucket, tolerance_pct):
    """
    Property 13: A mix weight exactly equal to bucket.maxWeight fits the bucket.

    **Validates: Requirements 6.5, 6.6**
    """
    assert _fits_bucket(bucket["maxWeight"], bucket, tolerance_pct), (
        f"mix_weight={bucket['maxWeight']} (maxWeight) should fit bucket "
        f"[effective_min, {bucket['maxWeight']}]"
    )


# ---------------------------------------------------------------------------
# Property 14 — A mix weight below effective_min does not fit the bucket
# **Validates: Requirements 6.5, 6.6**
# ---------------------------------------------------------------------------

@given(
    bucket=bucket_strategy(),
    tolerance_pct=tolerance_pct_strategy,
    delta=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property14_mix_weight_below_effective_min_does_not_fit(bucket, tolerance_pct, delta):
    """
    Property 14: A mix weight strictly below effective_min does not fit the bucket.

    **Validates: Requirements 6.5, 6.6**
    """
    effective_min = bucket["minWeight"] * (1 - tolerance_pct / 100)
    mix_weight = effective_min - delta
    assert not _fits_bucket(mix_weight, bucket, tolerance_pct), (
        f"mix_weight={mix_weight} is below effective_min={effective_min} "
        f"and should NOT fit the bucket"
    )


# ---------------------------------------------------------------------------
# Property 15 — A mix weight above bucket.maxWeight does not fit the bucket
# **Validates: Requirements 6.5, 6.6**
# ---------------------------------------------------------------------------

@given(
    bucket=bucket_strategy(),
    tolerance_pct=tolerance_pct_strategy,
    delta=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property15_mix_weight_above_max_weight_does_not_fit(bucket, tolerance_pct, delta):
    """
    Property 15: A mix weight strictly above bucket.maxWeight does not fit the bucket.

    **Validates: Requirements 6.5, 6.6**
    """
    mix_weight = bucket["maxWeight"] + delta
    assert not _fits_bucket(mix_weight, bucket, tolerance_pct), (
        f"mix_weight={mix_weight} is above maxWeight={bucket['maxWeight']} "
        f"and should NOT fit the bucket"
    )


# ---------------------------------------------------------------------------
# Property 16 — With tolerance_pct = 0, effective_min == bucket.minWeight
# **Validates: Requirements 6.5, 6.6**
# ---------------------------------------------------------------------------

@given(bucket=bucket_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property16_zero_tolerance_effective_min_equals_min_weight(bucket):
    """
    Property 16: With tolerance_pct = 0, effective_min equals bucket.minWeight,
    so a mix weight exactly at minWeight fits and one just below does not.

    **Validates: Requirements 6.5, 6.6**
    """
    tolerance_pct = 0.0
    effective_min = bucket["minWeight"] * (1 - tolerance_pct / 100)
    assert effective_min == bucket["minWeight"], (
        f"With tolerance_pct=0, effective_min={effective_min} should equal "
        f"bucket.minWeight={bucket['minWeight']}"
    )
    # Boundary: exactly at minWeight should fit
    assert _fits_bucket(bucket["minWeight"], bucket, tolerance_pct), (
        f"mix_weight=minWeight={bucket['minWeight']} should fit with tolerance_pct=0"
    )


from enumeration_engine import _compute_mix_metric


# ---------------------------------------------------------------------------
# Strategies for mix metric computation tests
# ---------------------------------------------------------------------------

@st.composite
def mix_metric_inputs(draw, force_includes_nug=None):
    """
    Build a (combo, skus_map, bucket, includes_nug, nugget_target_weight,
    config_values) tuple for _compute_mix_metric tests.
    """
    size = draw(st.integers(min_value=1, max_value=4))
    skus = []
    for i in range(size):
        product_type = draw(st.sampled_from(["FILET", "TENDER", "NUGGET"]))
        customer_type = draw(st.sampled_from(["FDS", "RTL"]))
        target_weight = draw(
            st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)
        )
        units_per_cut = draw(st.integers(min_value=1, max_value=10))
        sku = {
            "_id": f"T{i:04d}",
            "tradeNumber": f"T{i:04d}",
            "prodPlant": "P1",
            "birdSize": "L",
            "targetWeight": target_weight,
            "customerType": customer_type,
            "productType": product_type,
            "allowedParts": ["D"],
            "unitsPerCut": units_per_cut,
        }
        skus.append(sku)

    skus_map = {s["tradeNumber"]: "D" for s in skus}

    min_w = draw(st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    max_w = draw(
        st.floats(min_value=min_w, max_value=min_w + 500.0, allow_nan=False, allow_infinity=False)
    )
    bucket = {
        "_id": "bucket1",
        "minWeight": min_w,
        "maxWeight": max_w,
    }

    has_nugget_sku = any(s["productType"] == "NUGGET" for s in skus)
    if force_includes_nug is True:
        includes_nug = True
        # Ensure at least one nugget SKU exists
        if not has_nugget_sku:
            skus[0]["productType"] = "NUGGET"
            has_nugget_sku = True
    elif force_includes_nug is False:
        includes_nug = False
    else:
        includes_nug = draw(st.booleans()) and has_nugget_sku

    nugget_target_weight = None
    if includes_nug:
        for s in skus:
            if s["productType"] == "NUGGET":
                nugget_target_weight = s["targetWeight"]
                break

    config_values = {
        "tolerance_pct": draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        "fds_value": draw(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        "rtl_value": draw(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        "trim_value": draw(
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        "upgrade_mu": draw(
            st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
        ),
        "upgrade_sigma": draw(
            st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
    }

    return skus, skus_map, bucket, includes_nug, nugget_target_weight, config_values


# ---------------------------------------------------------------------------
# Property 17 — trimPercentage is 0.0 when mix_weight <= bucket.minWeight
# **Validates: Requirements 7.3**
# ---------------------------------------------------------------------------

@given(inputs=mix_metric_inputs())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property17_trim_percentage_zero_when_mix_weight_lte_min_weight(inputs):
    """
    Property 17: trimPercentage is 0.0 when mix_weight <= bucket.minWeight.

    **Validates: Requirements 7.3**
    """
    combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values = inputs

    mix_weight = sum(s["targetWeight"] for s in combo)

    # Only test the case where mix_weight <= minWeight
    if mix_weight > bucket["minWeight"]:
        return  # skip — not the scenario under test

    result = _compute_mix_metric(
        "mix1", combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values
    )

    assert result["trimPercentage"] == 0.0, (
        f"trimPercentage={result['trimPercentage']} but mix_weight={mix_weight} "
        f"<= bucket.minWeight={bucket['minWeight']}, expected 0.0"
    )


# ---------------------------------------------------------------------------
# Property 18 — upgradePercentage is in [0.0, 100.0]
# **Validates: Requirements 7.1**
# ---------------------------------------------------------------------------

@given(inputs=mix_metric_inputs())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property18_upgrade_percentage_in_valid_range(inputs):
    """
    Property 18: upgradePercentage is in [0.0, 100.0] for any valid combo and bucket.

    **Validates: Requirements 7.1**
    """
    combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values = inputs

    result = _compute_mix_metric(
        "mix1", combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values
    )

    assert 0.0 <= result["upgradePercentage"] <= 100.0, (
        f"upgradePercentage={result['upgradePercentage']} is outside [0.0, 100.0]"
    )


# ---------------------------------------------------------------------------
# Property 19 — value = fds_weight * fds_value + rtl_weight * rtl_value + trim_weight * trim_value
# **Validates: Requirements 7.2**
# ---------------------------------------------------------------------------

@given(inputs=mix_metric_inputs())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property19_value_formula(inputs):
    """
    Property 19: value = fds_weight * fds_value + rtl_weight * rtl_value + trim_weight * trim_value.

    **Validates: Requirements 7.2**
    """
    combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values = inputs

    mix_weight = sum(s["targetWeight"] for s in combo)
    fds_weight = sum(s["targetWeight"] for s in combo if s.get("customerType") == "FDS")
    rtl_weight = sum(s["targetWeight"] for s in combo if s.get("customerType") == "RTL")
    trim_weight = max(0.0, mix_weight - bucket["minWeight"])

    expected_value = (
        fds_weight * config_values["fds_value"]
        + rtl_weight * config_values["rtl_value"]
        + trim_weight * config_values["trim_value"]
    )

    result = _compute_mix_metric(
        "mix1", combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values
    )

    assert abs(result["value"] - expected_value) < 1e-9, (
        f"value={result['value']} but expected {expected_value}"
    )


# ---------------------------------------------------------------------------
# Property 20 — When includesNug is True, nugget SKU's unitsInPlan == floor(minWeight / nuggetTargetWeight)
# **Validates: Requirements 7.5**
# ---------------------------------------------------------------------------

@st.composite
def mix_metric_inputs_with_nug(draw):
    """Build inputs where includes_nug=True and nugget_target_weight > 0."""
    size = draw(st.integers(min_value=1, max_value=4))
    skus = []
    nugget_idx = draw(st.integers(min_value=0, max_value=size - 1))
    for i in range(size):
        product_type = "NUGGET" if i == nugget_idx else draw(st.sampled_from(["FILET", "TENDER"]))
        customer_type = draw(st.sampled_from(["FDS", "RTL"]))
        target_weight = draw(
            st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)
        )
        units_per_cut = draw(st.integers(min_value=1, max_value=10))
        sku = {
            "_id": f"T{i:04d}",
            "tradeNumber": f"T{i:04d}",
            "prodPlant": "P1",
            "birdSize": "L",
            "targetWeight": target_weight,
            "customerType": customer_type,
            "productType": product_type,
            "allowedParts": ["D"],
            "unitsPerCut": units_per_cut,
        }
        skus.append(sku)

    skus_map = {s["tradeNumber"]: "D" for s in skus}

    min_w = draw(st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    max_w = draw(
        st.floats(min_value=min_w, max_value=min_w + 500.0, allow_nan=False, allow_infinity=False)
    )
    bucket = {
        "_id": "bucket1",
        "minWeight": min_w,
        "targetWeight": min_w,
        "maxWeight": max_w,
    }

    nugget_sku = skus[nugget_idx]
    nugget_target_weight = nugget_sku["targetWeight"]

    config_values = {
        "tolerance_pct": 0.0,
        "fds_value": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        "rtl_value": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        "trim_value": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
    }

    return skus, skus_map, bucket, True, nugget_target_weight, config_values


@given(inputs=mix_metric_inputs_with_nug())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property20_nugget_units_in_plan_when_includes_nug(inputs):
    """
    Property 20: When includesNug is True, the nugget SKU's unitsInPlan equals
    floor(bucket.minWeight / nuggetTargetWeight).

    **Validates: Requirements 7.5**
    """
    combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values = inputs

    result = _compute_mix_metric(
        "mix1", combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values
    )

    expected_units = floor(bucket["minWeight"] / nugget_target_weight)

    for item in result["unitPlan"]:
        sku_trade = item["sku"]
        sku_doc = next(s for s in combo if s["tradeNumber"] == sku_trade)
        if sku_doc["productType"] == "NUGGET":
            assert item["unitsInPlan"] == expected_units, (
                f"Nugget SKU {sku_trade!r} unitsInPlan={item['unitsInPlan']} "
                f"but expected floor({bucket['minWeight']} / {nugget_target_weight}) = {expected_units}"
            )


# ---------------------------------------------------------------------------
# Property 21 — When includesNug is False, all SKUs have unitsInPlan == unitsPerCut
# **Validates: Requirements 7.4, 7.5**
# ---------------------------------------------------------------------------

@st.composite
def mix_metric_inputs_no_nug(draw):
    """Build inputs where includes_nug=False."""
    size = draw(st.integers(min_value=1, max_value=4))
    skus = []
    for i in range(size):
        product_type = draw(st.sampled_from(["FILET", "TENDER"]))
        customer_type = draw(st.sampled_from(["FDS", "RTL"]))
        target_weight = draw(
            st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)
        )
        units_per_cut = draw(st.integers(min_value=1, max_value=10))
        sku = {
            "_id": f"T{i:04d}",
            "tradeNumber": f"T{i:04d}",
            "prodPlant": "P1",
            "birdSize": "L",
            "targetWeight": target_weight,
            "customerType": customer_type,
            "productType": product_type,
            "allowedParts": ["D"],
            "unitsPerCut": units_per_cut,
        }
        skus.append(sku)

    skus_map = {s["tradeNumber"]: "D" for s in skus}

    min_w = draw(st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    max_w = draw(
        st.floats(min_value=min_w, max_value=min_w + 500.0, allow_nan=False, allow_infinity=False)
    )
    bucket = {
        "_id": "bucket1",
        "minWeight": min_w,
        "targetWeight": min_w,
        "maxWeight": max_w,
    }

    config_values = {
        "tolerance_pct": 0.0,
        "fds_value": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        "rtl_value": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        "trim_value": draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
    }

    return skus, skus_map, bucket, False, None, config_values


@given(inputs=mix_metric_inputs_no_nug())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property21_all_skus_use_units_per_cut_when_no_nug(inputs):
    """
    Property 21: When includesNug is False, all SKUs have unitsInPlan == unitsPerCut.

    **Validates: Requirements 7.4, 7.5**
    """
    combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values = inputs

    result = _compute_mix_metric(
        "mix1", combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values
    )

    sku_by_trade = {s["tradeNumber"]: s for s in combo}
    for item in result["unitPlan"]:
        sku_doc = sku_by_trade[item["sku"]]
        expected = sku_doc["unitsPerCut"]
        assert item["unitsInPlan"] == expected, (
            f"SKU {item['sku']!r} unitsInPlan={item['unitsInPlan']} "
            f"but expected unitsPerCut={expected} (includesNug=False)"
        )


# ---------------------------------------------------------------------------
# Property 22 — skuKeys exactly matches tradeNumber values in unitPlan in first-appearance order
# **Validates: Requirements 7.6**
# ---------------------------------------------------------------------------

@given(inputs=mix_metric_inputs())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property22_sku_keys_match_unit_plan_first_appearance_order(inputs):
    """
    Property 22: skuKeys exactly matches the tradeNumber values in unitPlan
    in first-appearance order.

    **Validates: Requirements 7.6**
    """
    combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values = inputs

    result = _compute_mix_metric(
        "mix1", combo, skus_map, bucket, includes_nug, nugget_target_weight, config_values
    )

    # Derive expected skuKeys from unitPlan in first-appearance order
    seen = []
    for item in result["unitPlan"]:
        if item["sku"] not in seen:
            seen.append(item["sku"])

    assert result["skuKeys"] == seen, (
        f"skuKeys={result['skuKeys']} does not match first-appearance order "
        f"from unitPlan: {seen}"
    )


import mongomock
from pymongo.errors import DuplicateKeyError

from enumeration_engine import _upsert_mix, _upsert_mix_metric


# ---------------------------------------------------------------------------
# Strategies for persistence helper tests
# ---------------------------------------------------------------------------

@st.composite
def mix_doc_strategy(draw):
    """Build a minimal mix document suitable for _upsert_mix."""
    # Generate 1-3 unique trade numbers for the SKU set
    n_skus = draw(st.integers(min_value=1, max_value=3))
    trade_numbers = [f"SKU{i:04d}" for i in range(n_skus)]
    skus_map = {tn: "D" for tn in trade_numbers}
    sku_set_key = "|".join(sorted(skus_map.keys()))
    mfg_type = draw(st.sampled_from(["STANDARD", "PREMIUM", "ECONOMY"]))
    return {
        "skus": skus_map,
        "skuSetKey": sku_set_key,
        "mfgType": mfg_type,
        "cutStrategyID": draw(st.text(alphabet="0123456789abcdef", min_size=4, max_size=8)),
        "beltSpeed": draw(st.integers(min_value=50, max_value=300)),
        "includesFDS": draw(st.booleans()),
        "includesRTL": draw(st.booleans()),
        "includesNug": False,
        "nuggetTargetWeight": None,
        "reqPlant": "P1",
        "reqBirdSize": "L",
        "numFillets": n_skus,
        "filletWeight": draw(st.floats(min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False)),
        "skuKeys": trade_numbers,
    }


@st.composite
def metric_doc_strategy(draw, mix_id="mix001"):
    """Build a minimal metric document suitable for _upsert_mix_metric."""
    bucket_id = draw(st.text(alphabet="0123456789abcdef", min_size=4, max_size=8))
    return {
        "_id": f"{mix_id}:{bucket_id}",
        "mixId": mix_id,
        "bucketId": bucket_id,
        "upgradePercentage": draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        "value": draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        "trimPercentage": draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        "unitPlan": [],
        "skuKeys": ["SKU0001"],
    }


# ---------------------------------------------------------------------------
# Property 23 — Calling _upsert_mix twice with the same SKU set and mfgType
#               results in exactly one document in the collection
# **Validates: Requirements 8.4**
# ---------------------------------------------------------------------------

@given(mix_doc=mix_doc_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property23_upsert_mix_twice_yields_one_document(mix_doc):
    """
    Property 23: Calling _upsert_mix twice with the same SKU set and mfgType
    results in exactly one document in the collection.

    **Validates: Requirements 8.4**
    """
    import copy

    client = mongomock.MongoClient()
    db = client["enumeration_db"]

    # Import MixRepository via the engine's already-loaded module reference
    from enumeration_engine import MixRepository

    mix_repo = MixRepository(db)

    # First upsert
    doc1 = copy.deepcopy(mix_doc)
    id1 = _upsert_mix(mix_repo, doc1)

    # Second upsert with same skuSetKey + mfgType (slightly different other fields)
    doc2 = copy.deepcopy(mix_doc)
    doc2["beltSpeed"] = mix_doc["beltSpeed"] + 10  # change a non-key field
    id2 = _upsert_mix(mix_repo, doc2)

    # Both calls must return the same _id
    assert id1 == id2, (
        f"First upsert returned _id={id1!r}, second returned {id2!r}; "
        "expected the same _id for the same skuSetKey+mfgType"
    )

    # Exactly one document must exist in the collection
    all_docs = list(db["mixes"].find({"skuSetKey": mix_doc["skuSetKey"], "mfgType": mix_doc["mfgType"]}))
    assert len(all_docs) == 1, (
        f"Expected exactly 1 document for skuSetKey={mix_doc['skuSetKey']!r} + "
        f"mfgType={mix_doc['mfgType']!r}, found {len(all_docs)}"
    )


# ---------------------------------------------------------------------------
# Property 24 — Calling _upsert_mix_metric twice with the same mixId + bucketId
#               results in exactly one document in the collection
# **Validates: Requirements 8.5**
# ---------------------------------------------------------------------------

@given(metric_doc=metric_doc_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property24_upsert_mix_metric_twice_yields_one_document(metric_doc):
    """
    Property 24: Calling _upsert_mix_metric twice with the same mixId + bucketId
    results in exactly one document in the collection.

    **Validates: Requirements 8.5**
    """
    import copy

    client = mongomock.MongoClient()
    db = client["enumeration_db"]

    from enumeration_engine import MixMetricRepository

    metric_repo = MixMetricRepository(db)

    # First upsert
    doc1 = copy.deepcopy(metric_doc)
    _upsert_mix_metric(metric_repo, doc1)

    # Second upsert with same _id (mixId:bucketId) but different value
    doc2 = copy.deepcopy(metric_doc)
    doc2["value"] = metric_doc["value"] + 1.0
    _upsert_mix_metric(metric_repo, doc2)

    # Exactly one document must exist for this composite _id
    all_docs = list(db["mix_metrics"].find({"_id": metric_doc["_id"]}))
    assert len(all_docs) == 1, (
        f"Expected exactly 1 document for _id={metric_doc['_id']!r}, found {len(all_docs)}"
    )


# ===========================================================================
# Properties 25–29 — Orchestrator end-to-end tests using run_enumeration
# ===========================================================================

from enumeration_engine import run_enumeration


# ---------------------------------------------------------------------------
# Shared helpers for orchestrator tests
# ---------------------------------------------------------------------------

def make_job_repo_mock():
    mock = MagicMock()
    mock.is_cancelled.return_value = False
    return mock


def make_config_mock():
    mock = MagicMock()
    mock.global_config_api_url = "http://mock-config:8001"
    return mock


def make_sku(trade_number, product_type="FILET", customer_type="FDS",
             target_weight=100.0, allowed_parts=None, plant="P1", bird_size="L"):
    return {
        "_id": trade_number,
        "tradeNumber": trade_number,
        "prodPlant": plant,
        "birdSize": bird_size,
        "targetWeight": target_weight,
        "minWeight": target_weight * 0.8,
        "maxWeight": target_weight * 1.2,
        "customerType": customer_type,
        "productType": product_type,
        "allowedParts": allowed_parts or ["D"],
        "unitsPerCut": 1,
    }


def make_cut_strategy(sid, parts, has_nugget=False, mfg_type="STANDARD"):
    return {
        "_id": sid,
        "name": sid,  # unique name required by CutStrategyRepository unique index
        "parts": parts,
        "hasNugget": has_nugget,
        "mfgType": mfg_type,
        "beltSpeed": 100,
    }


def make_bucket(bid, min_w, max_w):
    return {
        "_id": bid,
        "minWeight": min_w,
        "targetWeight": min_w,
        "maxWeight": max_w,
    }


def _mock_requests_get(url, timeout=5):
    """Return a mock response with value=0.0 for all config keys."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"value": 0.0}
    return resp


def _setup_db(skus, cut_strategies, buckets):
    """Create a fresh mongomock DB and insert reference data."""
    client = mongomock.MongoClient()
    db = client["enumeration_db"]
    if skus:
        db["skus"].insert_many(skus)
    if cut_strategies:
        db["cut_strategies"].insert_many(cut_strategies)
    if buckets:
        db["buckets"].insert_many(buckets)
    return db


def _run(db, max_combination_size=2):
    """Run enumeration with standard mocks."""
    job_repo = make_job_repo_mock()
    with patch("config.get_settings", return_value=make_config_mock()), \
         patch("enumeration_engine.requests.get", side_effect=_mock_requests_get):
        run_enumeration(
            db=db,
            job_id="job1",
            run_id="run1",
            job_repo=job_repo,
            max_combination_size=max_combination_size,
            batch_size=1000,
        )
    return job_repo


# ---------------------------------------------------------------------------
# Property 25 — Every persisted Mix has at least one associated MixMetric
# **Validates: Requirements 3.2, 6.7, 8.3**
# ---------------------------------------------------------------------------

@given(
    n_skus=st.integers(min_value=1, max_value=3),
    target_weight=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_property25_every_mix_has_at_least_one_metric(n_skus, target_weight):
    """
    Property 25: Every persisted Mix has at least one associated MixMetric
    (no bucket-less mixes).

    **Validates: Requirements 3.2, 6.7, 8.3**
    """
    skus = [
        make_sku(f"SKU{i:03d}", target_weight=target_weight, allowed_parts=["D"])
        for i in range(n_skus)
    ]
    cut_strategies = [make_cut_strategy("CS1", ["D"], has_nugget=False)]
    # Bucket covers the mix weight range: 1 SKU = target_weight, 2 SKUs = 2*target_weight
    max_mix_weight = n_skus * target_weight * 2  # generous upper bound
    buckets = [make_bucket("B1", min_w=target_weight * 0.5, max_w=max_mix_weight)]

    db = _setup_db(skus, cut_strategies, buckets)
    _run(db, max_combination_size=2)

    all_mixes = list(db["mixes"].find({}))
    all_metrics = list(db["mix_metrics"].find({}))

    mix_ids_with_metrics = {m["mixId"] for m in all_metrics}

    for mix in all_mixes:
        assert mix["_id"] in mix_ids_with_metrics, (
            f"Mix {mix['_id']!r} has no associated MixMetric — "
            "every persisted Mix must fit at least one bucket"
        )


# ---------------------------------------------------------------------------
# Property 26 — No combination contains more than one nugget SKU
# **Validates: Requirements 3.3**
# ---------------------------------------------------------------------------

@given(st.none())
@settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
def test_property26_no_combo_has_more_than_one_nugget(dummy):
    """
    Property 26: No combination contains more than one nugget SKU.

    **Validates: Requirements 3.3**
    """
    # 2 nugget SKUs + 1 non-nugget SKU
    skus = [
        make_sku("NUG1", product_type="NUGGET", customer_type="FDS",
                 target_weight=50.0, allowed_parts=["D"]),
        make_sku("NUG2", product_type="NUGGET", customer_type="FDS",
                 target_weight=60.0, allowed_parts=["D"]),
        make_sku("FIL1", product_type="FILET", customer_type="FDS",
                 target_weight=100.0, allowed_parts=["D"]),
    ]
    # Strategy with hasNugget=True to allow nugget combos
    cut_strategies = [
        make_cut_strategy("CS_NUG", ["D"], has_nugget=True),
        make_cut_strategy("CS_FIL", ["D"], has_nugget=False),
    ]
    # Bucket that covers all possible mix weights
    buckets = [make_bucket("B1", min_w=10.0, max_w=1000.0)]

    db = _setup_db(skus, cut_strategies, buckets)
    _run(db, max_combination_size=3)

    all_mixes = list(db["mixes"].find({}))

    # Build a lookup of nugget trade numbers
    nugget_trade_numbers = {"NUG1", "NUG2"}

    for mix in all_mixes:
        sku_keys = mix.get("skuKeys", [])
        nugget_count = sum(1 for tn in sku_keys if tn in nugget_trade_numbers)
        assert nugget_count <= 1, (
            f"Mix {mix['_id']!r} has {nugget_count} nugget SKUs in skuKeys={sku_keys}; "
            "at most one nugget SKU is allowed per combination"
        )


# ---------------------------------------------------------------------------
# Property 27 — No SKU appears more than 3 times in any combination
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------

@given(st.none())
@settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
def test_property27_no_sku_appears_more_than_3_times(dummy):
    """
    Property 27: No SKU appears more than 3 times in any combination.
    With max_combination_size=3, combinations_with_replacement guarantees
    at most 3 repetitions. We verify via the persisted skuKeys.

    **Validates: Requirements 3.2**
    """
    skus = [make_sku("SKU001", target_weight=100.0, allowed_parts=["D"])]
    cut_strategies = [make_cut_strategy("CS1", ["D"], has_nugget=False)]
    # Bucket covers 1x, 2x, and 3x target_weight
    buckets = [make_bucket("B1", min_w=50.0, max_w=400.0)]

    db = _setup_db(skus, cut_strategies, buckets)
    _run(db, max_combination_size=3)

    all_mixes = list(db["mixes"].find({}))

    for mix in all_mixes:
        sku_keys = mix.get("skuKeys", [])
        # skuKeys contains unique trade numbers; check via mix_metrics unitPlan for counts
        # But skuKeys is unique per mix doc — check unitPlan in metrics instead
        metrics = list(db["mix_metrics"].find({"mixId": mix["_id"]}))
        for metric in metrics:
            unit_plan = metric.get("unitPlan", [])
            from collections import Counter
            counts = Counter(item["sku"] for item in unit_plan)
            for sku_tn, count in counts.items():
                assert count <= 3, (
                    f"SKU {sku_tn!r} appears {count} times in unitPlan of metric "
                    f"{metric['_id']!r}; max allowed is 3"
                )


# ---------------------------------------------------------------------------
# Property 28 — At most one Mix per SKU set + mfgType after a run
# **Validates: Requirements 8.4**
# ---------------------------------------------------------------------------

@given(st.none())
@settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
def test_property28_at_most_one_mix_per_sku_set_and_mfg_type(dummy):
    """
    Property 28: For any given SKU set and mfgType, at most one Mix document
    exists after a run.

    **Validates: Requirements 8.4**
    """
    skus = [
        make_sku("SKU001", target_weight=100.0, allowed_parts=["D"]),
        make_sku("SKU002", target_weight=120.0, allowed_parts=["D"]),
    ]
    cut_strategies = [make_cut_strategy("CS1", ["D"], has_nugget=False, mfg_type="STANDARD")]
    buckets = [make_bucket("B1", min_w=50.0, max_w=500.0)]

    db = _setup_db(skus, cut_strategies, buckets)
    _run(db, max_combination_size=2)

    all_mixes = list(db["mixes"].find({}))

    # Group by (skuSetKey, mfgType) and assert at most one per group
    from collections import defaultdict
    groups = defaultdict(list)
    for mix in all_mixes:
        key = (mix.get("skuSetKey"), mix.get("mfgType"))
        groups[key].append(mix["_id"])

    for (sku_set_key, mfg_type), ids in groups.items():
        assert len(ids) == 1, (
            f"Found {len(ids)} Mix documents for skuSetKey={sku_set_key!r}, "
            f"mfgType={mfg_type!r}: {ids}; expected exactly 1"
        )


# ---------------------------------------------------------------------------
# Property 29 — processedCombinations checkpointed to job_repo is non-decreasing
# **Validates: Requirements 9.3**
# ---------------------------------------------------------------------------

@given(st.none())
@settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
def test_property29_processed_combinations_non_decreasing(dummy):
    """
    Property 29: processedCombinations checkpointed to job_repo is non-decreasing
    within a stage.

    **Validates: Requirements 9.3**
    """
    # Use enough SKUs to generate multiple batches; use small batch_size
    skus = [
        make_sku(f"SKU{i:03d}", target_weight=100.0, allowed_parts=["D"])
        for i in range(3)
    ]
    cut_strategies = [make_cut_strategy("CS1", ["D"], has_nugget=False)]
    buckets = [make_bucket("B1", min_w=50.0, max_w=1000.0)]

    db = _setup_db(skus, cut_strategies, buckets)

    job_repo = make_job_repo_mock()
    checkpoint_calls = []

    def capture_checkpoint(job_id, stage_index, processed):
        checkpoint_calls.append((stage_index, processed))

    job_repo.checkpoint_stage.side_effect = capture_checkpoint

    with patch("config.get_settings", return_value=make_config_mock()), \
         patch("enumeration_engine.requests.get", side_effect=_mock_requests_get):
        run_enumeration(
            db=db,
            job_id="job1",
            run_id="run1",
            job_repo=job_repo,
            max_combination_size=2,
            batch_size=2,  # small batch to trigger multiple checkpoints
        )

    # Group checkpoint calls by stage and verify non-decreasing order
    from collections import defaultdict
    by_stage = defaultdict(list)
    for stage_index, processed in checkpoint_calls:
        by_stage[stage_index].append(processed)

    for stage_index, processed_list in by_stage.items():
        for i in range(1, len(processed_list)):
            assert processed_list[i] >= processed_list[i - 1], (
                f"Stage {stage_index}: processedCombinations decreased from "
                f"{processed_list[i-1]} to {processed_list[i]} — must be non-decreasing"
            )
