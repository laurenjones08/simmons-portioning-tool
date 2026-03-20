"""
Property-based tests for pure validation functions extracted from page modules.
"""

import sys
import os
import importlib.util

from hypothesis import given, settings, assume
import hypothesis.strategies as st

# Load validate_bucket_weights from 1_Buckets.py (leading digit prevents normal import)
_buckets_path = os.path.join(os.path.dirname(__file__), "..", "pages", "1_Buckets.py")
_spec = importlib.util.spec_from_file_location("buckets_page", _buckets_path)
_buckets_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_buckets_module)
validate_bucket_weights = _buckets_module.validate_bucket_weights

# Load validate_sku_weights from 2_SKUs.py (leading digit prevents normal import)
_skus_path = os.path.join(os.path.dirname(__file__), "..", "pages", "2_SKUs.py")
_skus_spec = importlib.util.spec_from_file_location("skus_page", _skus_path)
_skus_module = importlib.util.module_from_spec(_skus_spec)
_skus_spec.loader.exec_module(_skus_module)
validate_sku_weights = _skus_module.validate_sku_weights

# Load validate_parts_unique from 3_Cut_Strategies.py (leading digit prevents normal import)
_cut_strategies_path = os.path.join(os.path.dirname(__file__), "..", "pages", "3_Cut_Strategies.py")
_cut_strategies_spec = importlib.util.spec_from_file_location("cut_strategies_page", _cut_strategies_path)
_cut_strategies_module = importlib.util.module_from_spec(_cut_strategies_spec)
_cut_strategies_spec.loader.exec_module(_cut_strategies_module)
validate_parts_unique = _cut_strategies_module.validate_parts_unique


# Feature: streamlit-settings-page, Property 2: Bucket weight validation rejects invalid ranges
@settings(max_examples=100)
@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_bucket_weight_validation_property(min_w: float, max_w: float) -> None:
    """Property 4: Bucket Tuple Ordering.

    Validates: Requirements 3.6

    - When min_w >= max_w, validate_bucket_weights must return a non-None, non-empty error string.
    - When min_w < max_w, validate_bucket_weights must return None.
    """
    result = validate_bucket_weights(min_w, max_w)

    if min_w >= max_w:
        assert result is not None, (
            f"Expected an error string for min_w={min_w} >= max_w={max_w}, got None"
        )
        assert isinstance(result, str) and len(result) > 0, (
            f"Expected a non-empty error string, got {result!r}"
        )
    else:
        assert result is None, (
            f"Expected None for valid min_w={min_w} < max_w={max_w}, got {result!r}"
        )


# Feature: streamlit-settings-page, Property 3: SKU weight validation enforces all three constraints
@settings(max_examples=100)
@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
)
def test_sku_weight_validation_property(min_w: float, target_w: float, max_w: float) -> None:
    """Property 3: SKU weight validation enforces all three constraints.

    Validates: Requirements 4.7

    - When min_w < max_w AND min_w <= target_w <= max_w → validate_sku_weights returns None.
    - When min_w >= max_w OR target_w < min_w OR target_w > max_w → returns a non-None, non-empty string.
    """
    result = validate_sku_weights(min_w, target_w, max_w)

    is_valid = (min_w < max_w) and (min_w <= target_w <= max_w)

    if is_valid:
        assert result is None, (
            f"Expected None for valid triple min_w={min_w}, target_w={target_w}, max_w={max_w}, got {result!r}"
        )
    else:
        assert result is not None, (
            f"Expected an error string for invalid triple min_w={min_w}, target_w={target_w}, max_w={max_w}, got None"
        )
        assert isinstance(result, str) and len(result) > 0, (
            f"Expected a non-empty error string, got {result!r}"
        )


# Feature: streamlit-settings-page, Property 5: Cut strategy parts duplicate validation
_PART_CODES = ["D", "R", "M", "T", "V", "K", "S", "U", "C", "J", "W", "G"]


@settings(max_examples=100)
@given(st.lists(st.sampled_from(_PART_CODES)))
def test_cut_strategy_parts_unique_no_duplicates(parts: list) -> None:
    """Property 5: Parts list without duplicates always passes validation.

    Validates: Requirements 5.7

    When all elements in parts are unique, validate_parts_unique must return None.
    """
    assume(len(parts) == len(set(parts)))

    result = validate_parts_unique(parts)

    assert result is None, (
        f"Expected None for unique parts list {parts!r}, got {result!r}"
    )


@settings(max_examples=100)
@given(st.lists(st.sampled_from(_PART_CODES), min_size=1))
def test_cut_strategy_parts_with_duplicates(parts: list) -> None:
    """Property 5: Parts list with duplicates always fails validation.

    Validates: Requirements 5.7

    When parts list contains at least one duplicate, validate_parts_unique must return
    a non-None, non-empty error string.
    """
    # Introduce a duplicate by appending the first element
    parts_with_dup = parts + [parts[0]]

    result = validate_parts_unique(parts_with_dup)

    assert result is not None, (
        f"Expected an error string for parts list with duplicate {parts_with_dup!r}, got None"
    )
    assert isinstance(result, str) and len(result) > 0, (
        f"Expected a non-empty error string, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Load validation helpers from 5_Mix_Generation.py
# ---------------------------------------------------------------------------

_mix_gen_path = os.path.join(os.path.dirname(__file__), "..", "pages", "5_Mix_Generation.py")
_mix_gen_spec = importlib.util.spec_from_file_location("mix_generation_page", _mix_gen_path)
_mix_gen_module = importlib.util.module_from_spec(_mix_gen_spec)
_mix_gen_spec.loader.exec_module(_mix_gen_module)
validate_max_combination_size = _mix_gen_module.validate_max_combination_size
validate_batch_size = _mix_gen_module.validate_batch_size
warn_if_no_filters = _mix_gen_module.warn_if_no_filters
cancel_button_visible = _mix_gen_module.cancel_button_visible


# Feature: streamlit-settings-page, Property 7: Job maxCombinationSize validation
@settings(max_examples=100)
@given(st.integers())
def test_max_combination_size_validation_property(n: int) -> None:
    """Property 7: validate_max_combination_size accepts iff 1 <= n <= 4.

    Validates: Requirements 7.5
    """
    result = validate_max_combination_size(n)
    if 1 <= n <= 4:
        assert result is None, f"Expected None for valid n={n}, got {result!r}"
    else:
        assert result is not None and isinstance(result, str) and len(result) > 0, (
            f"Expected error string for invalid n={n}, got {result!r}"
        )


# Feature: streamlit-settings-page, Property 8: Job batchSize validation
@settings(max_examples=100)
@given(st.integers())
def test_batch_size_validation_property(n: int) -> None:
    """Property 8: validate_batch_size accepts iff n >= 1.

    Validates: Requirements 7.6
    """
    result = validate_batch_size(n)
    if n >= 1:
        assert result is None, f"Expected None for valid n={n}, got {result!r}"
    else:
        assert result is not None and isinstance(result, str) and len(result) > 0, (
            f"Expected error string for invalid n={n}, got {result!r}"
        )


# Feature: streamlit-settings-page, Property 6: Job filter warning fires when both filters are absent
_optional_str = st.one_of(st.none(), st.text())


@settings(max_examples=100)
@given(_optional_str, _optional_str)
def test_warn_if_no_filters_property(plant_filter, bird_size_filter) -> None:
    """Property 6: warn_if_no_filters returns a warning iff both filters are absent/empty.

    Validates: Requirements 7.4
    """
    result = warn_if_no_filters(plant_filter, bird_size_filter)
    plant_present = bool(plant_filter and plant_filter.strip())
    bird_present = bool(bird_size_filter and bird_size_filter.strip())
    if not plant_present and not bird_present:
        assert result is not None and isinstance(result, str) and len(result) > 0, (
            f"Expected warning string when both filters absent, got {result!r}"
        )
    else:
        assert result is None, (
            f"Expected None when at least one filter present, got {result!r}"
        )


# Feature: streamlit-settings-page, Property 9: Cancel button visibility matches job status
_JOB_STATUSES = ["pending", "running", "completed", "failed", "cancelled"]


@settings(max_examples=100)
@given(st.sampled_from(_JOB_STATUSES))
def test_cancel_button_visible_property(status: str) -> None:
    """Property 9: cancel_button_visible returns True iff status in {pending, running}.

    Validates: Requirements 9.1, 9.4
    """
    result = cancel_button_visible(status)
    if status in {"pending", "running"}:
        assert result is True, f"Expected True for status={status!r}, got {result!r}"
    else:
        assert result is False, f"Expected False for status={status!r}, got {result!r}"


# ---------------------------------------------------------------------------
# Load pure helpers from 6_Global_Config.py
# ---------------------------------------------------------------------------

_global_config_path = os.path.join(os.path.dirname(__file__), "..", "pages", "6_Global_Config.py")
_global_config_spec = importlib.util.spec_from_file_location("global_config_page", _global_config_path)
_global_config_module = importlib.util.module_from_spec(_global_config_spec)
_global_config_spec.loader.exec_module(_global_config_module)
get_input_widget_type = _global_config_module.get_input_widget_type
validate_config_bounds = _global_config_module.validate_config_bounds
group_configs_by_prefix = _global_config_module.group_configs_by_prefix


# Feature: streamlit-settings-page, Property 11: Config input control type matches valueType
@settings(max_examples=100)
@given(st.sampled_from(["int", "float", "string", "bool"]))
def test_get_input_widget_type_property(value_type: str) -> None:
    """Property 11: get_input_widget_type returns the correct widget for each valueType.

    Validates: Requirements 11.1
    """
    result = get_input_widget_type(value_type)
    expected = {
        "int": "number_int",
        "float": "number_float",
        "string": "text",
        "bool": "checkbox",
    }[value_type]
    assert result == expected, (
        f"Expected {expected!r} for value_type={value_type!r}, got {result!r}"
    )


# Feature: streamlit-settings-page, Property 12: Config bounds enforcement
@settings(max_examples=100)
@given(
    st.floats(allow_nan=False, allow_infinity=False),
    st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
    st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
)
def test_validate_config_bounds_property(value: float, min_value, max_value) -> None:
    """Property 12: validate_config_bounds returns None iff value is within [min_value, max_value].

    Validates: Requirements 11.2
    """
    # Skip cases where min_value > max_value (undefined range)
    if min_value is not None and max_value is not None:
        assume(min_value <= max_value)

    result = validate_config_bounds(value, min_value, max_value)

    within_bounds = (
        (min_value is None or value >= min_value)
        and (max_value is None or value <= max_value)
    )

    if within_bounds:
        assert result is None, (
            f"Expected None for value={value} within [{min_value}, {max_value}], got {result!r}"
        )
    else:
        assert result is not None and isinstance(result, str) and len(result) > 0, (
            f"Expected error string for value={value} outside [{min_value}, {max_value}], got {result!r}"
        )


# Feature: streamlit-settings-page, Property 10: Config parameters are grouped by key prefix
@settings(max_examples=100)
@given(st.lists(st.fixed_dictionaries({"key": st.text(min_size=1)})))
def test_group_configs_by_prefix_property(configs: list) -> None:
    """Property 10: group_configs_by_prefix partitions configs correctly by key prefix.

    Validates: Requirements 10.2

    - All configs with the same prefix appear in the same group.
    - Configs with different prefixes appear in different groups.
    - Every config appears in exactly one group.
    """
    groups = group_configs_by_prefix(configs)

    # Every config must appear in exactly one group
    all_in_groups = [cfg for group in groups.values() for cfg in group]
    assert len(all_in_groups) == len(configs), (
        f"Expected {len(configs)} total configs across groups, got {len(all_in_groups)}"
    )

    # Each config in a group must share the same prefix as the group key
    for prefix, group in groups.items():
        for cfg in group:
            key = cfg.get("key", "")
            expected_prefix = key.split(".")[0] if "." in key else key
            assert expected_prefix == prefix, (
                f"Config with key={key!r} (prefix={expected_prefix!r}) found in group {prefix!r}"
            )

    # Configs with the same prefix must be in the same group
    for cfg in configs:
        key = cfg.get("key", "")
        expected_prefix = key.split(".")[0] if "." in key else key
        assert expected_prefix in groups, (
            f"Prefix {expected_prefix!r} not found in groups"
        )
        assert cfg in groups[expected_prefix], (
            f"Config {cfg!r} not found in its expected group {expected_prefix!r}"
        )
