"""
Consolidated property-based tests for the Streamlit Settings Page.

This file consolidates all property tests from:
- Task 1.3: API client URL configuration properties
- Task 2.2: Bucket weight validation (Property 2)
- Task 3.2: SKU weight validation (Property 3)
- Task 4.2: Cut strategy parts deduplication (Property 5)
- Task 7.4: Job form validation — maxCombinationSize and batchSize (Properties 7 & 8)
- Task 8.3: Config value type coercion / bounds enforcement (Properties 11 & 12)

All tests use Hypothesis with a minimum of 100 examples each.

Run with:
    python -m pytest streamlit-app/test_api_client_properties.py -v
"""

import os
import sys
import importlib
import importlib.util

from hypothesis import given, settings, assume
import hypothesis.strategies as st

# ---------------------------------------------------------------------------
# Re-export all property tests from tests/test_validation.py so they are
# discovered and executed when this file is run directly.
# ---------------------------------------------------------------------------

_validation_path = os.path.join(os.path.dirname(__file__), "tests", "test_validation.py")
_validation_spec = importlib.util.spec_from_file_location("test_validation", _validation_path)
_validation_module = importlib.util.module_from_spec(_validation_spec)
_validation_spec.loader.exec_module(_validation_module)

# Import each test function into this module's namespace so pytest discovers them.
test_bucket_weight_validation_property = _validation_module.test_bucket_weight_validation_property
test_sku_weight_validation_property = _validation_module.test_sku_weight_validation_property
test_cut_strategy_parts_unique_no_duplicates = _validation_module.test_cut_strategy_parts_unique_no_duplicates
test_cut_strategy_parts_with_duplicates = _validation_module.test_cut_strategy_parts_with_duplicates
test_max_combination_size_validation_property = _validation_module.test_max_combination_size_validation_property
test_batch_size_validation_property = _validation_module.test_batch_size_validation_property
test_warn_if_no_filters_property = _validation_module.test_warn_if_no_filters_property
test_cancel_button_visible_property = _validation_module.test_cancel_button_visible_property
test_get_input_widget_type_property = _validation_module.test_get_input_widget_type_property
test_validate_config_bounds_property = _validation_module.test_validate_config_bounds_property
test_group_configs_by_prefix_property = _validation_module.test_group_configs_by_prefix_property

# ---------------------------------------------------------------------------
# Task 1.3 — API client URL configuration properties
# ---------------------------------------------------------------------------

# The api_client module reads URLs at import time from os.getenv().
# To test URL configuration we reload the module after manipulating os.environ.

_api_client_path = os.path.join(os.path.dirname(__file__), "api_client.py")


def _load_api_client_with_env(env_overrides: dict):
    """Load api_client.py with the given environment variable overrides."""
    original = {}
    for key, value in env_overrides.items():
        original[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    try:
        spec = importlib.util.spec_from_file_location("api_client_fresh", _api_client_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        # Restore original environment
        for key, orig_value in original.items():
            if orig_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig_value


_URL_ENV_VARS = ["ENUMERATION_API_URL", "WORKER_API_URL", "CONFIG_API_URL"]
_URL_ATTRS = ["ENUMERATION_API_URL", "WORKER_API_URL", "CONFIG_API_URL"]


# Feature: streamlit-settings-page, Property 1a: URL defaults are valid localhost URLs
def test_api_client_url_defaults_are_localhost():
    """Property 1a: When env vars are absent, all base URLs default to localhost.

    Validates: Requirements 2.1, 2.2, 2.3, 2.4

    When ENUMERATION_API_URL, WORKER_API_URL, and CONFIG_API_URL are not set,
    each module-level constant must be a non-empty string starting with
    "http://localhost".
    """
    env_overrides = {var: None for var in _URL_ENV_VARS}
    module = _load_api_client_with_env(env_overrides)

    for attr in _URL_ATTRS:
        url = getattr(module, attr)
        assert isinstance(url, str) and len(url) > 0, (
            f"Expected a non-empty string for {attr}, got {url!r}"
        )
        assert url.startswith("http://localhost"), (
            f"Expected {attr} to start with 'http://localhost' when env var is absent, got {url!r}"
        )


# Feature: streamlit-settings-page, Property 1b: URL from env var is used verbatim
_printable_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
)


@settings(max_examples=100)
@given(_printable_text, _printable_text, _printable_text)
def test_api_client_url_from_env_var_used_verbatim(
    enum_url: str, worker_url: str, config_url: str
) -> None:
    """Property 1b: When env vars are set, the exact values are used as base URLs.

    Validates: Requirements 2.1, 2.2, 2.3, 2.4

    When ENUMERATION_API_URL, WORKER_API_URL, CONFIG_API_URL are set to arbitrary
    non-empty strings, the module-level constants must equal those exact strings.
    """
    env_overrides = {
        "ENUMERATION_API_URL": enum_url,
        "WORKER_API_URL": worker_url,
        "CONFIG_API_URL": config_url,
    }
    module = _load_api_client_with_env(env_overrides)

    assert module.ENUMERATION_API_URL == enum_url, (
        f"Expected ENUMERATION_API_URL={enum_url!r}, got {module.ENUMERATION_API_URL!r}"
    )
    assert module.WORKER_API_URL == worker_url, (
        f"Expected WORKER_API_URL={worker_url!r}, got {module.WORKER_API_URL!r}"
    )
    assert module.CONFIG_API_URL == config_url, (
        f"Expected CONFIG_API_URL={config_url!r}, got {module.CONFIG_API_URL!r}"
    )
