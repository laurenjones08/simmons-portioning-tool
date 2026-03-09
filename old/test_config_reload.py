"""Test configuration reload mechanism.

This test verifies that the configuration reload mechanism works correctly
after saving settings, ensuring that changes are reflected in the config module.
"""

from old.portioning import config
from old.portioning.config_manager import save_config, AppConfig, get_defaults, SETTINGS_FILE


def test_config_reload_after_save():
    """Test that config.reload_config() updates module-level constants.
    
    This test verifies that:
    1. Saving a new configuration to settings.json
    2. Calling config.reload_config()
    3. Results in updated module-level constants (BUCKETS, ILLEGAL_PAIRS, DEFAULTS)
    
    Validates: Requirements 8.1, 8.2
    """
    # Get initial defaults
    initial_config = get_defaults()
    
    # Create a modified configuration
    modified_config = AppConfig(
        buckets=[(100, 200), (300, 400)],  # Different from defaults
        illegal_pairs={"X": ["Y"], "Y": ["X"]},  # Different from defaults
        trim_cap=25,  # Different from default (15)
        time_limit_sec=120,  # Different from default (60)
        gap=0.01,  # Different from default (0.002)
        chunk_size=30,  # Different from default (20)
        pieces_per_min=initial_config.pieces_per_min,
        line_eff=initial_config.line_eff,
        dsi_variance=initial_config.dsi_variance,
        lines=initial_config.lines,
        cut_strategies=initial_config.cut_strategies,
        trim_dollar_value=initial_config.trim_dollar_value
    )
    
    # Save the modified configuration
    success = save_config(modified_config)
    assert success, "Failed to save modified configuration"
    
    # Reload configuration
    config.reload_config()
    
    # Verify that module-level constants are updated
    assert config.BUCKETS == [(100, 200), (300, 400)], "BUCKETS not updated after reload"
    assert config.ILLEGAL_PAIRS == {"X": ["Y"], "Y": ["X"]}, "ILLEGAL_PAIRS not updated after reload"
    assert config.DEFAULTS.trim_cap == 25, "DEFAULTS.trim_cap not updated after reload"
    assert config.DEFAULTS.time_limit_sec == 120, "DEFAULTS.time_limit_sec not updated after reload"
    assert config.DEFAULTS.gap == 0.01, "DEFAULTS.gap not updated after reload"
    assert config.DEFAULTS.chunk_size == 30, "DEFAULTS.chunk_size not updated after reload"
    
    # Clean up: restore defaults
    save_config(initial_config)
    config.reload_config()


def test_config_reload_idempotent():
    """Test that calling reload_config() multiple times is safe.
    
    This test verifies that:
    1. Calling reload_config() multiple times
    2. Does not cause errors or unexpected behavior
    3. Results in consistent configuration values
    
    Validates: Requirements 8.1
    """
    # Get initial state
    initial_buckets = config.BUCKETS.copy()
    initial_illegal_pairs = config.ILLEGAL_PAIRS.copy()
    initial_defaults = config.DEFAULTS
    
    # Reload multiple times
    config.reload_config()
    first_reload_buckets = config.BUCKETS.copy()
    first_reload_illegal_pairs = config.ILLEGAL_PAIRS.copy()
    first_reload_defaults = config.DEFAULTS
    
    config.reload_config()
    second_reload_buckets = config.BUCKETS.copy()
    second_reload_illegal_pairs = config.ILLEGAL_PAIRS.copy()
    second_reload_defaults = config.DEFAULTS
    
    # Verify that multiple reloads produce consistent results
    assert first_reload_buckets == second_reload_buckets, "BUCKETS changed between reloads"
    assert first_reload_illegal_pairs == second_reload_illegal_pairs, "ILLEGAL_PAIRS changed between reloads"
    assert first_reload_defaults.trim_cap == second_reload_defaults.trim_cap, "DEFAULTS.trim_cap changed between reloads"
    assert first_reload_defaults.time_limit_sec == second_reload_defaults.time_limit_sec, "DEFAULTS.time_limit_sec changed between reloads"
    assert first_reload_defaults.gap == second_reload_defaults.gap, "DEFAULTS.gap changed between reloads"
    assert first_reload_defaults.chunk_size == second_reload_defaults.chunk_size, "DEFAULTS.chunk_size changed between reloads"


def test_config_reload_with_missing_file():
    """Test that reload_config() handles missing settings.json gracefully.
    
    This test verifies that:
    1. If settings.json is deleted
    2. Calling reload_config() falls back to defaults
    3. No errors are raised
    
    Validates: Requirements 8.1, 9.1
    """
    # Save current config for restoration
    current_config = config._config
    
    # Delete settings.json if it exists
    if SETTINGS_FILE.exists():
        SETTINGS_FILE.unlink()
    
    # Reload configuration (should fall back to defaults)
    config.reload_config()
    
    # Verify that defaults are loaded
    defaults = get_defaults()
    assert config.BUCKETS == defaults.buckets, "BUCKETS not set to defaults when file missing"
    assert config.ILLEGAL_PAIRS == defaults.illegal_pairs, "ILLEGAL_PAIRS not set to defaults when file missing"
    assert config.DEFAULTS.trim_cap == defaults.trim_cap, "DEFAULTS.trim_cap not set to defaults when file missing"
    
    # Restore original config
    save_config(current_config)
    config.reload_config()


def test_config_reload_with_invalid_json():
    """Test that reload_config() handles invalid JSON gracefully.
    
    This test verifies that:
    1. If settings.json contains invalid JSON
    2. Calling reload_config() falls back to defaults
    3. No errors are raised
    
    Validates: Requirements 8.1, 10.5
    """
    # Save current config for restoration
    current_config = config._config
    
    # Write invalid JSON to settings.json
    with open(SETTINGS_FILE, 'w') as f:
        f.write("{ invalid json }")
    
    # Reload configuration (should fall back to defaults)
    config.reload_config()
    
    # Verify that defaults are loaded
    defaults = get_defaults()
    assert config.BUCKETS == defaults.buckets, "BUCKETS not set to defaults with invalid JSON"
    assert config.ILLEGAL_PAIRS == defaults.illegal_pairs, "ILLEGAL_PAIRS not set to defaults with invalid JSON"
    
    # Restore original config
    save_config(current_config)
    config.reload_config()


if __name__ == "__main__":
    print("Running configuration reload tests...")
    
    try:
        print("\n1. Testing config reload after save...")
        test_config_reload_after_save()
        print("   ✅ PASSED: Config reload after save works correctly")
    except AssertionError as e:
        print(f"   ❌ FAILED: {e}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    try:
        print("\n2. Testing config reload idempotency...")
        test_config_reload_idempotent()
        print("   ✅ PASSED: Config reload is idempotent")
    except AssertionError as e:
        print(f"   ❌ FAILED: {e}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    try:
        print("\n3. Testing config reload with missing file...")
        test_config_reload_with_missing_file()
        print("   ✅ PASSED: Config reload handles missing file gracefully")
    except AssertionError as e:
        print(f"   ❌ FAILED: {e}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    try:
        print("\n4. Testing config reload with invalid JSON...")
        test_config_reload_with_invalid_json()
        print("   ✅ PASSED: Config reload handles invalid JSON gracefully")
    except AssertionError as e:
        print(f"   ❌ FAILED: {e}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
    
    print("\n✅ All configuration reload tests completed!")
