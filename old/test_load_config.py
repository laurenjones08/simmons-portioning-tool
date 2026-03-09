"""Simple test script for load_config() function."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from old.portioning.config_manager import load_config, get_defaults


def test_load_config_file_not_exists():
    """Test load_config returns defaults when settings.json does not exist."""
    print("Test 1: File does not exist...")
    
    with patch('portioning.config_manager.SETTINGS_FILE') as mock_file:
        mock_file.exists.return_value = False
        
        config = load_config()
        defaults = get_defaults()
        
        assert config.buckets == defaults.buckets, "Buckets should match defaults"
        assert config.trim_cap == defaults.trim_cap, "trim_cap should match defaults"
        assert config.pieces_per_min == defaults.pieces_per_min, "pieces_per_min should match defaults"
        
    print("✓ Test 1 passed: Returns defaults when file doesn't exist")


def test_load_config_valid_json():
    """Test load_config successfully loads valid settings.json."""
    print("\nTest 2: Valid JSON file...")
    
    valid_config = {
        "buckets": [[0, 100], [101, 200]],
        "illegal_pairs": {"A": ["B"], "B": ["A"]},
        "defaults": {
            "trim_cap": 20,
            "time_limit_sec": 120,
            "gap": 0.003,
            "chunk_size": 25
        },
        "ui_parameters": {
            "pieces_per_min": 700.0,
            "line_eff": 0.90
        },
        "new_parameters": {
            "dsi_variance": 0.10,
            "lines": ["LineA", "LineB"],
            "cut_strategies": ["StrategyA", "StrategyB"],
            "trim_dollar_value": 2.0
        }
    }
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    test_file = Path(temp_dir) / "settings.json"
    
    try:
        with open(test_file, 'w') as f:
            json.dump(valid_config, f)
        
        with patch('portioning.config_manager.SETTINGS_FILE', test_file):
            config = load_config()
        
        # Verify loaded values
        assert config.buckets == [(0, 100), (101, 200)], "Buckets should be loaded correctly"
        assert config.illegal_pairs == {"A": ["B"], "B": ["A"]}, "Illegal pairs should be loaded"
        assert config.trim_cap == 20, "trim_cap should be 20"
        assert config.time_limit_sec == 120, "time_limit_sec should be 120"
        assert config.gap == 0.003, "gap should be 0.003"
        assert config.chunk_size == 25, "chunk_size should be 25"
        assert config.pieces_per_min == 700.0, "pieces_per_min should be 700.0"
        assert config.line_eff == 0.90, "line_eff should be 0.90"
        assert config.dsi_variance == 0.10, "dsi_variance should be 0.10"
        assert config.lines == ["LineA", "LineB"], "lines should be loaded"
        assert config.cut_strategies == ["StrategyA", "StrategyB"], "cut_strategies should be loaded"
        assert config.trim_dollar_value == 2.0, "trim_dollar_value should be 2.0"
        
        print("✓ Test 2 passed: Loads valid JSON correctly")
        
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        os.rmdir(temp_dir)


def test_load_config_invalid_json():
    """Test load_config returns defaults when JSON is invalid."""
    print("\nTest 3: Invalid JSON...")
    
    temp_dir = tempfile.mkdtemp()
    test_file = Path(temp_dir) / "settings.json"
    
    try:
        # Write invalid JSON
        with open(test_file, 'w') as f:
            f.write("{ invalid json }")
        
        with patch('portioning.config_manager.SETTINGS_FILE', test_file):
            config = load_config()
        
        defaults = get_defaults()
        
        # Should fall back to defaults
        assert config.buckets == defaults.buckets, "Should use default buckets"
        assert config.trim_cap == defaults.trim_cap, "Should use default trim_cap"
        
        print("✓ Test 3 passed: Falls back to defaults on invalid JSON")
        
    finally:
        if test_file.exists():
            test_file.unlink()
        os.rmdir(temp_dir)


def test_load_config_missing_fields():
    """Test load_config uses defaults for missing fields."""
    print("\nTest 4: Missing fields...")
    
    partial_config = {
        "buckets": [[0, 50]],
        "defaults": {
            "trim_cap": 25
        }
    }
    
    temp_dir = tempfile.mkdtemp()
    test_file = Path(temp_dir) / "settings.json"
    
    try:
        with open(test_file, 'w') as f:
            json.dump(partial_config, f)
        
        with patch('portioning.config_manager.SETTINGS_FILE', test_file):
            config = load_config()
        
        defaults = get_defaults()
        
        # Verify specified values are loaded
        assert config.buckets == [(0, 50)], "Should load specified buckets"
        assert config.trim_cap == 25, "Should load specified trim_cap"
        
        # Verify missing values use defaults
        assert config.illegal_pairs == defaults.illegal_pairs, "Should use default illegal_pairs"
        assert config.time_limit_sec == defaults.time_limit_sec, "Should use default time_limit_sec"
        assert config.pieces_per_min == defaults.pieces_per_min, "Should use default pieces_per_min"
        
        print("✓ Test 4 passed: Uses defaults for missing fields")
        
    finally:
        if test_file.exists():
            test_file.unlink()
        os.rmdir(temp_dir)


def test_load_config_wrong_types():
    """Test load_config returns defaults when data has wrong types."""
    print("\nTest 5: Wrong data types...")
    
    invalid_config = {
        "buckets": "not a list",
        "defaults": {
            "trim_cap": "not a number"
        }
    }
    
    temp_dir = tempfile.mkdtemp()
    test_file = Path(temp_dir) / "settings.json"
    
    try:
        with open(test_file, 'w') as f:
            json.dump(invalid_config, f)
        
        with patch('portioning.config_manager.SETTINGS_FILE', test_file):
            config = load_config()
        
        defaults = get_defaults()
        
        # Should fall back to defaults
        assert config.buckets == defaults.buckets, "Should use default buckets"
        assert config.trim_cap == defaults.trim_cap, "Should use default trim_cap"
        
        print("✓ Test 5 passed: Falls back to defaults on wrong types")
        
    finally:
        if test_file.exists():
            test_file.unlink()
        os.rmdir(temp_dir)


if __name__ == '__main__':
    print("Running load_config() tests...\n")
    print("=" * 60)
    
    try:
        test_load_config_file_not_exists()
        test_load_config_valid_json()
        test_load_config_invalid_json()
        test_load_config_missing_fields()
        test_load_config_wrong_types()
        
        print("\n" + "=" * 60)
        print("\n✓ All tests passed!")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
