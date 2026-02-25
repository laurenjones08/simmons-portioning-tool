"""Unit tests for settings page save functionality (Task 3.7)."""

import json
from pathlib import Path
from portioning.config_manager import AppConfig, save_config, load_config, validate_config


def test_save_functionality_with_valid_config(tmp_path, monkeypatch):
    """Test that save functionality works with valid configuration.
    
    This test validates Task 3.7 requirements:
    - Creates AppConfig from values
    - Calls validate_config() 
    - Calls save_config() if validation passes
    - Verifies successful save
    """
    # Change to temp directory for testing
    settings_file = tmp_path / "settings.json"
    monkeypatch.chdir(tmp_path)
    
    # Create a valid AppConfig
    config = AppConfig(
        buckets=[(0, 100), (101, 200)],
        illegal_pairs={"A": ["B"], "B": ["A"]},
        trim_cap=15,
        time_limit_sec=60,
        gap=0.002,
        chunk_size=20,
        pieces_per_min=600.0,
        line_eff=0.85,
        dsi_variance=0.05,
        lines=["Line1", "Line2"],
        cut_strategies=["Strategy1", "Strategy2"],
        trim_dollar_value=1.5
    )
    
    # Validate configuration (should pass)
    is_valid, errors = validate_config(config)
    assert is_valid, f"Configuration should be valid, but got errors: {errors}"
    assert len(errors) == 0
    
    # Save configuration (should succeed)
    success = save_config(config)
    assert success, "save_config should return True for valid configuration"
    
    # Verify file was created
    assert settings_file.exists(), "settings.json should be created"
    
    # Verify file contents
    with open(settings_file, 'r') as f:
        saved_data = json.load(f)
    
    assert saved_data["version"] == "1.0"
    assert saved_data["buckets"] == [[0, 100], [101, 200]]
    assert saved_data["illegal_pairs"] == {"A": ["B"], "B": ["A"]}
    assert saved_data["defaults"]["trim_cap"] == 15
    
    # Verify we can load it back
    loaded_config = load_config()
    assert loaded_config.trim_cap == 15
    assert loaded_config.buckets == [(0, 100), (101, 200)]


def test_save_functionality_with_invalid_config(tmp_path, monkeypatch):
    """Test that save functionality rejects invalid configuration.
    
    This test validates Task 3.7 requirements:
    - Calls validate_config() and detects errors
    - Does not save if validation fails
    """
    # Change to temp directory for testing
    settings_file = tmp_path / "settings.json"
    monkeypatch.chdir(tmp_path)
    
    # Create an invalid AppConfig (trim_cap out of range)
    config = AppConfig(
        buckets=[(0, 100)],
        illegal_pairs={},
        trim_cap=150,  # Invalid: should be 0-100
        time_limit_sec=60,
        gap=0.002,
        chunk_size=20,
        pieces_per_min=600.0,
        line_eff=0.85,
        dsi_variance=0.05,
        lines=["Line1"],
        cut_strategies=["Strategy1"],
        trim_dollar_value=1.5
    )
    
    # Validate configuration (should fail)
    is_valid, errors = validate_config(config)
    assert not is_valid, "Configuration should be invalid"
    assert len(errors) > 0, "Should have validation errors"
    assert any("trim_cap" in error for error in errors), "Should have trim_cap error"
    
    # Save configuration (should fail)
    success = save_config(config)
    assert not success, "save_config should return False for invalid configuration"
    
    # Verify file was not created
    assert not settings_file.exists(), "settings.json should not be created for invalid config"


def test_save_functionality_with_invalid_bucket():
    """Test that validation catches invalid bucket tuples.
    
    This test validates Task 3.7 requirements:
    - Validates bucket tuples (min < max)
    - Displays appropriate error messages
    """
    # Create config with invalid bucket (min >= max)
    config = AppConfig(
        buckets=[(100, 50)],  # Invalid: min >= max
        illegal_pairs={},
        trim_cap=15,
        time_limit_sec=60,
        gap=0.002,
        chunk_size=20,
        pieces_per_min=600.0,
        line_eff=0.85,
        dsi_variance=0.05,
        lines=["Line1"],
        cut_strategies=["Strategy1"],
        trim_dollar_value=1.5
    )
    
    # Validate configuration (should fail)
    is_valid, errors = validate_config(config)
    assert not is_valid, "Configuration with invalid bucket should be invalid"
    assert len(errors) > 0, "Should have validation errors"
    assert any("minimum" in error.lower() and "maximum" in error.lower() for error in errors), \
        "Should have bucket ordering error"


def test_save_functionality_displays_success_message():
    """Test that save functionality can indicate success.
    
    This simulates the UI behavior where:
    - Successful save returns True
    - UI can display success message based on return value
    """
    # This is tested implicitly in test_save_functionality_with_valid_config
    # The save_config function returns True on success, which the UI uses
    # to display the success message
    pass


def test_save_functionality_displays_error_message():
    """Test that save functionality can indicate failure.
    
    This simulates the UI behavior where:
    - Failed save returns False
    - Validation errors are available for display
    - UI can display error messages based on return value and errors
    """
    # This is tested implicitly in test_save_functionality_with_invalid_config
    # The save_config function returns False on failure, and validate_config
    # provides error messages that the UI can display
    pass


if __name__ == "__main__":
    import sys
    import tempfile
    import os
    
    # Simple monkeypatch class
    class Monkeypatch:
        def __init__(self):
            self.original_dir = None
        
        def chdir(self, path):
            self.original_dir = os.getcwd()
            os.chdir(path)
        
        def undo(self):
            if self.original_dir:
                os.chdir(self.original_dir)
    
    # Run tests manually
    print("Running test_save_functionality_with_valid_config...")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            monkeypatch = Monkeypatch()
            try:
                test_save_functionality_with_valid_config(Path(tmp_dir), monkeypatch)
                print("✓ PASSED")
            finally:
                monkeypatch.undo()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nRunning test_save_functionality_with_invalid_config...")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            monkeypatch = Monkeypatch()
            try:
                test_save_functionality_with_invalid_config(Path(tmp_dir), monkeypatch)
                print("✓ PASSED")
            finally:
                monkeypatch.undo()
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nRunning test_save_functionality_with_invalid_bucket...")
    try:
        test_save_functionality_with_invalid_bucket()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
