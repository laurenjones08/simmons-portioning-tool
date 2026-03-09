"""Unit tests for config_manager module."""

import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from old.portioning.config_manager import (
    AppConfig,
    load_config,
    get_defaults,
    save_config
)


class TestLoadConfig(unittest.TestCase):
    """Test cases for load_config() function."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_settings_file = Path(self.temp_dir) / "settings.json"
        
    def tearDown(self):
        """Clean up after each test."""
        # Remove test settings file if it exists
        if self.test_settings_file.exists():
            self.test_settings_file.unlink()
        # Remove temporary directory
        os.rmdir(self.temp_dir)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_load_config_file_not_exists(self, mock_settings_file):
        """Test load_config returns defaults when settings.json does not exist."""
        # Patch SETTINGS_FILE to point to non-existent file
        mock_settings_file.exists.return_value = False
        
        config = load_config()
        defaults = get_defaults()
        
        # Verify config matches defaults
        self.assertEqual(config.buckets, defaults.buckets)
        self.assertEqual(config.illegal_pairs, defaults.illegal_pairs)
        self.assertEqual(config.trim_cap, defaults.trim_cap)
        self.assertEqual(config.time_limit_sec, defaults.time_limit_sec)
        self.assertEqual(config.gap, defaults.gap)
        self.assertEqual(config.chunk_size, defaults.chunk_size)
        self.assertEqual(config.pieces_per_min, defaults.pieces_per_min)
        self.assertEqual(config.line_eff, defaults.line_eff)
        self.assertEqual(config.dsi_variance, defaults.dsi_variance)
        self.assertEqual(config.lines, defaults.lines)
        self.assertEqual(config.cut_strategies, defaults.cut_strategies)
        self.assertEqual(config.trim_dollar_value, defaults.trim_dollar_value)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_load_config_valid_json(self, mock_settings_file):
        """Test load_config successfully loads valid settings.json."""
        # Create valid settings file
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
        
        with open(self.test_settings_file, 'w') as f:
            json.dump(valid_config, f)
        
        # Patch SETTINGS_FILE to point to test file
        mock_settings_file.exists.return_value = True
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(valid_config))):
            config = load_config()
        
        # Verify loaded values match the file
        self.assertEqual(config.buckets, [(0, 100), (101, 200)])
        self.assertEqual(config.illegal_pairs, {"A": ["B"], "B": ["A"]})
        self.assertEqual(config.trim_cap, 20)
        self.assertEqual(config.time_limit_sec, 120)
        self.assertEqual(config.gap, 0.003)
        self.assertEqual(config.chunk_size, 25)
        self.assertEqual(config.pieces_per_min, 700.0)
        self.assertEqual(config.line_eff, 0.90)
        self.assertEqual(config.dsi_variance, 0.10)
        self.assertEqual(config.lines, ["LineA", "LineB"])
        self.assertEqual(config.cut_strategies, ["StrategyA", "StrategyB"])
        self.assertEqual(config.trim_dollar_value, 2.0)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_load_config_invalid_json(self, mock_settings_file):
        """Test load_config returns defaults when JSON is invalid."""
        # Patch SETTINGS_FILE to point to test file with invalid JSON
        mock_settings_file.exists.return_value = True
        
        with patch('builtins.open', unittest.mock.mock_open(read_data="{ invalid json }")):
            config = load_config()
        
        defaults = get_defaults()
        
        # Verify config matches defaults (fallback behavior)
        self.assertEqual(config.buckets, defaults.buckets)
        self.assertEqual(config.trim_cap, defaults.trim_cap)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_load_config_missing_fields(self, mock_settings_file):
        """Test load_config uses defaults for missing fields."""
        # Create settings file with only partial data
        partial_config = {
            "buckets": [[0, 50]],
            "defaults": {
                "trim_cap": 25
            }
        }
        
        mock_settings_file.exists.return_value = True
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(partial_config))):
            config = load_config()
        
        defaults = get_defaults()
        
        # Verify specified values are loaded
        self.assertEqual(config.buckets, [(0, 50)])
        self.assertEqual(config.trim_cap, 25)
        
        # Verify missing values use defaults
        self.assertEqual(config.illegal_pairs, defaults.illegal_pairs)
        self.assertEqual(config.time_limit_sec, defaults.time_limit_sec)
        self.assertEqual(config.pieces_per_min, defaults.pieces_per_min)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_load_config_wrong_types(self, mock_settings_file):
        """Test load_config returns defaults when data has wrong types."""
        # Create settings file with wrong data types
        invalid_config = {
            "buckets": "not a list",
            "defaults": {
                "trim_cap": "not a number"
            }
        }
        
        mock_settings_file.exists.return_value = True
        
        with patch('builtins.open', unittest.mock.mock_open(read_data=json.dumps(invalid_config))):
            config = load_config()
        
        defaults = get_defaults()
        
        # Verify config matches defaults (fallback behavior)
        self.assertEqual(config.buckets, defaults.buckets)
        self.assertEqual(config.trim_cap, defaults.trim_cap)


if __name__ == '__main__':
    unittest.main()



class TestValidationFunctions(unittest.TestCase):
    """Test cases for validation functions."""
    
    def test_validate_numeric_range_valid(self):
        """Test validate_numeric_range accepts values within range."""
        from old.portioning.config_manager import validate_numeric_range, VALIDATION_RULES
        
        # Test with valid integer
        is_valid, error = validate_numeric_range(15, "trim_cap", VALIDATION_RULES["trim_cap"])
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Test with valid float
        is_valid, error = validate_numeric_range(0.5, "line_eff", VALIDATION_RULES["line_eff"])
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Test boundary values
        is_valid, error = validate_numeric_range(0, "trim_cap", VALIDATION_RULES["trim_cap"])
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        is_valid, error = validate_numeric_range(100, "trim_cap", VALIDATION_RULES["trim_cap"])
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_numeric_range_out_of_range(self):
        """Test validate_numeric_range rejects values outside range."""
        from old.portioning.config_manager import validate_numeric_range, VALIDATION_RULES
        
        # Test value below minimum
        is_valid, error = validate_numeric_range(-1, "trim_cap", VALIDATION_RULES["trim_cap"])
        self.assertFalse(is_valid)
        self.assertIn("must be between", error)
        self.assertIn("-1", error)
        
        # Test value above maximum
        is_valid, error = validate_numeric_range(101, "trim_cap", VALIDATION_RULES["trim_cap"])
        self.assertFalse(is_valid)
        self.assertIn("must be between", error)
        self.assertIn("101", error)
        
        # Test float out of range
        is_valid, error = validate_numeric_range(1.5, "line_eff", VALIDATION_RULES["line_eff"])
        self.assertFalse(is_valid)
        self.assertIn("must be between", error)
    
    def test_validate_numeric_range_wrong_type(self):
        """Test validate_numeric_range rejects non-numeric values."""
        from old.portioning.config_manager import validate_numeric_range, VALIDATION_RULES
        
        # Test string
        is_valid, error = validate_numeric_range("15", "trim_cap", VALIDATION_RULES["trim_cap"])
        self.assertFalse(is_valid)
        self.assertIn("must be a number", error)
        
        # Test None
        is_valid, error = validate_numeric_range(None, "trim_cap", VALIDATION_RULES["trim_cap"])
        self.assertFalse(is_valid)
        self.assertIn("must be a number", error)
    
    def test_validate_bucket_tuple_valid(self):
        """Test validate_bucket_tuple accepts valid bucket tuples."""
        from old.portioning.config_manager import validate_bucket_tuple
        
        # Test valid tuple
        is_valid, error = validate_bucket_tuple((0, 100), 0)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        
        # Test valid list
        is_valid, error = validate_bucket_tuple([50, 150], 1)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_bucket_tuple_invalid_ordering(self):
        """Test validate_bucket_tuple rejects min >= max."""
        from old.portioning.config_manager import validate_bucket_tuple
        
        # Test min > max
        is_valid, error = validate_bucket_tuple((100, 50), 0)
        self.assertFalse(is_valid)
        self.assertIn("minimum", error)
        self.assertIn("must be less than", error)
        self.assertIn("100", error)
        self.assertIn("50", error)
        
        # Test min == max
        is_valid, error = validate_bucket_tuple((100, 100), 1)
        self.assertFalse(is_valid)
        self.assertIn("minimum", error)
        self.assertIn("must be less than", error)
    
    def test_validate_bucket_tuple_wrong_structure(self):
        """Test validate_bucket_tuple rejects invalid structures."""
        from old.portioning.config_manager import validate_bucket_tuple
        
        # Test wrong length
        is_valid, error = validate_bucket_tuple((0, 50, 100), 0)
        self.assertFalse(is_valid)
        self.assertIn("must be a tuple/list of 2 values", error)
        
        # Test single value
        is_valid, error = validate_bucket_tuple((50,), 1)
        self.assertFalse(is_valid)
        self.assertIn("must be a tuple/list of 2 values", error)
        
        # Test non-integer values
        is_valid, error = validate_bucket_tuple(("0", "100"), 2)
        self.assertFalse(is_valid)
        self.assertIn("must be integers", error)
    
    def test_validate_config_valid(self):
        """Test validate_config accepts valid configuration."""
        from old.portioning.config_manager import validate_config
        
        # Get defaults which should be valid
        defaults = get_defaults()
        is_valid, errors = validate_config(defaults)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_config_invalid_numeric_params(self):
        """Test validate_config rejects invalid numeric parameters."""
        from old.portioning.config_manager import validate_config
        
        # Create config with out-of-range trim_cap
        config = get_defaults()
        config.trim_cap = 150  # Max is 100
        
        is_valid, errors = validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("trim_cap" in err for err in errors))
        
        # Create config with out-of-range line_eff
        config = get_defaults()
        config.line_eff = 1.5  # Max is 1.0
        
        is_valid, errors = validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("line_eff" in err for err in errors))
    
    def test_validate_config_invalid_buckets(self):
        """Test validate_config rejects invalid bucket tuples."""
        from old.portioning.config_manager import validate_config
        
        # Create config with invalid bucket (min >= max)
        config = get_defaults()
        config.buckets = [(0, 100), (200, 150)]  # Second bucket is invalid
        
        is_valid, errors = validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("Bucket 1" in err and "must be less than" in err for err in errors))
    
    def test_validate_config_invalid_illegal_pairs(self):
        """Test validate_config rejects invalid illegal_pairs structure."""
        from old.portioning.config_manager import validate_config
        
        # Create config with non-dict illegal_pairs
        config = get_defaults()
        config.illegal_pairs = "not a dict"
        
        is_valid, errors = validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("illegal_pairs must be a dictionary" in err for err in errors))
        
        # Create config with non-list values in illegal_pairs
        config = get_defaults()
        config.illegal_pairs = {"A": "not a list"}
        
        is_valid, errors = validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("illegal_pairs[A] must be a list" in err for err in errors))
    
    def test_validate_config_empty_lists(self):
        """Test validate_config rejects empty required lists."""
        from old.portioning.config_manager import validate_config
        
        # Create config with empty lines
        config = get_defaults()
        config.lines = []
        
        is_valid, errors = validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("lines must be a non-empty list" in err for err in errors))
        
        # Create config with empty cut_strategies
        config = get_defaults()
        config.cut_strategies = []
        
        is_valid, errors = validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(any("cut_strategies must be a non-empty list" in err for err in errors))
    
    def test_validate_config_multiple_errors(self):
        """Test validate_config returns all errors when multiple validations fail."""
        from old.portioning.config_manager import validate_config
        
        # Create config with multiple issues
        config = get_defaults()
        config.trim_cap = 150  # Out of range
        config.buckets = [(100, 50)]  # Invalid ordering
        config.lines = []  # Empty list
        
        is_valid, errors = validate_config(config)
        self.assertFalse(is_valid)
        self.assertGreaterEqual(len(errors), 3)  # Should have at least 3 errors


class TestSaveConfig(unittest.TestCase):
    """Test cases for save_config() function."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_settings_file = Path(self.temp_dir) / "settings.json"
        
    def tearDown(self):
        """Clean up after each test."""
        # Remove test settings file if it exists
        if self.test_settings_file.exists():
            self.test_settings_file.unlink()
        # Remove temporary directory
        os.rmdir(self.temp_dir)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_save_config_valid(self, mock_settings_file):
        """Test save_config successfully saves valid configuration."""
        # Patch SETTINGS_FILE to point to test file
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        # Create valid config
        config = get_defaults()
        config.trim_cap = 25
        config.pieces_per_min = 750.0
        
        # Mock the file path for writing
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            result = save_config(config)
        
        # Verify save was successful
        self.assertTrue(result)
        
        # Verify file was created and contains correct data
        self.assertTrue(self.test_settings_file.exists())
        
        with open(self.test_settings_file, 'r') as f:
            saved_data = json.load(f)
        
        # Verify structure and values
        self.assertEqual(saved_data["version"], "1.0")
        self.assertEqual(saved_data["defaults"]["trim_cap"], 25)
        self.assertEqual(saved_data["ui_parameters"]["pieces_per_min"], 750.0)
        self.assertIn("buckets", saved_data)
        self.assertIn("illegal_pairs", saved_data)
        self.assertIn("new_parameters", saved_data)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_save_config_invalid(self, mock_settings_file):
        """Test save_config returns False for invalid configuration."""
        # Patch SETTINGS_FILE to point to test file
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        # Create invalid config (out of range value)
        config = get_defaults()
        config.trim_cap = 150  # Max is 100
        
        # Mock the file path for writing
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            result = save_config(config)
        
        # Verify save failed
        self.assertFalse(result)
        
        # Verify file was not created
        self.assertFalse(self.test_settings_file.exists())
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_save_config_json_format(self, mock_settings_file):
        """Test save_config creates properly formatted JSON with indentation."""
        # Patch SETTINGS_FILE to point to test file
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        # Create valid config
        config = get_defaults()
        
        # Mock the file path for writing
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            result = save_config(config)
        
        # Verify save was successful
        self.assertTrue(result)
        
        # Read file as text to check formatting
        with open(self.test_settings_file, 'r') as f:
            content = f.read()
        
        # Verify JSON is indented (contains newlines and spaces)
        self.assertIn('\n', content)
        self.assertIn('  ', content)  # 2-space indentation
        
        # Verify it's valid JSON
        json.loads(content)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_save_config_all_fields(self, mock_settings_file):
        """Test save_config includes all configuration fields."""
        # Patch SETTINGS_FILE to point to test file
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        # Create config with custom values
        config = AppConfig(
            buckets=[(0, 50), (51, 100)],
            illegal_pairs={"X": ["Y"], "Y": ["X"]},
            trim_cap=30,
            time_limit_sec=90,
            gap=0.004,
            chunk_size=15,
            pieces_per_min=800.0,
            line_eff=0.95,
            dsi_variance=0.08,
            lines=["L1", "L2"],
            cut_strategies=["S1", "S2", "S3"],
            trim_dollar_value=2.5
        )
        
        # Mock the file path for writing
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            result = save_config(config)
        
        # Verify save was successful
        self.assertTrue(result)
        
        # Load and verify all fields
        with open(self.test_settings_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["buckets"], [[0, 50], [51, 100]])
        self.assertEqual(saved_data["illegal_pairs"], {"X": ["Y"], "Y": ["X"]})
        self.assertEqual(saved_data["defaults"]["trim_cap"], 30)
        self.assertEqual(saved_data["defaults"]["time_limit_sec"], 90)
        self.assertEqual(saved_data["defaults"]["gap"], 0.004)
        self.assertEqual(saved_data["defaults"]["chunk_size"], 15)
        self.assertEqual(saved_data["ui_parameters"]["pieces_per_min"], 800.0)
        self.assertEqual(saved_data["ui_parameters"]["line_eff"], 0.95)
        self.assertEqual(saved_data["new_parameters"]["dsi_variance"], 0.08)
        self.assertEqual(saved_data["new_parameters"]["lines"], ["L1", "L2"])
        self.assertEqual(saved_data["new_parameters"]["cut_strategies"], ["S1", "S2", "S3"])
        self.assertEqual(saved_data["new_parameters"]["trim_dollar_value"], 2.5)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_save_config_validation_before_save(self, mock_settings_file):
        """Test save_config validates configuration before saving."""
        # Patch SETTINGS_FILE to point to test file
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        # Create config with multiple validation errors
        config = get_defaults()
        config.trim_cap = 200  # Out of range
        config.buckets = [(100, 50)]  # Invalid ordering
        config.lines = []  # Empty list
        
        # Mock the file path for writing
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            result = save_config(config)
        
        # Verify save failed due to validation
        self.assertFalse(result)
        
        # Verify file was not created
        self.assertFalse(self.test_settings_file.exists())



class TestResetToDefaults(unittest.TestCase):
    """Test cases for reset_to_defaults() function."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_settings_file = Path(self.temp_dir) / "settings.json"
        
    def tearDown(self):
        """Clean up after each test."""
        # Remove test settings file if it exists
        if self.test_settings_file.exists():
            self.test_settings_file.unlink()
        # Remove temporary directory
        os.rmdir(self.temp_dir)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_reset_to_defaults_success(self, mock_settings_file):
        """Test reset_to_defaults successfully resets configuration to defaults."""
        from old.portioning.config_manager import reset_to_defaults
        
        # Patch SETTINGS_FILE to point to test file
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        # First, save a modified configuration
        modified_config = get_defaults()
        modified_config.trim_cap = 35
        modified_config.pieces_per_min = 900.0
        modified_config.lines = ["CustomLine1", "CustomLine2"]
        
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            save_result = save_config(modified_config)
            self.assertTrue(save_result)
        
        # Verify modified config was saved
        with open(self.test_settings_file, 'r') as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data["defaults"]["trim_cap"], 35)
        self.assertEqual(saved_data["ui_parameters"]["pieces_per_min"], 900.0)
        
        # Now reset to defaults
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            reset_result = reset_to_defaults()
        
        # Verify reset was successful
        self.assertTrue(reset_result)
        
        # Verify file now contains default values
        with open(self.test_settings_file, 'r') as f:
            reset_data = json.load(f)
        
        defaults = get_defaults()
        self.assertEqual(reset_data["defaults"]["trim_cap"], defaults.trim_cap)
        self.assertEqual(reset_data["ui_parameters"]["pieces_per_min"], defaults.pieces_per_min)
        self.assertEqual(reset_data["new_parameters"]["lines"], defaults.lines)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_reset_to_defaults_creates_file(self, mock_settings_file):
        """Test reset_to_defaults creates settings file if it doesn't exist."""
        from old.portioning.config_manager import reset_to_defaults
        
        # Patch SETTINGS_FILE to point to test file (which doesn't exist yet)
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        # Verify file doesn't exist
        self.assertFalse(self.test_settings_file.exists())
        
        # Call reset_to_defaults
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            result = reset_to_defaults()
        
        # Verify reset was successful
        self.assertTrue(result)
        
        # Verify file was created
        self.assertTrue(self.test_settings_file.exists())
        
        # Verify file contains default values
        with open(self.test_settings_file, 'r') as f:
            data = json.load(f)
        
        defaults = get_defaults()
        self.assertEqual(data["defaults"]["trim_cap"], defaults.trim_cap)
        self.assertEqual(data["defaults"]["time_limit_sec"], defaults.time_limit_sec)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_reset_to_defaults_all_fields(self, mock_settings_file):
        """Test reset_to_defaults resets all configuration fields to defaults."""
        from old.portioning.config_manager import reset_to_defaults
        
        # Patch SETTINGS_FILE to point to test file
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        # Call reset_to_defaults
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            result = reset_to_defaults()
        
        # Verify reset was successful
        self.assertTrue(result)
        
        # Load saved data
        with open(self.test_settings_file, 'r') as f:
            data = json.load(f)
        
        # Get defaults for comparison
        defaults = get_defaults()
        
        # Verify all fields match defaults
        # Note: JSON stores tuples as lists, so convert for comparison
        self.assertEqual(data["buckets"], [list(b) for b in defaults.buckets])
        self.assertEqual(data["illegal_pairs"], defaults.illegal_pairs)
        self.assertEqual(data["defaults"]["trim_cap"], defaults.trim_cap)
        self.assertEqual(data["defaults"]["time_limit_sec"], defaults.time_limit_sec)
        self.assertEqual(data["defaults"]["gap"], defaults.gap)
        self.assertEqual(data["defaults"]["chunk_size"], defaults.chunk_size)
        self.assertEqual(data["ui_parameters"]["pieces_per_min"], defaults.pieces_per_min)
        self.assertEqual(data["ui_parameters"]["line_eff"], defaults.line_eff)
        self.assertEqual(data["new_parameters"]["dsi_variance"], defaults.dsi_variance)
        self.assertEqual(data["new_parameters"]["lines"], defaults.lines)
        self.assertEqual(data["new_parameters"]["cut_strategies"], defaults.cut_strategies)
        self.assertEqual(data["new_parameters"]["trim_dollar_value"], defaults.trim_dollar_value)
    
    @patch('portioning.config_manager.SETTINGS_FILE')
    def test_reset_to_defaults_idempotent(self, mock_settings_file):
        """Test reset_to_defaults is idempotent (calling multiple times has same result)."""
        from old.portioning.config_manager import reset_to_defaults
        
        # Patch SETTINGS_FILE to point to test file
        mock_settings_file.__fspath__ = lambda self: str(self.test_settings_file)
        
        # Call reset_to_defaults first time
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            result1 = reset_to_defaults()
        self.assertTrue(result1)
        
        # Read first result
        with open(self.test_settings_file, 'r') as f:
            data1 = json.load(f)
        
        # Call reset_to_defaults second time
        with patch('portioning.config_manager.SETTINGS_FILE', self.test_settings_file):
            result2 = reset_to_defaults()
        self.assertTrue(result2)
        
        # Read second result
        with open(self.test_settings_file, 'r') as f:
            data2 = json.load(f)
        
        # Verify both results are identical
        self.assertEqual(data1, data2)
