"""Configuration manager for the portioning application.

This module provides centralized configuration management with support for:
- Loading configuration from settings.json or falling back to defaults
- Validating configuration parameters
- Saving configuration to persistent storage
- Resetting configuration to default values
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import json
import os
from pathlib import Path


@dataclass
class AppConfig:
    """Complete application configuration.

    This dataclass contains all configurable parameters for the portioning application,
    including existing parameters from config.py, UI parameters, and new parameters.

    Attributes:
        buckets: List of (min, max) tuples defining WIP weight ranges in grams
        illegal_pairs: Dictionary mapping part codes to lists of incompatible part codes
        trim_cap: Maximum allowed trim percentage (0-100)
        pieces_per_min: Production rate in pieces per minute
        line_eff: Line efficiency factor (0.0-1.0)
        dsi_variance: DSI variance tolerance
        lines: List of production line identifiers
        cut_strategies: List of available cutting strategies
        trim_dollar_value: Dollar value per unit of trim
    """
    # Existing parameters from config.py
    buckets: List[Tuple[int, int]]
    illegal_pairs: Dict[str, List[str]]
    trim_cap: int

    # Existing UI parameters (currently hardcoded in ui.py)
    pieces_per_min: float
    line_eff: float

    # New parameters
    dsi_variance: float
    lines: List[str]
    cut_strategies: List[str]
    trim_dollar_value: float



# Configuration file path
SETTINGS_FILE = Path("settings.json")


# Validation rules for configuration parameters
VALIDATION_RULES = {
    "trim_cap": {
        "type": int,
        "min": 0,
        "max": 100,
        "description": "Trim percentage cap"
    },
    "pieces_per_min": {
        "type": float,
        "min": 100.0,
        "max": 2000.0,
        "description": "Production pieces per minute"
    },
    "line_eff": {
        "type": float,
        "min": 0.1,
        "max": 1.0,
        "description": "Line efficiency factor"
    },
    "dsi_variance": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "description": "DSI variance tolerance"
    },
    "trim_dollar_value": {
        "type": float,
        "min": 0.0,
        "max": 100.0,
        "description": "Dollar value per unit of trim"
    }
}


def get_defaults() -> AppConfig:
    """Get default configuration values.
    
    Returns default values from the original config.py plus hardcoded UI defaults
    and new parameter defaults.
    
    Returns:
        AppConfig: Default configuration object
    """
    # Define default values directly to avoid circular import
    default_buckets = [
        (0, 324),
        (325, 375),
        (376, 475),
        (476, 550),
        (551, 625),
        (626, 780),
        (390, 480),
        (481, 580),
    ]
    
    default_illegal_pairs = {
        "C": ["D"],
        "D": ["C", "T"],
        "R": ["V"],
        "V": ["R"],
        "M": ["K"],
        "K": ["M"],
        "T": ["D"],
    }
    
    return AppConfig(
        # Existing parameters from config.py
        buckets=default_buckets.copy(),
        illegal_pairs={k: v.copy() for k, v in default_illegal_pairs.items()},
        trim_cap=15,
        
        # UI parameters (hardcoded defaults from ui.py)
        pieces_per_min=600.0,
        line_eff=0.85,
        
        # New parameters (reasonable defaults)
        dsi_variance=0.05,
        lines=["Line1", "Line2", "Line3"],
        cut_strategies=["Strategy1", "Strategy2"],
        trim_dollar_value=1.5
    )


def load_config() -> AppConfig:
    """Load configuration from settings.json or fall back to defaults.
    
    Attempts to load configuration from the settings.json file. If the file
    does not exist or contains invalid JSON, falls back to default values.
    
    Returns:
        AppConfig: Loaded configuration object
    
    Behavior:
        1. Check if settings.json exists
        2. If exists, load and parse JSON
        3. If not exists or invalid, use defaults from config.py
        4. Return AppConfig instance
    """
    # Check if settings file exists
    if not SETTINGS_FILE.exists():
        return get_defaults()
    
    try:
        # Load and parse JSON
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
        
        # Get defaults to use as fallback for missing fields
        defaults_config = get_defaults()
        
        # Extract configuration sections
        buckets_data = data.get("buckets")
        illegal_pairs_data = data.get("illegal_pairs")
        ui_parameters = data.get("ui_parameters", {})
        new_parameters = data.get("new_parameters", {})
        
        # Convert bucket tuples (JSON arrays to tuples), or use defaults if missing/invalid
        if buckets_data is not None and isinstance(buckets_data, list):
            try:
                buckets = [tuple(b) for b in buckets_data]
            except (TypeError, ValueError):
                buckets = defaults_config.buckets
        else:
            buckets = defaults_config.buckets
        
        # Use loaded illegal_pairs or defaults if missing/invalid
        if illegal_pairs_data is not None and isinstance(illegal_pairs_data, dict):
            illegal_pairs = illegal_pairs_data
        else:
            illegal_pairs = defaults_config.illegal_pairs
        
        # Helper function to safely get typed values
        def get_typed_value(section, key, default_value, expected_type):
            """Get a value from section with type checking."""
            value = section.get(key)
            if value is not None and isinstance(value, expected_type):
                return value
            return default_value
        
        # Create AppConfig from loaded data with defaults as fallback
        config = AppConfig(
            buckets=buckets,
            illegal_pairs=illegal_pairs,
            trim_cap=get_typed_value(data, "trim_cap", defaults_config.trim_cap, int),
            pieces_per_min=get_typed_value(ui_parameters, "pieces_per_min", defaults_config.pieces_per_min, (int, float)),
            line_eff=get_typed_value(ui_parameters, "line_eff", defaults_config.line_eff, (int, float)),
            dsi_variance=get_typed_value(new_parameters, "dsi_variance", defaults_config.dsi_variance, (int, float)),
            lines=get_typed_value(new_parameters, "lines", defaults_config.lines, list),
            cut_strategies=get_typed_value(new_parameters, "cut_strategies", defaults_config.cut_strategies, list),
            trim_dollar_value=get_typed_value(new_parameters, "trim_dollar_value", defaults_config.trim_dollar_value, (int, float))
        )
        
        return config
        
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        # If JSON is invalid or has wrong structure, fall back to defaults
        return get_defaults()


def validate_numeric_range(value, param_name: str, rules: dict) -> Tuple[bool, Optional[str]]:
    """Validate that a numeric parameter is within acceptable range.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter being validated
        rules: Validation rules dictionary containing 'type', 'min', 'max'
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passes, False otherwise
        - error_message: None if valid, error description if invalid
    """
    expected_type = rules["type"]
    min_value = rules["min"]
    max_value = rules["max"]
    description = rules["description"]
    
    # Check type
    if not isinstance(value, (int, float)):
        return False, f"{param_name} must be a number, got {type(value).__name__}"
    
    # Check if value is within range
    if value < min_value or value > max_value:
        return False, f"{param_name} ({description}) must be between {min_value} and {max_value}, got {value}"
    
    return True, None


def validate_bucket_tuple(bucket: Tuple[int, int], index: int) -> Tuple[bool, Optional[str]]:
    """Validate that a bucket tuple has min < max.
    
    Args:
        bucket: Tuple of (min, max) values
        index: Index of the bucket in the list (for error messages)
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if validation passes, False otherwise
        - error_message: None if valid, error description if invalid
    """
    if not isinstance(bucket, (tuple, list)) or len(bucket) != 2:
        return False, f"Bucket {index} must be a tuple/list of 2 values, got {bucket}"
    
    min_val, max_val = bucket
    
    # Check types
    if not isinstance(min_val, int) or not isinstance(max_val, int):
        return False, f"Bucket {index} values must be integers, got ({type(min_val).__name__}, {type(max_val).__name__})"
    
    # Check ordering
    if min_val >= max_val:
        return False, f"Bucket {index} minimum ({min_val}) must be less than maximum ({max_val})"
    
    return True, None


def validate_config(config: AppConfig) -> Tuple[bool, List[str]]:
    """Validate all configuration parameters.
    
    Validates numeric parameters against defined ranges, bucket tuple ordering,
    and other constraints defined in VALIDATION_RULES.
    
    Args:
        config: Configuration object to validate
    
    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if all validations pass, False otherwise
        - error_messages: List of error descriptions (empty if valid)
    """
    errors = []
    
    # Validate numeric parameters with defined rules
    numeric_params = [
        ("trim_cap", config.trim_cap),
        ("pieces_per_min", config.pieces_per_min),
        ("line_eff", config.line_eff),
        ("dsi_variance", config.dsi_variance),
        ("trim_dollar_value", config.trim_dollar_value)
    ]
    
    for param_name, value in numeric_params:
        if param_name in VALIDATION_RULES:
            is_valid, error_msg = validate_numeric_range(value, param_name, VALIDATION_RULES[param_name])
            if not is_valid:
                errors.append(error_msg)
    
    # Validate bucket tuples
    if not isinstance(config.buckets, list):
        errors.append("buckets must be a list")
    else:
        for i, bucket in enumerate(config.buckets):
            is_valid, error_msg = validate_bucket_tuple(bucket, i)
            if not is_valid:
                errors.append(error_msg)
    
    # Validate illegal_pairs structure
    if not isinstance(config.illegal_pairs, dict):
        errors.append("illegal_pairs must be a dictionary")
    else:
        for key, value in config.illegal_pairs.items():
            if not isinstance(key, str):
                errors.append(f"illegal_pairs key must be a string, got {type(key).__name__}")
            if not isinstance(value, list):
                errors.append(f"illegal_pairs[{key}] must be a list, got {type(value).__name__}")
            else:
                for item in value:
                    if not isinstance(item, str):
                        errors.append(f"illegal_pairs[{key}] must contain strings, got {type(item).__name__}")
    
    # Validate lists are non-empty where required
    if not isinstance(config.lines, list) or len(config.lines) == 0:
        errors.append("lines must be a non-empty list")
    
    if not isinstance(config.cut_strategies, list) or len(config.cut_strategies) == 0:
        errors.append("cut_strategies must be a non-empty list")
    
    # Return validation result
    is_valid = len(errors) == 0
    return is_valid, errors


def save_config(config: AppConfig) -> bool:
    """Save configuration to settings.json.
    
    Validates the configuration before saving. If validation fails, the configuration
    is not saved and the function returns False.
    
    Args:
        config: Configuration object to save
    
    Returns:
        bool: True if save successful, False otherwise
    
    Behavior:
        1. Validate config using validate_config()
        2. Convert AppConfig to JSON-serializable dictionary
        3. Write to settings.json with pretty formatting (indentation)
        4. Return success status
    """
    # Validate configuration before saving
    is_valid, errors = validate_config(config)
    if not is_valid:
        # Configuration is invalid, don't save
        return False
    
    try:
        # Convert AppConfig to JSON-serializable dictionary
        config_dict = {
            "version": "1.0",
            "buckets": config.buckets,
            "illegal_pairs": config.illegal_pairs,
            "trim_cap": config.trim_cap,
            "ui_parameters": {
                "pieces_per_min": config.pieces_per_min,
                "line_eff": config.line_eff
            },
            "new_parameters": {
                "dsi_variance": config.dsi_variance,
                "lines": config.lines,
                "cut_strategies": config.cut_strategies,
                "trim_dollar_value": config.trim_dollar_value
            }
        }
        
        # Write to settings.json with indentation for readability
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return True
        
    except (IOError, OSError, TypeError) as e:
        # Failed to save (permission denied, disk full, etc.)
        return False


def reset_to_defaults() -> bool:
    """Reset configuration to defaults and save.
    
    Gets the default configuration values and saves them to settings.json,
    effectively resetting all configuration parameters to their original values.
    
    Returns:
        bool: True if reset successful, False otherwise
    
    Behavior:
        1. Get defaults using get_defaults()
        2. Save defaults using save_config()
        3. Return success status
    """
    # Get default configuration
    defaults = get_defaults()
    
    # Save defaults to settings.json
    return save_config(defaults)
