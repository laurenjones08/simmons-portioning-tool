# Design Document: Streamlit Settings Page

## Overview

This design implements a configurable settings page for the Streamlit portioning model application. The solution uses Streamlit's multipage app architecture to add a dedicated settings page, a JSON-based configuration file for persistence, and a configuration manager module to handle loading, validation, and saving of parameters.

The design maintains backward compatibility with the existing `config.py` while allowing runtime configuration changes through a user-friendly interface.

Ensure that our json file never pushes to Git. I.E make sure our config json file is added to the gitignore.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Application                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐              ┌──────────────┐            │
│  │   app.py     │              │   pages/     │            │
│  │  (Main Page) │              │  settings.py │            │
│  └──────┬───────┘              └──────┬───────┘            │
│         │                              │                     │
│         │         ┌────────────────────┘                     │
│         │         │                                          │
│         ▼         ▼                                          │
│  ┌─────────────────────────────┐                            │
│  │   portioning/config_manager.py │                         │
│  │  - load_config()             │                            │
│  │  - save_config()             │                            │
│  │  - validate_config()         │                            │
│  │  - get_defaults()            │                            │
│  └─────────────┬───────────────┘                            │
│                │                                              │
└────────────────┼──────────────────────────────────────────┘
                 │
                 ▼
         ┌──────────────────┐
         │  settings.json   │
         │  (Configuration) │
         └──────────────────┘
```

### Component Interaction Flow

1. **Application Startup**: `config_manager.load_config()` loads settings from `settings.json` or falls back to `config.py` defaults
2. **Settings Page Access**: User navigates to settings page via Streamlit sidebar
3. **Parameter Editing**: User modifies parameters through UI controls
4. **Validation**: `config_manager.validate_config()` validates inputs before saving
5. **Persistence**: `config_manager.save_config()` writes validated config to `settings.json`
6. **Application Use**: Main app and engines use config values from `config_manager`

## Components and Interfaces

### 1. Configuration Manager Module (`portioning/config_manager.py`)

The configuration manager provides a centralized interface for configuration operations.

#### Data Structures

```python
@dataclass
class AppConfig:
    """Complete application configuration."""
    # Existing parameters from config.py
    buckets: List[Tuple[int, int]]
    illegal_pairs: Dict[str, List[str]]
    trim_cap: int
    time_limit_sec: int
    gap: float
    chunk_size: int
    
    # Existing UI parameters (currently hardcoded)
    pieces_per_min: float
    line_eff: float
    
    # New parameters
    dsi_variance: float
    lines: List[str]
    cut_strategies: List[str]
    trim_dollar_value: float
```

#### Core Functions

```python
def load_config() -> AppConfig:
    """
    Load configuration from settings.json or fall back to defaults.
    
    Returns:
        AppConfig: Loaded configuration object
    
    Behavior:
        1. Check if settings.json exists
        2. If exists, load and validate JSON
        3. If not exists or invalid, use defaults from config.py
        4. Return AppConfig instance
    """

def save_config(config: AppConfig) -> bool:
    """
    Save configuration to settings.json.
    
    Args:
        config: Configuration object to save
    
    Returns:
        bool: True if save successful, False otherwise
    
    Behavior:
        1. Validate config using validate_config()
        2. Convert AppConfig to JSON-serializable dict
        3. Write to settings.json with pretty formatting
        4. Return success status
    """

def validate_config(config: AppConfig) -> Tuple[bool, List[str]]:
    """
    Validate all configuration parameters.
    
    Args:
        config: Configuration object to validate
    
    Returns:
        Tuple of (is_valid, error_messages)
    
    Validation Rules:
        - Numeric parameters within acceptable ranges
        - Bucket tuples have min < max
        - Percentages between 0 and 100
        - Lists are non-empty where required
        - Dictionary keys/values are valid strings
    """

def get_defaults() -> AppConfig:
    """
    Get default configuration from config.py.
    
    Returns:
        AppConfig: Default configuration object
    
    Behavior:
        1. Import values from config.py
        2. Add hardcoded UI defaults
        3. Add new parameter defaults
        4. Return AppConfig instance
    """

def reset_to_defaults() -> bool:
    """
    Reset configuration to defaults and save.
    
    Returns:
        bool: True if reset successful
    
    Behavior:
        1. Get defaults using get_defaults()
        2. Save defaults using save_config()
        3. Return success status
    """
```

### 2. Settings Page (`pages/settings.py`)

The settings page provides the UI for viewing and editing configuration.

#### Page Structure

```python
def render_settings_page():
    """
    Main entry point for settings page.
    
    Layout:
        1. Page header and description
        2. Action buttons (Save, Reset All)
        3. Configuration sections (tabs or expanders)
        4. Status messages (success/error)
    """

def render_buckets_section(config: AppConfig):
    """
    Render UI for editing BUCKETS list.
    
    Controls:
        - Display current buckets as editable rows
        - Add new bucket button
        - Remove bucket button for each row
        - Min/max number inputs for each bucket
    """

def render_illegal_pairs_section(config: AppConfig):
    """
    Render UI for editing ILLEGAL_PAIRS dictionary.
    
    Controls:
        - Display current pairs as editable rows
        - Add new pair button
        - Remove pair button for each row
        - Text inputs for part codes
        - Multi-select for illegal partners
    """

def render_defaults_section(config: AppConfig):
    """
    Render UI for editing Defaults parameters.
    
    Controls:
        - trim_cap: slider (0-40)
        - time_limit_sec: number input (10-600)
        - gap: number input (0.0-0.05)
        - chunk_size: number input (5-50)
    """

def render_ui_parameters_section(config: AppConfig):
    """
    Render UI for editing UI parameters.
    
    Controls:
        - pieces_per_min: number input (100-2000)
        - line_eff: number input (0.1-1.0)
    """

def render_new_parameters_section(config: AppConfig):
    """
    Render UI for editing new parameters.
    
    Controls:
        - dsi_variance: number input
        - lines: text area or multi-select
        - cut_strategies: text area or multi-select
        - trim_dollar_value: number input
    """
```

### 3. Modified Config Module (`portioning/config.py`)

The existing config module will be updated to use the config manager.

```python
# Import from config_manager instead of defining constants
from portioning.config_manager import load_config

# Load configuration at module level
_config = load_config()

# Expose configuration as module-level constants for backward compatibility
BUCKETS = _config.buckets
ILLEGAL_PAIRS = _config.illegal_pairs
DEFAULTS = Defaults(
    trim_cap=_config.trim_cap,
    time_limit_sec=_config.time_limit_sec,
    gap=_config.gap,
    chunk_size=_config.chunk_size
)

def reload_config():
    """Reload configuration from file (called after settings save)."""
    global _config, BUCKETS, ILLEGAL_PAIRS, DEFAULTS
    _config = load_config()
    BUCKETS = _config.buckets
    ILLEGAL_PAIRS = _config.illegal_pairs
    DEFAULTS = Defaults(
        trim_cap=_config.trim_cap,
        time_limit_sec=_config.time_limit_sec,
        gap=_config.gap,
        chunk_size=_config.chunk_size
    )
```

### 4. Modified UI Module (`portioning/ui.py`)

The UI module will be updated to use config manager for default values.

```python
from portioning.config_manager import load_config

def sidebar_controls(plants: Optional[list[str]], excel_sheets: tuple[str, ...]) -> UiState:
    """
    Render Streamlit sidebar controls.
    
    Changes:
        - Load config using config_manager
        - Use config values for defaults instead of hardcoded values
        - Use config.pieces_per_min and config.line_eff
    """
    config = load_config()
    
    # Use config values for defaults
    trim_cap = st.sidebar.slider(
        "Trim % allowed",
        min_value=0,
        max_value=40,
        value=config.trim_cap,  # From config instead of DEFAULTS.trim_cap
        step=1,
    )
    
    # ... rest of controls using config values ...
```

## Data Models

### Configuration File Format (settings.json)

```json
{
  "version": "1.0",
  "buckets": [
    [0, 324],
    [325, 375],
    [376, 475],
    [476, 550],
    [551, 625],
    [626, 780],
    [390, 480],
    [481, 580]
  ],
  "illegal_pairs": {
    "C": ["D"],
    "D": ["C", "T"],
    "R": ["V"],
    "V": ["R"],
    "M": ["K"],
    "K": ["M"],
    "T": ["D"]
  },
  "defaults": {
    "trim_cap": 15,
    "time_limit_sec": 60,
    "gap": 0.002,
    "chunk_size": 20
  },
  "ui_parameters": {
    "pieces_per_min": 600.0,
    "line_eff": 0.85
  },
  "new_parameters": {
    "dsi_variance": 0.05,
    "lines": ["Line1", "Line2", "Line3"],
    "cut_strategies": ["Strategy1", "Strategy2"],
    "trim_dollar_value": 1.5
  }
}
```

### Validation Rules

```python
VALIDATION_RULES = {
    "trim_cap": {
        "type": int,
        "min": 0,
        "max": 100,
        "description": "Trim percentage cap"
    },
    "time_limit_sec": {
        "type": int,
        "min": 10,
        "max": 600,
        "description": "CBC time limit in seconds"
    },
    "gap": {
        "type": float,
        "min": 0.0,
        "max": 0.05,
        "description": "CBC relative gap"
    },
    "chunk_size": {
        "type": int,
        "min": 5,
        "max": 50,
        "description": "Chunk size for processing"
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
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Configuration Load Fallback
*For any* application startup, if settings.json does not exist or is invalid, then loading configuration should return default values from config.py without error.
**Validates: Requirements 9.1, 10.5**

### Property 2: Configuration Save Round-Trip
*For any* valid AppConfig object, saving then loading the configuration should produce an equivalent configuration object.
**Validates: Requirements 5.1, 5.2, 5.3**

### Property 3: Validation Rejects Invalid Ranges
*For any* numeric parameter with defined min/max ranges, validation should reject values outside those ranges and accept values within ranges.
**Validates: Requirements 4.1, 4.3**

### Property 4: Bucket Tuple Ordering
*For any* bucket tuple (min, max), validation should reject the tuple if min >= max and accept if min < max.
**Validates: Requirements 4.2**

### Property 5: Reset Restores Defaults
*For any* modified configuration, calling reset_to_defaults() then loading configuration should return values equal to get_defaults().
**Validates: Requirements 6.3, 6.4, 6.6**

### Property 6: Backward Compatibility Preservation
*For any* existing code that references config.py constants (BUCKETS, ILLEGAL_PAIRS, DEFAULTS), those constants should have the same values as the loaded configuration.
**Validates: Requirements 9.2, 9.3**

### Property 7: Configuration Type Consistency
*For any* parameter in AppConfig, the loaded value should have the same type as the default value for that parameter.
**Validates: Requirements 9.3**

### Property 8: JSON Format Human Readability
*For any* saved settings.json file, the file should be valid JSON with indentation and should be parseable by standard JSON tools.
**Validates: Requirements 10.1, 10.2**

## Error Handling

### Configuration Loading Errors

1. **File Not Found**: Fall back to defaults, log info message
2. **Invalid JSON**: Fall back to defaults, log error with details
3. **Schema Mismatch**: Use defaults for missing fields, log warning
4. **Type Errors**: Fall back to defaults, log error

### Configuration Saving Errors

1. **Permission Denied**: Display error message to user, keep current config in memory
2. **Disk Full**: Display error message, keep current config in memory
3. **Invalid Path**: Create directory if needed, display error if creation fails

### Validation Errors

1. **Out of Range**: Display specific error message with valid range
2. **Type Mismatch**: Display error message with expected type
3. **Invalid Structure**: Display error message with expected format
4. **Empty Required Field**: Display error message indicating field is required

### UI Error Handling

1. **Concurrent Edits**: Use Streamlit session state to track unsaved changes
2. **Navigation During Edit**: Warn user about unsaved changes
3. **Invalid Input**: Disable save button until validation passes
4. **Network Issues**: Not applicable (local file system)

## Testing Strategy

### Unit Testing

Unit tests will verify specific examples, edge cases, and error conditions for individual functions.

**Test Cases**:
- Load config when settings.json exists with valid data
- Load config when settings.json does not exist (fallback to defaults)
- Load config when settings.json has invalid JSON (fallback to defaults)
- Save config with valid data
- Save config with invalid data (should fail validation)
- Validate config with all valid parameters
- Validate config with out-of-range numeric values
- Validate config with invalid bucket tuples (min >= max)
- Reset to defaults and verify all values match get_defaults()
- Backward compatibility: verify config.py constants match loaded config

### Property-Based Testing

Property tests will verify universal properties across all inputs using a property-based testing library. Each test will run a minimum of 100 iterations with randomized inputs.

**Library**: Use `hypothesis` for Python property-based testing

**Property Test Cases**:

1. **Configuration Round-Trip Property**
   - Generate random valid AppConfig objects
   - Save then load each config
   - Verify loaded config equals original
   - **Feature: streamlit-settings-page, Property 2: Configuration Save Round-Trip**

2. **Validation Range Property**
   - Generate random numeric values (both valid and invalid)
   - For each parameter with min/max, test validation
   - Verify values in range pass, values out of range fail
   - **Feature: streamlit-settings-page, Property 3: Validation Rejects Invalid Ranges**

3. **Bucket Ordering Property**
   - Generate random bucket tuples with various min/max combinations
   - Verify validation rejects when min >= max
   - Verify validation accepts when min < max
   - **Feature: streamlit-settings-page, Property 4: Bucket Tuple Ordering**

4. **Reset Idempotence Property**
   - Generate random modified configs
   - Call reset_to_defaults() then load_config()
   - Verify result equals get_defaults()
   - **Feature: streamlit-settings-page, Property 5: Reset Restores Defaults**

5. **Type Consistency Property**
   - Generate random valid configs
   - Save and load each config
   - Verify each parameter type matches default type
   - **Feature: streamlit-settings-page, Property 7: Configuration Type Consistency**

### Integration Testing

Integration tests will verify the interaction between components:
- Settings page UI updates when config changes
- Main app reflects config changes after save
- UI controls use correct default values from config
- Navigation between pages preserves state

### Manual Testing Checklist

- [ ] Open settings page from main app
- [ ] Edit each parameter type (numeric, list, dict, boolean)
- [ ] Save changes and verify success message
- [ ] Restart app and verify changes persisted
- [ ] Reset individual parameter and verify default restored
- [ ] Reset all parameters and verify all defaults restored
- [ ] Enter invalid values and verify validation errors
- [ ] Navigate away with unsaved changes and verify warning
- [ ] Manually edit settings.json and verify app loads correctly
- [ ] Delete settings.json and verify app uses defaults
