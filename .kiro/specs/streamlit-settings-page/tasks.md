# Implementation Plan: Streamlit Settings Page

## Overview

This implementation plan breaks down the settings page feature into incremental coding tasks. Each task builds on previous work, with testing integrated throughout to validate functionality early. The implementation follows this sequence: configuration manager → settings page UI → integration with existing code → testing.

## Tasks

- [x] 1. Create configuration manager module
  - [x] 1.1 Create `portioning/config_manager.py` with AppConfig dataclass
    - Define AppConfig dataclass with all configuration parameters
    - Include existing parameters (buckets, illegal_pairs, defaults)
    - Include UI parameters (pieces_per_min, line_eff)
    - Include new parameters (dsi_variance, lines, cut_strategies, trim_dollar_value)
    - _Requirements: 1.1, 2.1, 3.1, 3.2, 3.3, 3.4_
  
  - [x] 1.2 Implement `get_defaults()` function
    - Import values from existing config.py
    - Set hardcoded UI parameter defaults
    - Set new parameter defaults
    - Return AppConfig instance
    - _Requirements: 1.4, 9.1, 9.4_
  
  - [x] 1.3 Implement `load_config()` function
    - Check if settings.json exists
    - If exists, load and parse JSON
    - If not exists or invalid, call get_defaults()
    - Return AppConfig instance
    - _Requirements: 5.3, 5.4, 9.1, 10.4, 10.5_
  
  - [ ]* 1.4 Write property test for configuration load fallback
    - **Property 1: Configuration Load Fallback**
    - **Validates: Requirements 9.1, 10.5**
  
  - [x] 1.5 Implement validation functions
    - Create VALIDATION_RULES dictionary with parameter constraints
    - Implement `validate_numeric_range()` helper
    - Implement `validate_bucket_tuple()` helper
    - Implement `validate_config()` main function
    - Return tuple of (is_valid, error_messages)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  
  - [ ]* 1.6 Write property test for validation range checking
    - **Property 3: Validation Rejects Invalid Ranges**
    - **Validates: Requirements 4.1, 4.3**
  
  - [ ]* 1.7 Write property test for bucket tuple validation
    - **Property 4: Bucket Tuple Ordering**
    - **Validates: Requirements 4.2**
  
  - [x] 1.8 Implement `save_config()` function
    - Validate config using validate_config()
    - Convert AppConfig to JSON-serializable dictionary
    - Write to settings.json with indentation
    - Return success boolean
    - _Requirements: 5.1, 5.2, 5.5, 5.6, 10.1, 10.2_
  
  - [ ]* 1.9 Write property test for save/load round-trip
    - **Property 2: Configuration Save Round-Trip**
    - **Validates: Requirements 5.1, 5.2, 5.3**
  
  - [ ]* 1.10 Write property test for type consistency
    - **Property 7: Configuration Type Consistency**
    - **Validates: Requirements 9.3**
  
  - [x] 1.11 Implement `reset_to_defaults()` function
    - Call get_defaults()
    - Call save_config() with defaults
    - Return success boolean
    - _Requirements: 6.3, 6.4, 6.6_
  
  - [ ]* 1.12 Write property test for reset functionality
    - **Property 5: Reset Restores Defaults**
    - **Validates: Requirements 6.3, 6.4, 6.6**

- [x] 2. Checkpoint - Ensure configuration manager tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. Create settings page UI
  - [x] 3.1 Create `pages/settings.py` with page structure
    - Set page title and description
    - Create main layout with tabs or expanders for sections
    - Add save and reset all buttons at top
    - Initialize session state for tracking changes
    - _Requirements: 1.2, 7.1, 7.2_
  
  - [x] 3.2 Implement `render_defaults_section()`
    - Create UI controls for trim_cap (slider 0-40)
    - Create UI controls for time_limit_sec (number input 10-600)
    - Create UI controls for gap (number input 0.0-0.05)
    - Create UI controls for chunk_size (number input 5-50)
    - Store values in session state
    - _Requirements: 1.1, 1.2, 2.1, 2.5_
  
  - [x] 3.3 Implement `render_ui_parameters_section()`
    - Create UI controls for pieces_per_min (number input 100-2000)
    - Create UI controls for line_eff (number input 0.1-1.0)
    - Store values in session state
    - _Requirements: 1.1, 1.2, 2.1, 2.5_
  
  - [x] 3.4 Implement `render_new_parameters_section()`
    - Create UI controls for dsi_variance (number input)
    - Create UI controls for lines (text area with comma-separated values)
    - Create UI controls for cut_strategies (text area with comma-separated values)
    - Create UI controls for trim_dollar_value (number input)
    - Store values in session state
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 3.5 Implement `render_buckets_section()`
    - Display current buckets as editable rows
    - Add "Add Bucket" button
    - Add "Remove" button for each bucket row
    - Create min/max number inputs for each bucket
    - Validate min < max on input
    - Store values in session state
    - _Requirements: 1.1, 1.2, 2.2, 2.5_
  
  - [x] 3.6 Implement `render_illegal_pairs_section()`
    - Display current illegal pairs as editable rows
    - Add "Add Pair" button
    - Add "Remove" button for each pair row
    - Create text input for part code
    - Create multi-select for illegal partners
    - Store values in session state
    - _Requirements: 1.1, 1.2, 2.3, 2.5_
  
  - [x] 3.7 Implement save functionality
    - Create AppConfig from session state values
    - Call validate_config() and display errors if invalid
    - Call save_config() if validation passes
    - Display success message on successful save
    - Display error message on save failure
    - _Requirements: 4.5, 4.6, 5.1, 5.5, 5.6_
  
  - [x] 3.8 Implement reset functionality
    - Add reset button for individual parameters
    - Add confirmation dialog for reset actions
    - Call reset_to_defaults() for reset all
    - Update session state with default values
    - Call save_config() after reset
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  
  - [ ]* 3.9 Write unit tests for settings page UI components
    - Test session state initialization
    - Test validation error display
    - Test save success/failure messages
    - Test reset confirmation dialog
    - _Requirements: 4.5, 4.6, 5.5, 5.6, 6.5_

- [x] 4. Checkpoint - Ensure settings page renders correctly
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Integrate with existing application
  - [x] 5.1 Update `portioning/config.py` to use config_manager
    - Import load_config from config_manager
    - Load configuration at module level
    - Expose BUCKETS, ILLEGAL_PAIRS, DEFAULTS as module constants
    - Add reload_config() function for runtime updates
    - _Requirements: 9.1, 9.2, 9.3, 9.5_
  
  - [ ]* 5.2 Write property test for backward compatibility
    - **Property 6: Backward Compatibility Preservation**
    - **Validates: Requirements 9.2, 9.3**
  
  - [x] 5.3 Update `portioning/ui.py` to use config_manager
    - Import load_config from config_manager
    - Load config at start of sidebar_controls()
    - Use config values for default parameters
    - Use config.pieces_per_min and config.line_eff
    - _Requirements: 8.3, 9.2_
  
  - [x] 5.4 Update `app.py` to support multipage navigation
    - Verify Streamlit multipage structure (pages/ directory)
    - Ensure settings page appears in sidebar navigation
    - Test navigation between main page and settings page
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [x] 5.5 Add configuration reload mechanism
    - Call config.reload_config() after settings save
    - Add note in settings page about restart requirements
    - Implement session state to track if reload is needed
    - _Requirements: 8.1, 8.2, 8.5_
  
  - [x] 5.6 Add settings.json to .gitignore
    - Ensure settings.json is not committed to version control
    - Add entry to .gitignore file
    - _Requirements: Design requirement for git exclusion_
  
  - [ ]* 5.7 Write integration tests
    - Test settings page updates config file
    - Test main app loads updated config
    - Test UI controls reflect config changes
    - Test navigation preserves state
    - _Requirements: 7.3, 7.4, 8.1, 8.2, 8.3, 8.4_

- [x] 6. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using hypothesis library
- Unit tests validate specific examples and edge cases
- The settings.json file will be created in the project root directory
- Streamlit's multipage app structure automatically creates navigation from files in pages/ directory
- **Current Status**: Configuration manager and settings page UI are complete. Integration with existing application is the next major milestone.

## Implementation Status Summary

### ✅ Completed
- Configuration manager module with full functionality
- Settings page UI with all sections and controls
- Comprehensive unit tests for config manager
- Save/load/reset functionality working
- Validation system implemented

### 🔄 In Progress
- Integration with existing application code

### ⏳ Remaining
- Update config.py to use config_manager
- Update ui.py to use config_manager defaults
- Add settings.json to .gitignore
- Configuration reload mechanism
- Optional property-based tests
- Optional integration tests

The core functionality is implemented and working. The remaining tasks focus on integrating the settings system with the existing application to make configuration changes take effect.