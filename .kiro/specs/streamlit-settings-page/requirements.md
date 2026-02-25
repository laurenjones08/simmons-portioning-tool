# Requirements Document

## Introduction

This document specifies the requirements for a configurable settings page feature in a Streamlit-based portioning model application. The feature enables users to view, edit, and persist configuration parameters that control the application's behavior, replacing hardcoded values with a user-friendly interface.

## Glossary

- **Settings_Page**: The Streamlit UI component that displays and allows editing of configuration parameters
- **Config_Manager**: The system component responsible for loading, validating, and persisting configuration data
- **Parameter**: A configurable value that controls application behavior (e.g., trim_cap, time_limit_sec)
- **Settings_File**: The persistent storage file (JSON or YAML) containing configuration parameters
- **Validation_Rule**: A constraint that ensures parameter values are within acceptable ranges
- **Default_Values**: The original hardcoded values used when no custom settings exist

## Requirements

### Requirement 1: Configuration Parameter Management

**User Story:** As a user, I want to view all configuration parameters in one place, so that I can understand what settings control the application behavior.

#### Acceptance Criteria

1. WHEN a user opens the Settings_Page, THE Settings_Page SHALL display all existing configuration parameters from config.py
2. WHEN displaying parameters, THE Settings_Page SHALL show the parameter name, current value, and description
3. THE Settings_Page SHALL display parameters organized by category (Buckets, Illegal Pairs, Defaults, UI Parameters, New Parameters)
4. WHEN a parameter has a default value, THE Settings_Page SHALL indicate the default value alongside the current value

### Requirement 2: Parameter Editing

**User Story:** As a user, I want to edit configuration parameters through the UI, so that I can customize the application behavior without modifying code files.

#### Acceptance Criteria

1. WHEN a user modifies a numeric parameter, THE Settings_Page SHALL provide appropriate input controls (number input, slider, or text input)
2. WHEN a user modifies a list parameter (BUCKETS), THE Settings_Page SHALL provide controls to add, remove, and edit list items
3. WHEN a user modifies a dictionary parameter (ILLEGAL_PAIRS), THE Settings_Page SHALL provide controls to add, remove, and edit key-value pairs
4. WHEN a user modifies a boolean parameter, THE Settings_Page SHALL provide a checkbox or toggle control
5. WHEN a user edits a parameter, THE Settings_Page SHALL update the UI to reflect the pending change before saving

### Requirement 3: New Parameter Addition

**User Story:** As a user, I want to add new configuration parameters (DSI variance, lines, cut strategies, trim dollar value), so that I can extend the application's configurability.

#### Acceptance Criteria

1. THE Settings_Page SHALL provide input controls for DSI variance as a numeric parameter
2. THE Settings_Page SHALL provide input controls for lines configuration as a list or structured parameter
3. THE Settings_Page SHALL provide input controls for cut strategies as a list or selection parameter
4. THE Settings_Page SHALL provide input controls to assign allowed and prefered cut strategies to lines. 
5. THE Settings_Page SHALL provide input controls for trim dollar value as a numeric parameter
6. WHEN new parameters are added, THE Config_Manager SHALL store them in the Settings_File

### Requirement 4: Input Validation

**User Story:** As a user, I want the system to validate my configuration changes, so that I don't accidentally enter invalid values that break the application.

#### Acceptance Criteria

1. WHEN a user enters a numeric parameter, THE Config_Manager SHALL validate that the value is within acceptable ranges
2. WHEN a user enters a bucket tuple, THE Config_Manager SHALL validate that the minimum value is less than the maximum value
3. WHEN a user enters a percentage parameter, THE Config_Manager SHALL validate that the value is between 0 and 100
4. WHEN a user enters an illegal pair rule, THE Config_Manager SHALL validate that both parts are valid part codes
5. IF validation fails, THEN THE Settings_Page SHALL display an error message and prevent saving
6. WHEN validation succeeds, THE Settings_Page SHALL provide visual feedback indicating the input is valid

### Requirement 5: Configuration Persistence

**User Story:** As a user, I want my configuration changes to be saved permanently, so that my settings persist across application restarts.

#### Acceptance Criteria

1. WHEN a user clicks a save button, THE Config_Manager SHALL write all configuration parameters to the Settings_File
2. THE Config_Manager SHALL use JSON or YAML format for the Settings_File
3. WHEN the application starts, THE Config_Manager SHALL load configuration from the Settings_File if it exists
4. IF the Settings_File does not exist, THEN THE Config_Manager SHALL use default values from config.py
5. WHEN configuration is saved, THE Settings_Page SHALL display a success message
6. IF saving fails, THEN THE Settings_Page SHALL display an error message with details

### Requirement 6: Default Value Reset

**User Story:** As a user, I want to reset configuration parameters to their default values, so that I can recover from misconfiguration or start fresh.

#### Acceptance Criteria

1. THE Settings_Page SHALL provide a reset button for individual parameters
2. THE Settings_Page SHALL provide a reset all button to restore all parameters to defaults
3. WHEN a user clicks reset for a parameter, THE Settings_Page SHALL restore that parameter to its default value
4. WHEN a user clicks reset all, THE Settings_Page SHALL restore all parameters to their default values
5. WHEN reset is triggered, THE Settings_Page SHALL require user confirmation before applying the reset
6. WHEN reset is confirmed, THE Config_Manager SHALL update the Settings_File with default values

### Requirement 7: Settings Page Navigation

**User Story:** As a user, I want to easily access the settings page from the main application, so that I can modify configuration without disrupting my workflow.

#### Acceptance Criteria

1. THE Main_Application SHALL provide a navigation mechanism to access the Settings_Page
2. WHEN using Streamlit multipage apps, THE Settings_Page SHALL appear as a separate page in the sidebar navigation
3. WHEN a user navigates to the Settings_Page, THE Main_Application SHALL preserve the current application state
4. WHEN a user navigates away from the Settings_Page, THE Main_Application SHALL apply any saved configuration changes

### Requirement 8: Configuration Change Reflection

**User Story:** As a user, I want configuration changes to take effect in the application, so that I can see the impact of my settings immediately or after restart.

#### Acceptance Criteria

1. WHEN configuration is saved, THE Config_Manager SHALL reload the configuration into memory
2. WHEN the application restarts, THE Config_Manager SHALL load configuration from the Settings_File
3. WHEN configuration changes affect UI controls, THE UI_Controls SHALL reflect the new values
4. WHEN configuration changes affect engine behavior, THE Engines SHALL use the new parameter values
5. THE Settings_Page SHALL indicate whether changes require an application restart to take effect

### Requirement 9: Backward Compatibility

**User Story:** As a developer, I want the settings feature to work with the existing application structure, so that existing functionality is not broken.

#### Acceptance Criteria

1. WHEN the Settings_File does not exist, THE Config_Manager SHALL use values from config.py
2. WHEN existing code references config.py constants, THE Config_Manager SHALL provide those values from the Settings_File if available
3. THE Config_Manager SHALL maintain the same data types and structures as config.py
4. WHEN new parameters are added, THE Config_Manager SHALL provide default values for backward compatibility
5. THE Settings_Page SHALL not modify the original config.py file

### Requirement 10: Settings File Format

**User Story:** As a user, I want configuration stored in an easily editable format, so that I can manually edit settings if needed.

#### Acceptance Criteria

1. THE Config_Manager SHALL store configuration in JSON or YAML format
2. THE Settings_File SHALL be human-readable and well-formatted
3. THE Settings_File SHALL include comments or metadata indicating parameter purposes
4. WHEN the Settings_File is manually edited, THE Config_Manager SHALL validate the file on load
5. IF the Settings_File is corrupted or invalid, THEN THE Config_Manager SHALL fall back to default values and log an error
