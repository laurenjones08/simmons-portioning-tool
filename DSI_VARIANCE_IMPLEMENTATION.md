# DSI Variance Implementation

## Overview
Added a DSI variance parameter that specifies a percentage of machine variance. This parameter is used to reduce the bucket minimum (bmin) in the enumeration engine to account for production variability.

## Changes Made

### 1. Configuration (`portioning/config_manager.py`)
- DSI variance already exists in the configuration with default value of 0.05 (5%)
- No changes needed - parameter was already present

### 2. Engine Input (`portioning/engines/base.py`)
- Added `dsi_variance: float = 0.05` to `EngineInput` dataclass
- Default value is 5% (0.05 as decimal)

### 3. Enumeration Engine (`portioning/engines/enumeration_engine.py`)
- Added DSI variance extraction from input: `dsi_variance = float(getattr(inp, 'dsi_variance', 0.05))`
- Calculate adjusted bucket minimum: `bmin_adjusted = bmin * (1.0 - dsi_variance)`
- Replaced all `bmin` comparisons with `bmin_adjusted` in the enumeration logic
- Updated metadata to include both `bucket_min_adjusted` and `dsi_variance` for debugging

### 4. UI Controls (`portioning/ui.py`)
- Added `dsi_variance` field to `UiState` dataclass
- Added DSI variance input control in the sidebar under "Production Parameters"
- Input is displayed as percentage (0-100%) but stored as decimal (0.0-1.0)
- Help text: "Machine variance percentage - reduces bucket minimum by this amount"

### 5. Settings Page (`pages/settings.py`)
- Added DSI variance control to the UI Parameters section
- Displays as percentage with conversion to/from decimal
- Help text: "Machine variance percentage - reduces bucket minimum by this amount to account for production variability"

### 6. Main App (`app.py`)
- Added `dsi_variance=ui.dsi_variance` to the EngineInput construction

## How It Works

1. **User Input**: User sets DSI variance as a percentage (e.g., 5%)
2. **Conversion**: UI converts percentage to decimal (5% → 0.05)
3. **Adjustment**: Engine calculates adjusted minimum: `bmin_adjusted = bmin * (1 - 0.05) = bmin * 0.95`
4. **Application**: All bucket minimum checks use `bmin_adjusted` instead of `bmin`

## Example

If bucket is (390, 480) and DSI variance is 5%:
- Original bmin: 390g
- Adjusted bmin: 390 * 0.95 = 370.5g
- This allows combinations that produce 370.5g - 480g instead of 390g - 480g
- Accounts for machine variance that might produce slightly less than target

## Benefits

- Accounts for real-world production variability
- Allows more combinations to be considered valid
- Configurable per-run via UI or settings
- Stored in configuration for persistence across sessions
