"""Settings page for the portioning application.

This page provides a user interface for viewing and editing configuration parameters.
Users can modify settings, save changes, and reset parameters to default values.
"""

import streamlit as st
from portioning.config_manager import load_config, save_config, reset_to_defaults, AppConfig, validate_config


def initialize_session_state():
    """Initialize session state for tracking configuration changes.
    
    Session state is used to:
    - Store current configuration being edited
    - Track whether changes have been made
    - Store validation errors
    - Track save/reset status
    - Track whether a reload is needed (for restart notification)
    """
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "config_changed" not in st.session_state:
        st.session_state.config_changed = False

    if "validation_errors" not in st.session_state:
        st.session_state.validation_errors = []

    if "save_status" not in st.session_state:
        st.session_state.save_status = None

    if "reload_needed" not in st.session_state:
        st.session_state.reload_needed = False


def handle_save():
    """Handle save button click.
    
    Validates the current configuration and saves it to settings.json.
    Updates session state with success/error status.
    Reloads configuration into memory after successful save.
    """
    # Validate configuration
    is_valid, errors = validate_config(st.session_state.config)

    if not is_valid:
        st.session_state.validation_errors = errors
        st.session_state.save_status = "error"
        return

    # Save configuration
    success = save_config(st.session_state.config)

    if success:
        # Reload configuration into memory
        try:
            from portioning import config
            config.reload_config()
            st.session_state.reload_needed = False
        except Exception as e:
            # If reload fails, mark that a restart is needed
            st.session_state.reload_needed = True

        st.session_state.save_status = "success"
        st.session_state.config_changed = False
        st.session_state.validation_errors = []
    else:
        st.session_state.save_status = "error"
        st.session_state.validation_errors = ["Failed to save configuration file"]


def handle_reset_all():
    """Handle reset all button click.
    
    Resets all configuration parameters to their default values and saves.
    Updates session state with the default configuration.
    Reloads configuration into memory after successful reset.
    """
    success = reset_to_defaults()

    if success:
        # Reload configuration into memory
        try:
            from portioning import config
            config.reload_config()
            st.session_state.reload_needed = False
        except Exception as e:
            # If reload fails, mark that a restart is needed
            st.session_state.reload_needed = True

        st.session_state.config = load_config()
        st.session_state.config_changed = False
        st.session_state.save_status = "reset"
        st.session_state.validation_errors = []
    else:
        st.session_state.save_status = "error"
        st.session_state.validation_errors = ["Failed to reset configuration"]


def render_buckets_section():
    """Render UI for editing BUCKETS list.
    
    Creates UI controls for:
    - Display current buckets as editable rows
    - Add new bucket button
    - Remove bucket button for each row
    - Min/max number inputs for each bucket
    - Validation that min < max
    
    All values are stored in session state and marked as changed when modified.
    """
    config = st.session_state.config

    st.markdown("Define WIP weight ranges (in grams) for bucketing. Each bucket has a minimum and maximum value.")
    st.markdown("**Note:** Minimum value must be less than maximum value.")

    # Display existing buckets
    buckets_to_remove = []

    for idx, (min_val, max_val) in enumerate(config.buckets):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            new_min = st.number_input(
                f"Min (bucket {idx + 1})",
                min_value=0,
                max_value=10000,
                value=min_val,
                step=1,
                key=f"bucket_min_{idx}",
                help=f"Minimum value for bucket {idx + 1}"
            )

        with col2:
            new_max = st.number_input(
                f"Max (bucket {idx + 1})",
                min_value=0,
                max_value=10000,
                value=max_val,
                step=1,
                key=f"bucket_max_{idx}",
                help=f"Maximum value for bucket {idx + 1}"
            )

        with col3:
            # Show validation status
            if new_min >= new_max:
                st.error("❌ Min must be < Max")
            else:
                st.success("✅ Valid")

        with col4:
            if st.button("🗑️", key=f"remove_bucket_{idx}", help=f"Remove bucket {idx + 1}"):
                buckets_to_remove.append(idx)

        # Update bucket if values changed
        if (new_min, new_max) != (min_val, max_val):
            config.buckets[idx] = (new_min, new_max)
            st.session_state.config_changed = True

    # Remove buckets marked for deletion
    if buckets_to_remove:
        for idx in sorted(buckets_to_remove, reverse=True):
            config.buckets.pop(idx)
        st.session_state.config_changed = True
        st.rerun()

    st.divider()

    # Add new bucket button
    if st.button("➕ Add Bucket", use_container_width=True):
        # Add a new bucket with default values
        config.buckets.append((0, 100))
        st.session_state.config_changed = True
        st.rerun()


def render_illegal_pairs_section():
    """Render UI for editing ILLEGAL_PAIRS dictionary.
    
    Creates UI controls for:
    - Display current illegal pairs as editable rows
    - Add new pair button
    - Remove pair button for each row
    - Text input for part code
    - Multi-select for illegal partners
    
    All values are stored in session state and marked as changed when modified.
    """
    config = st.session_state.config

    st.markdown("Define part codes that cannot be paired together. Each part code can have multiple illegal partners.")
    st.markdown("**Note:** Part codes are case-sensitive.")

    # Display existing illegal pairs
    pairs_to_remove = []

    # Convert dict to list for easier manipulation
    illegal_pairs_list = list(config.illegal_pairs.items())

    for idx, (part_code, illegal_partners) in enumerate(illegal_pairs_list):
        col1, col2, col3 = st.columns([2, 4, 1])

        with col1:
            new_part_code = st.text_input(
                f"Part Code {idx + 1}",
                value=part_code,
                key=f"illegal_pair_code_{idx}",
                help=f"Part code for illegal pair {idx + 1}",
                placeholder="Enter part code (e.g., C, D, R)"
            )

        with col2:
            # Get all available part codes for multi-select
            # Include all existing part codes as options
            all_part_codes = sorted(set(config.illegal_pairs.keys()))

            # Create multi-select for illegal partners
            new_illegal_partners = st.multiselect(
                f"Illegal Partners {idx + 1}",
                options=all_part_codes,
                default=[p for p in illegal_partners if p in all_part_codes],
                key=f"illegal_partners_{idx}",
                help=f"Select part codes that cannot be paired with {new_part_code}",
                placeholder="Select illegal partners"
            )

        with col3:
            if st.button("🗑️", key=f"remove_illegal_pair_{idx}", help=f"Remove illegal pair {idx + 1}"):
                pairs_to_remove.append(part_code)

        # Update illegal pair if values changed
        if new_part_code != part_code or new_illegal_partners != illegal_partners:
            # Remove old entry if part code changed
            if new_part_code != part_code:
                if part_code in config.illegal_pairs:
                    del config.illegal_pairs[part_code]

            # Add/update new entry
            if new_part_code.strip():  # Only add if part code is not empty
                config.illegal_pairs[new_part_code] = new_illegal_partners

            st.session_state.config_changed = True

    # Remove pairs marked for deletion
    if pairs_to_remove:
        for part_code in pairs_to_remove:
            if part_code in config.illegal_pairs:
                del config.illegal_pairs[part_code]
        st.session_state.config_changed = True
        st.rerun()

    st.divider()

    # Add new illegal pair button
    if st.button("➕ Add Illegal Pair", use_container_width=True):
        # Add a new illegal pair with default values
        # Find a unique default part code
        new_code = "NEW"
        counter = 1
        while new_code in config.illegal_pairs:
            new_code = f"NEW{counter}"
            counter += 1

        config.illegal_pairs[new_code] = []
        st.session_state.config_changed = True
        st.rerun()


def render_portioning_section():
    """Render UI for editing new parameters.
    
    Creates UI controls for:
    - dsi_variance: number input (0.0-1.0)
    - lines: text area with comma-separated values
    - cut_strategies: text area with comma-separated values
    - trim_dollar_value: number input (0.0-100.0)
    
    All values are stored in session state and marked as changed when modified.
    """
    config = st.session_state.config

    # Trim cap slider
    st.markdown("#### Trim Cap")
    st.markdown("Default Maximum allowed trim percentage")
    trim_cap = st.slider(
        "Trim % allowed",
        min_value=0,
        max_value=40,
        value=config.trim_cap,
        step=1,
        key="trim_cap_slider",
        help="Default value for the maximum percentage of trim allowed in the optimization"
    )

    if trim_cap != config.trim_cap:
        st.session_state.config.trim_cap = trim_cap
        st.session_state.config_changed = True

    st.divider()

    # DSI variance number input
    st.markdown("#### DSI Variance")
    st.markdown("DSI variance tolerance (0.0 = no variance, 1.0 = 100% variance)")
    dsi_variance = st.number_input(
        "DSI variance",
        min_value=0.0,
        max_value=1.0,
        value=config.dsi_variance,
        step=0.01,
        format="%.2f",
        key="dsi_variance_input",
        help="Tolerance for DSI (Daily Shipping Index) variance"
    )

    if dsi_variance != config.dsi_variance:
        st.session_state.config.dsi_variance = dsi_variance
        st.session_state.config_changed = True

    st.divider()

    # Lines text area
    st.markdown("#### Production Lines")
    st.markdown("Enter production line identifiers (comma-separated)")
    lines_text = st.text_area(
        "Lines",
        value=", ".join(config.lines),
        key="lines_input",
        help="Enter production line identifiers separated by commas (e.g., Line1, Line2, Line3)",
        height=100
    )

    # Parse comma-separated lines
    lines_list = [line.strip() for line in lines_text.split(",") if line.strip()]
    if lines_list != config.lines:
        st.session_state.config.lines = lines_list
        st.session_state.config_changed = True

    st.divider()

    # Cut strategies text area
    st.markdown("#### Cut Strategies")
    st.markdown("Enter available cutting strategies (comma-separated)")
    cut_strategies_text = st.text_area(
        "Cut strategies",
        value=", ".join(config.cut_strategies),
        key="cut_strategies_input",
        help="Enter cutting strategy names separated by commas (e.g., Strategy1, Strategy2, Strategy3)",
        height=100
    )

    # Parse comma-separated cut strategies
    cut_strategies_list = [strategy.strip() for strategy in cut_strategies_text.split(",") if strategy.strip()]
    if cut_strategies_list != config.cut_strategies:
        st.session_state.config.cut_strategies = cut_strategies_list
        st.session_state.config_changed = True

    st.divider()

    # Trim dollar value number input
    st.markdown("#### Trim Dollar Value")
    st.markdown("Dollar value per unit of trim")
    trim_dollar_value = st.number_input(
        "Trim dollar value",
        min_value=0.0,
        max_value=100.0,
        value=config.trim_dollar_value,
        step=0.1,
        format="%.2f",
        key="trim_dollar_value_input",
        help="Dollar value assigned to each unit of trim material"
    )

    if trim_dollar_value != config.trim_dollar_value:
        st.session_state.config.trim_dollar_value = trim_dollar_value
        st.session_state.config_changed = True


def render_settings_page():
    """Main entry point for settings page.
    
    Renders the complete settings page UI including:
    - Page header and description
    - Action buttons (Save, Reset All)
    - Configuration sections organized in tabs
    - Status messages (success/error)
    """
    # Initialize session state
    initialize_session_state()

    # Page header
    st.title("⚙️ Application Settings")
    st.markdown("""
    Configure application parameters for the portioning model. Changes are saved to `settings.json` 
    and will persist across application restarts.
    """)

    # Action buttons at the top
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            handle_save()

    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            # Show confirmation dialog
            if st.session_state.get("confirm_reset", False):
                handle_reset_all()
                st.session_state.confirm_reset = False
            else:
                st.session_state.confirm_reset = True

    # Show confirmation message for reset
    if st.session_state.get("confirm_reset", False):
        st.warning("⚠️ Are you sure you want to reset all settings to defaults? Click 'Reset All' again to confirm.")

    # Display status messages
    if st.session_state.save_status == "success":
        st.success("✅ Configuration saved successfully!")
        if st.session_state.reload_needed:
            st.warning(
                "⚠️ Configuration reloaded. Some changes may require an application restart to take full effect.")
        st.session_state.save_status = None
    elif st.session_state.save_status == "reset":
        st.success("✅ Configuration reset to defaults!")
        if st.session_state.reload_needed:
            st.warning(
                "⚠️ Configuration reloaded. Some changes may require an application restart to take full effect.")
        st.session_state.save_status = None
    elif st.session_state.save_status == "error":
        st.error("❌ Error saving configuration:")
        for error in st.session_state.validation_errors:
            st.error(f"  • {error}")
        st.session_state.save_status = None

    # Show unsaved changes indicator
    if st.session_state.config_changed:
        st.info("ℹ️ You have unsaved changes")

    st.divider()

    # Configuration sections in tabs
    tab1, tab2, tab3 = st.tabs([
        "Buckets",
        "Illegal Pairs",
        "Portioning Parameters",
    ])

    with tab1:
        st.markdown("### Bucket Configuration")
        st.markdown("Define WIP weight ranges (in grams) for bucketing.")
        render_buckets_section()

    with tab2:
        st.markdown("### Illegal Pairs Configuration")
        st.markdown("Define part codes that cannot be paired together.")
        render_illegal_pairs_section()

    with tab3:
        st.markdown("### Portioning Parameters")
        st.markdown("Configure additional parameters related to portioning.")
        render_portioning_section()

    # Footer note about restart requirements
    st.divider()
    st.info("""
    **💡 Configuration Reload Information:**
    
    - Configuration changes are automatically reloaded into memory after saving
    - Most changes take effect immediately for new operations
    - Some changes (like bucket definitions or illegal pairs) may require restarting the application to fully apply to ongoing processes
    - If you experience unexpected behavior after changing settings, try restarting the application
    """)


# Main entry point
if __name__ == "__main__":
    render_settings_page()
else:
    # When imported as a Streamlit page
    render_settings_page()
