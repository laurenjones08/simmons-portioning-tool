"""
Global Config page — view, single-edit, and batch-edit configuration parameters.

Requirements: 10.1–10.4, 11.1–11.5, 12.1–12.5
"""

import sys
import os

import streamlit as st
import pandas as pd

# Allow importing api_client from the parent directory when running as a page
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import APIError, get_all_configs, update_config, batch_update_configs


# ---------------------------------------------------------------------------
# Pure helper functions (extracted for property-based testing)
# ---------------------------------------------------------------------------

def get_input_widget_type(value_type: str) -> str:
    """Return the widget type string for a given config valueType.

    Returns one of: "number_int", "number_float", "text", "checkbox".
    """
    mapping = {
        "int": "number_int",
        "float": "number_float",
        "string": "text",
        "bool": "checkbox",
    }
    return mapping.get(value_type, "text")


def validate_config_bounds(value, min_value, max_value) -> str | None:
    """Return None if value is within [min_value, max_value], else an error string.

    Only enforces bounds when the respective bound is not None.
    """
    if min_value is not None and value < min_value:
        return f"Value must be ≥ {min_value}."
    if max_value is not None and value > max_value:
        return f"Value must be ≤ {max_value}."
    return None


def group_configs_by_prefix(configs: list[dict]) -> dict[str, list[dict]]:
    """Group configs by the portion of the key before the first dot.

    Keys without a dot are grouped under their full key name.
    """
    groups: dict[str, list[dict]] = {}
    for cfg in configs:
        key = cfg.get("key", "")
        prefix = key.split(".")[0] if "." in key else key
        groups.setdefault(prefix, []).append(cfg)
    return groups


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "configs" not in st.session_state:
        st.session_state.configs = None  # None = not yet loaded
    if "batch_edits" not in st.session_state:
        st.session_state.batch_edits = {}  # key -> new value


def _handle_api_error(e: APIError, service: str = "Config API") -> None:
    if e.status_code == 0:
        st.error(f"Could not reach {service}. Check your connection.")
    elif e.status_code == 422:
        st.error(e.detail)
    else:
        st.error(e.detail)


def _load_configs() -> None:
    try:
        st.session_state.configs = get_all_configs()
    except APIError as e:
        _handle_api_error(e)
        st.session_state.configs = []


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Global Config", page_icon="⚙️")
st.title("Global Config")

_init_state()

# Load configs on first page visit
if st.session_state.configs is None:
    _load_configs()

# ---------------------------------------------------------------------------
# Toolbar — Refresh + mode toggle
# ---------------------------------------------------------------------------

col_refresh, col_mode, _ = st.columns([1, 2, 4])
with col_refresh:
    if st.button("🔄 Refresh"):
        _load_configs()
        st.session_state.batch_edits = {}

with col_mode:
    edit_mode = st.radio(
        "Edit mode",
        options=["Single edit", "Batch edit"],
        horizontal=True,
        label_visibility="collapsed",
    )

configs: list[dict] = st.session_state.configs or []

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

if not configs:
    st.info("No configuration parameters defined.")
    st.stop()

# ---------------------------------------------------------------------------
# Config table — grouped by key prefix
# ---------------------------------------------------------------------------

st.subheader("Configuration Parameters")

groups = group_configs_by_prefix(configs)

for prefix, group_configs in groups.items():
    with st.expander(f"**{prefix}** ({len(group_configs)} parameter{'s' if len(group_configs) != 1 else ''})", expanded=True):
        rows = []
        for cfg in group_configs:
            rows.append({
                "key": cfg.get("key", ""),
                "value": cfg.get("value"),
                "valueType": cfg.get("valueType", ""),
                "description": cfg.get("description", ""),
                "minValue": cfg.get("minValue"),
                "maxValue": cfg.get("maxValue"),
                "updatedAt": cfg.get("updatedAt", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Single-edit mode
# ---------------------------------------------------------------------------

if edit_mode == "Single edit":
    st.subheader("Edit Parameter")

    config_keys = [cfg.get("key", "") for cfg in configs]
    selected_key = st.selectbox("Select parameter to edit", options=[""] + config_keys)

    if selected_key:
        # Find the selected config
        selected_cfg = next((c for c in configs if c.get("key") == selected_key), None)

        if selected_cfg:
            value_type = selected_cfg.get("valueType", "string")
            current_value = selected_cfg.get("value")
            min_val = selected_cfg.get("minValue")
            max_val = selected_cfg.get("maxValue")
            description = selected_cfg.get("description", "")

            if description:
                st.caption(description)

            # Build bounds hint
            bounds_parts = []
            if min_val is not None:
                bounds_parts.append(f"min: {min_val}")
            if max_val is not None:
                bounds_parts.append(f"max: {max_val}")
            bounds_hint = f"Bounds: {', '.join(bounds_parts)}" if bounds_parts else ""

            widget_type = get_input_widget_type(value_type)

            with st.form("single_edit_form"):
                if widget_type == "number_int":
                    kwargs = {"label": f"Value ({value_type})", "value": int(current_value) if current_value is not None else 0, "step": 1}
                    if min_val is not None:
                        kwargs["min_value"] = int(min_val)
                    if max_val is not None:
                        kwargs["max_value"] = int(max_val)
                    new_value = st.number_input(**kwargs)
                    if bounds_hint:
                        st.caption(bounds_hint)

                elif widget_type == "number_float":
                    kwargs = {"label": f"Value ({value_type})", "value": float(current_value) if current_value is not None else 0.0, "step": 0.01, "format": "%.6f"}
                    if min_val is not None:
                        kwargs["min_value"] = float(min_val)
                    if max_val is not None:
                        kwargs["max_value"] = float(max_val)
                    new_value = st.number_input(**kwargs)
                    if bounds_hint:
                        st.caption(bounds_hint)

                elif widget_type == "checkbox":
                    new_value = st.checkbox(f"Value ({value_type})", value=bool(current_value))

                else:  # text
                    new_value = st.text_input(f"Value ({value_type})", value=str(current_value) if current_value is not None else "")

                submitted = st.form_submit_button("💾 Save")

            if submitted:
                # Client-side bounds check for numeric types
                if widget_type in ("number_int", "number_float"):
                    bounds_err = validate_config_bounds(new_value, min_val, max_val)
                    if bounds_err:
                        st.warning(bounds_err)
                    else:
                        try:
                            update_config(selected_key, {"value": new_value})
                            st.success(f"✅ `{selected_key}` updated successfully.")
                            _load_configs()
                        except APIError as e:
                            _handle_api_error(e)
                else:
                    try:
                        update_config(selected_key, {"value": new_value})
                        st.success(f"✅ `{selected_key}` updated successfully.")
                        _load_configs()
                    except APIError as e:
                        _handle_api_error(e)

# ---------------------------------------------------------------------------
# Batch-edit mode
# ---------------------------------------------------------------------------

else:
    st.subheader("Batch Edit")
    st.caption("Modify multiple values below, then validate or submit all changes at once.")

    batch_edits: dict = st.session_state.batch_edits

    with st.form("batch_edit_form"):
        for cfg in configs:
            key = cfg.get("key", "")
            value_type = cfg.get("valueType", "string")
            current_value = cfg.get("value")
            min_val = cfg.get("minValue")
            max_val = cfg.get("maxValue")
            description = cfg.get("description", "")

            # Use buffered value if already edited, else current API value
            buffered_value = batch_edits.get(key, current_value)

            widget_type = get_input_widget_type(value_type)

            bounds_parts = []
            if min_val is not None:
                bounds_parts.append(f"min: {min_val}")
            if max_val is not None:
                bounds_parts.append(f"max: {max_val}")
            bounds_hint = f" ({', '.join(bounds_parts)})" if bounds_parts else ""

            label = f"{key}{bounds_hint}"
            help_text = description or None

            if widget_type == "number_int":
                kwargs = {"label": label, "value": int(buffered_value) if buffered_value is not None else 0, "step": 1, "key": f"batch_{key}", "help": help_text}
                if min_val is not None:
                    kwargs["min_value"] = int(min_val)
                if max_val is not None:
                    kwargs["max_value"] = int(max_val)
                st.number_input(**kwargs)

            elif widget_type == "number_float":
                kwargs = {"label": label, "value": float(buffered_value) if buffered_value is not None else 0.0, "step": 0.01, "format": "%.6f", "key": f"batch_{key}", "help": help_text}
                if min_val is not None:
                    kwargs["min_value"] = float(min_val)
                if max_val is not None:
                    kwargs["max_value"] = float(max_val)
                st.number_input(**kwargs)

            elif widget_type == "checkbox":
                st.checkbox(label, value=bool(buffered_value), key=f"batch_{key}", help=help_text)

            else:
                st.text_input(label, value=str(buffered_value) if buffered_value is not None else "", key=f"batch_{key}", help=help_text)

        col_validate, col_submit = st.columns(2)
        with col_validate:
            validate_clicked = st.form_submit_button("🔍 Validate Only")
        with col_submit:
            submit_clicked = st.form_submit_button("💾 Submit All Changes")

    def _collect_batch_payload() -> list[dict]:
        """Collect all modified values from session state into a batch payload."""
        payload = []
        for cfg in configs:
            key = cfg.get("key", "")
            widget_key = f"batch_{key}"
            if widget_key in st.session_state:
                new_val = st.session_state[widget_key]
                payload.append({"key": key, "value": new_val})
        return payload

    def _display_batch_result(result: dict, validate_only: bool) -> None:
        """Display batch update summary."""
        total = result.get("total", 0)
        successful = result.get("successful", 0)
        failed = result.get("failed", 0)
        errors = result.get("errors", {})

        action = "Validation" if validate_only else "Batch update"

        if failed == 0:
            st.success(f"✅ {action} complete — {successful}/{total} succeeded.")
        else:
            st.warning(f"⚠️ {action} complete — {successful}/{total} succeeded, {failed} failed.")

        if errors:
            st.markdown("**Per-key errors:**")
            for err_key, err_detail in errors.items():
                st.error(f"`{err_key}`: {err_detail}")

    if validate_clicked:
        payload = _collect_batch_payload()
        if not payload:
            st.info("No changes to validate.")
        else:
            try:
                result = batch_update_configs(payload, validate_only=True)
                _display_batch_result(result, validate_only=True)
            except APIError as e:
                _handle_api_error(e)

    if submit_clicked:
        payload = _collect_batch_payload()
        if not payload:
            st.info("No changes to submit.")
        else:
            try:
                result = batch_update_configs(payload, validate_only=False)
                _display_batch_result(result, validate_only=False)
                # Refresh config list after successful batch update
                _load_configs()
                st.session_state.batch_edits = {}
            except APIError as e:
                _handle_api_error(e)
