"""
Cut Strategies page — CRUD management for manufacturing cut strategy configurations.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
"""

import sys
import os

import streamlit as st
import pandas as pd

# Allow importing api_client from the parent directory when running as a page
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import (
    APIError,
    search_cut_strategies,
    create_cut_strategy,
    update_cut_strategy,
    delete_cut_strategy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PART_CODES = ["D", "R", "M", "T", "V", "K", "S", "U", "C", "J", "W", "G"]
LINE_TYPE_OPTIONS = ["DB20", "DSI884", "DSI888"]

# ---------------------------------------------------------------------------
# Pure validation helper (extracted for property-based testing)
# ---------------------------------------------------------------------------

def validate_parts_unique(parts: list) -> str | None:
    """Return None if no duplicates, or an error string if duplicates exist."""
    seen = set()
    duplicates = set()
    for part in parts:
        if part in seen:
            duplicates.add(part)
        seen.add(part)
    if duplicates:
        return f"Parts list contains duplicates: {', '.join(sorted(duplicates))}."
    return None


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "strategies" not in st.session_state:
        st.session_state.strategies = []
    if "strategies_loaded" not in st.session_state:
        st.session_state.strategies_loaded = False


def _load_strategies() -> None:
    """Call POST /cut-strategies/search with empty body and store results."""
    try:
        st.session_state.strategies = search_cut_strategies({})
        st.session_state.strategies_loaded = True
    except APIError as e:
        if e.status_code == 0:
            st.error("Could not reach Enumeration API. Check your connection.")
        elif e.status_code == 409:
            st.error(e.detail)
        else:
            st.error(f"Failed to load cut strategies: {e.detail}")


def _handle_api_error(e: APIError, action: str) -> None:
    if e.status_code == 0:
        st.error("Could not reach Enumeration API. Check your connection.")
    elif e.status_code == 404:
        st.warning("Resource not found.")
    elif e.status_code == 409:
        st.error(e.detail)
    else:
        st.error(f"Failed to {action}: {e.detail}")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Cut Strategies", page_icon="✂️")
st.title("Cut Strategies")

_init_state()

# Load on first render
if not st.session_state.strategies_loaded:
    _load_strategies()

# Refresh button
if st.button("🔄 Refresh"):
    _load_strategies()

# ---------------------------------------------------------------------------
# Cut Strategies table
# ---------------------------------------------------------------------------

strategies = st.session_state.strategies

if strategies:
    df = pd.DataFrame(
        [
            {
                "name": s.get("name", ""),
                "lineType": s.get("lineType", s.get("mfgType", "")),
                "hasNugget": s.get("hasNugget", False),
                "beltSpeed": s.get("beltSpeed"),
                "parts": ", ".join(s.get("parts", [])),
            }
            for s in strategies
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("No cut strategies found.")

# ---------------------------------------------------------------------------
# Create form
# ---------------------------------------------------------------------------

st.subheader("Create Cut Strategy")

with st.form("create_strategy_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    with c1:
        new_name = st.text_input("Name *")
        new_description = st.text_input("Description")
        new_line_type = st.selectbox("lineType *", options=LINE_TYPE_OPTIONS)
    with c2:
        new_has_nugget = st.checkbox("hasNugget")
        new_belt_speed = st.number_input("beltSpeed *", value=0.0, step=0.1, min_value=0.0)
        new_parts = st.multiselect("Parts *", options=VALID_PART_CODES)
    create_submitted = st.form_submit_button("Create")

if create_submitted:
    if not new_name.strip():
        st.warning("Name is required.")
    else:
        err = validate_parts_unique(new_parts)
        if err:
            st.warning(err)
        else:
            payload = {
                "name": new_name.strip(),
                "description": new_description.strip(),
                "lineType": new_line_type,
                "hasNugget": new_has_nugget,
                "beltSpeed": new_belt_speed,
                "parts": new_parts,
            }
            try:
                create_cut_strategy(payload)
                st.success("Cut strategy created.")
                _load_strategies()
            except APIError as e:
                _handle_api_error(e, "create cut strategy")

# ---------------------------------------------------------------------------
# Edit / Delete section
# ---------------------------------------------------------------------------

st.subheader("Edit / Delete Cut Strategy")

if not strategies:
    st.caption("No cut strategies available to edit or delete.")
else:
    strategy_options = {s.get("_id", ""): s for s in strategies}
    selected_id = st.selectbox(
        "Select cut strategy",
        options=list(strategy_options.keys()),
        format_func=lambda sid: (
            f"{strategy_options[sid].get('name', sid)}  "
            f"(lineType={strategy_options[sid].get('lineType', strategy_options[sid].get('mfgType', ''))})"
        ),
    )

    sel = strategy_options.get(selected_id, {})

    # Edit form
    with st.form("edit_strategy_form"):
        ec1, ec2 = st.columns(2)
        with ec1:
            edit_name = st.text_input("Name *", value=sel.get("name", ""))
            edit_description = st.text_input("Description", value=sel.get("description", ""))
            current_line_type = sel.get("lineType", sel.get("mfgType", LINE_TYPE_OPTIONS[0]))
            line_type_index = LINE_TYPE_OPTIONS.index(current_line_type) if current_line_type in LINE_TYPE_OPTIONS else 0
            edit_line_type = st.selectbox("lineType *", options=LINE_TYPE_OPTIONS, index=line_type_index)
        with ec2:
            edit_has_nugget = st.checkbox("hasNugget", value=bool(sel.get("hasNugget", False)))
            edit_belt_speed = st.number_input(
                "beltSpeed *",
                value=float(sel.get("beltSpeed", 0.0)),
                step=0.1,
                min_value=0.0,
            )
            current_parts = sel.get("parts", [])
            edit_parts = st.multiselect(
                "Parts *", options=VALID_PART_CODES, default=current_parts
            )
        save_clicked = st.form_submit_button("Save Changes")

    if save_clicked:
        if not edit_name.strip():
            st.warning("Name is required.")
        else:
            err = validate_parts_unique(edit_parts)
            if err:
                st.warning(err)
            else:
                payload = {
                    "name": edit_name.strip(),
                    "description": edit_description.strip(),
                    "lineType": edit_line_type,
                    "hasNugget": edit_has_nugget,
                    "beltSpeed": edit_belt_speed,
                    "parts": edit_parts,
                }
                try:
                    update_cut_strategy(selected_id, payload)
                    st.success("Cut strategy updated.")
                    _load_strategies()
                except APIError as e:
                    _handle_api_error(e, "update cut strategy")

    # Delete button (outside form to allow displaying response details)
    if st.button("🗑️ Delete Selected Cut Strategy", type="secondary"):
        try:
            result = delete_cut_strategy(selected_id)
            deleted_mixes = result.get("deletedMixes", 0)
            deleted_metrics = result.get("deletedMetrics", 0)
            st.success(
                f"Cut strategy deleted. "
                f"Cascade deleted: {deleted_mixes} mix(es), {deleted_metrics} metric(s)."
            )
            _load_strategies()
        except APIError as e:
            _handle_api_error(e, "delete cut strategy")
