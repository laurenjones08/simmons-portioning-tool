"""
Buckets page — CRUD management for weight bucket definitions.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

import sys
import os

import streamlit as st
import pandas as pd

# Allow importing api_client from the parent directory when running as a page
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import (
    APIError,
    search_buckets,
    create_bucket,
    update_bucket,
    delete_bucket,
)

# ---------------------------------------------------------------------------
# Pure validation helper (extracted for property-based testing)
# ---------------------------------------------------------------------------

def validate_bucket_weights(min_w: float, target_w: float, max_w: float) -> str | None:
    """Return an error message string if invalid, or None if valid.

    A bucket is valid when minWeight < maxWeight and minWeight <= targetWeight <= maxWeight.
    """
    if min_w >= max_w:
        return f"minWeight ({min_w}) must be less than maxWeight ({max_w})."
    if target_w < min_w or target_w > max_w:
        return (
            f"targetWeight ({target_w}) must be between "
            f"minWeight ({min_w}) and maxWeight ({max_w})."
        )
    return None


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "buckets" not in st.session_state:
        st.session_state.buckets = []
    if "edit_bucket" not in st.session_state:
        st.session_state.edit_bucket = None  # dict of the bucket being edited


def _load_buckets() -> None:
    """Call POST /buckets/search and store results in session state."""
    try:
        st.session_state.buckets = search_buckets({})
    except APIError as e:
        if e.status_code == 0:
            st.error("Could not reach Enumeration API. Check your connection.")
        elif e.status_code == 409:
            st.error(e.detail)
        else:
            st.error(f"Failed to load buckets: {e.detail}")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Buckets", page_icon="🪣")
st.title("Buckets")

_init_state()

# Load on first render
if not st.session_state.buckets and st.session_state.edit_bucket is None:
    _load_buckets()

# Refresh button
if st.button("🔄 Refresh"):
    st.session_state.edit_bucket = None
    _load_buckets()

# ---------------------------------------------------------------------------
# Buckets table
# ---------------------------------------------------------------------------

buckets = st.session_state.buckets

if buckets:
    df = pd.DataFrame(
        [
            {
                "id": b.get("_id", ""),
                "minWeight": b.get("minWeight"),
                "targetWeight": b.get("targetWeight"),
                "maxWeight": b.get("maxWeight"),
            }
            for b in buckets
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("No buckets found.")

# ---------------------------------------------------------------------------
# Create form
# ---------------------------------------------------------------------------

st.subheader("Create Bucket")

with st.form("create_bucket_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        new_min = st.number_input("minWeight", value=0.0, step=0.1, key="create_min")
    with col2:
        new_target = st.number_input("targetWeight", value=0.5, step=0.1, key="create_target")
    with col3:
        new_max = st.number_input("maxWeight", value=1.0, step=0.1, key="create_max")
    submitted = st.form_submit_button("Create")

if submitted:
    err = validate_bucket_weights(new_min, new_target, new_max)
    if err:
        st.warning(err)
    else:
        try:
            create_bucket(
                {"minWeight": new_min, "targetWeight": new_target, "maxWeight": new_max}
            )
            st.success("Bucket created.")
            _load_buckets()
        except APIError as e:
            if e.status_code == 0:
                st.error("Could not reach Enumeration API. Check your connection.")
            elif e.status_code == 409:
                st.error(e.detail)
            else:
                st.error(f"Failed to create bucket: {e.detail}")

# ---------------------------------------------------------------------------
# Edit / Delete section
# ---------------------------------------------------------------------------

st.subheader("Edit / Delete Bucket")

if not buckets:
    st.caption("No buckets available to edit or delete.")
else:
    bucket_options = {b.get("_id", ""): b for b in buckets}
    selected_id = st.selectbox(
        "Select bucket",
        options=list(bucket_options.keys()),
        format_func=lambda bid: (
            f"{bid}  (min={bucket_options[bid].get('minWeight')}, "
            f"target={bucket_options[bid].get('targetWeight')}, "
            f"max={bucket_options[bid].get('maxWeight')})"
        ),
    )

    selected_bucket = bucket_options.get(selected_id, {})

    # Edit form
    with st.form("edit_bucket_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            edit_min = st.number_input(
                "minWeight",
                value=float(selected_bucket.get("minWeight", 0.0)),
                step=0.1,
                key="edit_min",
            )
        with col2:
            edit_target = st.number_input(
                "targetWeight",
                value=float(selected_bucket.get("targetWeight", 0.0)),
                step=0.1,
                key="edit_target",
            )
        with col3:
            edit_max = st.number_input(
                "maxWeight",
                value=float(selected_bucket.get("maxWeight", 1.0)),
                step=0.1,
                key="edit_max",
            )
        save_clicked = st.form_submit_button("Save Changes")

    if save_clicked:
        err = validate_bucket_weights(edit_min, edit_target, edit_max)
        if err:
            st.warning(err)
        else:
            try:
                update_bucket(
                    selected_id,
                    {"minWeight": edit_min, "targetWeight": edit_target, "maxWeight": edit_max},
                )
                st.success("Bucket updated.")
                _load_buckets()
            except APIError as e:
                if e.status_code == 0:
                    st.error("Could not reach Enumeration API. Check your connection.")
                elif e.status_code == 404:
                    st.warning("Resource not found.")
                elif e.status_code == 409:
                    st.error(e.detail)
                else:
                    st.error(f"Failed to update bucket: {e.detail}")

    # Delete button (outside form so it can show warnings from response)
    if st.button("🗑️ Delete Selected Bucket", type="secondary"):
        try:
            result = delete_bucket(selected_id)
            # Display any recomputation warning returned by the API
            warning_msg = result.get("warning") or result.get("message")
            if warning_msg:
                st.warning(warning_msg)
            else:
                st.success("Bucket deleted.")
            _load_buckets()
        except APIError as e:
            if e.status_code == 0:
                st.error("Could not reach Enumeration API. Check your connection.")
            elif e.status_code == 404:
                st.warning("Resource not found.")
            elif e.status_code == 409:
                st.error(e.detail)
            else:
                st.error(f"Failed to delete bucket: {e.detail}")
