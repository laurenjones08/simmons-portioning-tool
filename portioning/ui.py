from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import re

import streamlit as st

from portioning.config import BUCKETS, DEFAULTS
from portioning.config_manager import load_config


# -----------------------------
# Bucket parsing / formatting
# -----------------------------
_BUCKET_RE = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*\)$")


def _format_bucket(bucket: Tuple[int, int]) -> str:
    """Format (min,max) bucket tuple into a stable UI label."""
    return f"({bucket[0]}, {bucket[1]})"


def _parse_bucket(label: str) -> Optional[Tuple[int, int]]:
    """
    Safely parse a bucket label like '(390, 480)' into (390, 480).

    We do NOT use eval() for safety and robustness.
    Returns None if the string is invalid.
    """
    m = _BUCKET_RE.match(label.strip())
    if not m:
        return None
    lo = int(m.group(1))
    hi = int(m.group(2))
    if lo >= hi:
        return None
    return (lo, hi)


# -----------------------------
# Sidebar state model
# -----------------------------
@dataclass(frozen=True)
class UiState:
    """
    Immutable view-model representing all sidebar inputs.

    Why this exists:
    - Keeps Streamlit UI code clean by collecting all inputs in one object
    - Makes it harder to "forget" to pass new controls into the engines
    - Provides a stable typed contract between UI and solver logic

    Fields
    ------
    bucket:
        WIP bucket (min_g, max_g) for enumeration engine.

    bird_size:
        "SB", "BB", or "ALL" for enumeration filtering.

    min_nuggets:
        Require at least this many nugget items in any returned combination.

    trim_cap:
        Maximum allowed Trim_% for a returned result.

    customer_constraint:
        "NONE", "RTL", "FDS" for enumeration must-include logic.

    plant:
        If set, restrict enumeration candidates to a single plant. None = all.

    pieces_per_min / line_eff:
        UI parameters for production calculations.
    
    dsi_variance:
        Machine variance as a decimal (0.0-1.0) - reduces bucket minimum.
    """
    bucket: Optional[Tuple[int, int]]
    bird_size: str
    min_nuggets: int
    trim_cap: float
    customer_constraint: str
    plant: Optional[str]


def sidebar_controls(plants: Optional[list[str]], excel_sheets: tuple[str, ...]) -> UiState:
    """
    Render Streamlit sidebar controls and return an immutable UiState.

    Parameters
    ----------
    plants:
        Optional list of plant codes discovered from the input file (e.g., ["FSP", "XYZ"]).
        If provided, we show a Plant dropdown for enumeration.
    excel_sheets:
        Sheet names discovered from an uploaded Excel file. Empty tuple for CSV inputs.

    Returns
    -------
    UiState
        Captured sidebar state for use by app.py and downstream engines.
    """
    # Load configuration from config_manager
    config = load_config()
    
    st.sidebar.header("Inputs")

    trim_cap = st.sidebar.slider(
        "Trim % allowed",
        min_value=0,
        max_value=40,
        value=DEFAULTS.trim_cap,
        step=1,
    )

    # -----------------------------
    # Defaults for all controls
    # -----------------------------
    bucket: Optional[Tuple[int, int]] = None
    bird_size = "ALL"
    customer_constraint = "NONE"
    plant: Optional[str] = None
    min_nuggets = 0
    pieces_per_min = config.pieces_per_min
    line_eff = config.line_eff

    # -----------------------------
    # Enumeration controls
    # -----------------------------
    st.sidebar.subheader("Enumeration settings")

    bucket_mode = st.sidebar.radio(
        "Bucket mode",
        ["Preset buckets", "Custom bucket"],
        horizontal=True,
        help="Choose from preset bucket list or define your own min/max (grams).",
    )

    if bucket_mode == "Preset buckets":
        bucket_strs = [_format_bucket(b) for b in BUCKETS]

        default_bucket = (390, 480)
        default_str = _format_bucket(default_bucket)
        default_idx = bucket_strs.index(default_str) if default_str in bucket_strs else 0

        bucket_sel = st.sidebar.selectbox("Bucket (WIP range)", bucket_strs, index=default_idx)
        bucket = _parse_bucket(bucket_sel)

        if bucket is None:
            st.sidebar.error("Selected bucket is invalid. Try another value.")

    else:
        c1, c2 = st.sidebar.columns(2)
        lo = c1.number_input("Bucket min (g)", min_value=0, max_value=2500, value=390, step=1)
        hi = c2.number_input("Bucket max (g)", min_value=1, max_value=3000, value=480, step=1)
        bucket = (int(lo), int(hi)) if lo < hi else None
        if bucket is None:
            st.sidebar.error("Bucket min must be less than bucket max.")

    bird_size = st.sidebar.radio("Bird size", ["SB", "BB", "ALL"], horizontal=True)

    customer_constraint = st.sidebar.selectbox("Customer constraint", ["NONE", "RTL", "FDS"])

    min_nuggets = st.sidebar.number_input(
        "Min nuggets per nugget SKU",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        key="min_nuggets",
        help="If a combination includes any NUGGET SKU, that SKU must be produced at least this many units.",
    )

    if plants:
        plant = st.sidebar.selectbox("Plant", ["ALL"] + plants)
        if plant == "ALL":
            plant = None

    # -----------------------------
    # Final state object
    # -----------------------------
    return UiState(
        bucket=bucket,
        bird_size=bird_size,
        min_nuggets=int(min_nuggets),
        trim_cap=float(trim_cap),
        customer_constraint=customer_constraint,
        plant=plant,
    )
