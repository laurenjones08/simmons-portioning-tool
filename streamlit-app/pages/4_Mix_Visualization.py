"""
Mix Visualization page — browse and filter enumeration mix results.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import sys
import os

import streamlit as st
import pandas as pd

# Allow importing api_client from the parent directory when running as a page
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import APIError, search_mixes, search_cut_strategies

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIRD_SIZE_OPTIONS = ["", "SB", "BB", "ALL"]
LINE_TYPE_OPTIONS = ["", "DB20", "DSI884", "DSI888"]

# Tri-state options for boolean filters
TRISTATE_OPTIONS = ["(not filtered)", "True", "False"]


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "mixes" not in st.session_state:
        st.session_state.mixes = None  # None = not yet searched
    if "mix_cut_strategies" not in st.session_state:
        st.session_state.mix_cut_strategies = []
    if "mix_cut_strategies_loaded" not in st.session_state:
        st.session_state.mix_cut_strategies_loaded = False


def _handle_api_error(e: APIError) -> None:
    if e.status_code == 0:
        st.error("Could not reach Enumeration API. Check your connection.")
    else:
        st.error(e.detail)


def _load_cut_strategies() -> None:
    try:
        st.session_state.mix_cut_strategies = search_cut_strategies({})
        st.session_state.mix_cut_strategies_loaded = True
    except APIError as e:
        _handle_api_error(e)
        st.session_state.mix_cut_strategies = []


def _format_cut_strategy_label(strategy: dict) -> str:
    name = strategy.get("name", "")
    line_type = strategy.get("lineType", strategy.get("mfgType", ""))
    parts = ", ".join(strategy.get("parts", []))

    label_parts = [part for part in [name, line_type] if part]
    label = " - ".join(label_parts) if label_parts else "Cut Strategy"
    if parts:
        label = f"{label} [{parts}]"
    return label


def _strategy_options(line_type: str | None = None) -> tuple[list[str], dict[str, str]]:
    strategies = st.session_state.mix_cut_strategies
    if line_type:
        strategies = [
            strategy
            for strategy in strategies
            if strategy.get("lineType", strategy.get("mfgType")) == line_type
        ]

    option_ids = [strategy.get("_id", "") for strategy in strategies if strategy.get("_id")]
    labels = {
        strategy.get("_id", ""): _format_cut_strategy_label(strategy)
        for strategy in strategies
        if strategy.get("_id")
    }
    return option_ids, labels


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Mix Visualization", page_icon="🔀")
st.title("Mix Visualization")

_init_state()
if not st.session_state.mix_cut_strategies_loaded:
    _load_cut_strategies()

# ---------------------------------------------------------------------------
# Filter panel
# ---------------------------------------------------------------------------

st.subheader("Filters")

if not st.session_state.mix_cut_strategies:
    st.warning("No cut strategies are currently available from the Enumeration API. The cutStrategyID filter will stay unfiltered.")

with st.form("mix_search_form"):
    col1, col2 = st.columns(2)
    cut_strategy_ids, cut_strategy_labels = _strategy_options()

    with col1:
        req_plant = st.text_input("reqPlant", placeholder="e.g. PLANT1")
        req_bird_size = st.selectbox("reqBirdSize", options=BIRD_SIZE_OPTIONS)
        mfg_type = st.selectbox("lineType", options=LINE_TYPE_OPTIONS)
        filtered_strategy_ids, filtered_strategy_labels = _strategy_options(mfg_type or None)
        strategy_ids_to_show = filtered_strategy_ids or cut_strategy_ids
        strategy_labels_to_show = filtered_strategy_labels or cut_strategy_labels
        cut_strategy_id = st.selectbox(
            "cutStrategyID",
            options=[""] + strategy_ids_to_show,
            format_func=lambda strategy_id: "(not filtered)" if not strategy_id else strategy_labels_to_show.get(strategy_id, strategy_id),
        )

    with col2:
        sku_trade_number = st.text_input("skuTradeNumber", placeholder="e.g. 12345")
        includes_fds_str = st.selectbox("includesFDS", options=TRISTATE_OPTIONS)
        includes_rtl_str = st.selectbox("includesRTL", options=TRISTATE_OPTIONS)
        includes_nug_str = st.selectbox("includesNug", options=TRISTATE_OPTIONS)

    search_clicked = st.form_submit_button("🔍 Search")


def _tristate_to_bool(value: str):
    """Convert tri-state string to bool or None."""
    if value == "True":
        return True
    if value == "False":
        return False
    return None


if search_clicked:
    criteria = {}

    if req_plant.strip():
        criteria["reqPlant"] = req_plant.strip()
    if req_bird_size:
        criteria["reqBirdSize"] = req_bird_size
    if mfg_type:
        criteria["mfgType"] = mfg_type
    if cut_strategy_id:
        criteria["cutStrategyID"] = cut_strategy_id
    if sku_trade_number.strip():
        criteria["skuTradeNumber"] = sku_trade_number.strip()

    fds_val = _tristate_to_bool(includes_fds_str)
    rtl_val = _tristate_to_bool(includes_rtl_str)
    nug_val = _tristate_to_bool(includes_nug_str)

    if fds_val is not None:
        criteria["includesFDS"] = fds_val
    if rtl_val is not None:
        criteria["includesRTL"] = rtl_val
    if nug_val is not None:
        criteria["includesNug"] = nug_val

    try:
        st.session_state.mixes = search_mixes(criteria)
    except APIError as e:
        _handle_api_error(e)
        st.session_state.mixes = []

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

mixes = st.session_state.mixes

if mixes is None:
    st.info("Apply filters and click Search to view mixes.")
elif len(mixes) == 0:
    st.warning("No mixes found for the given filters.")
else:
    st.subheader(f"Results ({len(mixes)} mix{'es' if len(mixes) != 1 else ''})")

    rows = []
    for m in mixes:
        sku_trade_numbers = ", ".join(m.get("skus", {}).keys())
        rows.append({
            "Mix ID": m.get("_id", ""),
            "reqPlant": m.get("reqPlant", ""),
            "reqBirdSize": m.get("reqBirdSize", ""),
            "mfgType": m.get("mfgType", ""),
            "numFillets": m.get("numFillets"),
            "filletWeight": m.get("filletWeight"),
            "includesFDS": m.get("includesFDS"),
            "includesRTL": m.get("includesRTL"),
            "includesNug": m.get("includesNug"),
            "SKU Trade Numbers": sku_trade_numbers,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    # ---------------------------------------------------------------------------
    # Row detail — select a mix by ID
    # ---------------------------------------------------------------------------

    st.subheader("Mix Detail")

    mix_ids = [m.get("_id", "") for m in mixes]
    selected_mix_id = st.selectbox("Select a mix to view full detail", options=mix_ids)

    selected_mix = next((m for m in mixes if m.get("_id") == selected_mix_id), None)

    if selected_mix:
        with st.expander("Full Mix Detail", expanded=True):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**cutStrategyID**")
                st.write(selected_mix.get("cutStrategyID", "N/A"))

                st.markdown("**beltSpeed**")
                st.write(selected_mix.get("beltSpeed", "N/A"))

                st.markdown("**nuggetTargetWeight**")
                nug_weight = selected_mix.get("nuggetTargetWeight")
                st.write(nug_weight if nug_weight is not None else "N/A")

            with detail_col2:
                st.markdown("**SKUs Map** (tradeNumber → partCode)")
                skus = selected_mix.get("skus", {})
                if skus:
                    skus_df = pd.DataFrame(
                        [{"tradeNumber": k, "partCode": v} for k, v in skus.items()]
                    )
                    st.dataframe(skus_df, width="stretch", hide_index=True)
                else:
                    st.write("No SKUs")
