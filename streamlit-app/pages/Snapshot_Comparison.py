"""Snapshot Comparison — side-by-side mix snapshot comparison."""
import os
import sys

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui.theme import apply_theme
from ui.layout import render_sidebar, render_header
from api_client import search_mixes, search_mix_metrics


st.set_page_config(page_title="Snapshot Comparison", page_icon="🗂️", layout="wide")
apply_theme()
current = render_sidebar(selected="Snapshot Comparison")
render_header("Snapshot Comparison", "Compare two enumeration snapshots side-by-side")

st.markdown("---")

try:
    mixes = search_mixes({})
except Exception:
    mixes = []

mix_ids = [m.get("_id") for m in mixes]
left_id = st.selectbox("Left snapshot", options=[""] + mix_ids, key="left_snap")
right_id = st.selectbox("Right snapshot", options=[""] + mix_ids, key="right_snap")

if left_id and right_id:
    left_metrics = search_mix_metrics({"mixId": left_id}) or []
    right_metrics = search_mix_metrics({"mixId": right_id}) or []

    left_df = pd.DataFrame(left_metrics)
    right_df = pd.DataFrame(right_metrics)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Left: {left_id}**")
        if not left_df.empty:
            st.dataframe(left_df)
        else:
            st.info("No metrics for left snapshot.")
    with col2:
        st.markdown(f"**Right: {right_id}**")
        if not right_df.empty:
            st.dataframe(right_df)
        else:
            st.info("No metrics for right snapshot.")

    # simple comparison metrics
    def avg_upgrade(df: pd.DataFrame):
        if df.empty or "upgradePercentage" not in df.columns:
            return None
        return df["upgradePercentage"].dropna().mean()

    a_left = avg_upgrade(left_df)
    a_right = avg_upgrade(right_df)
    st.markdown("---")
    st.markdown(f"Avg Upgrade — Left: **{a_left:.2f}%** | Right: **{a_right:.2f}%**" if a_left is not None and a_right is not None else "Upgrade data not available for comparison.")
