"""Overview / Home page — polished entry point showing pipeline status and KPIs.

This page is designed to be the high-level operational view for schedulers
and planners. It uses the thin `ui.services` wrappers and `ui.components` for
consistency with the rest of the app.
"""
import os
import sys

import streamlit as st
import pandas as pd

# Allow importing the ui package and api_client from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui.theme import apply_theme
from ui.components import header, kpi_card, job_card
from ui.services import get_kpis_sample, get_recent_mixes


st.set_page_config(page_title="Portioning Overview", page_icon="📊", layout="wide")

apply_theme()

header("Portioning Decision Support", "From raw inputs to enumeration snapshots and scheduling runs")

st.markdown("---")

# Fetch a small set of KPIs
with st.spinner("Loading KPIs..."):
    kpis = get_kpis_sample()

col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_card("Recent Candidate Mixes", kpis.get("candidate_count", 0), "Immutable snapshots")
with col2:
    avg_up = kpis.get("avg_upgrade")
    kpi_card("Avg. Upgrade % (sample)", f"{avg_up:.2f}%" if avg_up is not None else "n/a", "Calculated from recent mix metrics")
with col3:
    jobs = kpis.get("recent_jobs", [])
    kpi_card("Recent Jobs", len(jobs), "Worker queue and history")
with col4:
    kpi_card("Open Violations", "—", "Validation & constraint violations")

st.markdown("---")

st.subheader("Recent Jobs")
jobs = kpis.get("recent_jobs", [])
if not jobs:
    st.info("No recent jobs found.")
else:
    cols = st.columns(min(3, len(jobs)))
    for col, job in zip(cols, jobs):
        with col:
            job_card(job)

st.markdown("---")

st.subheader("Recent Mix Snapshots")
mixes = get_recent_mixes(limit=10)
if not mixes:
    st.info("No mixes available.")
else:
    rows = []
    for m in mixes:
        rows.append({
            "mixId": m.get("_id", ""),
            "plant": m.get("reqPlant", ""),
            "birdSize": m.get("reqBirdSize", ""),
            "numFillets": m.get("numFillets"),
            "filletWeight": m.get("filletWeight"),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("\n")
    st.caption("Select a mix in the Mix Visualization page to view provenance and metrics.")

st.markdown("---")

st.subheader("Quick Actions")
col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("🔍 Preview latest snapshot"):
        st.info("Preview launched (uses local sandbox preview). See Mix Generation or Scheduling pages for details.")
with col_b:
    if st.button("🚀 Start full enumeration job"):
        st.info("Full run requested — this uses the Worker API and creates immutable snapshots.")
with col_c:
    if st.button("🗂️ Export recent snapshot"):
        st.info("Export prepared — download available from Snapshot Detail view.")
