"""Consolidated Scheduling Dashboard page."""
import os
import sys
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui.theme import apply_theme
from ui.layout import render_sidebar, render_header
from ui.services import get_recent_mixes
from api_client import list_jobs


st.set_page_config(page_title="Scheduling Dashboard", page_icon="🗓️", layout="wide")
apply_theme()
current = render_sidebar(selected="Scheduling Dashboard")
render_header("Scheduling Dashboard", "Scheduling Decision Support", plant=None, snapshot=None, last_run=None)

st.markdown("---")

mixes = get_recent_mixes(limit=20)
mix_options = {m.get("_id", f"mix-{i}"): m for i, m in enumerate(mixes)}

selected_mix_id = st.selectbox("Select enumeration snapshot (mixId)", options=[""] + list(mix_options.keys()))
selected_mix = mix_options.get(selected_mix_id)

col_left, col_right = st.columns([3, 1])
with col_left:
    st.subheader("Preview / Commit Controls")
    st.caption("Use Preview to run a fast local schedule. Commit should create a persisted scheduling job.")
    preview = st.button("🔍 Preview Schedule (local sandbox)")
    commit = st.button("🚀 Commit Full Scheduling Job")
    st.markdown("\n")
    if preview:
        if not selected_mix:
            st.warning("Select a mix to preview a schedule.")
        else:
            st.success("Running local preview...")
            # simple sample schedule
            sku_list = list(selected_mix.get("skus", {}).keys()) or ["sku-1","sku-2"]
            rows = []
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            for i, sku in enumerate(sku_list):
                start = now + timedelta(hours=i)
                rows.append({"lineId": f"LINE-{(i%3)+1}", "sku": sku, "start": start, "end": start + timedelta(hours=1), "units": 100 + i*10})
            df = pd.DataFrame(rows)
            st.session_state["sched_preview"] = df

    if commit:
        st.info("Commit path not yet wired — implement scheduling API call to persist runs.")

with col_right:
    st.subheader("Job Status")
    try:
        jobs = list_jobs() or []
        if not jobs:
            st.info("No scheduling/worker jobs found (worker API).")
        else:
            for j in sorted(jobs, key=lambda x: x.get("createdAt", ""), reverse=True)[:5]:
                st.markdown(f"- **{j.get('runId', j.get('jobId', 'job'))}** — {j.get('status', '')} \n  {j.get('createdAt', '')}")
    except Exception:
        st.error("Could not fetch jobs from Worker API.")

st.markdown("---")

df = st.session_state.get("sched_preview", pd.DataFrame())
left, right = st.columns(2)
with left:
    if df.empty:
        st.info("No preview schedule available.")
    else:
        df["start_str"] = df["start"].dt.strftime("%Y-%m-%d %H:%M")
        df["end_str"] = df["end"].dt.strftime("%Y-%m-%d %H:%M")
        st.markdown("**Gantt (table preview)**")
        st.dataframe(df[["lineId","sku","start_str","end_str","units"]], hide_index=True)
with right:
    if df.empty:
        st.info("No assignment matrix.")
    else:
        pivot = df.pivot_table(index="lineId", columns="sku", values="units", aggfunc="sum", fill_value=0)
        st.markdown("**Assignment Matrix**")
        st.dataframe(pivot)
