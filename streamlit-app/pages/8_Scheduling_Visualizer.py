"""Scheduling Visualizer — Gantt-style timeline, assignment matrix, and job controls.

This page provides a scaffolded visualizer that consumes enumeration snapshots
and demonstrates a preview (sandbox) schedule using lightweight logic.

It intentionally does not modify backend artifacts; the Preview button runs a
local non-persistent scheduler for quick demos. The Commit path should be
implemented later by calling the scheduling API to create a persisted run.
"""
import os
import sys
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui.theme import apply_theme
from ui.components import header
from ui.services import get_recent_mixes
from ui.state import set as ui_set, get as ui_get
from api_client import list_jobs

# Attempt to import shared scheduling models for validation (optional)
try:
    from scheduling_shared.models.scheduling_output import SchedulingOutputCreate
    from pydantic import ValidationError
    _HAS_SCHED_MODEL = True
except Exception:
    SchedulingOutputCreate = None  # type: ignore
    ValidationError = Exception  # type: ignore
    _HAS_SCHED_MODEL = False


def _make_sample_schedule(mix: dict) -> pd.DataFrame:
    """Create a simple demo schedule DataFrame from a mix snapshot.

    Columns: lineId, sku, start, end, units
    """
    # Simple deterministic pseudo-scheduler: distribute SKUs across 3 lines
    skus_map = mix.get("skus", {})
    sku_list = list(skus_map.keys()) or ["sample-sku-1", "sample-sku-2"]
    lines = ["LINE-A", "LINE-B", "LINE-C"]
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    rows = []
    for i, sku in enumerate(sku_list):
        for j in range(2):  # two batches per SKU for demo
            line = lines[(i + j) % len(lines)]
            start = now + timedelta(hours=(i * 2 + j))
            end = start + timedelta(hours=1)
            units = 100 + i * 10 + j * 5
            rows.append({"lineId": line, "sku": sku, "start": start, "end": end, "units": units})

    df = pd.DataFrame(rows)
    return df


def _render_gantt(df: pd.DataFrame) -> None:
    """Render a simple textual Gantt-style chart using DataFrame transformations.

    This is a lightweight placeholder using `st.dataframe` and colored labels.
    A richer implementation can use Plotly/Altair for true Gantt visuals.
    """
    if df.empty:
        st.info("No schedule to display.")
        return

    # Normalize times and show a table with start/end and duration
    df = df.copy()
    df["start_str"] = df["start"].dt.strftime("%Y-%m-%d %H:%M")
    df["end_str"] = df["end"].dt.strftime("%Y-%m-%d %H:%M")
    df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60

    st.markdown("**Gantt (table preview)**")
    st.dataframe(df[["lineId", "sku", "start_str", "end_str", "duration_min", "units"]], hide_index=True)


def _render_assignment_matrix(df: pd.DataFrame) -> None:
    st.markdown("**Assignment Matrix (lines × SKUs)**")
    if df.empty:
        st.info("No assignments.")
        return
    pivot = df.pivot_table(index="lineId", columns="sku", values="units", aggfunc="sum", fill_value=0)
    st.dataframe(pivot, hide_index=False)


st.set_page_config(page_title="Scheduling Visualizer", page_icon="🗓️", layout="wide")
apply_theme()

header("Scheduling Visualizer", "Preview and inspect schedule assignments derived from enumeration snapshots")

st.markdown("---")

# Select a recent mix snapshot
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
            schedule_df = _make_sample_schedule(selected_mix)
            # Validate each schedule row against scheduling_shared model when available
            validation_rows = []
            if _HAS_SCHED_MODEL:
                decision_id = f"{selected_mix_id}-preview"
                for _, row in schedule_df.iterrows():
                    payload = {
                        "decisionId": decision_id,
                        "skuId": row["sku"],
                        "lbsProduced": float(row["units"]),
                        "contractLbs": float(row["units"]),
                        "date": row["start"].date(),
                    }
                    try:
                        SchedulingOutputCreate(**payload)
                        validation_rows.append({"ok": True, "error": None, **payload})
                    except ValidationError as err:
                        validation_rows.append({"ok": False, "error": str(err), **payload})
            else:
                for _, row in schedule_df.iterrows():
                    validation_rows.append({"ok": True, "error": None, "skuId": row["sku"], "lbsProduced": float(row["units"]), "contractLbs": float(row["units"]), "date": row["start"].date(),})

            st.session_state["ui_last_preview_schedule"] = schedule_df
            st.session_state["ui_last_preview_validation"] = validation_rows

    if commit:
        st.info("Commit path not yet wired — implement scheduling API call to persist runs.")

with col_right:
    st.subheader("Job Status")
    try:
        jobs = list_jobs() or []
        if not jobs:
            st.info("No scheduling/worker jobs found (worker API).")
        else:
            # show recent 5 jobs
            for j in sorted(jobs, key=lambda x: x.get("createdAt", ""), reverse=True)[:5]:
                st.markdown(f"- **{j.get('runId', j.get('jobId', 'job'))}** — {j.get('status', '')} \n  {j.get('createdAt', '')}")
    except Exception:
        st.error("Could not fetch jobs from Worker API.")

st.markdown("---")

schedule_df = st.session_state.get("ui_last_preview_schedule", pd.DataFrame())
validation_rows = st.session_state.get("ui_last_preview_validation", [])

left, right = st.columns(2)
with left:
    _render_gantt(schedule_df)
with right:
    _render_assignment_matrix(schedule_df)
    if validation_rows:
        st.markdown("**Preview Validation**")
        vr_df = pd.DataFrame(validation_rows)
        # show ok/error summary
        ok_count = int(vr_df[vr_df["ok"] == True].shape[0]) if not vr_df.empty else 0
        total = vr_df.shape[0]
        st.markdown(f"Validation: **{ok_count}/{total}** rows OK")
        # show errors if any
        errs = vr_df[vr_df["ok"] == False]
        if not errs.empty:
            st.markdown("**Errors**")
            for idx, e in errs.iterrows():
                st.error(f"Row {idx}: {e.get('error')}")

st.markdown("---")
st.subheader("Notes & Next Steps")
st.markdown(
    "- Preview runs are local and non-persistent; wire the scheduling API to commit.\n"
    "- Replace the table-based Gantt with a Plotly/Altair Gantt for richer interactions.\n"
    "- Add manual edit controls: drag-drop assignments, then persist an edited run as a child artifact."
)
