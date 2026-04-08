"""
Mix Generation page — job submission, monitoring, and cancellation.

Requirements: 7.1–7.6, 8.1–8.6, 9.1–9.4
"""

import sys
import os

import streamlit as st
import pandas as pd

# Allow importing api_client from the parent directory when running as a page
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import APIError, list_jobs, get_job, submit_job, cancel_job


# ---------------------------------------------------------------------------
# Pure validation helpers (extracted for property-based testing)
# ---------------------------------------------------------------------------

def validate_max_combination_size(n: int) -> str | None:
    """Return None if 1 <= n <= 4, else an error string."""
    if 1 <= n <= 4:
        return None
    return "maxCombinationSize must be between 1 and 4 inclusive."


def validate_batch_size(n: int) -> str | None:
    """Return None if n >= 1, else an error string."""
    if n >= 1:
        return None
    return "batchSize must be a positive integer (>= 1)."


def warn_if_no_filters(plant_filter: str | None, bird_size_filter: str | None) -> str | None:
    """Return a warning string if both filters are absent/empty, else None."""
    plant_present = bool(plant_filter and plant_filter.strip())
    bird_present = bool(bird_size_filter and bird_size_filter.strip())
    if not plant_present and not bird_present:
        return (
            "Neither plantFilter nor birdSizeFilter is set. "
            "Submitting without filters may produce a very large or failed job."
        )
    return None


def cancel_button_visible(status: str) -> bool:
    """Return True if the job status allows cancellation."""
    return status in {"pending", "running"}


def _format_job_option_label(job: dict) -> str:
    """Build a friendly dropdown label from runId and submitted date."""
    run_id = job.get("runId") or "Unnamed job"
    created_at = job.get("createdAt") or ""
    submitted_date = created_at[:10] if isinstance(created_at, str) and created_at else "unknown date"
    return f"{run_id} - submitted {submitted_date}"


def _stage_label(stage_num: int | str) -> str:
    """Translate a 1-4 stage number into a user-friendly combination label."""
    labels = {
        1: "single-SKU combinations",
        2: "two-SKU combinations",
        3: "three-SKU combinations",
        4: "four-SKU combinations",
    }
    try:
        return labels[int(stage_num)]
    except (TypeError, ValueError, KeyError):
        return "combination stage"


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "jobs" not in st.session_state:
        st.session_state.jobs = None  # None = not yet loaded
    if "selected_job_detail" not in st.session_state:
        st.session_state.selected_job_detail = None


def _handle_api_error(e: APIError, service: str = "Worker API") -> None:
    if e.status_code == 0:
        st.error(f"Could not reach {service}. Check your connection.")
    elif e.status_code == 409:
        st.warning(e.detail)
    elif e.status_code == 404:
        st.warning("Job not found or already in terminal state.")
    else:
        st.error(e.detail)


def _load_jobs() -> None:
    try:
        st.session_state.jobs = list_jobs()
        st.session_state.selected_job_detail = None
    except APIError as e:
        _handle_api_error(e)
        st.session_state.jobs = []


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Mix Generation", page_icon="⚙️")
st.title("Mix Generation")

_init_state()

# Load jobs on first page visit
if st.session_state.jobs is None:
    _load_jobs()

# ---------------------------------------------------------------------------
# Section 1 — Job Submission Form
# ---------------------------------------------------------------------------

st.subheader("Submit New Job")

with st.form("job_submission_form"):
    run_id = st.text_input("runId *", placeholder="e.g. run-2024-01")
    plant_filter = st.text_input("plantFilter (optional)", placeholder="e.g. PLANT1")
    bird_size_filter = st.text_input("birdSizeFilter (optional)", placeholder="e.g. SB")
    max_combination_size = st.number_input(
        "Largest combination size to enumerate (1–4)", min_value=1, max_value=4, value=4, step=1
    )
    st.caption("Stage 1 = single-SKU combos, Stage 2 = pairs, Stage 3 = triples, Stage 4 = quadruples.")
    batch_size = st.number_input(
        "batchSize (≥ 1)", min_value=1, value=1000, step=1
    )
    submit_clicked = st.form_submit_button("🚀 Submit Job")

if submit_clicked:
    # Client-side validation
    max_combo_err = validate_max_combination_size(int(max_combination_size))
    batch_err = validate_batch_size(int(batch_size))

    if max_combo_err:
        st.warning(max_combo_err)
    elif batch_err:
        st.warning(batch_err)
    else:
        # Warn if no filters provided
        filter_warning = warn_if_no_filters(plant_filter, bird_size_filter)
        if filter_warning:
            st.warning(filter_warning)

        payload: dict = {
            "runId": run_id.strip(),
            "maxCombinationSize": int(max_combination_size),
            "batchSize": int(batch_size),
        }
        if plant_filter.strip():
            payload["plantFilter"] = plant_filter.strip()
        if bird_size_filter.strip():
            payload["birdSizeFilter"] = bird_size_filter.strip()

        try:
            result = submit_job(payload)
            st.success(f"Job submitted. Status: **{result.get('status', 'unknown')}** — jobId: `{result.get('jobId', '')}`")
            # Refresh job list after submission
            _load_jobs()
        except APIError as e:
            if e.status_code == 409:
                st.warning("Only one job can run at a time. " + e.detail)
            elif e.status_code == 0:
                st.error("Could not reach Worker API. Check your connection.")
            else:
                st.error(e.detail)

# ---------------------------------------------------------------------------
# Section 2 — Job Monitoring
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Job List")

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh"):
        _load_jobs()

jobs = st.session_state.jobs

if not jobs:
    st.info("No jobs found.")
else:
    # Build display dataframe
    rows = []
    for j in jobs:
        rows.append({
            "jobId": j.get("jobId", ""),
            "runId": j.get("runId", ""),
            "status": j.get("status", ""),
            "createdAt": j.get("createdAt", ""),
            "skuCount": j.get("skuCount"),
            "plantFilter": j.get("plantFilter", ""),
            "birdSizeFilter": j.get("birdSizeFilter", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    # Job selection for detail view
    job_options = {j.get("jobId", ""): j for j in jobs if j.get("jobId")}
    job_ids = list(job_options.keys())
    selected_job_id = st.selectbox(
        "Select a job to view details",
        options=[""] + job_ids,
        format_func=lambda job_id: "Select a job..." if not job_id else _format_job_option_label(job_options[job_id]),
    )

    if selected_job_id:
        try:
            job_detail = get_job(selected_job_id)
            st.session_state.selected_job_detail = job_detail
        except APIError as e:
            _handle_api_error(e)
            st.session_state.selected_job_detail = None

    # ---------------------------------------------------------------------------
    # Section 2b — Selected Job Detail
    # ---------------------------------------------------------------------------

    detail = st.session_state.selected_job_detail
    if detail:
        st.subheader(f"Job Detail — `{detail.get('jobId', '')}`")

        status = detail.get("status", "")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**runId:** {detail.get('runId', 'N/A')}")
            st.markdown(f"**status:** {status}")
            st.markdown(f"**createdAt:** {detail.get('createdAt', 'N/A')}")
            st.markdown(f"**maxCombinationSize:** {detail.get('maxCombinationSize', 'N/A')}")
            st.markdown(f"**batchSize:** {detail.get('batchSize', 'N/A')}")
        with col2:
            st.markdown(f"**plantFilter:** {detail.get('plantFilter') or '—'}")
            st.markdown(f"**birdSizeFilter:** {detail.get('birdSizeFilter') or '—'}")
            st.markdown(f"**skuCount:** {detail.get('skuCount', 'N/A')}")
            if detail.get("startedAt"):
                st.markdown(f"**startedAt:** {detail['startedAt']}")
            if detail.get("updatedAt"):
                st.markdown(f"**updatedAt:** {detail['updatedAt']}")

        # Status-specific display
        if status == "running":
            stages = detail.get("stages", [])
            if stages:
                st.markdown("**Stage Progress:**")
                for s in stages:
                    stage_num = s.get("stage", "?")
                    stage_status = s.get("status", "")
                    processed = s.get("processedCombinations", 0)
                    total = s.get("totalCombinations", 0)
                    pct = int(processed / total * 100) if total > 0 else 0
                    st.markdown(
                        f"Stage {stage_num} - {_stage_label(stage_num)} "
                        f"({stage_status}): {processed}/{total} combinations"
                    )
                    st.progress(pct / 100)
            else:
                st.info("Job is running — no stage data yet.")

        elif status == "failed":
            error_msg = detail.get("errorMessage")
            if error_msg:
                st.error(f"Error: {error_msg}")
            else:
                st.error("Job failed with no error message.")

        elif status == "completed":
            st.success(
                f"Job completed at **{detail.get('finishedAt', 'N/A')}** "
                f"with **{detail.get('skuCount', 0)}** SKUs."
            )

        # ---------------------------------------------------------------------------
        # Section 3 — Job Cancellation
        # ---------------------------------------------------------------------------

        if cancel_button_visible(status):
            st.divider()
            if st.button(f"🛑 Cancel Job `{detail.get('jobId', '')}`", type="primary"):
                try:
                    cancel_job(detail["jobId"])
                    st.success("Cancellation requested. Refreshing job status...")
                    # Refresh job list and re-fetch detail
                    _load_jobs()
                    try:
                        st.session_state.selected_job_detail = get_job(detail["jobId"])
                    except APIError:
                        pass
                    st.rerun()
                except APIError as e:
                    if e.status_code == 404:
                        st.warning("Job not found or already in terminal state.")
                    elif e.status_code == 0:
                        st.error("Could not reach Worker API. Check your connection.")
                    else:
                        st.error(e.detail)
