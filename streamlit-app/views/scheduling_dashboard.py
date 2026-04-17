"""Scheduling landing page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import list_scheduling_jobs
from views.scheduling_shared import completed_job, latest_job, sort_records_by_date, status_badge


def _go(page: str) -> None:
    try:
        st.query_params["page"] = page
    except Exception:
        st.session_state.ui_selected_page = page
        st.session_state.ui_sidebar_nav = page
    st.rerun()


def _recent_jobs_table(jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for job in sort_records_by_date(jobs, ("finishedAt", "updatedAt", "createdAt"))[:6]:
        rows.append(
            {
                "Run ID": job.get("runId", "—"),
                "Plant": job.get("plantId", "—"),
                "Status": job.get("status", "—"),
                "Plan Start": job.get("planStartDate", "—"),
                "Horizon": job.get("horizonDays", "—"),
                "Created": str(job.get("createdAt", "—"))[:19],
            }
        )
    return pd.DataFrame(rows)


def render():
    st.markdown(
        "<div class='simmons-card' style='margin-top:12px;margin-bottom:18px'>"
        "<div style='display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-start'>"
        "<div style='min-width:320px;flex:1'>"
        "<div style='font-size:30px;font-weight:800;color:#00264F'>Scheduling control center</div>"
        "<div class='simmons-small' style='margin-top:6px'>"
        "Choose between launching a new run and exploring the scheduling tables and metrics."
        "</div></div>"
        "<div style='min-width:320px;flex:1'>"
        "<div class='simmons-small'>"
        "The create view focuses on inputs, while the insights view focuses on cuts, demand, and utilization."
        "</div></div></div></div>",
        unsafe_allow_html=True,
    )

    try:
        jobs = list_scheduling_jobs() or []
    except Exception:
        jobs = []

    active_jobs = [job for job in jobs if str(job.get("status", "")).lower() in {"pending", "running"}]
    completed_jobs = [job for job in jobs if str(job.get("status", "")).lower() == "completed"]
    last_job = latest_job(jobs)
    last_completed = completed_job(jobs)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"<div class='simmons-card'><div class='simmons-small'>Total scheduling runs</div>"
            f"<div style='font-size:34px;font-weight:800;color:#0046AD'>{len(jobs)}</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div class='simmons-card'><div class='simmons-small'>Active jobs</div>"
            f"<div style='font-size:34px;font-weight:800;color:#0046AD'>{len(active_jobs)}</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"<div class='simmons-card'><div class='simmons-small'>Completed runs</div>"
            f"<div style='font-size:34px;font-weight:800;color:#0046AD'>{len(completed_jobs)}</div></div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "<div class='simmons-card' style='min-height:220px'>"
            "<div style='font-size:18px;font-weight:800;color:#00264F'>View 1: Create a schedule</div>"
            "<div class='simmons-small' style='margin-top:8px'>"
            "Choose the plant, pick the SKUs that belong to that plant, set the planning window, and submit the worker job."
            "</div>"
            "<div class='simmons-small' style='margin-top:12px'>"
            "This is the clean entry point for planners who just want to build and launch a run."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Open schedule builder", key="open_schedule_builder", type="primary", use_container_width=True):
            _go("Scheduling Create")

    with c2:
        st.markdown(
            "<div class='simmons-card' style='min-height:220px'>"
            "<div style='font-size:18px;font-weight:800;color:#00264F'>View 2: Explore scheduling data</div>"
            "<div class='simmons-small' style='margin-top:8px'>"
            "Inspect today's cut schedule, upcoming cuts, demand progress, line utilization, bucket usage, and the raw tables behind the scheduler."
            "</div>"
            "<div class='simmons-small' style='margin-top:12px'>"
            "This workspace is built for understanding what the schedule is doing and why."
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Open analytics workspace", key="open_schedule_insights", type="primary", use_container_width=True):
            _go("Scheduling Insights")

    bottom_left, bottom_right = st.columns([1.05, 0.95])
    with bottom_left:
        st.markdown("#### Latest scheduling job")
        if last_job:
            st.markdown(
                f"<div class='simmons-card'>"
                f"<div style='font-weight:700'>{last_job.get('runId', '—')}</div>"
                f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:6px'>"
                f"<div class='simmons-small'>Plant: <strong>{last_job.get('plantId', '—')}</strong></div>"
                f"<div class='simmons-small'>Plan start: <strong>{last_job.get('planStartDate', '—')}</strong></div>"
                f"<div>{status_badge(last_job.get('status', 'unknown'))}</div>"
                f"</div>"
                f"<div class='simmons-small' style='margin-top:6px'>Job ID: <code>{last_job.get('jobId') or last_job.get('_id', '—')}</code></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No scheduling jobs found yet.")

        st.markdown("#### Last completed run")
        if last_completed:
            st.markdown(
                f"<div class='simmons-card'>"
                f"<div style='font-weight:700'>{last_completed.get('runId', '—')}</div>"
                f"<div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:6px'>"
                f"<div class='simmons-small'>Plant: <strong>{last_completed.get('plantId', '—')}</strong></div>"
                f"<div class='simmons-small'>Finished: <strong>{str(last_completed.get('finishedAt', '—'))[:19]}</strong></div>"
                f"<div>{status_badge(last_completed.get('status', 'unknown'))}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No completed scheduling run is available yet.")

    with bottom_right:
        st.markdown("#### Recent runs")
        recent_df = _recent_jobs_table(jobs)
        if recent_df.empty:
            st.info("No recent jobs to display.")
        else:
            st.dataframe(recent_df, use_container_width=True, hide_index=True, height=320)

    st.caption("Use View 1 to launch a run and View 2 to understand how the scheduler is using demand, lines, and buckets.")
