"""Scheduling analytics view."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from api_client import (
    APIError,
    get_all_configs,
    list_lines,
    list_scheduling_jobs,
    search_available_wip,
    search_bucket_usage,
    search_monthly_contract_demands_bulk,
    search_scheduling_decisions,
    search_scheduling_outputs,
    search_sku_demands,
)
from views.scheduling_shared import (
    build_bucket_usage_summary,
    build_demand_progress,
    build_line_utilization,
    completed_job,
    filter_date_range,
    focus_window,
    format_date,
    format_timestamp,
    job_label,
    latest_job,
    parse_date,
    safe_float,
    sort_records_by_date,
    status_badge,
    table_or_empty,
)


def _handle_api_error(error: APIError, action: str) -> None:
    if error.status_code == 0:
        st.error("Could not reach the Scheduling APIs. Check your connection.")
    elif error.status_code == 404:
        st.warning("Resource not found.")
    else:
        st.error(f"Failed to {action}: {error.detail}")


@st.cache_data(show_spinner=False, ttl=60)
def _load_jobs() -> list[dict]:
    try:
        jobs = list_scheduling_jobs() or []
        return sort_records_by_date(jobs, ("finishedAt", "updatedAt", "createdAt"))
    except APIError as error:
        _handle_api_error(error, "load scheduling jobs")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_decisions() -> list[dict]:
    try:
        return search_scheduling_decisions({}) or []
    except APIError as error:
        _handle_api_error(error, "load scheduling decisions")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_outputs() -> list[dict]:
    try:
        return search_scheduling_outputs({}) or []
    except APIError as error:
        _handle_api_error(error, "load scheduling outputs")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_sku_demands(criteria_key: tuple[str, ...]) -> list[dict]:
    try:
        criteria: dict[str, object] = {}
        sku_ids = [value for value in criteria_key if value and not value.startswith("month:")]
        if sku_ids:
            criteria["skuIds"] = sku_ids
        return search_sku_demands(criteria) or []
    except APIError as error:
        _handle_api_error(error, "load SKU demands")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_monthly_contracts(criteria_key: tuple[str, ...]) -> list[dict]:
    try:
        criteria: dict[str, object] = {}
        sku_ids = [value for value in criteria_key if not value.startswith("month:") and value]
        months = [value[6:] for value in criteria_key if value.startswith("month:")]
        if sku_ids:
            criteria["skuIds"] = sku_ids
        if months:
            criteria["yearMonths"] = months
        return search_monthly_contract_demands_bulk(criteria) or []
    except APIError as error:
        _handle_api_error(error, "load monthly contract demands")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_available_wip(plant_name: str) -> list[dict]:
    try:
        criteria = {"plantName": plant_name} if plant_name else {}
        return search_available_wip(criteria) or []
    except APIError as error:
        _handle_api_error(error, "load available WIP")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_bucket_usage() -> list[dict]:
    try:
        return search_bucket_usage({}) or []
    except APIError as error:
        _handle_api_error(error, "load bucket usage")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_lines() -> list[dict]:
    try:
        return list_lines() or []
    except APIError as error:
        _handle_api_error(error, "load lines")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_configs() -> list[dict]:
    try:
        return get_all_configs() or []
    except APIError as error:
        _handle_api_error(error, "load config values")
        return []


def _init_state() -> None:
    defaults = {
        "scheduling_insights_job_id": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _job_lookup(jobs: list[dict]) -> dict[str, dict]:
    lookup = {}
    for job in jobs:
        job_id = str(job.get("jobId") or job.get("_id") or "").strip()
        if job_id:
            lookup[job_id] = job
    return lookup


def _job_options(jobs: list[dict]) -> list[str]:
    return [str(job.get("jobId") or job.get("_id") or "").strip() for job in jobs if str(job.get("jobId") or job.get("_id") or "").strip()]


def _default_job(jobs: list[dict]) -> dict | None:
    focused = completed_job(jobs)
    if focused is not None:
        return focused
    return latest_job(jobs)


def _job_months(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    months = pd.period_range(start=start, end=end, freq="M")
    return [period.strftime("%Y-%m") for period in months]


def _job_context(job: dict | None, decisions_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, list[str], str]:
    dates = decisions_df["date"].tolist() if not decisions_df.empty and "date" in decisions_df.columns else []
    start, end = focus_window(job, dates)
    sku_ids = [str(value).strip() for value in (job.get("skuIds", []) if job else []) if str(value).strip()]
    plant_id = str(job.get("plantId", "")).strip() if job else ""
    return start, end, sku_ids, plant_id


def _line_config_frame(lines: list[dict]) -> pd.DataFrame:
    if not lines:
        return pd.DataFrame()
    df = pd.DataFrame(lines).copy()
    rename = {
        "lineId": "lineId",
        "friendlyName": "friendlyName",
        "lineType": "lineType",
        "plant": "plant",
        "hoursOfLaborAvailablePerShift": "hoursOfLaborAvailablePerShift",
        "unitsAvailable": "unitsAvailable",
        "lineThroughput": "lineThroughput",
        "isActive": "isActive",
    }
    return df.rename(columns=rename)


def _derived_cut_schedule(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "date" not in decisions.columns:
        return pd.DataFrame(columns=["date", "line", "cuts", "planned_lbs", "total_duration_hours"])

    df = decisions.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if "lineId" in df.columns and "line" not in df.columns:
        df = df.rename(columns={"lineId": "line"})
    if "mixId" not in df.columns:
        df["mixId"] = ""
    if "lbsProduced" not in df.columns:
        df["lbsProduced"] = 0.0
    if "duration" not in df.columns:
        df["duration"] = 0.0

    def _cut_text(frame: pd.DataFrame) -> str:
        parts = []
        for _, row in frame.iterrows():
            mix = str(row.get("mixId", "")).strip() or "Unknown mix"
            lbs = safe_float(row.get("lbsProduced"), 0.0)
            duration = safe_float(row.get("duration"), 0.0)
            parts.append(f"{mix} ({lbs:,.0f} lbs, {duration:.1f}h)")
        return ", ".join(parts)

    summary = (
        df.groupby(["date", "line"], as_index=False)
        .agg(planned_lbs=("lbsProduced", "sum"), total_duration_hours=("duration", "sum"))
        .sort_values(["date", "line"])
    )

    cuts = (
        df.groupby(["date", "line"])
        .apply(_cut_text)
        .reset_index(name="cuts")
    )

    merged = summary.merge(cuts, on=["date", "line"], how="left")
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged[["date", "line", "cuts", "planned_lbs", "total_duration_hours"]]


def _summary_card(title: str, value: str, detail: str) -> None:
    st.markdown(
        "<div class='simmons-card' style='min-height:132px'>"
        f"<div style='font-size:14px;font-weight:700;color:#00264F'>{title}</div>"
        f"<div style='font-size:34px;font-weight:800;color:#0046AD;line-height:1.1;margin-top:8px'>{value}</div>"
        f"<div class='simmons-small' style='margin-top:6px'>{detail}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_table(df: pd.DataFrame, label: str, file_name: str, height: int = 320) -> None:
    if df.empty:
        st.info(f"No {label.lower()} found for the current selection.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)
    st.download_button(
        f"Download {label} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


def render():
    st.markdown(
        "<div class='simmons-card' style='margin-top:12px;margin-bottom:16px'>"
        "<div style='display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-start'>"
        "<div style='min-width:280px;flex:1'>"
        "<div style='font-size:28px;font-weight:800;color:#00264F'>Scheduling analytics</div>"
        "<div class='simmons-small' style='margin-top:6px'>"
        "Review the scheduler's decisions, demand coverage, bucket usage, and line utilization in one place."
        "</div></div>"
        "<div style='min-width:280px;flex:1'>"
        "<div class='simmons-small'>"
        "Use the selector below to focus the dashboard on a specific run. The page will anchor its date window to that run's plan start date and horizon."
        "</div></div></div></div>",
        unsafe_allow_html=True,
    )

    _init_state()
    jobs = _load_jobs()
    job_lookup = _job_lookup(jobs)
    job_options = _job_options(jobs)
    default_job = _default_job(jobs)
    default_job_id = str(default_job.get("jobId") or default_job.get("_id") or "").strip() if default_job else ""

    selection_col, refresh_col = st.columns([3, 1])
    with selection_col:
        selected_job_id = st.selectbox(
            "Focus job",
            options=[""] + job_options,
            index=([""] + job_options).index(st.session_state.scheduling_insights_job_id)
            if st.session_state.scheduling_insights_job_id in job_options
            else (([""] + job_options).index(default_job_id) if default_job_id in job_options else 0),
            format_func=lambda job_id: "Latest completed run" if not job_id else job_label(job_lookup[job_id]),
        )
    with refresh_col:
        if st.button("Refresh data", key="refresh_scheduling_insights"):
            _load_jobs.clear()
            _load_decisions.clear()
            _load_outputs.clear()
            _load_bucket_usage.clear()
            _load_available_wip.clear()
            _load_sku_demands.clear()
            _load_monthly_contracts.clear()
            _load_lines.clear()
            _load_configs.clear()
            st.rerun()

    st.session_state.scheduling_insights_job_id = selected_job_id
    selected_job = job_lookup.get(selected_job_id) if selected_job_id else default_job

    decisions_df = table_or_empty(_load_decisions())
    outputs_df = table_or_empty(_load_outputs())
    lines_df = table_or_empty(_load_lines())

    window_start, window_end, sku_ids, plant_id = _job_context(selected_job, decisions_df)
    sku_key = tuple(sku_ids)
    month_keys = tuple([*sku_ids, *[f"month:{month}" for month in _job_months(window_start, window_end)]])

    sku_demands_df = table_or_empty(_load_sku_demands(sku_key))
    monthly_contract_df = table_or_empty(_load_monthly_contracts(month_keys))
    available_wip_df = table_or_empty(_load_available_wip(plant_id))
    bucket_usage_df = table_or_empty(_load_bucket_usage())

    if not decisions_df.empty:
        decisions_df = filter_date_range(decisions_df, "date", window_start, window_end)
    if not outputs_df.empty:
        outputs_df = filter_date_range(outputs_df, "date", window_start, window_end)
    if not sku_demands_df.empty and "dueDate" in sku_demands_df.columns:
        sku_demands_df = filter_date_range(sku_demands_df, "dueDate", window_start, window_end)
    if not bucket_usage_df.empty:
        bucket_usage_df = filter_date_range(bucket_usage_df, "date", window_start, window_end)

    cut_schedule_df = _derived_cut_schedule(decisions_df)
    line_load_raw = pd.DataFrame()
    if not decisions_df.empty and not lines_df.empty:
        line_load_raw = decisions_df.copy()
        line_load_raw["date"] = pd.to_datetime(line_load_raw["date"], errors="coerce").dt.normalize()
        line_load_raw = line_load_raw.rename(columns={"lineId": "line"})
        lines_for_join = _line_config_frame(lines_df)
        if "lineId" in lines_for_join.columns:
            lines_for_join = lines_for_join.rename(columns={"lineId": "line"})
        if "line" in lines_for_join.columns:
            line_load_raw = line_load_raw.merge(
                lines_for_join[
                    [
                        "line",
                        "friendlyName",
                        "hoursOfLaborAvailablePerShift",
                        "unitsAvailable",
                        "lineThroughput",
                    ]
                ],
                on="line",
                how="left",
            )
        else:
            line_load_raw["friendlyName"] = ""
            line_load_raw["hoursOfLaborAvailablePerShift"] = 0.0
            line_load_raw["unitsAvailable"] = 1
            line_load_raw["lineThroughput"] = None

        line_load_raw["duration"] = pd.to_numeric(line_load_raw.get("duration", 0.0), errors="coerce").fillna(0.0)
        line_load_raw["lbsProduced"] = pd.to_numeric(line_load_raw.get("lbsProduced", 0.0), errors="coerce").fillna(0.0)
        line_load_raw["capacity_hours"] = pd.to_numeric(line_load_raw.get("hoursOfLaborAvailablePerShift", 0.0), errors="coerce").fillna(0.0) * pd.to_numeric(
            line_load_raw.get("unitsAvailable", 1), errors="coerce"
        ).fillna(1.0)
        line_load_raw["util_pct"] = line_load_raw.apply(
            lambda row: round((safe_float(row.get("duration")) / safe_float(row.get("capacity_hours"))) * 100, 1)
            if safe_float(row.get("capacity_hours")) > 0
            else 0.0,
            axis=1,
        )
        line_load_raw["throughput_capacity_lbs"] = pd.to_numeric(line_load_raw.get("lineThroughput", 0.0), errors="coerce").fillna(0.0) * line_load_raw["capacity_hours"]
        line_load_raw["throughput_util_pct"] = line_load_raw.apply(
            lambda row: round((safe_float(row.get("lbsProduced")) / safe_float(row.get("throughput_capacity_lbs"))) * 100, 1)
            if safe_float(row.get("throughput_capacity_lbs")) > 0
            else 0.0,
            axis=1,
        )
        line_load_raw = line_load_raw[
            [
                "date",
                "line",
                "friendlyName",
                "duration",
                "capacity_hours",
                "util_pct",
                "lbsProduced",
                "throughput_capacity_lbs",
                "throughput_util_pct",
            ]
        ]
        line_load_raw["date"] = line_load_raw["date"].dt.strftime("%Y-%m-%d")

    line_utilization_df = build_line_utilization(
        line_load_raw.rename(columns={"date": "date", "line": "line", "util_pct": "util_pct", "throughput_util_pct": "throughput_util_pct", "duration": "hours_used", "capacity_hours": "hours_available"}),
        window_start,
        window_end,
    )
    bucket_summary_df = build_bucket_usage_summary(bucket_usage_df, window_start, window_end)
    demand_progress_df = build_demand_progress(outputs_df, sku_demands_df, window_start, window_end)

    today_rows = cut_schedule_df[cut_schedule_df["date"] == window_start.strftime("%Y-%m-%d")] if not cut_schedule_df.empty else pd.DataFrame()
    if today_rows.empty and not cut_schedule_df.empty:
        today_rows = cut_schedule_df.head(min(5, len(cut_schedule_df)))
    upcoming_rows = cut_schedule_df[cut_schedule_df["date"] > window_start.strftime("%Y-%m-%d")] if not cut_schedule_df.empty else pd.DataFrame()

    produced_total = float(pd.to_numeric(outputs_df.get("lbsProduced", 0.0), errors="coerce").fillna(0.0).sum()) if not outputs_df.empty else 0.0
    demand_total = float(pd.to_numeric(sku_demands_df.get("demandValue", 0.0), errors="coerce").fillna(0.0).sum()) if not sku_demands_df.empty else 0.0
    coverage_pct = round((produced_total / demand_total) * 100, 1) if demand_total else 0.0
    avg_line_util = round(float(pd.to_numeric(line_utilization_df.get("avg_util_pct", 0.0), errors="coerce").fillna(0.0).mean()), 1) if not line_utilization_df.empty else 0.0
    avg_bucket_util = round(float(pd.to_numeric(bucket_summary_df.get("util_pct", 0.0), errors="coerce").fillna(0.0).mean()), 1) if not bucket_summary_df.empty else 0.0

    if selected_job:
        detail_line = f"Run {selected_job.get('runId', '—')} | {status_badge(selected_job.get('status', 'unknown'))}"
        detail_subline = f"Plant {selected_job.get('plantId', '—')} | Start {selected_job.get('planStartDate', '—')} | Horizon {selected_job.get('horizonDays', '—')} days"
    else:
        detail_line = "No completed run selected"
        detail_subline = "Showing the latest available scheduling data."

    st.markdown(
        f"<div class='simmons-card' style='margin-bottom:16px'>"
        f"<div style='display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap'>"
        f"<div style='min-width:260px;flex:1'>"
        f"<div style='font-weight:800;font-size:16px;color:#00264F'>{detail_line}</div>"
        f"<div class='simmons-small' style='margin-top:6px'>{detail_subline}</div>"
        f"</div>"
        f"<div style='min-width:260px;flex:1'>"
        f"<div class='simmons-small'>Focus window: <strong>{format_date(window_start)}</strong> to <strong>{format_date(window_end)}</strong></div>"
        f"<div class='simmons-small'>SKU scope: <strong>{len(sku_ids) or 'All'}</strong> | Lines: <strong>{len(lines_df) or 'Unknown'}</strong></div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        _summary_card("Today's cut rows", str(len(today_rows)), "Scheduled cuts on the focus date")
    with k2:
        _summary_card("Upcoming cuts", str(len(upcoming_rows)), "Rows after the focus date")
    with k3:
        _summary_card("Demand coverage", f"{coverage_pct:.1f}%", "Produced lbs vs. demand lbs")
    with k4:
        _summary_card("Avg line util", f"{avg_line_util:.1f}%", "Duration-based line utilization")
    with k5:
        _summary_card("Bucket util", f"{avg_bucket_util:.1f}%", "Weighted across bucket usage")
    with k6:
        _summary_card("Active lines", str(len(line_utilization_df) or len(lines_df)), "Lines represented in the data")

    tabs = st.tabs(["Today's Schedule", "Demand", "Utilization", "Raw Tables"])

    with tabs[0]:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("#### Today's cut schedule")
            _render_table(today_rows, "Today's cut schedule", "today-cut-schedule.csv", height=330)
        with right:
            st.markdown("#### Upcoming cuts")
            _render_table(upcoming_rows.head(12), "Upcoming cuts", "upcoming-cuts.csv", height=330)

    with tabs[1]:
        dc1, dc2 = st.columns([1.1, 0.9])
        with dc1:
            st.markdown("#### Demand progress by date")
            if demand_progress_df.empty:
                st.info("No demand progress data available for the current window.")
            else:
                chart_df = demand_progress_df.copy()
                chart_df["date"] = pd.to_datetime(chart_df["date"])
                chart_df = chart_df.set_index("date")[["produced_lbs", "demand_lbs"]]
                st.bar_chart(chart_df, use_container_width=True)
        with dc2:
            st.markdown("#### Production vs demand")
            _render_table(demand_progress_df, "Production vs demand", "production-vs-demand.csv", height=330)

        st.markdown("#### Monthly contract demand")
        monthly_contract_view = monthly_contract_df.copy()
        if not monthly_contract_view.empty:
            if "yearMonth" in monthly_contract_view.columns:
                monthly_contract_view = monthly_contract_view.sort_values(["yearMonth", "skuId"])
        _render_table(monthly_contract_view, "Monthly contract demand", "monthly-contract-demand.csv", height=260)

    with tabs[2]:
        lc1, lc2 = st.columns([1.1, 0.9])
        with lc1:
            st.markdown("#### Line utilization")
            if line_utilization_df.empty:
                st.info("No line utilization data available for the current window.")
            else:
                chart_df = line_utilization_df.copy()
                chart_df = chart_df.set_index("line")[["avg_util_pct", "avg_throughput_util_pct"]]
                st.bar_chart(chart_df, use_container_width=True)
        with lc2:
            st.markdown("#### Bucket utilization")
            if bucket_summary_df.empty:
                st.info("No bucket usage data available for the current window.")
            else:
                chart_df = bucket_summary_df.copy().set_index("bucket")[["util_pct"]]
                st.bar_chart(chart_df, use_container_width=True)

        st.markdown("#### Line utilization detail")
        _render_table(line_utilization_df, "Line utilization", "line-utilization.csv", height=260)
        st.markdown("#### Bucket usage detail")
        _render_table(bucket_summary_df, "Bucket usage", "bucket-usage.csv", height=260)

        if not available_wip_df.empty:
            st.markdown("#### Available WIP")
            _render_table(available_wip_df, "Available WIP", "available-wip.csv", height=220)

    with tabs[3]:
        table_tabs = st.tabs(
            [
                "Decisions",
                "Outputs",
                "SKU Demand",
                "Monthly Contracts",
                "Cut Schedule",
                "Available WIP",
                "Bucket Usage",
                "Lines",
                "Jobs",
            ]
        )

        with table_tabs[0]:
            decisions_view = decisions_df.copy()
            if not decisions_view.empty:
                decisions_view["date"] = decisions_view["date"].astype(str)
            _render_table(decisions_view, "Scheduling decisions", "scheduling-decisions.csv", height=320)

        with table_tabs[1]:
            outputs_view = outputs_df.copy()
            if not outputs_view.empty:
                outputs_view["date"] = outputs_view["date"].astype(str)
            _render_table(outputs_view, "Scheduling outputs", "scheduling-outputs.csv", height=320)

        with table_tabs[2]:
            sku_demand_view = sku_demands_df.copy()
            if not sku_demand_view.empty and "dueDate" in sku_demand_view.columns:
                sku_demand_view["dueDate"] = sku_demand_view["dueDate"].astype(str)
            _render_table(sku_demand_view, "SKU demand", "sku-demand.csv", height=320)

        with table_tabs[3]:
            monthly_contract_view = monthly_contract_df.copy()
            _render_table(monthly_contract_view, "Monthly contract demand", "monthly-contract-demand.csv", height=320)

        with table_tabs[4]:
            _render_table(cut_schedule_df, "Cut schedule", "cut-schedule.csv", height=320)

        with table_tabs[5]:
            _render_table(available_wip_df, "Available WIP", "available-wip.csv", height=320)

        with table_tabs[6]:
            _render_table(bucket_summary_df, "Bucket usage", "bucket-usage.csv", height=320)

        with table_tabs[7]:
            _render_table(lines_df, "Lines", "lines.csv", height=320)

        with table_tabs[8]:
            jobs_view = pd.DataFrame(
                [
                    {
                        "jobId": job.get("jobId", ""),
                        "runId": job.get("runId", ""),
                        "plantId": job.get("plantId", ""),
                        "status": job.get("status", ""),
                        "planStartDate": job.get("planStartDate", ""),
                        "horizonDays": job.get("horizonDays", ""),
                        "createdAt": format_timestamp(job.get("createdAt")),
                        "finishedAt": format_timestamp(job.get("finishedAt")),
                    }
                    for job in jobs
                ]
            )
            _render_table(jobs_view, "Jobs", "scheduling-jobs.csv", height=320)

