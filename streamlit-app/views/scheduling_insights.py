"""Scheduling analytics view."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import streamlit as st

from api_client import (
    APIError,
    get_all_configs,
    list_lines,
    list_scheduling_jobs,
    search_available_wip,
    search_buckets,
    search_bucket_usage,
    search_mix_metrics,
    search_mixes,
    search_monthly_contract_demands,
    search_monthly_contract_demands_bulk,
    search_scheduling_decisions,
    search_scheduling_outputs,
    search_sku_demands,
)
from views.scheduling_shared import (
    build_bucket_usage_summary,
    build_demand_progress,
    build_line_utilization,
    build_upcoming_batches,
    filter_date_range,
    focus_window,
    format_date,
    format_timestamp,
    safe_float,
    sort_records_by_date,
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
        sku_ids = [value for value in criteria_key if value and not value.startswith("month:")]
        months = [value[6:] for value in criteria_key if value.startswith("month:")]
        if sku_ids and months:
            criteria["skuIds"] = sku_ids
            criteria["yearMonths"] = months
            return search_monthly_contract_demands_bulk(criteria) or []
        if months:
            records: list[dict] = []
            for month in months:
                records.extend(search_monthly_contract_demands({"yearMonth": month}) or [])
            return records
        if sku_ids:
            records: list[dict] = []
            for sku_id in sku_ids:
                records.extend(search_monthly_contract_demands({"skuId": sku_id}) or [])
            return records
        return search_monthly_contract_demands({}) or []
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
def _load_buckets() -> list[dict]:
    try:
        return search_buckets({}) or []
    except APIError as error:
        _handle_api_error(error, "load buckets")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_mixes() -> list[dict]:
    try:
        return search_mixes({}) or []
    except APIError as error:
        _handle_api_error(error, "load mixes")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_mix_metrics() -> list[dict]:
    try:
        return search_mix_metrics({}) or []
    except APIError as error:
        _handle_api_error(error, "load mix metrics")
        return []


@st.cache_data(show_spinner=False, ttl=60)
def _load_configs() -> list[dict]:
    try:
        return get_all_configs() or []
    except APIError as error:
        _handle_api_error(error, "load config values")
        return []


def _job_months(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    months = pd.period_range(start=start, end=end, freq="M")
    return [period.strftime("%Y-%m") for period in months]


def _global_context(decisions_df: pd.DataFrame, outputs_df: pd.DataFrame, sku_demands_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    candidate_dates: list[object] = []
    for frame, column in (
        (decisions_df, "date"),
        (outputs_df, "date"),
        (sku_demands_df, "dueDate"),
    ):
        if not frame.empty and column in frame.columns:
            candidate_dates.extend(frame[column].tolist())
    return focus_window(None, candidate_dates)


def _line_config_frame(lines: list[dict] | pd.DataFrame) -> pd.DataFrame:
    if lines is None:
        return pd.DataFrame()
    if isinstance(lines, pd.DataFrame):
        if lines.empty:
            return pd.DataFrame()
        df = lines.copy()
    else:
        if not lines:
            return pd.DataFrame()
        df = pd.DataFrame(lines).copy()
    return df.rename(
        columns={
            "lineId": "lineId",
            "friendlyName": "friendlyName",
            "lineType": "lineType",
            "plant": "plant",
            "hoursOfLaborAvailablePerShift": "hoursOfLaborAvailablePerShift",
            "unitsAvailable": "unitsAvailable",
            "lineThroughput": "lineThroughput",
            "isActive": "isActive",
        }
    )


def _mix_label_map(mixes: list[dict] | pd.DataFrame) -> dict[str, str]:
    if mixes is None:
        return {}
    if isinstance(mixes, pd.DataFrame):
        if mixes.empty:
            return {}
        rows = mixes.to_dict(orient="records")
    else:
        rows = mixes

    labels: dict[str, str] = {}
    for mix in rows:
        mix_id = str(mix.get("_id") or mix.get("mixId") or "").strip()
        if not mix_id:
            continue
        line = str(mix.get("mfgType", "")).strip()
        plant = str(mix.get("reqPlant", "")).strip()
        bird_size = str(mix.get("reqBirdSize", "")).strip()
        sku_keys = mix.get("skuKeys")
        if isinstance(sku_keys, list):
            sku_count = len([sku for sku in sku_keys if str(sku).strip()])
        else:
            skus = mix.get("skus", {})
            sku_count = len(skus) if isinstance(skus, dict) else 0

        parts = [part for part in (line, plant, bird_size) if part]
        if sku_count:
            parts.append(f"{sku_count} SKU{'s' if sku_count != 1 else ''}")
        labels[mix_id] = " | ".join(parts) if parts else mix_id
    return labels


def _bucket_label_map(buckets: list[dict] | pd.DataFrame) -> dict[str, str]:
    if buckets is None:
        return {}
    if isinstance(buckets, pd.DataFrame):
        if buckets.empty:
            return {}
        rows = buckets.to_dict(orient="records")
    else:
        rows = buckets

    labels: dict[str, str] = {}
    for bucket in rows:
        bucket_id = str(bucket.get("_id") or bucket.get("bucketId") or "").strip()
        if not bucket_id:
            continue
        min_weight = bucket.get("minWeight")
        max_weight = bucket.get("maxWeight")
        if min_weight is None or max_weight is None:
            labels[bucket_id] = bucket_id
            continue
        labels[bucket_id] = f"{bucket_id} [{float(min_weight):g}, {float(max_weight):g}]"
    return labels


def _unit_plan_sku_summary(unit_plan: object) -> str:
    if not isinstance(unit_plan, list):
        return ""
    sku_ids = []
    seen = set()
    for item in unit_plan:
        if not isinstance(item, dict):
            continue
        sku = str(item.get("sku", "")).strip()
        if not sku or sku in seen:
            continue
        seen.add(sku)
        sku_ids.append(sku)
    return ", ".join(sku_ids)


def _unit_plan_text(unit_plan: object) -> str:
    if not isinstance(unit_plan, list):
        return ""
    parts: list[str] = []
    for item in unit_plan:
        if not isinstance(item, dict):
            continue
        sku = str(item.get("sku", "")).strip() or "?"
        part_code = str(item.get("partCode", "")).strip() or "?"
        units = int(safe_float(item.get("unitsInPlan", 0), 0.0))
        weight = safe_float(item.get("totalWeightInPlan", 0.0), 0.0)
        parts.append(f"{sku} {part_code} x{units} ({weight:.0f}g)")
    return "; ".join(parts)


def _mix_metric_context_map(
    mix_metrics: list[dict] | pd.DataFrame,
    mix_labels: dict[str, str],
    bucket_labels: dict[str, str],
) -> dict[str, dict[str, str]]:
    if mix_metrics is None:
        return {}
    if isinstance(mix_metrics, pd.DataFrame):
        if mix_metrics.empty:
            return {}
        rows = mix_metrics.to_dict(orient="records")
    else:
        rows = mix_metrics

    context: dict[str, dict[str, str]] = {}
    for metric in rows:
        metric_id = str(metric.get("_id") or metric.get("metricId") or "").strip()
        mix_id = str(metric.get("mixId") or "").strip()
        bucket_id = str(metric.get("bucketId") or "").strip()
        if not metric_id:
            continue
        context[metric_id] = {
            "mixLabel": mix_labels.get(mix_id, mix_id),
            "bucketLabel": bucket_labels.get(bucket_id, bucket_id),
            "skuIds": _unit_plan_sku_summary(metric.get("unitPlan")),
            "unitPlan": _unit_plan_text(metric.get("unitPlan")),
        }
    return context


def _display_cut_name(value: object, mix_labels: dict[str, str], metric_context: dict[str, dict[str, str]] | None = None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "Unknown mix"

    if metric_context and raw in metric_context:
        context = metric_context[raw]
        details = [context.get("mixLabel", ""), context.get("bucketLabel", ""), context.get("skuIds", "")]
        return " | ".join([detail for detail in details if detail])

    metric_mix_id = raw.split(":", 1)[0]
    return mix_labels.get(metric_mix_id) or mix_labels.get(raw) or metric_mix_id or raw


def _enrich_decision_frame(
    decisions: pd.DataFrame,
    mix_labels: dict[str, str],
    metric_context: dict[str, dict[str, str]],
) -> pd.DataFrame:
    if decisions.empty:
        return decisions.copy()
    df = decisions.copy()
    if "mixId" not in df.columns:
        return df

    mix_metric_ids = df["mixId"].astype(str)
    df["mixMetricId"] = mix_metric_ids
    df["mix"] = mix_metric_ids.map(
        lambda value: (
            metric_context.get(value, {}).get("mixLabel")
            or mix_labels.get(value.split(":", 1)[0], value.split(":", 1)[0])
        )
    )
    df["bucket"] = mix_metric_ids.map(lambda value: metric_context.get(value, {}).get("bucketLabel", ""))
    df["skuIds"] = mix_metric_ids.map(lambda value: metric_context.get(value, {}).get("skuIds", ""))
    df["unitPlan"] = mix_metric_ids.map(lambda value: metric_context.get(value, {}).get("unitPlan", ""))
    return df


def _derived_cut_schedule(
    decisions: pd.DataFrame,
    mix_labels: dict[str, str] | None = None,
    metric_context: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    if decisions.empty or "date" not in decisions.columns:
        return pd.DataFrame(columns=["date", "line", "cuts", "planned_lbs", "total_duration_hours"])

    df = decisions.copy()
    mix_labels = mix_labels or {}
    metric_context = metric_context or {}
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
            mix = _display_cut_name(row.get("mixId", ""), mix_labels, metric_context)
            lbs = safe_float(row.get("lbsProduced"), 0.0)
            duration = safe_float(row.get("duration"), 0.0)
            parts.append(f"{mix} ({lbs:,.0f} lbs, {duration:.1f}h)")
        return ", ".join(parts)

    summary = (
        df.groupby(["date", "line"], as_index=False)
        .agg(planned_lbs=("lbsProduced", "sum"), total_duration_hours=("duration", "sum"))
        .sort_values(["date", "line"])
    )
    cuts = df.groupby(["date", "line"]).apply(_cut_text).reset_index(name="cuts")
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


def _render_table(df: pd.DataFrame, label: str, file_name: str, key: str, height: int = 320) -> None:
    if df.empty:
        st.info(f"No {label.lower()} found for the current selection.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)
    st.download_button(
        f"Download {label} CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


def render():
    st.markdown(
        "<div class='simmons-card' style='margin-top:12px;margin-bottom:16px'>"
        "<div style='display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:flex-start'>"
        "<div style='min-width:280px;flex:1'>"
        "<div style='font-size:28px;font-weight:800;color:#00264F'>Integrated scheduling data</div>"
        "<div class='simmons-small' style='margin-top:6px'>"
        "Review the scheduler's decisions, outputs, demand, WIP, bucket usage, line setup, and job history in one place."
        "</div></div>"
        "<div style='min-width:280px;flex:1'>"
        "<div class='simmons-small'>"
        "This workspace is global. It does not scope the results to a specific scheduling job."
        "</div></div></div></div>",
        unsafe_allow_html=True,
    )

    if st.button("Refresh data", key="refresh_scheduling_insights"):
        _load_jobs.clear()
        _load_decisions.clear()
        _load_outputs.clear()
        _load_bucket_usage.clear()
        _load_available_wip.clear()
        _load_sku_demands.clear()
        _load_monthly_contracts.clear()
        _load_lines.clear()
        _load_buckets.clear()
        _load_mixes.clear()
        _load_mix_metrics.clear()
        _load_configs.clear()
        st.rerun()

    jobs = _load_jobs()
    decisions_df = table_or_empty(_load_decisions())
    outputs_df = table_or_empty(_load_outputs())
    sku_demands_df = table_or_empty(_load_sku_demands(tuple()))
    lines_df = table_or_empty(_load_lines())
    mix_label_lookup = _mix_label_map(_load_mixes())
    bucket_label_lookup = _bucket_label_map(_load_buckets())
    metric_context_lookup = _mix_metric_context_map(_load_mix_metrics(), mix_label_lookup, bucket_label_lookup)
    decisions_df = _enrich_decision_frame(decisions_df, mix_label_lookup, metric_context_lookup)
    window_start, window_end = _global_context(decisions_df, outputs_df, sku_demands_df)
    recent_window_start = pd.Timestamp.today().normalize()
    recent_window_end = recent_window_start + timedelta(days=6)
    month_keys = tuple(f"month:{month}" for month in _job_months(window_start, window_end))

    monthly_contract_df = table_or_empty(_load_monthly_contracts(month_keys))
    available_wip_df = table_or_empty(_load_available_wip(""))
    bucket_usage_df = table_or_empty(_load_bucket_usage())

    decisions_window_df = filter_date_range(decisions_df, "date", recent_window_start, recent_window_end) if not decisions_df.empty else pd.DataFrame()
    outputs_window_df = filter_date_range(outputs_df, "date", recent_window_start, recent_window_end) if not outputs_df.empty else pd.DataFrame()
    sku_demands_window_df = (
        filter_date_range(sku_demands_df, "dueDate", recent_window_start, recent_window_end)
        if not sku_demands_df.empty and "dueDate" in sku_demands_df.columns
        else pd.DataFrame()
    )
    bucket_usage_window_df = filter_date_range(bucket_usage_df, "date", recent_window_start, recent_window_end) if not bucket_usage_df.empty else pd.DataFrame()

    cut_schedule_df = _derived_cut_schedule(decisions_window_df, mix_label_lookup, metric_context_lookup)
    line_load_raw = pd.DataFrame()
    if not decisions_window_df.empty and not lines_df.empty:
        line_load_raw = decisions_window_df.copy()
        line_load_raw["date"] = pd.to_datetime(line_load_raw["date"], errors="coerce").dt.normalize()
        line_load_raw = line_load_raw.rename(columns={"lineId": "line"})
        lines_for_join = _line_config_frame(lines_df)
        if "lineId" in lines_for_join.columns:
            lines_for_join = lines_for_join.rename(columns={"lineId": "line"})
        if "line" in lines_for_join.columns:
            line_load_raw = line_load_raw.merge(
                lines_for_join[["line", "friendlyName", "hoursOfLaborAvailablePerShift", "unitsAvailable", "lineThroughput"]],
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
        line_load_raw["capacity_hours"] = (
            pd.to_numeric(line_load_raw.get("hoursOfLaborAvailablePerShift", 0.0), errors="coerce").fillna(0.0)
            * pd.to_numeric(line_load_raw.get("unitsAvailable", 1), errors="coerce").fillna(1.0)
        )
        line_load_raw["util_pct"] = line_load_raw.apply(
            lambda row: round((safe_float(row.get("duration")) / safe_float(row.get("capacity_hours"))) * 100, 1)
            if safe_float(row.get("capacity_hours")) > 0
            else 0.0,
            axis=1,
        )
        line_load_raw["throughput_capacity_lbs"] = (
            pd.to_numeric(line_load_raw.get("lineThroughput", 0.0), errors="coerce").fillna(0.0)
            * line_load_raw["capacity_hours"]
        )
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
        line_load_raw.rename(
            columns={
                "date": "date",
                "line": "line",
                "util_pct": "util_pct",
                "throughput_util_pct": "throughput_util_pct",
                "duration": "hours_used",
                "capacity_hours": "hours_available",
            }
        ),
        recent_window_start,
        recent_window_end,
    )
    bucket_summary_df = build_bucket_usage_summary(bucket_usage_window_df, recent_window_start, recent_window_end)
    demand_progress_df = build_demand_progress(outputs_window_df, sku_demands_window_df, recent_window_start, recent_window_end)

    today_rows = cut_schedule_df[cut_schedule_df["date"] == recent_window_start.strftime("%Y-%m-%d")] if not cut_schedule_df.empty else pd.DataFrame()
    if today_rows.empty and not cut_schedule_df.empty:
        today_rows = cut_schedule_df.head(min(5, len(cut_schedule_df)))
    upcoming_rows = cut_schedule_df[cut_schedule_df["date"] > recent_window_start.strftime("%Y-%m-%d")] if not cut_schedule_df.empty else pd.DataFrame()
    upcoming_batches_df = build_upcoming_batches(decisions_df, recent_window_start, days=4)
    if not upcoming_batches_df.empty and "mixId" in upcoming_batches_df.columns:
        upcoming_batches_df = upcoming_batches_df.rename(columns={"mixId": "mixMetricId"})
        original_bucket_values = upcoming_batches_df["bucket"] if "bucket" in upcoming_batches_df.columns else pd.Series("", index=upcoming_batches_df.index)
        upcoming_batches_df["mix"] = upcoming_batches_df["mixMetricId"].map(
            lambda value: (
                metric_context_lookup.get(str(value), {}).get("mixLabel")
                or mix_label_lookup.get(str(value).split(":", 1)[0], str(value).split(":", 1)[0])
            )
        )
        upcoming_batches_df["bucket"] = [
            metric_context_lookup.get(str(metric_id), {}).get("bucketLabel") or original_bucket_values.iloc[index]
            for index, metric_id in enumerate(upcoming_batches_df["mixMetricId"])
        ]
        upcoming_batches_df["skuIds"] = upcoming_batches_df["mixMetricId"].map(
            lambda value: metric_context_lookup.get(str(value), {}).get("skuIds", "")
        )
        upcoming_batches_df["unitPlan"] = upcoming_batches_df["mixMetricId"].map(
            lambda value: metric_context_lookup.get(str(value), {}).get("unitPlan", "")
        )
        ordered_columns = [
            "date",
            "line",
            "mix",
            "bucket",
            "skuIds",
            "unitPlan",
            "mixMetricId",
            "lbsProduced",
            "duration",
            "upgradePercentage",
            "trimPercentage",
        ]
        available_columns = [column for column in ordered_columns if column in upcoming_batches_df.columns]
        upcoming_batches_df = upcoming_batches_df[available_columns]

    produced_total = float(pd.to_numeric(outputs_window_df.get("lbsProduced", 0.0), errors="coerce").fillna(0.0).sum()) if not outputs_window_df.empty else 0.0
    demand_total = float(pd.to_numeric(sku_demands_window_df.get("demandValue", 0.0), errors="coerce").fillna(0.0).sum()) if not sku_demands_window_df.empty else 0.0
    coverage_pct = round((produced_total / demand_total) * 100, 1) if demand_total else 0.0
    avg_line_util = round(float(pd.to_numeric(line_utilization_df.get("avg_util_pct", 0.0), errors="coerce").fillna(0.0).mean()), 1) if not line_utilization_df.empty else 0.0
    avg_bucket_util = round(float(pd.to_numeric(bucket_summary_df.get("util_pct", 0.0), errors="coerce").fillna(0.0).mean()), 1) if not bucket_summary_df.empty else 0.0

    st.markdown(
        f"<div class='simmons-card' style='margin-bottom:16px'>"
        f"<div style='display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap'>"
        f"<div style='min-width:260px;flex:1'>"
        f"<div style='font-weight:800;font-size:16px;color:#00264F'>Global scheduling view</div>"
        f"<div class='simmons-small' style='margin-top:6px'>Showing all available scheduling data across runs.</div>"
        f"</div>"
        f"<div style='min-width:260px;flex:1'>"
        f"<div class='simmons-small'>Recent window: <strong>{format_date(recent_window_start)}</strong> to <strong>{format_date(recent_window_end)}</strong></div>"
        f"<div class='simmons-small'>Historical span: <strong>{format_date(window_start)}</strong> to <strong>{format_date(window_end)}</strong> | Lines: <strong>{len(lines_df) or 'Unknown'}</strong></div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        _summary_card("Today's cut rows", str(len(today_rows)), "Scheduled cuts for today")
    with k2:
        _summary_card("Upcoming batches", str(len(upcoming_batches_df)), "Scheduling decisions in the next 4 days")
    with k3:
        _summary_card("Demand coverage", f"{coverage_pct:.1f}%", "Produced lbs vs. demand lbs this week")
    with k4:
        _summary_card("Avg line util", f"{avg_line_util:.1f}%", "Duration-based utilization this week")
    with k5:
        _summary_card("Bucket util", f"{avg_bucket_util:.1f}%", "Weighted bucket utilization this week")
    with k6:
        _summary_card("Scheduling jobs", str(len(jobs)), "Jobs represented in the history table")

    tabs = st.tabs(["Upcoming Batches", "Demand", "Utilization", "Data Grids"])

    with tabs[0]:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("#### Today's cut schedule")
            _render_table(today_rows, "Today's cut schedule", "today-cut-schedule.csv", key="insights_today_cut_schedule_download", height=330)
        with right:
            st.markdown("#### Upcoming batches")
            _render_table(upcoming_batches_df, "Upcoming batches", "upcoming-batches.csv", key="insights_upcoming_batches_download", height=330)

        st.markdown("#### Upcoming cut schedule")
        _render_table(upcoming_rows.head(12), "Upcoming cuts", "upcoming-cuts.csv", key="insights_upcoming_cuts_download", height=260)

    with tabs[1]:
        dc1, dc2 = st.columns([1.1, 0.9])
        with dc1:
            st.markdown("#### Demand progress by date")
            if demand_progress_df.empty:
                st.info("No demand progress data available for the recent window.")
            else:
                chart_df = demand_progress_df.copy()
                chart_df["date"] = pd.to_datetime(chart_df["date"])
                chart_df = chart_df.set_index("date")[["produced_lbs", "demand_lbs"]]
                st.bar_chart(chart_df, use_container_width=True)
        with dc2:
            st.markdown("#### Production vs demand")
            _render_table(demand_progress_df, "Production vs demand", "production-vs-demand.csv", key="insights_production_vs_demand_download", height=330)

        st.markdown("#### Monthly contract demand")
        monthly_contract_view = monthly_contract_df.copy()
        if not monthly_contract_view.empty and "yearMonth" in monthly_contract_view.columns:
            monthly_contract_view = monthly_contract_view.sort_values(["yearMonth", "skuId"])
        _render_table(monthly_contract_view, "Monthly contract demand", "monthly-contract-demand.csv", key="insights_monthly_contract_demand_window_download", height=260)

    with tabs[2]:
        lc1, lc2 = st.columns([1.1, 0.9])
        with lc1:
            st.markdown("#### Line utilization")
            if line_utilization_df.empty:
                st.info("No line utilization data available for the recent window.")
            else:
                chart_df = line_utilization_df.copy().set_index("line")[["avg_util_pct", "avg_throughput_util_pct"]]
                st.bar_chart(chart_df, use_container_width=True)
        with lc2:
            st.markdown("#### Bucket utilization")
            if bucket_summary_df.empty:
                st.info("No bucket usage data available for the recent window.")
            else:
                chart_df = bucket_summary_df.copy().set_index("bucket")[["util_pct"]]
                st.bar_chart(chart_df, use_container_width=True)

        st.markdown("#### Line utilization detail")
        _render_table(line_utilization_df, "Line utilization", "line-utilization.csv", key="insights_line_utilization_detail_download", height=260)
        st.markdown("#### Bucket usage detail")
        _render_table(bucket_summary_df, "Bucket usage", "bucket-usage.csv", key="insights_bucket_usage_detail_download", height=260)

        if not available_wip_df.empty:
            st.markdown("#### Available WIP")
            _render_table(available_wip_df, "Available WIP", "available-wip.csv", key="insights_available_wip_detail_download", height=220)

    with tabs[3]:
        decisions_view = decisions_df.copy()
        if not decisions_view.empty and "date" in decisions_view.columns:
            decisions_view["date"] = decisions_view["date"].astype(str)
        st.markdown("#### Scheduling decisions")
        if not decisions_view.empty:
            preferred_columns = [
                "date",
                "lineId",
                "mix",
                "bucket",
                "skuIds",
                "unitPlan",
                "mixMetricId",
                "lbsProduced",
                "duration",
                "upgradePct",
            ]
            available_columns = [column for column in preferred_columns if column in decisions_view.columns]
            decisions_view = decisions_view[available_columns]
        _render_table(decisions_view, "Scheduling decisions", "scheduling-decisions.csv", key="insights_scheduling_decisions_download", height=320)

        outputs_view = outputs_df.copy()
        if not outputs_view.empty and "date" in outputs_view.columns:
            outputs_view["date"] = outputs_view["date"].astype(str)
        st.markdown("#### Scheduling outputs")
        _render_table(outputs_view, "Scheduling outputs", "scheduling-outputs.csv", key="insights_scheduling_outputs_download", height=320)

        sku_demand_view = sku_demands_df.copy()
        if not sku_demand_view.empty and "dueDate" in sku_demand_view.columns:
            sku_demand_view["dueDate"] = sku_demand_view["dueDate"].astype(str)
        st.markdown("#### SKU demand")
        _render_table(sku_demand_view, "SKU demand", "sku-demand.csv", key="insights_sku_demand_download", height=320)

        st.markdown("#### Monthly contract demand")
        _render_table(monthly_contract_df.copy(), "Monthly contract demand", "monthly-contract-demand.csv", key="insights_monthly_contract_demand_grid_download", height=320)

        st.markdown("#### Available WIP")
        _render_table(available_wip_df, "Available WIP", "available-wip.csv", key="insights_available_wip_grid_download", height=320)

        st.markdown("#### Bucket usage")
        _render_table(bucket_usage_df, "Bucket usage", "bucket-usage.csv", key="insights_bucket_usage_grid_download", height=320)

        st.markdown("#### Lines")
        _render_table(lines_df, "Lines", "lines.csv", key="insights_lines_download", height=320)

        jobs_view = pd.DataFrame(
            [
                {
                    "jobId": job.get("jobId", ""),
                    "runId": job.get("runId", ""),
                    "plantId": job.get("plantId", ""),
                    "status": job.get("status", ""),
                    "planStartDate": job.get("planStartDate", ""),
                    "horizonDays": job.get("horizonDays", ""),
                    "maxTrimPercentage": job.get("maxTrimPercentage", ""),
                    "createdAt": format_timestamp(job.get("createdAt")),
                    "finishedAt": format_timestamp(job.get("finishedAt")),
                }
                for job in jobs
            ]
        )
        st.markdown("#### Scheduling jobs")
        _render_table(jobs_view, "Jobs", "scheduling-jobs.csv", key="insights_jobs_download", height=320)
