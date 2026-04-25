"""Shared helpers for scheduling Streamlit views."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable

import pandas as pd


def _clean_text(value: Any) -> str:
    return str(value).strip()


def parse_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    if isinstance(value, date):
        return pd.Timestamp(value)
    try:
        parsed = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def format_date(value: Any, fallback: str = "—") -> str:
    parsed = parse_date(value)
    if parsed is None:
        return fallback
    return parsed.strftime("%Y-%m-%d")


def format_timestamp(value: Any, fallback: str = "—") -> str:
    if value in (None, ""):
        return fallback
    try:
        parsed = pd.Timestamp(value)
        return parsed.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def status_badge(status: str) -> str:
    colors = {
        "completed": ("#1d7a3a", "#e9f7ee"),
        "running": ("#0046AD", "#e3f0ff"),
        "pending": ("#9a6700", "#fff7e6"),
        "failed": ("#b42318", "#fdecec"),
        "cancelled": ("#5b6572", "#f3f4f6"),
    }
    fg, bg = colors.get(str(status).lower(), ("#333333", "#eeeeee"))
    return (
        f"<span style='background:{bg};color:{fg};padding:3px 10px;"
        f"border-radius:999px;font-size:12px;font-weight:700;letter-spacing:0.02em'>"
        f"{str(status).upper()}</span>"
    )


def job_label(job: dict) -> str:
    run_id = _clean_text(job.get("runId") or job.get("run_id") or "Unnamed run")
    plant = _clean_text(job.get("plantId") or job.get("plant_id") or "No plant")
    status = _clean_text(job.get("status") or "unknown")
    created_at = format_timestamp(job.get("createdAt") or job.get("created_at"))
    return f"{run_id} | {plant} | {status} | {created_at}"


def sort_records_by_date(records: list[dict], field_names: Iterable[str]) -> list[dict]:
    fields = list(field_names)

    def _sort_key(record: dict) -> tuple:
        for field in fields:
            parsed = parse_date(record.get(field))
            if parsed is not None:
                return (parsed, _clean_text(record.get("_id") or record.get("jobId") or record.get("runId")))
        return (pd.Timestamp.min, _clean_text(record.get("_id") or record.get("jobId") or record.get("runId")))

    return sorted(records or [], key=_sort_key, reverse=True)


def latest_job(jobs: list[dict]) -> dict | None:
    if not jobs:
        return None
    ordered = sort_records_by_date(jobs, ("finishedAt", "updatedAt", "createdAt"))
    return ordered[0] if ordered else None


def completed_job(jobs: list[dict]) -> dict | None:
    completed = [job for job in jobs if _clean_text(job.get("status")).lower() == "completed"]
    if not completed:
        return latest_job(jobs)
    return latest_job(completed)


def focus_window(job: dict | None, dates: Iterable[Any] | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    candidate_dates = [parse_date(value) for value in (dates or []) if parse_date(value) is not None]
    if job is not None:
        start = parse_date(job.get("planStartDate") or job.get("plan_start_date"))
        if start is None and candidate_dates:
            start = min(candidate_dates)
        elif start is None:
            start = pd.Timestamp.today().normalize()
        horizon = int(safe_float(job.get("horizonDays") or job.get("horizon_days"), 7))
        horizon = max(horizon, 1)
        end = start + timedelta(days=horizon - 1)
        return start, end

    if candidate_dates:
        start = min(candidate_dates)
        end = max(candidate_dates)
        return start, end

    today = pd.Timestamp.today().normalize()
    return today, today + timedelta(days=6)


def _with_parsed_date(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame.copy()
    df = frame.copy()
    df["_parsed_date"] = pd.to_datetime(df[column], errors="coerce").dt.normalize()
    return df


def filter_date_range(frame: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame.iloc[0:0].copy()
    df = _with_parsed_date(frame, column)
    df = df[df["_parsed_date"].notna()]
    return df[(df["_parsed_date"] >= start) & (df["_parsed_date"] <= end)].drop(columns=["_parsed_date"])


def table_or_empty(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _sum_series(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def build_todays_cut_schedule(line_schedule: pd.DataFrame, focus_date: pd.Timestamp) -> pd.DataFrame:
    if line_schedule.empty or "date" not in line_schedule.columns:
        return pd.DataFrame(columns=["date", "line", "cuts"])
    df = _with_parsed_date(line_schedule, "date")
    today_rows = df[df["_parsed_date"] == focus_date].drop(columns=["_parsed_date"])
    if not today_rows.empty:
        return today_rows.reset_index(drop=True)

    fallback = df.sort_values("_parsed_date").drop(columns=["_parsed_date"])
    if fallback.empty:
        return pd.DataFrame(columns=["date", "line", "cuts"])
    return fallback.head(max(5, min(len(fallback), 8))).reset_index(drop=True)


def build_upcoming_cuts(line_schedule: pd.DataFrame, focus_date: pd.Timestamp, days: int = 3) -> pd.DataFrame:
    if line_schedule.empty or "date" not in line_schedule.columns:
        return pd.DataFrame(columns=["date", "line", "cuts"])
    df = _with_parsed_date(line_schedule, "date")
    upcoming = df[(df["_parsed_date"] > focus_date) & (df["_parsed_date"] <= focus_date + timedelta(days=days))]
    return upcoming.sort_values(["_parsed_date", "line"]).drop(columns=["_parsed_date"]).reset_index(drop=True)


def build_upcoming_batches(decisions: pd.DataFrame, focus_date: pd.Timestamp, days: int = 3) -> pd.DataFrame:
    if decisions.empty or "date" not in decisions.columns:
        return pd.DataFrame(columns=["date", "line", "mixId", "lbsProduced", "duration"])

    df = _with_parsed_date(decisions, "date")
    df = df[(df["_parsed_date"] > focus_date) & (df["_parsed_date"] <= focus_date + timedelta(days=days))]
    if df.empty:
        return pd.DataFrame(columns=["date", "line", "mixId", "lbsProduced", "duration"])

    if "lineId" in df.columns and "line" not in df.columns:
        df = df.rename(columns={"lineId": "line"})
    if "line" not in df.columns:
        df["line"] = ""
    if "mixId" not in df.columns:
        df["mixId"] = ""
    if "lbsProduced" not in df.columns:
        df["lbsProduced"] = 0.0
    if "duration" not in df.columns:
        df["duration"] = 0.0

    preferred_columns = [
        "date",
        "line",
        "mixId",
        "skuId",
        "bucket",
        "lbsProduced",
        "duration",
        "upgradePercentage",
        "trimPercentage",
    ]
    available_columns = [column for column in preferred_columns if column in df.columns]
    upcoming = df.sort_values(["_parsed_date", "line", "mixId"]).drop(columns=["_parsed_date"])
    upcoming["date"] = pd.to_datetime(upcoming["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return upcoming[available_columns].reset_index(drop=True)


def build_demand_progress(outputs: pd.DataFrame, sku_demands: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if outputs.empty or "date" not in outputs.columns:
        return pd.DataFrame(columns=["date", "produced_lbs", "demand_lbs", "coverage_pct", "gap_lbs"])

    output_window = filter_date_range(outputs, "date", start, end)
    if output_window.empty:
        return pd.DataFrame(columns=["date", "produced_lbs", "demand_lbs", "coverage_pct", "gap_lbs"])

    produced = (
        _with_parsed_date(output_window, "date")
        .groupby("_parsed_date", as_index=False)["lbsProduced"]
        .sum()
        .rename(columns={"_parsed_date": "date", "lbsProduced": "produced_lbs"})
    )

    demand = pd.DataFrame(columns=["date", "demand_lbs"])
    if not sku_demands.empty and "dueDate" in sku_demands.columns:
        demand_window = filter_date_range(sku_demands, "dueDate", start, end)
        if not demand_window.empty:
            demand = (
                _with_parsed_date(demand_window, "dueDate")
                .groupby("_parsed_date", as_index=False)["demandValue"]
                .sum()
                .rename(columns={"_parsed_date": "date", "demandValue": "demand_lbs"})
            )

    merged = produced.merge(demand, on="date", how="left").fillna({"demand_lbs": 0.0})
    merged["coverage_pct"] = merged.apply(
        lambda row: round((row["produced_lbs"] / row["demand_lbs"]) * 100, 1) if row["demand_lbs"] else 0.0,
        axis=1,
    )
    merged["gap_lbs"] = merged["produced_lbs"] - merged["demand_lbs"]
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged[["date", "produced_lbs", "demand_lbs", "coverage_pct", "gap_lbs"]].sort_values("date").reset_index(drop=True)


def build_line_utilization(line_load: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if line_load.empty or "date" not in line_load.columns:
        return pd.DataFrame(columns=["line", "avg_util_pct", "avg_throughput_util_pct", "total_hours_used", "total_hours_available"])
    window = filter_date_range(line_load, "date", start, end)
    if window.empty:
        return pd.DataFrame(columns=["line", "avg_util_pct", "avg_throughput_util_pct", "total_hours_used", "total_hours_available"])

    grouped = (
        window.groupby("line", as_index=False)
        .agg(
            avg_util_pct=("util_pct", "mean"),
            avg_throughput_util_pct=("throughput_util_pct", "mean"),
            total_hours_used=("hours_used", "sum"),
            total_hours_available=("hours_available", "sum"),
        )
        .sort_values("avg_util_pct", ascending=False)
    )
    for column in ["avg_util_pct", "avg_throughput_util_pct", "total_hours_used", "total_hours_available"]:
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce").fillna(0.0).round(1)
    return grouped.reset_index(drop=True)


def build_bucket_usage_summary(bucket_usage: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if bucket_usage.empty or "date" not in bucket_usage.columns:
        return pd.DataFrame(columns=["bucket", "used_lbs", "available_lbs", "remaining_lbs", "util_pct"])
    window = filter_date_range(bucket_usage, "date", start, end)
    if window.empty:
        return pd.DataFrame(columns=["bucket", "used_lbs", "available_lbs", "remaining_lbs", "util_pct"])

    bucket_column = "bucket" if "bucket" in window.columns else "bucketId" if "bucketId" in window.columns else None
    if bucket_column is None:
        return pd.DataFrame(columns=["bucket", "used_lbs", "available_lbs", "remaining_lbs", "util_pct"])
    if bucket_column != "bucket":
        window = window.copy()
        window["bucket"] = window[bucket_column]

    source_used = (
        "usedLbs"
        if "usedLbs" in window.columns
        else "utilizedLbs"
        if "utilizedLbs" in window.columns
        else "used_lbs"
        if "used_lbs" in window.columns
        else None
    )
    source_available = "availableLbs" if "availableLbs" in window.columns else "available_lbs" if "available_lbs" in window.columns else None
    source_util = "utilPct" if "utilPct" in window.columns else "util_pct" if "util_pct" in window.columns else None

    if source_used is None:
        window["used_lbs"] = 0.0
        source_used = "used_lbs"
    if source_available is None:
        window["available_lbs"] = 0.0
        source_available = "available_lbs"
    if source_util is None:
        window["util_pct"] = 0.0
        source_util = "util_pct"

    summary = (
        window.groupby("bucket", as_index=False)
        .agg(
            used_lbs=(source_used, "sum"),
            available_lbs=(source_available, "mean"),
            util_pct=(source_util, "mean"),
        )
        .sort_values("util_pct", ascending=False)
        .reset_index(drop=True)
    )
    summary["remaining_lbs"] = summary["available_lbs"] - summary["used_lbs"]
    return summary[["bucket", "used_lbs", "available_lbs", "remaining_lbs", "util_pct"]]


def aggregate_job_metrics(
    decisions: pd.DataFrame,
    outputs: pd.DataFrame,
    sku_demands: pd.DataFrame,
    line_load: pd.DataFrame,
    bucket_usage: pd.DataFrame,
    focus_date: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict[str, float]:
    metrics = {
        "today_schedule_rows": float(len(build_todays_cut_schedule(decisions, focus_date))),
        "upcoming_rows": float(len(build_upcoming_cuts(decisions, focus_date))),
        "produced_lbs": _sum_series(outputs, "lbsProduced"),
        "demand_lbs": _sum_series(sku_demands, "demandValue"),
        "line_util_pct": 0.0,
        "bucket_util_pct": 0.0,
    }

    line_summary = build_line_utilization(line_load, focus_date, window_end)
    if not line_summary.empty and "avg_util_pct" in line_summary.columns:
        metrics["line_util_pct"] = round(float(pd.to_numeric(line_summary["avg_util_pct"], errors="coerce").fillna(0.0).mean()), 1)

    bucket_summary = build_bucket_usage_summary(bucket_usage, focus_date, window_end)
    if not bucket_summary.empty and "util_pct" in bucket_summary.columns:
        metrics["bucket_util_pct"] = round(float(pd.to_numeric(bucket_summary["util_pct"], errors="coerce").fillna(0.0).mean()), 1)

    metrics["coverage_pct"] = round((metrics["produced_lbs"] / metrics["demand_lbs"]) * 100, 1) if metrics["demand_lbs"] else 0.0
    return metrics
