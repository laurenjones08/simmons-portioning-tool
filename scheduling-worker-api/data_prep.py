from __future__ import annotations

import json
import os
import random
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from urllib import error, request

import pandas as pd


# -----------------------------------------------------------------------------
# API client helpers
# -----------------------------------------------------------------------------


def _env_url(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.rstrip("/")
    return default.rstrip("/")


def _env_float(*names: str, default: float) -> float:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default


@dataclass(frozen=True)
class ApiEndpoints:
    scheduling_api_url: str = field(default_factory=lambda: _env_url("SCHEDULING_API_URL", default="http://api-gateway:80/api/scheduling"))
    global_config_api_url: str = field(default_factory=lambda: _env_url("GLOBAL_CONFIG_API_URL", default="http://api-gateway:80/api/config"))
    enumeration_api_url: str = field(default_factory=lambda: _env_url("ENUMERATION_API_URL", default="http://api-gateway:80/api/enumeration"))
    timeout_seconds: float = field(
        default_factory=lambda: _env_float(
            "DATA_PREP_TIMEOUT_SECONDS",
            "SCHEDULING_WORKER_API_TIMEOUT_SECONDS",
            default=120.0,
        )
    )


class ApiClientError(RuntimeError):
    pass


class BaseApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request_json(self, path: str, method: str = "GET", payload: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=body, headers=headers, method=method)

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ApiClientError(f"HTTP {exc.code} while calling {url}: {detail}") from exc
        except error.URLError as exc:
            raise ApiClientError(f"Could not reach {url}: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise ApiClientError(f"Timed out after {self.timeout_seconds:.0f}s while calling {url}") from exc
        except json.JSONDecodeError as exc:
            raise ApiClientError(f"Invalid JSON returned from {url}") from exc


class SchedulingApiClient(BaseApiClient):
    def search_sku_demands(self, criteria: Optional[dict] = None) -> List[dict]:
        return self._request_json("/sku-demands/search", method="POST", payload=criteria or {}) or []

    def bulk_create_sku_demands(self, demands: Sequence[dict]) -> Dict[str, Any]:
        payload = {"demands": list(demands)}
        response = self._request_json("/sku-demands/bulk", method="POST", payload=payload)
        return response or {"total": 0, "successful": 0, "failed": 0, "errors": []}

    def search_available_wip(self, criteria: Optional[dict] = None) -> List[dict]:
        return self._request_json("/available-wip/search", method="POST", payload=criteria or {}) or []

    def list_bucket_usage(self) -> List[dict]:
        return self._request_json("/bucket-usage/search", method="POST", payload={}) or []

    def bulk_search_monthly_contracts(self, sku_ids: Sequence[str], year_months: Sequence[str]) -> List[dict]:
        payload = {"skuIds": list(sku_ids), "yearMonths": list(year_months)}
        return self._request_json("/monthly-contracts/bulk-search", method="POST", payload=payload) or []

    def list_scheduling_decisions(self) -> List[dict]:
        return self._request_json("/scheduling-decisions/search", method="POST", payload={}) or []

    def list_scheduling_outputs(self) -> List[dict]:
        return self._request_json("/scheduling-outputs/search", method="POST", payload={}) or []


class GlobalConfigApiClient(BaseApiClient):
    def list_configs(self) -> List[dict]:
        return self._request_json("/config", method="GET") or []

    def list_lines(self) -> List[dict]:
        return self._request_json("/lines", method="GET") or []


class EnumerationApiClient(BaseApiClient):
    def list_skus(self) -> List[dict]:
        return self._request_json("/skus/search", method="POST", payload={}) or []

    def list_mixes(self) -> List[dict]:
        return self._request_json("/mixes/search", method="POST", payload={}) or []

    def list_buckets(self) -> List[dict]:
        return self._request_json("/buckets/search", method="POST", payload={}) or []

    def list_cut_strategies(self) -> List[dict]:
        return self._request_json("/cut-strategies/search", method="POST", payload={}) or []

    def list_mix_metrics(self, sku_trade_numbers: Optional[Sequence[str]] = None) -> List[dict]:
        if not sku_trade_numbers:
            return self._request_json("/metrics/search", method="POST", payload={}) or []

        metrics_by_id: Dict[str, dict] = {}
        for sku_trade_number in _normalize_ids(sku_trade_numbers):
            if not sku_trade_number:
                continue
            payload = {"skuTradeNumber": sku_trade_number}
            for metric in self._request_json("/metrics/search", method="POST", payload=payload) or []:
                metric_id = str(metric.get("_id") or metric.get("metricId") or "").strip()
                if metric_id:
                    metrics_by_id[metric_id] = metric
        return list(metrics_by_id.values())


@dataclass
class DataPrepSources:
    skus: List[dict] = field(default_factory=list)
    mixes: List[dict] = field(default_factory=list)
    buckets: List[dict] = field(default_factory=list)
    cut_strategies: List[dict] = field(default_factory=list)
    mix_metrics: List[dict] = field(default_factory=list)
    lines: List[dict] = field(default_factory=list)
    configs: List[dict] = field(default_factory=list)
    sku_demands: List[dict] = field(default_factory=list)
    available_wip: List[dict] = field(default_factory=list)
    bucket_usage: List[dict] = field(default_factory=list)
    scheduling_decisions: List[dict] = field(default_factory=list)
    scheduling_outputs: List[dict] = field(default_factory=list)
    api_timings: Dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Demo fallbacks / business-logic hook points
# -----------------------------------------------------------------------------


def _demo_parts() -> List[str]:
    return ["A", "B", "C", "D", "E", "F", "G", "H"]


def _emit_progress(
    progress_callback: Optional[Callable[[str, Optional[str], Optional[Dict[str, Any]]], None]],
    stage: str,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, message, details or {})


def _demo_decisions() -> List[str]:
    return [
        "P1_DSI888",
        "P1_DB20",
        "P2_DSI888",
        "P2_DSI884",
        "P3_DSI884",
        "P3_DB20",
        "P4_DB20",
        "P5_DSI888",
        "pack_DSI884",
    ]


def _demo_lines() -> List[str]:
    return ["DSI 888", "DSI 884", "DB20"]


def _demo_buckets() -> List[str]:
    return [
        "B 0-390",
        "B 390-440",
        "B 440-490",
        "B 490-540",
        "B 540-590",
        "B 590-640",
        "B 640-690",
        "B 690-1000",
    ]


def _demo_monthly_contract() -> Dict[tuple[str, pd.Period], float]:
    return {
        ("A", pd.Period("2026-01", freq="M")): 3000,
        ("A", pd.Period("2026-02", freq="M")): 3200,
        ("B", pd.Period("2026-01", freq="M")): 2200,
        ("B", pd.Period("2026-02", freq="M")): 2400,
        ("C", pd.Period("2026-01", freq="M")): 2600,
        ("C", pd.Period("2026-02", freq="M")): 2700,
        ("D", pd.Period("2026-01", freq="M")): 2100,
        ("D", pd.Period("2026-02", freq="M")): 2300,
        ("E", pd.Period("2026-01", freq="M")): 3400,
        ("E", pd.Period("2026-02", freq="M")): 3500,
        ("F", pd.Period("2026-01", freq="M")): 2000,
        ("F", pd.Period("2026-02", freq="M")): 2150,
        ("G", pd.Period("2026-01", freq="M")): 3100,
        ("G", pd.Period("2026-02", freq="M")): 3250,
        ("H", pd.Period("2026-01", freq="M")): 2500,
        ("H", pd.Period("2026-02", freq="M")): 2600,
    }


def _demo_bucket_of_k() -> Dict[str, str]:
    return {
        "P1_DSI888": "B 0-390",
        "P1_DB20": "B 0-390",
        "P2_DSI888": "B 440-490",
        "P2_DSI884": "B 440-490",
        "P3_DSI884": "B 490-540",
        "P3_DB20": "B 490-540",
        "P4_DB20": "B 540-590",
        "P5_DSI888": "B 590-640",
        "pack_DSI884": "B 390-440",
    }


def _demo_line_of_k() -> Dict[str, str]:
    return {
        "P1_DSI888": "DSI 888",
        "P1_DB20": "DB20",
        "P2_DSI888": "DSI 888",
        "P2_DSI884": "DSI 884",
        "P3_DSI884": "DSI 884",
        "P3_DB20": "DB20",
        "P4_DB20": "DB20",
        "P5_DSI888": "DSI 888",
        "pack_DSI884": "DSI 884",
    }


def _demo_yield_map() -> Dict[tuple[str, str], float]:
    return {
        ("A", "P1_DSI888"): 0.60, ("A", "P1_DB20"): 0.60, ("A", "pack_DSI884"): 0.40,
        ("A", "P2_DSI888"): 0.55, ("A", "P2_DSI884"): 0.55,
        ("A", "P3_DSI884"): 0.30, ("A", "P3_DB20"): 0.30,
        ("A", "P4_DB20"): 0.20, ("A", "P5_DSI888"): 0.10,
        ("B", "P1_DSI888"): 0.50, ("B", "P1_DB20"): 0.50, ("B", "pack_DSI884"): 0.50,
        ("B", "P2_DSI888"): 0.45, ("B", "P2_DSI884"): 0.45,
        ("B", "P3_DSI884"): 0.35, ("B", "P3_DB20"): 0.35,
        ("B", "P4_DB20"): 0.25, ("B", "P5_DSI888"): 0.15,
        ("C", "P1_DSI888"): 0.40, ("C", "P1_DB20"): 0.40, ("C", "pack_DSI884"): 0.60,
        ("C", "P2_DSI888"): 0.50, ("C", "P2_DSI884"): 0.50,
        ("C", "P3_DSI884"): 0.45, ("C", "P3_DB20"): 0.45,
        ("C", "P4_DB20"): 0.20, ("C", "P5_DSI888"): 0.10,
        ("D", "P1_DSI888"): 0.55, ("D", "P1_DB20"): 0.55, ("D", "pack_DSI884"): 0.35,
        ("D", "P2_DSI888"): 0.60, ("D", "P2_DSI884"): 0.60,
        ("D", "P3_DSI884"): 0.25, ("D", "P3_DB20"): 0.25,
        ("D", "P4_DB20"): 0.15, ("D", "P5_DSI888"): 0.10,
        ("E", "P1_DSI888"): 0.30, ("E", "P1_DB20"): 0.30, ("E", "pack_DSI884"): 0.65,
        ("E", "P2_DSI888"): 0.40, ("E", "P2_DSI884"): 0.40,
        ("E", "P3_DSI884"): 0.50, ("E", "P3_DB20"): 0.50,
        ("E", "P4_DB20"): 0.35, ("E", "P5_DSI888"): 0.20,
        ("F", "P1_DSI888"): 0.45, ("F", "P1_DB20"): 0.45, ("F", "pack_DSI884"): 0.40,
        ("F", "P2_DSI888"): 0.50, ("F", "P2_DSI884"): 0.50,
        ("F", "P3_DSI884"): 0.30, ("F", "P3_DB20"): 0.30,
        ("F", "P4_DB20"): 0.25, ("F", "P5_DSI888"): 0.15,
        ("G", "P1_DSI888"): 0.50, ("G", "P1_DB20"): 0.50, ("G", "pack_DSI884"): 0.55,
        ("G", "P2_DSI888"): 0.45, ("G", "P2_DSI884"): 0.45,
        ("G", "P3_DSI884"): 0.40, ("G", "P3_DB20"): 0.40,
        ("G", "P4_DB20"): 0.30, ("G", "P5_DSI888"): 0.20,
        ("H", "P1_DSI888"): 0.35, ("H", "P1_DB20"): 0.35, ("H", "pack_DSI884"): 0.45,
        ("H", "P2_DSI888"): 0.55, ("H", "P2_DSI884"): 0.55,
        ("H", "P3_DSI884"): 0.50, ("H", "P3_DB20"): 0.50,
        ("H", "P4_DB20"): 0.25, ("H", "P5_DSI888"): 0.15,
    }


def _demo_value_map() -> Dict[str, float]:
    return {
        "P1_DSI888": 2.0,
        "P1_DB20": 2.0,
        "pack_DSI884": 3.0,
        "P2_DSI888": 2.4,
        "P2_DSI884": 2.4,
        "P3_DSI884": 2.8,
        "P3_DB20": 2.8,
        "P4_DB20": 3.2,
        "P5_DSI888": 1.9,
    }


def _demo_rate_map() -> Dict[str, float]:
    return {
        "P1_DSI888": 100,
        "P1_DB20": 100,
        "pack_DSI884": 80,
        "P2_DSI888": 90,
        "P2_DSI884": 90,
        "P3_DSI884": 85,
        "P3_DB20": 85,
        "P4_DB20": 75,
        "P5_DSI888": 110,
    }


def _demo_hours_map() -> Dict[str, float]:
    return {"DSI 888": 8, "DSI 884": 8, "DB20": 8}


def _demo_line_throughput() -> Dict[str, float]:
    return {"DSI 888": 9000, "DSI 884": 3000, "DB20": 67000}


def _demo_bird_eligibility(parts: Sequence[str]) -> tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    big_allowed = {
        "A": 1,
        "B": 1,
        "C": 0,
        "D": 1,
        "E": 0,
        "F": 1,
        "G": 1,
        "H": 0,
    }
    small_allowed = {
        "A": 0,
        "B": 1,
        "C": 1,
        "D": 0,
        "E": 1,
        "F": 1,
        "G": 0,
        "H": 1,
    }
    normalized_parts = [str(part).strip() for part in parts if str(part).strip()]
    bird_type: Dict[str, str] = {}
    for part in normalized_parts:
        if big_allowed.get(part, 0) and small_allowed.get(part, 0):
            bird_type[part] = "all"
        elif big_allowed.get(part, 0):
            bird_type[part] = "big"
        elif small_allowed.get(part, 0):
            bird_type[part] = "small"
        else:
            bird_type[part] = "none"
    return big_allowed, small_allowed, bird_type


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _unique(seq: Iterable[Any]) -> List[Any]:
    seen = set()
    values = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        values.append(item)
    return values


def _clean_str(value: Any) -> str:
    return str(value).strip()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.Timestamp(value).normalize()
    except Exception:
        return None


def _config_map(configs: Sequence[dict]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for item in configs:
        key = item.get("key")
        if key:
            mapping[str(key)] = item.get("value")
    return mapping


def _job_value(job: Any, *names: str, default: Any = None) -> Any:
    if job is None:
        return default
    for name in names:
        if isinstance(job, dict) and name in job:
            value = job.get(name)
            if value is not None:
                return value
        if hasattr(job, name):
            value = getattr(job, name)
            if value is not None:
                return value
    return default


def _normalize_ids(values: Optional[Sequence[Any]]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in values or []:
        cleaned = _clean_str(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


def _month_strings(months: Sequence[pd.Period]) -> List[str]:
    return [month.strftime("%Y-%m") for month in months]


def _date_strings(dates: Sequence[pd.Timestamp]) -> List[str]:
    return [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates]


def _metric_id(metric: Dict[str, Any]) -> str:
    return _clean_str(metric.get("_id") or metric.get("metricId") or f"{metric.get('mixId')}:{metric.get('bucketId')}")


def _metric_mix_id(metric: Dict[str, Any]) -> str:
    return _clean_str(metric.get("mixId") or metric.get("mix_id"))


def _metric_bucket_id(metric: Dict[str, Any]) -> str:
    return _clean_str(metric.get("bucketId") or metric.get("bucket_id"))


def _metric_sku_keys(metric: Dict[str, Any]) -> List[str]:
    keys = metric.get("skuKeys") or metric.get("sku_keys") or []
    if isinstance(keys, str):
        keys = [keys]
    normalized = _normalize_ids(keys)
    if normalized:
        return normalized

    unit_plan = metric.get("unitPlan") or metric.get("unit_plan") or []
    sku_values = []
    for item in unit_plan:
        sku = _clean_str(item.get("sku"))
        if sku:
            sku_values.append(sku)
    return _normalize_ids(sku_values)


def _metric_unit_plan(metric: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(metric.get("unitPlan") or metric.get("unit_plan") or [])


def _mix_bird_size(mix: Dict[str, Any]) -> str:
    return _clean_str(mix.get("reqBirdSize") or mix.get("birdSize") or mix.get("req_bird_size")).upper()


def _normalize_line_type(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(_clean_str(value).upper().split())


def _line_id(line: Dict[str, Any]) -> str:
    return _clean_str(line.get("lineId") or line.get("_id") or line.get("friendlyName"))


def _line_type(line: Dict[str, Any]) -> str:
    return _normalize_line_type(
        line.get("lineType")
        or line.get("mfgType")
        or line.get("lineId")
        or line.get("friendlyName")
        or line.get("line_id")
    )


def _cut_strategy_id(mix: Dict[str, Any]) -> str:
    return _clean_str(mix.get("cutStrategyID") or mix.get("cutStrategyId") or mix.get("cut_strategy_id"))


def _cut_strategy_line_type(cut_strategy: Dict[str, Any]) -> str:
    return _normalize_line_type(cut_strategy.get("lineType") or cut_strategy.get("mfgType"))


def _line_permitted_cut_strategy_ids(line: Dict[str, Any]) -> List[str]:
    values = line.get("permittedCutStrategyIds") or line.get("permitted_cut_strategy_ids") or []
    if isinstance(values, str):
        values = [values]
    return _normalize_ids(values)


def _line_type_matches(line_type: Any, mfg_type: Any) -> bool:
    normalized_line = _normalize_line_type(line_type)
    normalized_mfg = _normalize_line_type(mfg_type)
    if not normalized_line or not normalized_mfg:
        return False
    return normalized_line == normalized_mfg


def _sku_bird_size(sku: Dict[str, Any]) -> str:
    return _clean_str(sku.get("birdSize") or sku.get("bird_size")).upper()


def _normalize_column_name(name: Any) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _find_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    lookup = {_normalize_column_name(column): column for column in df.columns}
    for candidate in candidates:
        column = lookup.get(_normalize_column_name(candidate))
        if column is not None:
            return column
    return None


def _load_short_term_demand_frame(short_term_file: Path) -> pd.DataFrame:
    suffix = short_term_file.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        separator = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(short_term_file, sep=separator)
    return pd.read_excel(short_term_file, sheet_name=0)


def extract_short_term_demand_records(
    short_term_file: Path,
    P: Sequence[str],
    allowed_due_dates: Optional[Sequence[pd.Timestamp]] = None,
) -> List[Dict[str, Any]]:
    """Return normalized short-term demand rows from a flat demand file."""

    df = _load_short_term_demand_frame(short_term_file)
    if df.empty:
        return []

    sku_col = _find_column(df, "sku", "skuId", "sku_id")
    demand_col = _find_column(df, "demand", "demandValue", "demand_value", "amount")
    type_col = _find_column(df, "type", "demandType", "demand_type")
    due_date_col = _find_column(df, "dueDate", "due_date", "due")

    missing_columns = [
        label
        for label, column in [
            ("sku", sku_col),
            ("demand", demand_col),
            ("type", type_col),
            ("dueDate", due_date_col),
        ]
        if column is None
    ]
    if missing_columns:
        raise ValueError(
            "Short-term demand file must contain columns for "
            + ", ".join(missing_columns)
            + ". Expected a flat file with one demand record per row."
        )

    sku_set = set(P)
    allowed_due_date_set = set(allowed_due_dates or [])
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        sku = _clean_str(row.get(sku_col))
        demand_type = _clean_str(row.get(type_col))
        due_date = _as_date(row.get(due_date_col))
        amount = _safe_float(row.get(demand_col))

        if not sku or sku not in sku_set or due_date is None or amount is None:
            continue
        if allowed_due_date_set and due_date not in allowed_due_date_set:
            continue
        if demand_type and demand_type.lower() != "short":
            continue

        records.append(
            {
                "skuId": sku,
                "demandValue": amount,
                "demandType": "Short",
                "dueDate": due_date.strftime("%Y-%m-%d"),
            }
        )

    return records


# -----------------------------------------------------------------------------
# API-backed data prep
# -----------------------------------------------------------------------------


@dataclass
class SchedulingWorkerDataPrep:
    endpoints: ApiEndpoints = field(default_factory=ApiEndpoints)
    use_demo_fallbacks: bool = True
    scheduling_api: SchedulingApiClient = field(init=False)
    config_api: GlobalConfigApiClient = field(init=False)
    enumeration_api: EnumerationApiClient = field(init=False)

    def __post_init__(self) -> None:
        self.scheduling_api = SchedulingApiClient(self.endpoints.scheduling_api_url, self.endpoints.timeout_seconds)
        self.config_api = GlobalConfigApiClient(self.endpoints.global_config_api_url, self.endpoints.timeout_seconds)
        self.enumeration_api = EnumerationApiClient(self.endpoints.enumeration_api_url, self.endpoints.timeout_seconds)

    def load_sources(
        self,
        sku_ids: Sequence[str] | None = None,
        due_dates: Sequence[pd.Timestamp] | None = None,
        progress_callback: Optional[Callable[[str, Optional[str], Optional[Dict[str, Any]]], None]] = None,
    ) -> DataPrepSources:
        sku_demand_criteria: Dict[str, Any] = {"demandType": "Short"}
        if sku_ids:
            sku_demand_criteria["skuIds"] = list(sku_ids)
        if due_dates:
            sku_demand_criteria["dueDates"] = _date_strings(due_dates)
        api_timings: Dict[str, float] = {}

        def timed_fetch(name: str, loader: Callable[[], List[dict]]) -> List[dict]:
            _emit_progress(progress_callback, "data_prep_api_call", f"Loading {name} from upstream API.", {"resource": name})
            started = time.perf_counter()
            rows = loader()
            api_timings[name] = round(time.perf_counter() - started, 3)
            _emit_progress(
                progress_callback,
                "data_prep_api_call_complete",
                f"Loaded {name}.",
                {"resource": name, "count": len(rows), "seconds": api_timings[name]},
            )
            return rows

        return DataPrepSources(
            skus=timed_fetch("skus", self.enumeration_api.list_skus),
            mixes=timed_fetch("mixes", self.enumeration_api.list_mixes),
            buckets=timed_fetch("buckets", self.enumeration_api.list_buckets),
            cut_strategies=timed_fetch("cut_strategies", self.enumeration_api.list_cut_strategies),
            mix_metrics=timed_fetch("mix_metrics", lambda: self.enumeration_api.list_mix_metrics(sku_ids)),
            lines=timed_fetch("lines", self.config_api.list_lines),
            configs=timed_fetch("configs", self.config_api.list_configs),
            sku_demands=timed_fetch("sku_demands", lambda: self.scheduling_api.search_sku_demands(sku_demand_criteria)),
            api_timings=api_timings,
        )

    def _prepare_demo_inputs(self, P: Sequence[str], T: Sequence[pd.Timestamp], week1_dates: Sequence[pd.Timestamp], M: Sequence[pd.Period]) -> Dict[str, Any]:
        K = _demo_decisions()
        L = _demo_lines()
        B = _demo_buckets()
        monthly_contract = {
            (part, month): _demo_monthly_contract().get((part, month), 0.0)
            for part in P
            for month in M
        }
        demo_wip = {
            "B 0-390": 30468.11,
            "B 390-440": 39254.81,
            "B 440-490": 56783.94,
            "B 490-540": 60929.4,
            "B 540-590": 48495.82,
            "B 590-640": 28630.86,
            "B 640-690": 12536.14,
            "B 690-1000": 5250.097,
        }
        WIP = {(bucket, day): demo_wip.get(bucket, 0.0) for bucket in B for day in T}
        demo_y = _demo_yield_map()
        Y = {(sku, decision): demo_y.get((sku, decision), 0.0) for sku in P for decision in K}
        big_allowed, small_allowed, bird_type = _demo_bird_eligibility(P)
        return {
            "P": list(P),
            "T": list(T),
            "K": K,
            "L": L,
            "B": B,
            "M": list(M),
            "week1_dates": list(week1_dates),
            "future_dates": list(T[len(week1_dates):]),
            "month_of_day": {day: day.to_period("M") for day in T},
            "bucket_of_k": _demo_bucket_of_k(),
            "line_of_k": _demo_line_of_k(),
            "base_wip_by_bucket": demo_wip,
            "WIP": WIP,
            "base_hours_by_line": _demo_hours_map(),
            "D_short": {},
            "D_week1": build_week1_demand(P, week1_dates, {}),
            "monthly_contract": monthly_contract,
            "Y": Y,
            "V": _demo_value_map(),
            "R": _demo_rate_map(),
            "H": {(line, day): _demo_hours_map().get(line, 0.0) for line in L for day in T},
            "line_throughput": _demo_line_throughput(),
            "big_allowed": big_allowed,
            "small_allowed": small_allowed,
            "bird_type": bird_type,
            "gamma": 0.1,
        }

    def _select_mix_metrics(self, sources: DataPrepSources, selected_sku_ids: Sequence[str]) -> List[dict]:
        selected = set(_normalize_ids(selected_sku_ids))
        metrics: List[dict] = []
        for metric in sources.mix_metrics:
            sku_keys = set(_metric_sku_keys(metric))
            if selected and not sku_keys:
                continue
            if selected and not sku_keys.issubset(selected):
                continue
            if not sku_keys and not selected:
                continue
            metrics.append(metric)
        return metrics

    def _select_mixes(self, sources: DataPrepSources, mix_ids: Sequence[str]) -> List[dict]:
        wanted = set(_normalize_ids(mix_ids))
        mixes: List[dict] = []
        for mix in sources.mixes:
            mix_id = _clean_str(mix.get("_id") or mix.get("mixId"))
            if mix_id in wanted:
                mixes.append(mix)
        return mixes

    def _select_lines(self, sources: DataPrepSources, plant_id: Optional[str]) -> List[dict]:
        lines: List[dict] = []
        for line in sources.lines:
            if not line.get("isActive", True):
                continue
            line_plant = _clean_str(line.get("plant") or line.get("reqPlant"))
            if plant_id and line_plant != plant_id:
                continue
            lines.append(line)
        lines.sort(key=lambda line: _line_id(line))
        return lines

    def _select_buckets(self, sources: DataPrepSources, metrics: Sequence[dict], wip_rows: Sequence[dict]) -> List[str]:
        buckets = [_clean_str(bucket.get("_id") or bucket.get("bucketId") or bucket.get("name")) for bucket in sources.buckets]
        buckets.extend(_metric_bucket_id(metric) for metric in metrics)
        buckets.extend(_clean_str(row.get("bucketId") or row.get("bucket")) for row in wip_rows)
        return _normalize_ids(buckets)

    def _build_base_wip_by_bucket(self, wip_rows: Sequence[dict], plant_id: Optional[str], buckets: Sequence[str]) -> Dict[str, float]:
        bucket_set = set(buckets)
        base: Dict[str, float] = {}
        for row in wip_rows:
            bucket_id = _clean_str(row.get("bucketId") or row.get("bucket_id") or row.get("bucket"))
            if not bucket_id or bucket_id not in bucket_set:
                continue
            row_plant = _clean_str(row.get("plantName") or row.get("plant_name"))
            if plant_id and row_plant != plant_id:
                continue
            lbs = _safe_float(row.get("availableLbs") or row.get("available_lbs"))
            if lbs is None:
                continue
            base[bucket_id] = lbs
        return base

    def _build_hours_and_throughput(self, lines: Sequence[dict], dates: Sequence[pd.Timestamp]) -> tuple[Dict[str, float], Dict[tuple[str, pd.Timestamp], float], Dict[str, float]]:
        base_hours_by_line: Dict[str, float] = {}
        line_throughput: Dict[str, float] = {}
        for line in lines:
            line_id = _line_id(line)
            if not line_id:
                continue
            hours = _safe_float(line.get("hoursOfLaborAvailablePerShift") or line.get("hours_of_labor_available_per_shift"))
            if hours is not None:
                base_hours_by_line[line_id] = hours
            throughput = _safe_float(line.get("lineThroughput") or line.get("line_throughput"))
            if throughput is not None:
                line_throughput[line_id] = throughput
        if not line_throughput and self.use_demo_fallbacks:
            line_throughput = _demo_line_throughput()
        if not base_hours_by_line and self.use_demo_fallbacks:
            base_hours_by_line = _demo_hours_map()
        for line in lines:
            line_id = _line_id(line)
            if not line_id:
                continue
            base_hours_by_line.setdefault(line_id, 8.0)
            line_throughput.setdefault(line_id, 1.0)
            if base_hours_by_line[line_id] <= 0:
                base_hours_by_line[line_id] = 8.0
            if line_throughput[line_id] <= 0:
                line_throughput[line_id] = 1.0
        H = {(line_id, day): base_hours_by_line.get(line_id, 0.0) for line_id in base_hours_by_line for day in dates}
        return base_hours_by_line, H, line_throughput
    def _build_short_term_demand(
        self,
        sources: DataPrepSources,
        week1_dates: Sequence[pd.Timestamp],
        parts: Sequence[str],
        short_term_file: Optional[str],
    ) -> Dict[tuple[str, pd.Timestamp], float]:
        demand: Dict[tuple[str, pd.Timestamp], float] = {}
        week1_date_set = set(week1_dates)

        for record in sources.sku_demands:
            sku = _clean_str(record.get("skuId") or record.get("sku_id"))
            due_date = _as_date(record.get("dueDate") or record.get("due_date"))
            demand_type = _clean_str(record.get("demandType") or record.get("demand_type"))
            amount = _safe_float(record.get("demandValue") or record.get("demand_value"))
            if not sku or due_date is None or amount is None:
                continue
            if sku not in parts:
                continue
            if week1_date_set and due_date not in week1_date_set:
                continue
            if demand_type and demand_type.lower() != "short":
                continue
            demand[(sku, due_date)] = demand.get((sku, due_date), 0.0) + amount

        if demand:
            return demand

        if short_term_file:
            path = Path(short_term_file)
            if path.exists():
                return load_short_term_demand_from_table(path, parts, week1_dates)

        return demand

    def _build_wip_map(self, sources: DataPrepSources, buckets: Sequence[str], dates: Sequence[pd.Timestamp], plant_id: Optional[str]) -> Dict[tuple[str, pd.Timestamp], float]:
        wip: Dict[tuple[str, pd.Timestamp], float] = {}

        for record in sources.available_wip:
            bucket = _clean_str(record.get("bucketId") or record.get("bucket_id") or record.get("bucket"))
            if not bucket or bucket not in buckets:
                continue
            if plant_id and _clean_str(record.get("plantName") or record.get("plant_name")) not in ("", plant_id):
                continue
            available = _safe_float(record.get("availableLbs") or record.get("available_lbs"))
            if available is None:
                continue
            for date in dates:
                wip[(bucket, date)] = available

        if wip:
            return wip

        if self.use_demo_fallbacks:
            demo = {
                "B 0-390": 30468.11,
                "B 390-440": 39254.81,
                "B 440-490": 56783.94,
                "B 490-540": 60929.4,
                "B 540-590": 48495.82,
                "B 590-640": 28630.86,
                "B 640-690": 12536.14,
                "B 690-1000": 5250.097,
            }
            return {(bucket, date): demo.get(bucket, 0.0) for bucket in buckets for date in dates}
        return wip

    def _build_bucket_usage_map(self, sources: DataPrepSources) -> Dict[tuple[str, pd.Timestamp], float]:
        usage: Dict[tuple[str, pd.Timestamp], float] = {}
        for record in sources.bucket_usage:
            bucket = _clean_str(record.get("bucketId") or record.get("bucket_id"))
            date = _as_date(record.get("date") or record.get("usage_date"))
            if not bucket or date is None:
                continue
            utilized = _safe_float(record.get("utilizedLbs") or record.get("utilized_lbs"))
            if utilized is None:
                continue
            usage[(bucket, date)] = usage.get((bucket, date), 0.0) + utilized
        return usage

    def _build_monthly_contract(
        self,
        sku_ids: Sequence[str],
        months: Sequence[pd.Period],
    ) -> Dict[tuple[str, pd.Period], float]:
        contract: Dict[tuple[str, pd.Period], float] = {}
        if not sku_ids or not months:
            return contract

        rows = self.scheduling_api.bulk_search_monthly_contracts(sku_ids, _month_strings(months))
        for row in rows:
            sku_id = _clean_str(row.get("skuId") or row.get("sku_id"))
            year_month = _clean_str(row.get("yearMonth") or row.get("year_month"))
            if not sku_id or not year_month:
                continue
            if sku_id not in sku_ids:
                continue
            try:
                month = pd.Period(year_month, freq="M")
            except Exception:
                continue
            if month not in months:
                continue
            demand = _safe_float(row.get("demandLbs") or row.get("demand_lbs"))
            if demand is None:
                continue
            contract[(sku_id, month)] = demand

        if contract:
            return contract

        if self.use_demo_fallbacks:
            defaults = _demo_monthly_contract()
            return {(sku_id, month): defaults.get((sku_id, month), 0.0) for sku_id in sku_ids for month in months}
        return contract

    def _build_bucket_of_k(self, metrics: Sequence[dict]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for metric in metrics:
            metric_id = _metric_id(metric)
            bucket_id = _metric_bucket_id(metric)
            if metric_id and bucket_id:
                mapping[metric_id] = bucket_id
        if mapping:
            return mapping
        return _demo_bucket_of_k() if self.use_demo_fallbacks else {}

    def _build_line_of_k(
            self,
            mixes: Sequence[dict],
            lines: Sequence[dict],
            cut_strategies: Sequence[dict],
    ) -> Dict[str, str]:
        cut_strategy_lookup = {
            _clean_str(strategy.get("_id") or strategy.get("cutStrategyId")): strategy
            for strategy in cut_strategies
            if _clean_str(strategy.get("_id") or strategy.get("cutStrategyId"))
        }
        line_candidates: List[tuple[str, str, List[str]]] = []
        for line in lines:
            line_type = _line_type(line)
            line_id = _line_id(line)
            if not line_type or not line_id:
                continue
            line_candidates.append((line_type, line_id, _line_permitted_cut_strategy_ids(line)))

        mapping: Dict[str, str] = {}
        for mix in mixes:
            mix_id = _clean_str(mix.get("_id") or mix.get("mixId"))
            if not mix_id:
                continue
            mfg_type = _normalize_line_type(mix.get("mfgType") or mix.get("mfg_type"))
            mix_cut_strategy_id = _cut_strategy_id(mix)
            cut_strategy = cut_strategy_lookup.get(mix_cut_strategy_id, {})
            effective_line_type = _cut_strategy_line_type(cut_strategy) or mfg_type

            strategy_candidates = [
                line_id
                for line_type, line_id, permitted_cut_strategy_ids in line_candidates
                if mix_cut_strategy_id and mix_cut_strategy_id in permitted_cut_strategy_ids
            ]
            candidates = strategy_candidates or [
                line_id
                for line_type, line_id, _permitted in line_candidates
                if _line_type_matches(line_type, effective_line_type)
            ]
            if candidates:
                mapping[mix_id] = random.choice(candidates)

        if mapping:
            return mapping
        return _demo_line_of_k() if self.use_demo_fallbacks else {}

    def _filter_runnable_metrics(
        self,
        metrics: Sequence[dict],
        line_of_k: Dict[str, str],
    ) -> List[dict]:
        runnable_mix_ids = set(line_of_k)
        return [metric for metric in metrics if _metric_mix_id(metric) in runnable_mix_ids]

    def _build_yield_map(self, metrics: Sequence[dict], parts: Sequence[str]) -> Dict[tuple[str, str], float]:
        part_set = set(parts)
        yield_map: Dict[tuple[str, str], float] = {}
        for metric in metrics:
            metric_id = _metric_id(metric)
            unit_plan = _metric_unit_plan(metric)
            metric_total = sum(_safe_float(item.get("totalWeightInPlan") or item.get("total_weight_in_plan")) or 0.0 for item in unit_plan)
            for item in unit_plan:
                sku = _clean_str(item.get("sku"))
                if sku not in part_set:
                    continue
                pct = _safe_float(item.get("pctOfTotal") or item.get("pct_of_total"))
                if pct is None:
                    weight = _safe_float(item.get("totalWeightInPlan") or item.get("total_weight_in_plan")) or 0.0
                    pct = (weight / metric_total * 100.0) if metric_total > 0 else 0.0
                yield_map[(sku, metric_id)] = yield_map.get((sku, metric_id), 0.0) + (pct / 100.0)
        if yield_map:
            full_map: Dict[tuple[str, str], float] = {}
            metric_ids = [_metric_id(metric) for metric in metrics]
            for sku in parts:
                for metric_id in metric_ids:
                    full_map[(sku, metric_id)] = yield_map.get((sku, metric_id), 0.0)
            return full_map
        if self.use_demo_fallbacks:
            return _demo_yield_map()
        return {}

    def _build_value_map(self, metrics: Sequence[dict]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for metric in metrics:
            metric_id = _metric_id(metric)
            value = _safe_float(metric.get("value"))
            if value is not None:
                values[metric_id] = value
        if values:
            return values
        return _demo_value_map() if self.use_demo_fallbacks else {}

    def _build_rate_map(self, mixes: Sequence[dict], metrics: Sequence[dict]) -> Dict[str, float]:
        mix_lookup = {_clean_str(mix.get("_id") or mix.get("mixId")): mix for mix in mixes}
        rates: Dict[str, float] = {}
        for metric in metrics:
            metric_id = _metric_id(metric)
            mix_id = _metric_mix_id(metric)
            mix_doc = mix_lookup.get(mix_id, {})
            belt_speed = _safe_float(mix_doc.get("beltSpeed") or mix_doc.get("belt_speed"))
            if belt_speed is None or belt_speed <= 0:
                belt_speed = 1.0
            # TODO: convert belt speed (feet/minute) to lbs/hour using the real production formula. For now, we will just use the belt speed as a proxy for the rate.
            rates[metric_id] = belt_speed
        if rates:
            return rates
        return _demo_rate_map() if self.use_demo_fallbacks else {}

    def _build_bird_eligibility(
        self,
        mixes: Sequence[dict],
        skus: Sequence[dict],
        metrics: Sequence[dict],
        parts: Sequence[str],
    ) -> tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
        part_list = [str(part).strip() for part in parts if str(part).strip()]
        part_set = set(part_list)
        big_allowed = {part: 0 for part in part_list}
        small_allowed = {part: 0 for part in part_list}

        mix_lookup = {_clean_str(mix.get("_id") or mix.get("mixId")): mix for mix in mixes}
        sku_bird_lookup = {
            _clean_str(sku.get("tradeNumber") or sku.get("trade_number") or sku.get("_id")): _sku_bird_size(sku)
            for sku in skus
            if _clean_str(sku.get("tradeNumber") or sku.get("trade_number") or sku.get("_id"))
        }

        def apply_requirement(sku_id: str, requirement: str) -> None:
            req = requirement.upper()
            if req == "ALL":
                big_allowed[sku_id] = 1
                small_allowed[sku_id] = 1
            elif req == "BB":
                big_allowed[sku_id] = 1
            elif req == "SB":
                small_allowed[sku_id] = 1

        for metric in metrics:
            mix_id = _metric_mix_id(metric)
            mix_doc = mix_lookup.get(mix_id, {})
            requirement = _mix_bird_size(mix_doc)
            if not requirement:
                continue

            allowed_skus = set(_metric_sku_keys(metric))
            if not allowed_skus:
                skus_map = mix_doc.get("skus") or {}
                if isinstance(skus_map, dict):
                    allowed_skus = set(_normalize_ids(skus_map.keys()))

            for sku_id in allowed_skus:
                if sku_id in part_set:
                    apply_requirement(sku_id, requirement)

        for sku_id, requirement in sku_bird_lookup.items():
            if sku_id in part_set and not (big_allowed[sku_id] or small_allowed[sku_id]):
                apply_requirement(sku_id, requirement)

        bird_type: Dict[str, str] = {}
        for part in part_list:
            if big_allowed[part] and small_allowed[part]:
                bird_type[part] = "all"
            elif big_allowed[part]:
                bird_type[part] = "big"
            elif small_allowed[part]:
                bird_type[part] = "small"
            else:
                bird_type[part] = "none"

        return big_allowed, small_allowed, bird_type

    def _build_gamma(self, configs: Sequence[dict]) -> float:
        config_values = _config_map(configs)
        value = _safe_float(config_values.get("scheduling.gamma_value"))
        if value is None:
            value = _safe_float(config_values.get("scheduling.gamma"))
        if value is not None:
            return value
        return 0.1

    def prepare(
        self,
        job: Any = None,
        short_term_file: Optional[str] = None,
        plan_start_date: Optional[str] = None,
        horizon_days: Optional[int] = None,
        plant_id: Optional[str] = None,
        sku_ids: Optional[Sequence[str]] = None,
        progress_callback: Optional[Callable[[str, Optional[str], Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        overall_started = time.perf_counter()
        prep_timings: Dict[str, float] = {}
        plan_start_date = _job_value(job, "plan_start_date", "planStartDate", default=plan_start_date or "2026-01-05")
        horizon_days = int(_job_value(job, "horizon_days", "horizonDays", default=horizon_days or 12))
        plant_id = _clean_str(_job_value(job, "plant_id", "plantId", default=plant_id))
        short_term_file = _job_value(job, "short_term_file", "shortTermFile", default=short_term_file)
        sku_ids = _normalize_ids(_job_value(job, "sku_ids", "skuIds", default=sku_ids))

        if not sku_ids and self.use_demo_fallbacks:
            sku_ids = _demo_parts()

        start_date = pd.Timestamp(plan_start_date)
        all_calendar_days = pd.date_range(start=start_date, periods=max(horizon_days * 2, 40), freq="D")
        production_dates = [d.normalize() for d in all_calendar_days if d.weekday() < 6][:horizon_days]
        T = production_dates
        week1_dates = T[:6]
        future_dates = T[6:]
        M = sorted({d.to_period("M") for d in T})
        month_of_day = {d: d.to_period("M") for d in T}

        _emit_progress(progress_callback, "data_prep_loading_sources", "Loading dataprep sources from upstream APIs.")
        started = time.perf_counter()
        sources = self.load_sources(sku_ids, week1_dates, progress_callback=progress_callback)
        prep_timings["load_sources"] = round(time.perf_counter() - started, 3)
        available_wip_rows = self.scheduling_api.search_available_wip({"plantName": plant_id}) if plant_id else self.scheduling_api.search_available_wip()
        prep_timings["available_wip"] = round(time.perf_counter() - started - prep_timings["load_sources"], 3)
        selected_metrics = self._select_mix_metrics(sources, sku_ids)
        if not selected_metrics:
            if self.use_demo_fallbacks:
                return self._prepare_demo_inputs(sku_ids, T, week1_dates, M)
            raise ValueError("No mix metrics found for the selected skuIds")

        selected_mix_ids = _normalize_ids(_metric_mix_id(metric) for metric in selected_metrics)
        selected_mixes = self._select_mixes(sources, selected_mix_ids)
        selected_lines = self._select_lines(sources, plant_id)
        if not selected_lines:
            if self.use_demo_fallbacks:
                return self._prepare_demo_inputs(sku_ids, T, week1_dates, M)
            raise ValueError(f"No active lines found for plantId '{plant_id}'")

        _emit_progress(
            progress_callback,
            "data_prep_filtering_metrics",
            "Filtering mix metrics to plant-runnable decisions.",
            {
                "selectedMetricCount": len(selected_metrics),
                "selectedMixCount": len(selected_mixes),
                "selectedLineCount": len(selected_lines),
            },
        )
        mix_line_of_k = self._build_line_of_k(selected_mixes, selected_lines, sources.cut_strategies)
        selected_metrics = self._filter_runnable_metrics(selected_metrics, mix_line_of_k)
        if not selected_metrics:
            if self.use_demo_fallbacks:
                return self._prepare_demo_inputs(sku_ids, T, week1_dates, M)
            raise ValueError(
                f"No runnable mix metrics found for skuIds {sku_ids} at plantId '{plant_id}'. "
                "The selected plant does not have a matching active line for the available mixes."
            )

        selected_mix_ids = _normalize_ids(_metric_mix_id(metric) for metric in selected_metrics)
        selected_mixes = self._select_mixes(sources, selected_mix_ids)
        line_of_k = {
            _metric_id(metric): mix_line_of_k[_metric_mix_id(metric)]
            for metric in selected_metrics
            if _metric_mix_id(metric) in mix_line_of_k
        }
        prep_timings["filter_metrics"] = round(time.perf_counter() - overall_started - sum(prep_timings.values()), 3)

        selected_buckets = self._select_buckets(sources, selected_metrics, available_wip_rows)
        if not selected_buckets:
            if self.use_demo_fallbacks:
                return self._prepare_demo_inputs(sku_ids, T, week1_dates, M)
            raise ValueError("No buckets were available for the selected mix metrics")

        _emit_progress(
            progress_callback,
            "data_prep_building_inputs",
            "Building demand, capacity, yield, and contract inputs.",
            {
                "runnableMetricCount": len(selected_metrics),
                "selectedBucketCount": len(selected_buckets),
            },
        )
        D_short = self._build_short_term_demand(sources, week1_dates, sku_ids, short_term_file)
        D_week1 = build_week1_demand(sku_ids, week1_dates, D_short)
        base_wip_by_bucket = self._build_base_wip_by_bucket(available_wip_rows, plant_id, selected_buckets)
        if not base_wip_by_bucket and self.use_demo_fallbacks:
            base_wip_by_bucket = {
                "B 0-390": 30468.11,
                "B 390-440": 39254.81,
                "B 440-490": 56783.94,
                "B 490-540": 60929.4,
                "B 540-590": 48495.82,
                "B 590-640": 28630.86,
                "B 640-690": 12536.14,
                "B 690-1000": 5250.097,
            }

        WIP = {(bucket, day): base_wip_by_bucket.get(bucket, 0.0) for bucket in selected_buckets for day in T}
        monthly_contract = self._build_monthly_contract(sku_ids, M)
        bucket_of_k = self._build_bucket_of_k(selected_metrics)
        Y = self._build_yield_map(selected_metrics, sku_ids)
        V = self._build_value_map(selected_metrics)
        R = self._build_rate_map(selected_mixes, selected_metrics)
        base_hours_by_line, H, line_throughput = self._build_hours_and_throughput(selected_lines, T)
        gamma = self._build_gamma(sources.configs)
        prep_timings["assemble_inputs"] = round(time.perf_counter() - overall_started - sum(prep_timings.values()), 3)

        metric_ids = [_metric_id(metric) for metric in selected_metrics]
        missing_buckets = [metric_id for metric_id in metric_ids if metric_id not in bucket_of_k]
        missing_lines = [metric_id for metric_id in metric_ids if metric_id not in line_of_k]
        if missing_buckets or missing_lines:
            raise ValueError(
                "Unable to derive bucket_of_k/line_of_k for all selected mix metrics. "
                f"Missing buckets: {missing_buckets or 'none'}, missing lines: {missing_lines or 'none'}."
            )

        if self.use_demo_fallbacks and not line_throughput:
            line_throughput = _demo_line_throughput()

        big_allowed, small_allowed, bird_type = self._build_bird_eligibility(
            selected_mixes,
            sources.skus,
            selected_metrics,
            sku_ids,
        )
        prep_timings["total_data_prep"] = round(time.perf_counter() - overall_started, 3)
        _emit_progress(
            progress_callback,
            "data_prep_complete",
            "Finished dataprep.",
            {
                "timings": {**sources.api_timings, **prep_timings},
                "counts": {
                    "skuCount": len(sku_ids),
                    "metricCount": len(selected_metrics),
                    "mixCount": len(selected_mixes),
                    "lineCount": len(selected_lines),
                    "bucketCount": len(selected_buckets),
                    "dayCount": len(T),
                },
            },
        )

        return {
            "P": sku_ids,
            "T": T,
            "K": [ _metric_id(metric) for metric in selected_metrics ],
            "L": [_clean_str(line.get("lineId") or line.get("friendlyName") or line.get("lineType")) for line in selected_lines],
            "B": selected_buckets,
            "M": M,
            "week1_dates": week1_dates,
            "future_dates": future_dates,
            "month_of_day": month_of_day,
            "bucket_of_k": bucket_of_k,
            "line_of_k": line_of_k,
            "base_wip_by_bucket": base_wip_by_bucket,
            "WIP": WIP,
            "base_hours_by_line": base_hours_by_line,
            "D_short": D_short,
            "D_week1": D_week1,
            "monthly_contract": monthly_contract,
            "Y": Y,
            "V": V,
            "R": R,
            "H": H,
            "L_delay": {},
            "line_throughput": line_throughput,
            "big_allowed": big_allowed,
            "small_allowed": small_allowed,
            "bird_type": bird_type,
            "gamma": gamma,
            "dataPrepTimings": {**sources.api_timings, **prep_timings},
            "dataPrepCounts": {
                "skuCount": len(sku_ids),
                "metricCount": len(selected_metrics),
                "mixCount": len(selected_mixes),
                "lineCount": len(selected_lines),
                "bucketCount": len(selected_buckets),
                "dayCount": len(T),
            },
            "sources": sources,
        }


# -----------------------------------------------------------------------------
# Short-term demand file fallback for local/manual scenarios
# -----------------------------------------------------------------------------


def load_short_term_demand_from_table(
    short_term_file: Path,
    P: Sequence[str],
    week1_dates: Sequence[pd.Timestamp],
) -> Dict[tuple[str, pd.Timestamp], float]:
    """Load short-term demand rows from a flat file.

    The file is expected to contain one demand record per row with columns
    similar to: sku, demand, type, dueDate.
    """

    demand: Dict[tuple[str, pd.Timestamp], float] = {}
    records = extract_short_term_demand_records(short_term_file, P, week1_dates)
    for record in records:
        sku = _clean_str(record.get("skuId"))
        due_date = _as_date(record.get("dueDate"))
        amount = _safe_float(record.get("demandValue"))
        if not sku or due_date is None or amount is None:
            continue
        demand[(sku, due_date)] = demand.get((sku, due_date), 0.0) + amount

    return demand


load_short_term_demand_from_excel = load_short_term_demand_from_table


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def build_week1_demand(P: Sequence[str], week1_dates: Sequence[pd.Timestamp], D_short: Dict[tuple[str, pd.Timestamp], float]) -> Dict[tuple[str, pd.Timestamp], float]:
    demand = {}
    for part in P:
        for day in week1_dates:
            demand[(part, day)] = D_short.get((part, day), 0.0)
    return demand


def get_model_inputs(
    job: Any = None,
    short_term_file: Optional[str] = None,
    plan_start_date: str = "2026-01-05",
    horizon_days: int = 12,
    plant_id: Optional[str] = None,
    sku_ids: Optional[Sequence[str]] = None,
    endpoints: Optional[ApiEndpoints] = None,
    use_demo_fallbacks: bool = True,
    progress_callback: Optional[Callable[[str, Optional[str], Optional[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    prep = SchedulingWorkerDataPrep(
        endpoints=endpoints or ApiEndpoints(),
        use_demo_fallbacks=use_demo_fallbacks,
    )
    return prep.prepare(
        job=job,
        short_term_file=short_term_file,
        plan_start_date=plan_start_date,
        horizon_days=horizon_days,
        plant_id=plant_id,
        sku_ids=sku_ids,
        progress_callback=progress_callback,
    )
