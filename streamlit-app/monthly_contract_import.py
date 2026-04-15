from __future__ import annotations

import csv
import io
import math
from datetime import date, datetime
from typing import Any

import pandas as pd


_MONTHLY_CONTRACT_FIELD_KEY_MAP = {
    "skuid": "skuId",
    "tradenumber": "skuId",
    "code": "skuId",
    "sku": "skuId",
    "skuid": "skuId",
    "yearmonth": "yearMonth",
    "month": "yearMonth",
    "demandlbs": "demandLbs",
    "demandvalue": "demandLbs",
    "lbs": "demandLbs",
}

_SKU_HEADER_KEYS = {"skuid", "tradenumber", "code", "sku"}
_FLAT_HEADER_KEYS = {"skuid", "tradenumber", "yearmonth", "month", "demandlbs", "demandvalue", "lbs"}


def _normalize_header_key(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch not in {" ", "_"})


def _sniff_dialect(content: str) -> csv.Dialect:
    sample = content[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
    except csv.Error:
        return csv.get_dialect("excel")


def _read_tabular_content(content: str) -> pd.DataFrame:
    dialect = _sniff_dialect(content)
    return pd.read_csv(
        io.StringIO(content),
        sep=dialect.delimiter,
        dtype=str,
        keep_default_na=False,
    )


def _parse_year_month_value(value: Any) -> str:
    if isinstance(value, date):
        return value.strftime("%Y-%m")

    text = str(value).strip()
    if not text:
        raise ValueError("yearMonth is required.")

    for fmt in (
        "%Y-%m",
        "%b-%y",
        "%B-%y",
        "%b-%Y",
        "%B-%Y",
        "%Y/%m",
        "%m/%Y",
        "%b %y",
        "%B %y",
    ):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m")
        except ValueError:
            continue

    try:
        return pd.Period(pd.to_datetime(text, errors="raise"), freq="M").strftime("%Y-%m")
    except Exception as exc:
        raise ValueError("yearMonth must be a recognizable month label.") from exc


def _parse_float_value(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc

    if math.isnan(parsed) or math.isinf(parsed):
        raise ValueError(f"{field_name} must be numeric.")
    if parsed < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")
    return parsed


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if key is None:
            continue
        key_text = str(key).strip()
        if not key_text:
            continue
        normalized[_normalize_header_key(key_text)] = value
    return normalized


def _valid_sku_check(sku_id: str, valid_sku_ids: set[str]) -> None:
    if valid_sku_ids and sku_id not in valid_sku_ids:
        raise ValueError(f"skuId '{sku_id}' does not exist in the Enumeration SKU API.")


def _parse_flat_monthly_contract_rows(
    rows: list[dict[str, Any]],
    valid_sku_ids: set[str],
) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []

    for row_idx, row in enumerate(rows, start=2):
        normalized = _normalize_row(row)
        try:
            sku_id = str(
                normalized.get("skuid")
                or normalized.get("tradenumber")
                or normalized.get("code")
                or normalized.get("sku")
                or ""
            ).strip()
            if not sku_id:
                raise ValueError("skuId is required.")
            _valid_sku_check(sku_id, valid_sku_ids)

            year_month_value = normalized.get("yearmonth") or normalized.get("month") or ""
            year_month = _parse_year_month_value(year_month_value)
            demand_lbs = _parse_float_value(
                normalized.get("demandlbs") or normalized.get("demandvalue") or normalized.get("lbs") or "",
                "demandLbs",
            )
            records.append(
                {
                    "skuId": sku_id,
                    "yearMonth": year_month,
                    "demandLbs": demand_lbs,
                }
            )
        except Exception as exc:
            errors.append(f"Row {row_idx}: {exc}")

    return records, errors


def _parse_wide_monthly_contract_rows(
    rows: list[dict[str, Any]],
    header_names: list[str],
    valid_sku_ids: set[str],
) -> tuple[list[dict], list[str], str | None]:
    sku_column = next(
        (name for name in header_names if _normalize_header_key(name) in _SKU_HEADER_KEYS),
        None,
    )
    if sku_column is None:
        return [], [], "Could not find a SKU column. Use a header like Code, skuId, or tradeNumber."

    month_columns = [name for name in header_names if name != sku_column]
    if not month_columns:
        return [], [], "Could not find any month columns."

    month_labels: dict[str, str] = {}
    for column in month_columns:
        try:
            month_labels[column] = _parse_year_month_value(column)
        except Exception:
            return [], [], f"Column '{column}' is not a recognizable month header."

    records: list[dict] = []
    errors: list[str] = []

    for row_idx, row in enumerate(rows, start=2):
        sku_id = str(row.get(sku_column, "")).strip()
        if not sku_id:
            errors.append(f"Row {row_idx}: skuId is required.")
            continue

        try:
            _valid_sku_check(sku_id, valid_sku_ids)
        except Exception as exc:
            errors.append(f"Row {row_idx}: {exc}")
            continue

        for column in month_columns:
            raw_value = str(row.get(column, "")).strip()
            if not raw_value:
                continue
            try:
                records.append(
                    {
                        "skuId": sku_id,
                        "yearMonth": month_labels[column],
                        "demandLbs": _parse_float_value(raw_value, "demandLbs"),
                    }
                )
            except Exception as exc:
                errors.append(f"Row {row_idx}, column {column}: {exc}")

    return records, errors, None


def parse_monthly_contract_upload_content(
    content: str,
    valid_sku_ids: set[str] | None = None,
) -> tuple[list[dict], list[str], str | None]:
    valid_sku_ids = valid_sku_ids or set()
    if not content.strip():
        return [], [], "CSV file is empty."

    try:
        table = _read_tabular_content(content)
    except Exception as exc:
        return [], [], f"Unable to read the file: {exc}"

    if table.empty and len(table.columns) == 0:
        return [], [], "CSV file is missing a header row."

    header_names = [str(column) for column in table.columns]
    normalized_headers = {_normalize_header_key(column) for column in header_names}
    if _FLAT_HEADER_KEYS.intersection(normalized_headers):
        records, errors = _parse_flat_monthly_contract_rows(table.to_dict(orient="records"), valid_sku_ids)
        return records, errors, None

    return _parse_wide_monthly_contract_rows(table.to_dict(orient="records"), header_names, valid_sku_ids)
