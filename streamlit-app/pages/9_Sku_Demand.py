"""
SKU Demand page - CRUD management and CSV upload for plant scheduling demand.

This page references the Enumeration SKU API so demand rows can be tied to real
SKU trade numbers.
"""

import csv
import io
import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import (  # noqa: E402
    APIError,
    bulk_create_sku_demands,
    create_sku_demand,
    delete_sku_demand,
    search_sku_demands,
    search_skus,
    update_sku_demand,
)


_SKU_FIELD_KEY_MAP = {
    "skuid": "skuId",
    "tradenumber": "skuId",
    "demandvalue": "demandValue",
    "demandlbs": "demandValue",
    "lbs": "demandValue",
    "demandtype": "demandType",
    "duedate": "dueDate",
}


def _normalize_keys(record: dict) -> dict:
    normalized: dict = {}
    for key, value in record.items():
        if key is None:
            continue
        key_text = str(key).strip()
        if not key_text:
            continue
        normalized[_SKU_FIELD_KEY_MAP.get(key_text.lower(), key_text)] = value
    return normalized


def _parse_demand_type(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"short", "s"}:
        return "Short"
    if text in {"long", "l"}:
        return "Long"
    raise ValueError(f"Invalid demandType '{value}'. Use Short or Long.")


def _parse_date_value(value) -> date:
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("dueDate is required.")
    return date.fromisoformat(text)


def _parse_float_value(value, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")
    return parsed


def _load_skus() -> list[dict]:
    try:
        return search_skus({})
    except APIError as error:
        if error.status_code == 0:
            st.error("Could not reach Enumeration API. Check your connection.")
        else:
            st.error(f"Failed to load SKUs: {error.detail}")
        return []


def _load_demands(criteria: dict | None = None) -> list[dict]:
    try:
        return search_sku_demands(criteria or {})
    except APIError as error:
        if error.status_code == 0:
            st.error("Could not reach Scheduling API. Check your connection.")
        else:
            st.error(f"Failed to load SKU demands: {error.detail}")
        return []


def _handle_api_error(error: APIError, action: str) -> None:
    if error.status_code == 0:
        st.error("Could not reach Scheduling API. Check your connection.")
    elif error.status_code == 404:
        st.warning("Resource not found.")
    elif error.status_code == 409:
        st.error(error.detail)
    else:
        st.error(f"Failed to {action}: {error.detail}")


def _build_search_criteria(sku_id: str, demand_type: str, due_date: date | str | None) -> dict:
    criteria: dict[str, object] = {}
    if sku_id.strip():
        criteria["skuId"] = sku_id.strip()
    if demand_type.strip():
        criteria["demandType"] = demand_type.strip()
    if due_date:
        if isinstance(due_date, date):
            criteria["dueDate"] = due_date.isoformat()
        else:
            criteria["dueDate"] = str(due_date).strip()
    return criteria


def _parse_csv_payload(uploaded_file, valid_sku_ids: set[str]) -> tuple[list[dict], list[str], str | None]:
    file_name = uploaded_file.name.lower()
    if not file_name.endswith(".csv"):
        return [], [], "Please upload a CSV file."

    content = uploaded_file.getvalue().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        return [], [], "CSV file is missing a header row."

    records: list[dict] = []
    errors: list[str] = []

    for row_idx, row in enumerate(reader, start=2):
        normalized = _normalize_keys(dict(row))
        try:
            sku_id = str(normalized.get("skuId", "")).strip()
            if not sku_id:
                raise ValueError("skuId is required.")
            if valid_sku_ids and sku_id not in valid_sku_ids:
                raise ValueError(f"skuId '{sku_id}' does not exist in the Enumeration SKU API.")
            demand_value = _parse_float_value(normalized.get("demandValue", ""), "demandValue")
            demand_type = _parse_demand_type(normalized.get("demandType", ""))
            due_date = _parse_date_value(normalized.get("dueDate", ""))
            records.append(
                {
                    "skuId": sku_id,
                    "demandValue": demand_value,
                    "demandType": demand_type,
                    "dueDate": due_date.isoformat(),
                }
            )
        except Exception as exc:
            errors.append(f"Row {row_idx}: {exc}")

    return records, errors, None


def _build_template_csv() -> str:
    return (
        "skuId,demandValue,demandType,dueDate\n"
        "50624,1500,Short,2026-04-15\n"
        "50625,2500,Long,2026-04-16\n"
    )


st.set_page_config(page_title="SKU Demand", page_icon="d")
st.title("SKU Demand")
st.caption("Manage demand records by SKU, demand type, and due date.")

if "sku_demand_rows" not in st.session_state:
    st.session_state.sku_demand_rows = None
if "sku_options" not in st.session_state:
    st.session_state.sku_options = []
if "sku_demand_filters" not in st.session_state:
    st.session_state.sku_demand_filters = {"skuId": "", "demandType": "", "dueDate": ""}

if st.session_state.sku_options is None or not st.session_state.sku_options:
    st.session_state.sku_options = _load_skus()

sku_options = st.session_state.sku_options or []
sku_lookup = {sku.get("tradeNumber", sku.get("_id", "")): sku for sku in sku_options if sku.get("tradeNumber")}
valid_sku_ids = set(sku_lookup.keys())

if st.session_state.sku_demand_rows is None:
    st.session_state.sku_demand_rows = _load_demands()

with st.expander("SKU Reference from Enumeration API", expanded=False):
    if sku_options:
        st.write(f"Loaded {len(sku_options)} SKU record(s) from the Enumeration SKU API.")
        sku_df = pd.DataFrame(
            [
                {
                    "skuId": sku.get("tradeNumber", ""),
                    "customerName": sku.get("customerName", ""),
                    "customerType": sku.get("customerType", ""),
                    "productType": sku.get("productType", ""),
                    "prodPlant": sku.get("prodPlant", ""),
                    "birdSize": sku.get("birdSize", ""),
                }
                for sku in sku_options
            ]
        )
        st.dataframe(sku_df, width="stretch", hide_index=True)
    else:
        st.info("No SKUs returned by the Enumeration SKU API.")

with st.expander("Search Filters", expanded=True):
    with st.form("search_sku_demand_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            search_sku_id = st.text_input(
                "SKU ID",
                value=st.session_state.sku_demand_filters.get("skuId", ""),
                placeholder="50624",
            )
        with col2:
            search_demand_type = st.selectbox(
                "Demand Type",
                options=["", "Short", "Long"],
                index=["", "Short", "Long"].index(st.session_state.sku_demand_filters.get("demandType", "")),
            )
        with col3:
            search_due_date = st.text_input(
                "Due Date (YYYY-MM-DD)",
                value=st.session_state.sku_demand_filters.get("dueDate", ""),
                placeholder="2026-04-15",
            )
        search_submitted = st.form_submit_button("Search")

    if search_submitted:
        st.session_state.sku_demand_filters = {
            "skuId": search_sku_id.strip(),
            "demandType": search_demand_type.strip(),
            "dueDate": search_due_date.strip(),
        }
        st.session_state.sku_demand_rows = _load_demands(
            _build_search_criteria(search_sku_id, search_demand_type, search_due_date)
        )

    if st.button("Reset Filters"):
        st.session_state.sku_demand_filters = {"skuId": "", "demandType": "", "dueDate": ""}
        st.session_state.sku_demand_rows = _load_demands({})

if st.button("Refresh"):
    filters = st.session_state.sku_demand_filters
    st.session_state.sku_demand_rows = _load_demands(
        _build_search_criteria(filters.get("skuId", ""), filters.get("demandType", ""), filters.get("dueDate"))
    )

rows = st.session_state.sku_demand_rows or []

if rows:
    df = pd.DataFrame(
        [
            {
                "id": row.get("_id", ""),
                "skuId": row.get("skuId", ""),
                "demandValue": row.get("demandValue", 0.0),
                "demandType": row.get("demandType", ""),
                "dueDate": row.get("dueDate", ""),
            }
            for row in rows
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("No SKU demand rows found.")

st.divider()

st.subheader("Create SKU Demand")

sku_id_options = [sku.get("tradeNumber", "") for sku in sku_options if sku.get("tradeNumber")]

with st.form("create_sku_demand_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    with c1:
        if sku_id_options:
            selected_sku_id = st.selectbox("SKU ID", options=sku_id_options)
        else:
            selected_sku_id = st.text_input("SKU ID", placeholder="50624")
        create_demand_type = st.selectbox("Demand Type", options=["Short", "Long"])
    with c2:
        create_demand_value = st.number_input("Demand Value (lbs)", min_value=0.0, value=0.0, step=1.0)
        create_due_date = st.date_input("Due Date")
    create_clicked = st.form_submit_button("Create")

if create_clicked:
    if not selected_sku_id.strip():
        st.warning("SKU ID is required.")
    elif valid_sku_ids and selected_sku_id.strip() not in valid_sku_ids:
        st.warning("Select a SKU that exists in the Enumeration SKU API.")
    else:
        try:
            create_sku_demand(
                {
                    "skuId": selected_sku_id.strip(),
                    "demandValue": float(create_demand_value),
                    "demandType": create_demand_type,
                    "dueDate": create_due_date.isoformat(),
                }
            )
            st.success("SKU demand created.")
            st.session_state.sku_demand_rows = _load_demands(
                _build_search_criteria(
                    st.session_state.sku_demand_filters.get("skuId", ""),
                    st.session_state.sku_demand_filters.get("demandType", ""),
                    st.session_state.sku_demand_filters.get("dueDate"),
                )
            )
        except APIError as error:
            _handle_api_error(error, "create SKU demand")

st.subheader("CSV Upload")
st.caption("Upload a CSV with columns skuId or tradeNumber, demandValue, demandType, and dueDate.")

st.download_button(
    "Download CSV Template",
    data=_build_template_csv(),
    file_name="sku_demand_template.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    parsed_rows, row_errors, parse_error = _parse_csv_payload(uploaded_file, valid_sku_ids)

    if parse_error:
        st.error(parse_error)
    elif row_errors:
        st.error("Parsing failed for one or more rows:")
        for error in row_errors:
            st.error(error)
    elif not parsed_rows:
        st.warning("The uploaded CSV contains no rows.")
    else:
        st.info(f"Parsed {len(parsed_rows)} demand row(s). Click Import to submit.")

        if st.button("Import SKU Demands"):
            try:
                result = bulk_create_sku_demands({"demands": parsed_rows})
                success_count = int(result.get("successful", 0))
                failed_count = int(result.get("failed", 0))
                errors = result.get("errors", [])

                st.success(
                    f"Imported {success_count} SKU demand row(s). Failed: {failed_count}."
                )
                if errors:
                    st.error("Some rows failed to import:")
                    for error in errors:
                        if isinstance(error, dict):
                            row_index = error.get("rowIndex", "?")
                            sku_id = error.get("skuId")
                            message = error.get("error") or error.get("detail") or "Import failed"
                            if sku_id:
                                st.error(f"Row {row_index} ({sku_id}): {message}")
                            else:
                                st.error(f"Row {row_index}: {message}")
                        else:
                            st.error(str(error))
                st.session_state.sku_demand_rows = _load_demands(
                    _build_search_criteria(
                        st.session_state.sku_demand_filters.get("skuId", ""),
                        st.session_state.sku_demand_filters.get("demandType", ""),
                        st.session_state.sku_demand_filters.get("dueDate"),
                    )
                )
            except APIError as error:
                _handle_api_error(error, "bulk import SKU demands")

st.subheader("Edit / Delete SKU Demand")

if not rows:
    st.caption("No demand rows available to edit or delete.")
else:
    demand_lookup = {row.get("_id", ""): row for row in rows}
    selected_id = st.selectbox(
        "Select demand row",
        options=list(demand_lookup.keys()),
        format_func=lambda demand_id: (
            f"{demand_lookup[demand_id].get('skuId', '')} / "
            f"{demand_lookup[demand_id].get('demandType', '')} / "
            f"{demand_lookup[demand_id].get('dueDate', '')}"
        ),
    )
    selected_row = demand_lookup.get(selected_id, {})

    with st.form("edit_sku_demand_form"):
        e1, e2 = st.columns(2)
        with e1:
            edit_sku_id = st.selectbox(
                "SKU ID",
                options=sku_id_options or [selected_row.get("skuId", "")],
                index=(sku_id_options.index(selected_row.get("skuId", "")) if selected_row.get("skuId", "") in sku_id_options else 0),
            ) if sku_id_options else st.text_input("SKU ID", value=selected_row.get("skuId", ""))
            edit_demand_type = st.selectbox(
                "Demand Type",
                options=["Short", "Long"],
                index=["Short", "Long"].index(selected_row.get("demandType", "Short")) if selected_row.get("demandType", "Short") in ["Short", "Long"] else 0,
            )
        with e2:
            edit_demand_value = st.number_input(
                "Demand Value (lbs)",
                min_value=0.0,
                value=float(selected_row.get("demandValue", 0.0)),
                step=1.0,
            )
            due_date_value = selected_row.get("dueDate", date.today().isoformat())
            edit_due_date = st.date_input(
                "Due Date",
                value=date.fromisoformat(str(due_date_value)),
            )
        save_clicked = st.form_submit_button("Save Changes")

    if save_clicked:
        if not edit_sku_id.strip():
            st.warning("SKU ID is required.")
        elif valid_sku_ids and edit_sku_id.strip() not in valid_sku_ids:
            st.warning("Select a SKU that exists in the Enumeration SKU API.")
        else:
            try:
                update_sku_demand(
                    selected_id,
                    {
                        "skuId": edit_sku_id.strip(),
                        "demandValue": float(edit_demand_value),
                        "demandType": edit_demand_type,
                        "dueDate": edit_due_date.isoformat(),
                    },
                )
                st.success("SKU demand updated.")
                st.session_state.sku_demand_rows = _load_demands(
                    _build_search_criteria(
                        st.session_state.sku_demand_filters.get("skuId", ""),
                        st.session_state.sku_demand_filters.get("demandType", ""),
                        st.session_state.sku_demand_filters.get("dueDate"),
                    )
                )
            except APIError as error:
                _handle_api_error(error, "update SKU demand")

    if st.button("Delete Selected Demand", type="secondary"):
        try:
            delete_sku_demand(selected_id)
            st.success("SKU demand deleted.")
            st.session_state.sku_demand_rows = _load_demands(
                _build_search_criteria(
                    st.session_state.sku_demand_filters.get("skuId", ""),
                    st.session_state.sku_demand_filters.get("demandType", ""),
                    st.session_state.sku_demand_filters.get("dueDate"),
                )
            )
        except APIError as error:
            _handle_api_error(error, "delete SKU demand")
