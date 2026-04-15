"""
Monthly Contract Demand page - CRUD management and CSV upload for SKU/month demand.

This page references the Enumeration SKU API so monthly contract rows can be tied
to real SKU trade numbers.
"""

import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import (  # noqa: E402
    APIError,
    bulk_create_monthly_contract_demands,
    create_monthly_contract_demand,
    delete_monthly_contract_demand,
    search_monthly_contract_demands,
    search_skus,
    update_monthly_contract_demand,
)
from monthly_contract_import import parse_monthly_contract_upload_content  # noqa: E402


def _load_skus() -> list[dict]:
    try:
        return search_skus({})
    except APIError as error:
        if error.status_code == 0:
            st.error("Could not reach Enumeration API. Check your connection.")
        else:
            st.error(f"Failed to load SKUs: {error.detail}")
        return []


def _load_contracts(criteria: dict | None = None) -> list[dict]:
    try:
        return search_monthly_contract_demands(criteria or {})
    except APIError as error:
        if error.status_code == 0:
            st.error("Could not reach Scheduling API. Check your connection.")
        else:
            st.error(f"Failed to load monthly contract demand rows: {error.detail}")
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


def _build_search_criteria(sku_id: str, year_month: str) -> dict:
    criteria: dict[str, object] = {}
    if sku_id.strip():
        criteria["skuId"] = sku_id.strip()
    if year_month.strip():
        criteria["yearMonth"] = year_month.strip()
    return criteria


def _build_template_csv() -> str:
    return (
        "skuId,yearMonth,demandLbs\n"
        "50624,2026-01,3000\n"
        "50625,2026-02,3200\n"
    )


st.set_page_config(page_title="Monthly Contract Demand", page_icon="m", layout="wide")
st.title("Monthly Contract Demand")
st.caption("Manage monthly contract demand by SKU and year-month. Each SKU and month combination must be unique.")

if "monthly_contract_rows" not in st.session_state:
    st.session_state.monthly_contract_rows = None
if "monthly_contract_filters" not in st.session_state:
    st.session_state.monthly_contract_filters = {"skuId": "", "yearMonth": ""}
if "monthly_contract_skus" not in st.session_state:
    st.session_state.monthly_contract_skus = []

if st.session_state.monthly_contract_skus is None or not st.session_state.monthly_contract_skus:
    st.session_state.monthly_contract_skus = _load_skus()

sku_options = st.session_state.monthly_contract_skus or []
sku_lookup = {sku.get("tradeNumber", sku.get("_id", "")): sku for sku in sku_options if sku.get("tradeNumber")}
valid_sku_ids = set(sku_lookup.keys())

if st.session_state.monthly_contract_rows is None:
    st.session_state.monthly_contract_rows = _load_contracts()

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
    with st.form("search_monthly_contract_form"):
        col1, col2 = st.columns(2)
        with col1:
            search_sku_id = st.text_input(
                "SKU ID",
                value=st.session_state.monthly_contract_filters.get("skuId", ""),
                placeholder="50624",
            )
        with col2:
            search_year_month = st.text_input(
                "Year-Month (YYYY-MM)",
                value=st.session_state.monthly_contract_filters.get("yearMonth", ""),
                placeholder="2026-01",
            )
        search_submitted = st.form_submit_button("Search")

    if search_submitted:
        st.session_state.monthly_contract_filters = {
            "skuId": search_sku_id.strip(),
            "yearMonth": search_year_month.strip(),
        }
        st.session_state.monthly_contract_rows = _load_contracts(
            _build_search_criteria(search_sku_id, search_year_month)
        )

    if st.button("Reset Filters"):
        st.session_state.monthly_contract_filters = {"skuId": "", "yearMonth": ""}
        st.session_state.monthly_contract_rows = _load_contracts({})

if st.button("Refresh"):
    filters = st.session_state.monthly_contract_filters
    st.session_state.monthly_contract_rows = _load_contracts(
        _build_search_criteria(filters.get("skuId", ""), filters.get("yearMonth", ""))
    )

rows = st.session_state.monthly_contract_rows or []

if rows:
    df = pd.DataFrame(
        [
            {
                "id": row.get("_id", ""),
                "skuId": row.get("skuId", ""),
                "yearMonth": row.get("yearMonth", ""),
                "demandLbs": row.get("demandLbs", 0.0),
            }
            for row in rows
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("No monthly contract demand rows found.")

st.divider()

st.subheader("Create Monthly Contract Demand")

sku_id_options = [sku.get("tradeNumber", "") for sku in sku_options if sku.get("tradeNumber")]

with st.form("create_monthly_contract_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        if sku_id_options:
            selected_sku_id = st.selectbox("SKU ID", options=sku_id_options)
        else:
            selected_sku_id = st.text_input("SKU ID", placeholder="50624")
        create_month_value = st.date_input("Contract Month", value=date.today().replace(day=1))
    with col2:
        create_demand_lbs = st.number_input("Demand (lbs)", min_value=0.0, value=0.0, step=1.0)
    create_clicked = st.form_submit_button("Create")

if create_clicked:
    if not selected_sku_id.strip():
        st.warning("SKU ID is required.")
    elif valid_sku_ids and selected_sku_id.strip() not in valid_sku_ids:
        st.warning("Select a SKU that exists in the Enumeration SKU API.")
    else:
        try:
            create_monthly_contract_demand(
                {
                    "skuId": selected_sku_id.strip(),
                    "yearMonth": create_month_value.strftime("%Y-%m"),
                    "demandLbs": float(create_demand_lbs),
                }
            )
            st.success("Monthly contract demand created.")
            filters = st.session_state.monthly_contract_filters
            st.session_state.monthly_contract_rows = _load_contracts(
                _build_search_criteria(filters.get("skuId", ""), filters.get("yearMonth", ""))
            )
        except APIError as error:
            _handle_api_error(error, "create monthly contract demand")

st.subheader("CSV Upload")
st.caption(
    "Upload a CSV or TSV with either flat rows (skuId, yearMonth, demandLbs) or a wide sheet "
    "with a SKU column and month headers like Code, Feb-26, Mar-26, ..."
)

st.download_button(
    "Download CSV Template",
    data=_build_template_csv(),
    file_name="monthly_contract_template.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv", "tsv", "txt"])
if uploaded_file is not None:
    file_contents = uploaded_file.getvalue().decode("utf-8-sig")
    parsed_rows, row_errors, parse_error = parse_monthly_contract_upload_content(
        file_contents,
        valid_sku_ids,
    )

    if parse_error:
        st.error(parse_error)
    elif row_errors:
        st.error("Parsing failed for one or more rows:")
        for error in row_errors:
            st.error(error)
    elif not parsed_rows:
        st.warning("The uploaded CSV contains no rows.")
    else:
        st.info(f"Parsed {len(parsed_rows)} monthly contract row(s). Click Import to submit.")

        if st.button("Import Monthly Contract Rows"):
            try:
                result = bulk_create_monthly_contract_demands({"demands": parsed_rows})
                success_count = int(result.get("successful", 0))
                failed_count = int(result.get("failed", 0))
                errors = result.get("errors", [])

                st.success(
                    f"Imported {success_count} monthly contract row(s). Failed: {failed_count}."
                )
                if errors:
                    st.error("Some rows failed to import:")
                    for error in errors:
                        if isinstance(error, dict):
                            row_index = error.get("rowIndex", "?")
                            sku_id = error.get("skuId")
                            year_month = error.get("yearMonth")
                            message = error.get("error") or error.get("detail") or "Import failed"
                            if sku_id and year_month:
                                st.error(f"Row {row_index} ({sku_id} / {year_month}): {message}")
                            else:
                                st.error(f"Row {row_index}: {message}")
                        else:
                            st.error(str(error))
                filters = st.session_state.monthly_contract_filters
                st.session_state.monthly_contract_rows = _load_contracts(
                    _build_search_criteria(filters.get("skuId", ""), filters.get("yearMonth", ""))
                )
            except APIError as error:
                _handle_api_error(error, "bulk import monthly contract demand")

st.subheader("Edit / Delete Monthly Contract Demand")

if not rows:
    st.caption("No contract rows available to edit or delete.")
else:
    contract_lookup = {row.get("_id", ""): row for row in rows}
    selected_id = st.selectbox(
        "Select contract row",
        options=list(contract_lookup.keys()),
        format_func=lambda contract_id: (
            f"{contract_lookup[contract_id].get('skuId', '')} / "
            f"{contract_lookup[contract_id].get('yearMonth', '')} / "
            f"{contract_lookup[contract_id].get('demandLbs', 0.0)} lbs"
        ),
    )
    selected_row = contract_lookup.get(selected_id, {})
    selected_sku = selected_row.get("skuId", "")
    selected_year_month = selected_row.get("yearMonth", "")

    with st.form("edit_monthly_contract_form"):
        col1, col2 = st.columns(2)
        with col1:
            if sku_id_options:
                edit_sku_id = st.selectbox(
                    "SKU ID",
                    options=sku_id_options,
                    index=sku_id_options.index(selected_sku) if selected_sku in sku_id_options else 0,
                )
            else:
                edit_sku_id = st.text_input("SKU ID", value=selected_sku)

            try:
                edit_month_default = pd.Period(selected_year_month, freq="M").to_timestamp().date()
            except Exception:
                edit_month_default = date.today().replace(day=1)
            edit_month_value = st.date_input("Contract Month", value=edit_month_default)
        with col2:
            edit_demand_lbs = st.number_input(
                "Demand (lbs)",
                min_value=0.0,
                value=float(selected_row.get("demandLbs", 0.0)),
                step=1.0,
            )
        save_clicked = st.form_submit_button("Save Changes")

    if save_clicked:
        if not edit_sku_id.strip():
            st.warning("SKU ID is required.")
        elif valid_sku_ids and edit_sku_id.strip() not in valid_sku_ids:
            st.warning("Select a SKU that exists in the Enumeration SKU API.")
        else:
            try:
                update_monthly_contract_demand(
                    selected_id,
                    {
                        "skuId": edit_sku_id.strip(),
                        "yearMonth": edit_month_value.strftime("%Y-%m"),
                        "demandLbs": float(edit_demand_lbs),
                    },
                )
                st.success("Monthly contract demand updated.")
                filters = st.session_state.monthly_contract_filters
                st.session_state.monthly_contract_rows = _load_contracts(
                    _build_search_criteria(filters.get("skuId", ""), filters.get("yearMonth", ""))
                )
            except APIError as error:
                _handle_api_error(error, "update monthly contract demand")

    if st.button("Delete Selected Contract", type="secondary"):
        try:
            delete_monthly_contract_demand(selected_id)
            st.success("Monthly contract demand deleted.")
            filters = st.session_state.monthly_contract_filters
            st.session_state.monthly_contract_rows = _load_contracts(
                _build_search_criteria(filters.get("skuId", ""), filters.get("yearMonth", ""))
            )
        except APIError as error:
            _handle_api_error(error, "delete monthly contract demand")
