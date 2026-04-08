"""
SKUs page — search, CRUD management, and bulk import for SKU records.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
"""

import sys
import os
import io
import json
import csv

import streamlit as st
import pandas as pd

# Allow importing api_client from the parent directory when running as a page
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import (
    APIError,
    search_skus,
    create_sku,
    update_sku,
    delete_sku,
    batch_import_skus,
)

# ---------------------------------------------------------------------------
# Pure validation helper (extracted for property-based testing)
# ---------------------------------------------------------------------------

def validate_sku_weights(min_w: float, target_w: float, max_w: float) -> str | None:
    """Return an error message string if invalid, or None if valid.

    Valid when: minWeight < maxWeight AND minWeight <= targetWeight <= maxWeight
    """
    if min_w >= max_w:
        return f"minWeight ({min_w}) must be less than maxWeight ({max_w})."
    if target_w < min_w or target_w > max_w:
        return (
            f"targetWeight ({target_w}) must be between "
            f"minWeight ({min_w}) and maxWeight ({max_w})."
        )
    return None


_SKU_FIELD_KEY_MAP = {
    "tradenumber": "tradeNumber",
    "customername": "customerName",
    "customertype": "customerType",
    "producttype": "productType",
    "unitspercut": "unitsPerCut",
    "prodplant": "prodPlant",
    "minweight": "minWeight",
    "maxweight": "maxWeight",
    "targetweight": "targetWeight",
    "birdsize": "birdSize",
    "allowedparts": "allowedParts",
}


def _normalize_sku_record_keys(record: dict) -> dict:
    normalized: dict = {}
    for key, value in record.items():
        if key is None:
            continue
        key_text = str(key).strip()
        if not key_text:
            continue
        mapped_key = _SKU_FIELD_KEY_MAP.get(key_text.lower(), key_text)
        normalized[mapped_key] = value
    return normalized


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "skus" not in st.session_state:
        st.session_state.skus = []
    if "sku_searched" not in st.session_state:
        st.session_state.sku_searched = False


def _search_skus(criteria: dict) -> None:
    """Call POST /skus/search and store results in session state."""
    try:
        st.session_state.skus = search_skus(criteria)
        st.session_state.sku_searched = True
    except APIError as e:
        if e.status_code == 0:
            st.error("Could not reach Enumeration API. Check your connection.")
        else:
            st.error(f"Search failed: {e.detail}")


def _handle_api_error(e: APIError, action: str) -> None:
    if e.status_code == 0:
        st.error("Could not reach Enumeration API. Check your connection.")
    elif e.status_code == 404:
        st.warning("Resource not found.")
    elif e.status_code in (400, 409):
        st.error(e.detail)
    else:
        st.error(f"Failed to {action}: {e.detail}")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="SKUs", page_icon="📦")
st.title("SKUs")

_init_state()

# ---------------------------------------------------------------------------
# Search form
# ---------------------------------------------------------------------------

st.subheader("Search SKUs")

with st.form("search_skus_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        filter_prod_plant = st.text_input("prodPlant", value="")
    with col2:
        filter_bird_size = st.text_input("birdSize", value="")
    with col3:
        filter_customer_type = st.text_input("customerType", value="")
    with col4:
        filter_product_type = st.text_input("productType", value="")
    search_submitted = st.form_submit_button("Search")

if search_submitted:
    criteria: dict = {}
    if filter_prod_plant.strip():
        criteria["prodPlant"] = filter_prod_plant.strip()
    if filter_bird_size.strip():
        criteria["birdSize"] = filter_bird_size.strip()
    if filter_customer_type.strip():
        criteria["customerType"] = filter_customer_type.strip()
    if filter_product_type.strip():
        criteria["productType"] = filter_product_type.strip()
    _search_skus(criteria)

# ---------------------------------------------------------------------------
# SKUs table
# ---------------------------------------------------------------------------

skus = st.session_state.skus

if st.session_state.sku_searched:
    if skus:
        df = pd.DataFrame(
            [
                {
                    "id": s.get("_id", ""),
                    "tradeNumber": s.get("tradeNumber", ""),
                    "customerName": s.get("customerName", ""),
                    "customerType": s.get("customerType", ""),
                    "productType": s.get("productType", ""),
                    "prodPlant": s.get("prodPlant", ""),
                    "birdSize": s.get("birdSize", ""),
                    "minWeight": s.get("minWeight"),
                    "targetWeight": s.get("targetWeight"),
                    "maxWeight": s.get("maxWeight"),
                    "unitsPerCut": s.get("unitsPerCut"),
                    "allowedParts": ", ".join(s.get("allowedParts", [])),
                }
                for s in skus
            ]
        )
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.info("No SKUs found for the given filters.")

# ---------------------------------------------------------------------------
# Create SKU form
# ---------------------------------------------------------------------------

st.subheader("Create SKU")

with st.form("create_sku_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    with c1:
        new_trade_number = st.text_input("tradeNumber *")
        new_customer_name = st.text_input("customerName *")
        new_customer_type = st.text_input("customerType *")
        new_product_type = st.text_input("productType *")
        new_units_per_cut = st.number_input("unitsPerCut *", min_value=1, value=1, step=1)
        new_prod_plant = st.text_input("prodPlant *")
    with c2:
        new_bird_size = st.text_input("birdSize *")
        new_min_weight = st.number_input("minWeight *", value=0.0, step=0.1)
        new_target_weight = st.number_input("targetWeight *", value=0.5, step=0.1)
        new_max_weight = st.number_input("maxWeight *", value=1.0, step=0.1)
        new_allowed_parts = st.text_input(
            "allowedParts (comma-separated)", value=""
        )
    create_submitted = st.form_submit_button("Create SKU")

if create_submitted:
    err = validate_sku_weights(new_min_weight, new_target_weight, new_max_weight)
    if err:
        st.warning(err)
    elif not new_trade_number.strip():
        st.warning("tradeNumber is required.")
    else:
        parts = [p.strip() for p in new_allowed_parts.split(",") if p.strip()]
        payload = {
            "tradeNumber": new_trade_number.strip(),
            "customerName": new_customer_name.strip(),
            "customerType": new_customer_type.strip(),
            "productType": new_product_type.strip(),
            "unitsPerCut": int(new_units_per_cut),
            "prodPlant": new_prod_plant.strip(),
            "birdSize": new_bird_size.strip(),
            "minWeight": new_min_weight,
            "targetWeight": new_target_weight,
            "maxWeight": new_max_weight,
            "allowedParts": parts,
        }
        try:
            create_sku(payload)
            st.success("SKU created.")
        except APIError as e:
            _handle_api_error(e, "create SKU")

# ---------------------------------------------------------------------------
# Edit / Delete section
# ---------------------------------------------------------------------------

st.subheader("Edit / Delete SKU")

if not skus:
    st.caption("Run a search above to load SKUs for editing or deletion.")
else:
    sku_options = {s.get("_id", ""): s for s in skus}
    selected_sku_id = st.selectbox(
        "Select SKU",
        options=list(sku_options.keys()),
        format_func=lambda sid: (
            f"{sku_options[sid].get('tradeNumber', sid)}  "
            f"(plant={sku_options[sid].get('prodPlant', '')}, "
            f"birdSize={sku_options[sid].get('birdSize', '')})"
        ),
    )

    sel = sku_options.get(selected_sku_id, {})

    with st.form("edit_sku_form"):
        ec1, ec2 = st.columns(2)
        with ec1:
            edit_trade_number = st.text_input("tradeNumber *", value=sel.get("tradeNumber", ""))
            edit_customer_name = st.text_input("customerName *", value=sel.get("customerName", ""))
            edit_customer_type = st.text_input("customerType *", value=sel.get("customerType", ""))
            edit_product_type = st.text_input("productType *", value=sel.get("productType", ""))
            edit_units_per_cut = st.number_input(
                "unitsPerCut *", min_value=1, value=int(sel.get("unitsPerCut", 1)), step=1
            )
            edit_prod_plant = st.text_input("prodPlant *", value=sel.get("prodPlant", ""))
        with ec2:
            edit_bird_size = st.text_input("birdSize *", value=sel.get("birdSize", ""))
            edit_min_weight = st.number_input(
                "minWeight *", value=float(sel.get("minWeight", 0.0)), step=0.1
            )
            edit_target_weight = st.number_input(
                "targetWeight *", value=float(sel.get("targetWeight", 0.5)), step=0.1
            )
            edit_max_weight = st.number_input(
                "maxWeight *", value=float(sel.get("maxWeight", 1.0)), step=0.1
            )
            edit_allowed_parts = st.text_input(
                "allowedParts (comma-separated)",
                value=", ".join(sel.get("allowedParts", [])),
            )
        save_clicked = st.form_submit_button("Save Changes")

    if save_clicked:
        err = validate_sku_weights(edit_min_weight, edit_target_weight, edit_max_weight)
        if err:
            st.warning(err)
        elif not edit_trade_number.strip():
            st.warning("tradeNumber is required.")
        else:
            parts = [p.strip() for p in edit_allowed_parts.split(",") if p.strip()]
            payload = {
                "tradeNumber": edit_trade_number.strip(),
                "customerName": edit_customer_name.strip(),
                "customerType": edit_customer_type.strip(),
                "productType": edit_product_type.strip(),
                "unitsPerCut": int(edit_units_per_cut),
                "prodPlant": edit_prod_plant.strip(),
                "birdSize": edit_bird_size.strip(),
                "minWeight": edit_min_weight,
                "targetWeight": edit_target_weight,
                "maxWeight": edit_max_weight,
                "allowedParts": parts,
            }
            try:
                update_sku(selected_sku_id, payload)
                st.success("SKU updated.")
                _search_skus({})
            except APIError as e:
                _handle_api_error(e, "update SKU")

    if st.button("🗑️ Delete Selected SKU", type="secondary"):
        try:
            delete_sku(selected_sku_id)
            st.success("SKU deleted.")
            # Refresh with last search — re-run empty search to reload
            _search_skus({})
        except APIError as e:
            _handle_api_error(e, "delete SKU")

# ---------------------------------------------------------------------------
# Bulk Import section
# ---------------------------------------------------------------------------

st.subheader("Bulk Import SKUs")
st.caption("Upload a CSV or JSON file containing SKU records.")

uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    records: list[dict] = []
    parse_error: str | None = None
    row_parse_errors: list[str] = []

    try:
        if file_name.endswith(".json"):
            content = uploaded_file.read().decode("utf-8")
            data = json.loads(content)
            if isinstance(data, list):
                for idx, item in enumerate(data, start=1):
                    if isinstance(item, dict):
                        records.append(_normalize_sku_record_keys(item))
                    else:
                        row_parse_errors.append(
                            f"JSON item {idx}: expected an object, got {type(item).__name__}"
                        )
            elif isinstance(data, dict):
                if isinstance(data.get("skus"), list):
                    for idx, item in enumerate(data["skus"], start=1):
                        if isinstance(item, dict):
                            records.append(_normalize_sku_record_keys(item))
                        else:
                            row_parse_errors.append(
                                f"JSON 'skus' item {idx}: expected an object, got {type(item).__name__}"
                            )
                else:
                    records = [_normalize_sku_record_keys(data)]
            else:
                parse_error = "JSON file must contain an array of SKU objects or a single SKU object."
        else:
            # CSV
            content = uploaded_file.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row_idx, row in enumerate(reader, start=2):
                # Coerce numeric fields
                record: dict = _normalize_sku_record_keys(dict(row))
                for float_field in ("minWeight", "maxWeight", "targetWeight"):
                    if float_field in record and record[float_field] != "":
                        try:
                            record[float_field] = float(record[float_field])
                        except ValueError:
                            row_parse_errors.append(
                                f"Row {row_idx}, field '{float_field}': expected a number, got '{record[float_field]}'"
                            )
                if "unitsPerCut" in record and record["unitsPerCut"] != "":
                    try:
                        record["unitsPerCut"] = int(record["unitsPerCut"])
                    except ValueError:
                        row_parse_errors.append(
                            f"Row {row_idx}, field 'unitsPerCut': expected an integer, got '{record['unitsPerCut']}'"
                        )
                if "allowedParts" in record and isinstance(record["allowedParts"], str):
                    record["allowedParts"] = [
                        p.strip() for p in record["allowedParts"].split(",") if p.strip()
                    ]
                records.append(record)
    except Exception as exc:
        parse_error = f"Failed to parse file: {exc}"

    if parse_error:
        st.error(parse_error)
    elif row_parse_errors:
        st.error("Parsing failed for one or more rows:")
        for row_error in row_parse_errors:
            st.error(row_error)
    elif not records:
        st.warning("The uploaded file contains no records.")
    else:
        st.info(f"Parsed {len(records)} record(s). Click Import to submit.")

        if st.button("📥 Import SKUs"):
            try:
                result = batch_import_skus(records)
                total = result.get("total", len(records))
                successful = result.get("successful", 0)
                failed = result.get("failed", 0)
                errors = result.get("errors", [])

                st.success(
                    f"Import complete — Total: {total}, Successful: {successful}, Failed: {failed}"
                )
                if errors:
                    st.error("Errors during import:")
                    for err_item in errors:
                        if isinstance(err_item, dict):
                            row = err_item.get("index", "?")
                            field = err_item.get("field")
                            message = err_item.get("error") or err_item.get("detail") or str(err_item)
                            if field:
                                st.error(f"  Row {row}, {field}: {message}")
                            else:
                                st.error(f"  Row {row}: {message}")
                        else:
                            st.error(f"  {err_item}")
            except APIError as e:
                _handle_api_error(e, "bulk import SKUs")
