"""
Available WIP page - CRUD management for plant and bucket level WIP.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import (  # noqa: E402
    APIError,
    get_all_configs,
    create_available_wip,
    delete_available_wip,
    search_available_wip,
    search_buckets,
    update_available_wip,
)


def validate_available_wip_lbs(value: float) -> str | None:
    if value < 0:
        return "availableLbs must be greater than or equal to 0."
    return None


def _init_state() -> None:
    if "available_wip_rows" not in st.session_state:
        st.session_state.available_wip_rows = None
    if "available_wip_filters" not in st.session_state:
        st.session_state.available_wip_filters = {"plantName": "", "bucketId": ""}
    if "available_wip_plants" not in st.session_state:
        st.session_state.available_wip_plants = []
    if "available_wip_buckets" not in st.session_state:
        st.session_state.available_wip_buckets = []


def _load_reference_data() -> None:
    try:
        configs = get_all_configs()
        plant_config = next((cfg for cfg in configs if cfg.get("key") == "mix.availablePlants"), None)
        plant_value = str(plant_config.get("value", "")) if plant_config else ""
        plants = [part.strip() for part in plant_value.split(",") if part.strip()]
        st.session_state.available_wip_plants = plants
    except APIError as error:
        _handle_api_error(error)
        st.session_state.available_wip_plants = []

    try:
        buckets = search_buckets({})
        st.session_state.available_wip_buckets = buckets
    except APIError as error:
        _handle_api_error(error)
        st.session_state.available_wip_buckets = []


def _handle_api_error(error: APIError) -> None:
    if error.status_code == 0:
        st.error("Could not reach Scheduling API. Check your connection.")
    elif error.status_code == 409:
        st.error(error.detail)
    else:
        st.error(f"Request failed: {error.detail}")


def _build_search_criteria(plant_name: str, bucket_id: str) -> dict:
    criteria: dict[str, str] = {}
    plant_name = plant_name.strip()
    bucket_id = bucket_id.strip()
    if plant_name:
        criteria["plantName"] = plant_name
    if bucket_id:
        criteria["bucketId"] = bucket_id
    return criteria


def _load_available_wip(criteria: dict | None = None) -> None:
    try:
        st.session_state.available_wip_rows = search_available_wip(criteria or {})
    except APIError as error:
        _handle_api_error(error)
        st.session_state.available_wip_rows = []


st.set_page_config(page_title="Available WIP", layout="wide")
st.title("Available Daily WIP")
st.caption("Manage WIP by plant and bucket. Each plant and bucket combination must be unique.")

_init_state()
_load_reference_data()

if st.session_state.available_wip_rows is None:
    _load_available_wip()

search_plant = st.session_state.available_wip_filters.get("plantName", "")
search_bucket = st.session_state.available_wip_filters.get("bucketId", "")
plant_options = [""] + list(st.session_state.available_wip_plants or [])
bucket_options = list(st.session_state.available_wip_buckets or [])


def _bucket_label(bucket: dict) -> str:
    min_weight = bucket.get("minWeight", "")
    max_weight = bucket.get("maxWeight", "")
    return f"{min_weight} - {max_weight}"


def _selected_index(options: list[str], selected_value: str) -> int:
    try:
        return options.index(selected_value)
    except ValueError:
        return 0

with st.expander("Search Filters", expanded=True):
    with st.form("search_available_wip_form"):
        col1, col2 = st.columns(2)
        with col1:
            search_plant = st.selectbox(
                "Plant Name",
                options=plant_options,
                index=_selected_index(plant_options, search_plant),
            )
        with col2:
            search_bucket = st.selectbox(
                "Bucket",
                options=[""] + [bucket.get("_id", "") for bucket in bucket_options],
                index=_selected_index([""] + [bucket.get("_id", "") for bucket in bucket_options], search_bucket),
                format_func=lambda bucket_id: (
                    "" if not bucket_id else _bucket_label(next((b for b in bucket_options if b.get("_id", "") == bucket_id), {}))
                ),
            )
        submitted = st.form_submit_button("Search")

    if submitted:
        criteria = _build_search_criteria(search_plant, search_bucket)
        st.session_state.available_wip_filters = {
            "plantName": search_plant.strip(),
            "bucketId": search_bucket.strip(),
        }
        _load_available_wip(criteria)

    if st.button("Reset Filters"):
        st.session_state.available_wip_filters = {"plantName": "", "bucketId": ""}
        _load_available_wip({})

if st.button("Refresh"):
    _load_available_wip(
        _build_search_criteria(
            st.session_state.available_wip_filters.get("plantName", ""),
            st.session_state.available_wip_filters.get("bucketId", ""),
        )
    )

records = st.session_state.available_wip_rows or []

if records:
    df = pd.DataFrame(
        [
            {
                "id": row.get("_id", ""),
                "plantName": row.get("plantName", ""),
                "bucketId": row.get("bucketId", ""),
                "availableLbs": row.get("availableLbs", 0.0),
            }
            for row in records
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("No available WIP rows found.")

st.divider()

st.subheader("Create Available WIP")

with st.form("create_available_wip_form", clear_on_submit=True):
    col1, col2, col3 = st.columns(3)
    plant_choices = [plant for plant in plant_options if plant]
    bucket_id_options = [bucket.get("_id", "") for bucket in bucket_options]
    with col1:
        if plant_choices:
            create_plant = st.selectbox("Plant Name", options=plant_choices)
        else:
            create_plant = st.text_input("Plant Name", placeholder="FSP")
    with col2:
        if bucket_id_options:
            create_bucket = st.selectbox(
                "Bucket",
                options=bucket_id_options,
                format_func=lambda bucket_id: _bucket_label(next((b for b in bucket_options if b.get("_id", "") == bucket_id), {})),
            )
        else:
            create_bucket = st.text_input("Bucket ID", placeholder="B 0-390")
    with col3:
        create_lbs = st.number_input("Available Lbs", min_value=0.0, value=0.0, step=1.0)
    create_clicked = st.form_submit_button("Create")

if create_clicked:
    lbs_error = validate_available_wip_lbs(create_lbs)
    if lbs_error:
        st.warning(lbs_error)
    elif not create_plant.strip() or not create_bucket.strip():
        st.warning("Plant Name and Bucket ID are required.")
    else:
        try:
            create_available_wip(
                {
                    "plantName": create_plant.strip(),
                    "bucketId": create_bucket.strip(),
                    "availableLbs": float(create_lbs),
                }
            )
            st.success("Available WIP created.")
            _load_available_wip(
                _build_search_criteria(
                    st.session_state.available_wip_filters.get("plantName", ""),
                    st.session_state.available_wip_filters.get("bucketId", ""),
                )
            )
        except APIError as error:
            _handle_api_error(error)

st.subheader("Edit / Delete Available WIP")

if not records:
    st.caption("No WIP rows available to edit or delete.")
else:
    record_lookup = {row.get("_id", ""): row for row in records}
    selected_id = st.selectbox(
        "Select WIP row",
        options=list(record_lookup.keys()),
        format_func=lambda row_id: (
            f"{record_lookup[row_id].get('plantName', '')} / "
            f"{record_lookup[row_id].get('bucketId', '')} "
            f"({record_lookup[row_id].get('availableLbs', 0.0)} lbs)"
        ),
    )
    selected_row = record_lookup.get(selected_id, {})
    selected_plant = selected_row.get("plantName", "")
    selected_bucket = selected_row.get("bucketId", "")

    with st.form("edit_available_wip_form"):
        col1, col2, col3 = st.columns(3)
        plant_choices = [plant for plant in plant_options if plant]
        bucket_id_options = [bucket.get("_id", "") for bucket in bucket_options]
        with col1:
            if plant_choices:
                edit_plant = st.selectbox(
                    "Plant Name",
                    options=plant_choices,
                    index=plant_choices.index(selected_plant) if selected_plant in plant_choices else 0,
                )
            else:
                edit_plant = st.text_input("Plant Name", value=selected_plant)
        with col2:
            if bucket_id_options:
                edit_bucket = st.selectbox(
                    "Bucket",
                    options=bucket_id_options,
                    index=bucket_id_options.index(selected_bucket) if selected_bucket in bucket_id_options else 0,
                    format_func=lambda bucket_id: _bucket_label(next((b for b in bucket_options if b.get("_id", "") == bucket_id), {})),
                )
            else:
                edit_bucket = st.text_input("Bucket ID", value=selected_bucket)
        with col3:
            edit_lbs = st.number_input(
                "Available Lbs",
                min_value=0.0,
                value=float(selected_row.get("availableLbs", 0.0)),
                step=1.0,
            )
        save_clicked = st.form_submit_button("Save Changes")

    if save_clicked:
        lbs_error = validate_available_wip_lbs(edit_lbs)
        if lbs_error:
            st.warning(lbs_error)
        elif not edit_plant.strip() or not edit_bucket.strip():
            st.warning("Plant Name and Bucket ID are required.")
        else:
            try:
                update_available_wip(
                    selected_id,
                    {
                        "plantName": edit_plant.strip(),
                        "bucketId": edit_bucket.strip(),
                        "availableLbs": float(edit_lbs),
                    },
                )
                st.success("Available WIP updated.")
                _load_available_wip(
                    _build_search_criteria(
                        st.session_state.available_wip_filters.get("plantName", ""),
                        st.session_state.available_wip_filters.get("bucketId", ""),
                    )
                )
            except APIError as error:
                _handle_api_error(error)

    if st.button("Delete Selected WIP", type="secondary"):
        try:
            delete_available_wip(selected_id)
            st.success("Available WIP deleted.")
            _load_available_wip(
                _build_search_criteria(
                    st.session_state.available_wip_filters.get("plantName", ""),
                    st.session_state.available_wip_filters.get("bucketId", ""),
                )
            )
        except APIError as error:
            _handle_api_error(error)
