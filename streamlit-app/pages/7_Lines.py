"""Lines page - CRUD management for production lines in Global Config."""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import (
    APIError,
    list_lines,
    create_line,
    update_line,
    delete_line,
    search_cut_strategies,
)

LINE_TYPE_OPTIONS = ["DB20", "DSI884", "DSI888"]


def _init_state() -> None:
    if "lines_loaded" not in st.session_state:
        st.session_state.lines_loaded = False
    if "lines" not in st.session_state:
        st.session_state.lines = []
    if "line_cut_strategies" not in st.session_state:
        st.session_state.line_cut_strategies = []


def _handle_api_error(e: APIError, service: str) -> None:
    if e.status_code == 0:
        st.error(f"Could not reach {service}. Check your connection.")
    else:
        st.error(e.detail)


def _load_lines() -> None:
    try:
        st.session_state.lines = list_lines()
        st.session_state.lines_loaded = True
    except APIError as e:
        _handle_api_error(e, "Config API")
        st.session_state.lines = []


def _load_cut_strategies() -> None:
    try:
        st.session_state.line_cut_strategies = search_cut_strategies({})
    except APIError as e:
        _handle_api_error(e, "Enumeration API")
        st.session_state.line_cut_strategies = []


def _format_cut_strategy_label(strategy: dict) -> str:
    name = strategy.get("name", "")
    line_type = strategy.get("lineType", strategy.get("mfgType", ""))
    parts = ", ".join(strategy.get("parts", []))

    label_parts = [part for part in [name, line_type] if part]
    label = " - ".join(label_parts) if label_parts else "Cut Strategy"
    if parts:
        label = f"{label} [{parts}]"
    return label


def _strategy_options(line_type: str | None = None) -> tuple[list[str], dict[str, str]]:
    strategies = st.session_state.line_cut_strategies
    if line_type:
        strategies = [
            strategy
            for strategy in strategies
            if strategy.get("lineType", strategy.get("mfgType")) == line_type
        ]
    option_ids = [strategy.get("_id", "") for strategy in strategies if strategy.get("_id")]
    labels = {
        strategy.get("_id", ""): _format_cut_strategy_label(strategy)
        for strategy in strategies
        if strategy.get("_id")
    }
    return option_ids, labels


st.set_page_config(page_title="Lines", page_icon="🏭")
st.title("Lines")

_init_state()
if not st.session_state.lines_loaded:
    _load_lines()
    _load_cut_strategies()

if st.button("Refresh"):
    _load_lines()
    _load_cut_strategies()

lines = st.session_state.lines
strategy_ids, strategy_labels = _strategy_options()

if not strategy_ids:
    st.warning("No cut strategies are currently available from the Enumeration API, so new lines cannot yet be assigned permitted strategies.")

st.subheader("Configured Lines")
if lines:
    df = pd.DataFrame(
        [
            {
                "lineId": line.get("lineId", ""),
                "friendlyName": line.get("friendlyName", ""),
                "lineType": line.get("lineType", ""),
                "plant": line.get("plant", ""),
                "isActive": line.get("isActive", False),
                "permittedCutStrategyIds": ", ".join(line.get("permittedCutStrategyIds", [])),
            }
            for line in lines
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("No lines configured.")

st.subheader("Create Line")
with st.form("create_line_form", clear_on_submit=True):
    create_line_id = st.text_input("Line ID *", help="Unique identifier used by other APIs.")
    create_friendly_name = st.text_input("Friendly name *")
    create_line_type = st.selectbox("Line type *", options=LINE_TYPE_OPTIONS)
    create_strategy_ids, create_strategy_labels = _strategy_options(create_line_type)
    create_plant = st.text_input("Plant *")
    create_is_active = st.checkbox("Active", value=True)
    create_cut_strategy_ids = st.multiselect(
        "Permitted cut strategies",
        options=create_strategy_ids,
        format_func=lambda strategy_id: create_strategy_labels.get(strategy_id, strategy_id),
    )
    create_clicked = st.form_submit_button("Create Line")

if create_clicked:
    payload = {
        "lineId": create_line_id.strip(),
        "friendlyName": create_friendly_name.strip(),
        "lineType": create_line_type,
        "plant": create_plant.strip(),
        "isActive": create_is_active,
        "permittedCutStrategyIds": create_cut_strategy_ids,
    }
    try:
        create_line(payload)
        st.success("Line created.")
        _load_lines()
    except APIError as e:
        _handle_api_error(e, "Config API")

st.subheader("Edit / Delete Line")
if not lines:
    st.caption("No lines available to edit or delete.")
else:
    line_options = {line.get("lineId", ""): line for line in lines if line.get("lineId")}
    selected_line_id = st.selectbox(
        "Select line",
        options=list(line_options.keys()),
        format_func=lambda line_id: f"{line_id} - {line_options[line_id].get('friendlyName', line_id)}",
    )
    selected_line = line_options[selected_line_id]

    with st.form("edit_line_form"):
        edit_friendly_name = st.text_input("Friendly name *", value=selected_line.get("friendlyName", ""))
        current_line_type = selected_line.get("lineType", LINE_TYPE_OPTIONS[0])
        edit_line_type = st.selectbox(
            "Line type *",
            options=LINE_TYPE_OPTIONS,
            index=LINE_TYPE_OPTIONS.index(current_line_type) if current_line_type in LINE_TYPE_OPTIONS else 0,
        )
        edit_strategy_ids, edit_strategy_labels = _strategy_options(edit_line_type)
        edit_plant = st.text_input("Plant *", value=selected_line.get("plant", ""))
        edit_is_active = st.checkbox("Active", value=bool(selected_line.get("isActive", True)))
        edit_cut_strategy_ids = st.multiselect(
            "Permitted cut strategies",
            options=edit_strategy_ids,
            default=[strategy_id for strategy_id in selected_line.get("permittedCutStrategyIds", []) if strategy_id in edit_strategy_ids],
            format_func=lambda strategy_id: edit_strategy_labels.get(strategy_id, strategy_id),
        )
        save_clicked = st.form_submit_button("Save Changes")

    if save_clicked:
        payload = {
            "friendlyName": edit_friendly_name.strip(),
            "lineType": edit_line_type,
            "plant": edit_plant.strip(),
            "isActive": edit_is_active,
            "permittedCutStrategyIds": edit_cut_strategy_ids,
        }
        try:
            update_line(selected_line_id, payload)
            st.success("Line updated.")
            _load_lines()
        except APIError as e:
            _handle_api_error(e, "Config API")

    if st.button("Delete Selected Line"):
        try:
            delete_line(selected_line_id)
            st.success("Line deleted.")
            _load_lines()
        except APIError as e:
            _handle_api_error(e, "Config API")
