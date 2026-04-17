"""Advanced Settings view for single-page router."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Callable

import pandas as pd
import streamlit as st

from api_client import (
    batch_import_skus,
    create_bucket,
    get_all_configs,
    list_lines,
    search_buckets,
    search_cut_strategies,
    search_skus,
    update_bucket,
    update_config,
    update_cut_strategy,
    update_line,
    update_sku,
)


def _safe_load(action: Callable[[], list[dict]]) -> list[dict]:
    try:
        return action()
    except Exception:
        return []


def _format_list_value(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    if isinstance(values, list):
        return ", ".join(str(value) for value in values if str(value).strip())
    return str(values)


def _parse_list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def _parse_optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _parse_optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _require_float(value: Any, field_name: str) -> float:
    parsed = _parse_optional_float(value)
    if parsed is None:
        raise ValueError(f"{field_name} is required.")
    return parsed


def _require_int(value: Any, field_name: str) -> int:
    parsed = _parse_optional_int(value)
    if parsed is None:
        raise ValueError(f"{field_name} is required.")
    return parsed


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot interpret '{value}' as a boolean value.")


def _parse_config_value(value: Any, value_type: str) -> Any:
    normalized_type = str(value_type).strip().lower()
    if normalized_type == "int":
        return _require_int(value, "value")
    if normalized_type == "float":
        return _require_float(value, "value")
    if normalized_type == "bool":
        return _parse_bool(value)
    if normalized_type == "string":
        return "" if value is None else str(value)
    raise ValueError(f"Unsupported valueType '{value_type}'.")


def _build_editor_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _save_updates(
    edited_rows: list[dict[str, Any]],
    original_rows: list[dict[str, Any]],
    id_field: str,
    update_fn: Callable[[str, dict[str, Any]], dict],
    build_payload_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    label: str,
) -> None:
    original_by_id = {
        str(row.get(id_field, "")).strip(): row
        for row in original_rows
        if str(row.get(id_field, "")).strip()
    }

    changed = 0
    succeeded = 0
    errors: list[str] = []

    for row in edited_rows:
        row_id = str(row.get(id_field, "")).strip()
        if not row_id:
            continue

        original = original_by_id.get(row_id)
        if original is None:
            continue

        try:
            payload = build_payload_fn(row, original)
            if payload == {k: original.get(k) for k in payload}:
                continue
            changed += 1
            update_fn(row_id, payload)
            succeeded += 1
        except Exception as exc:
            errors.append(f"{row_id}: {exc}")

    if changed == 0:
        st.info(f"No {label.lower()} changes to save.")
        return

    if succeeded:
        st.success(f"Saved {succeeded} {label.lower()} change(s).")
    if errors:
        st.error(f"Failed to save {len(errors)} {label.lower()} row(s):")
        for error in errors:
            st.error(error)
    if succeeded and not errors:
        st.rerun()


def _bucket_rows(buckets: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "bucketId": bucket.get("_id", ""),
            "minWeight": bucket.get("minWeight", 0.0),
            "targetWeight": bucket.get("targetWeight", 0.0),
            "maxWeight": bucket.get("maxWeight", 0.0),
        }
        for bucket in buckets
    ]


def _bucket_payload(row: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    return {
        "minWeight": _require_float(row.get("minWeight", original.get("minWeight", 0.0)), "minWeight"),
        "targetWeight": _require_float(row.get("targetWeight", original.get("targetWeight", 0.0)), "targetWeight"),
        "maxWeight": _require_float(row.get("maxWeight", original.get("maxWeight", 0.0)), "maxWeight"),
    }


def _sku_rows(skus: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "tradeNumber": sku.get("tradeNumber", ""),
            "customerName": sku.get("customerName", ""),
            "customerType": sku.get("customerType", ""),
            "productType": sku.get("productType", ""),
            "unitsPerCut": sku.get("unitsPerCut", 1),
            "prodPlant": sku.get("prodPlant", ""),
            "minWeight": sku.get("minWeight", 0.0),
            "maxWeight": sku.get("maxWeight", 0.0),
            "targetWeight": sku.get("targetWeight", 0.0),
            "birdSize": sku.get("birdSize", ""),
            "allowedParts": _format_list_value(sku.get("allowedParts", [])),
        }
        for sku in skus
    ]


def _sku_payload(row: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    return {
        "tradeNumber": str(row.get("tradeNumber", original.get("tradeNumber", ""))).strip(),
        "customerName": str(row.get("customerName", original.get("customerName", ""))).strip(),
        "customerType": str(row.get("customerType", original.get("customerType", ""))).strip(),
        "productType": str(row.get("productType", original.get("productType", ""))).strip(),
        "unitsPerCut": _require_int(row.get("unitsPerCut", original.get("unitsPerCut", 1)), "unitsPerCut"),
        "prodPlant": str(row.get("prodPlant", original.get("prodPlant", ""))).strip(),
        "minWeight": _require_float(row.get("minWeight", original.get("minWeight", 0.0)), "minWeight"),
        "maxWeight": _require_float(row.get("maxWeight", original.get("maxWeight", 0.0)), "maxWeight"),
        "targetWeight": _require_float(row.get("targetWeight", original.get("targetWeight", 0.0)), "targetWeight"),
        "birdSize": str(row.get("birdSize", original.get("birdSize", ""))).strip(),
        "allowedParts": _parse_list_value(row.get("allowedParts", original.get("allowedParts", []))),
    }


def _strategy_rows(strategies: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "strategyId": strategy.get("_id", ""),
            "name": strategy.get("name", ""),
            "description": strategy.get("description") or "",
            "lineType": strategy.get("lineType", strategy.get("mfgType", "")),
            "hasNugget": strategy.get("hasNugget", False),
            "beltSpeed": strategy.get("beltSpeed", 0.0),
            "parts": _format_list_value(strategy.get("parts", [])),
        }
        for strategy in strategies
    ]


def _format_cut_strategy_label(strategy: dict) -> str:
    name = strategy.get("name", "")
    line_type = strategy.get("lineType", strategy.get("mfgType", ""))
    parts = ", ".join(strategy.get("parts", []))

    label_parts = [part for part in [name, line_type] if part]
    label = " - ".join(label_parts) if label_parts else "Cut Strategy"
    if parts:
        label = f"{label} [{parts}]"
    return label


def _plant_options(configs: list[dict]) -> list[str]:
    plant_config = next((cfg for cfg in configs if cfg.get("key") == "mix.availablePlants"), None)
    plant_value = str(plant_config.get("value", "")) if plant_config else ""
    return [part.strip() for part in plant_value.split(",") if part.strip()]


def _strategy_id_labels(strategies: list[dict]) -> tuple[list[str], dict[str, str]]:
    option_ids = [strategy.get("_id", "") for strategy in strategies if strategy.get("_id")]
    labels = {
        strategy.get("_id", ""): _format_cut_strategy_label(strategy)
        for strategy in strategies
        if strategy.get("_id")
    }
    return option_ids, labels


def _strategy_options_for_line(
    strategies: list[dict], line_type: str | None = None
) -> tuple[list[str], dict[str, str]]:
    if line_type:
        strategies = [
            strategy
            for strategy in strategies
            if strategy.get("lineType", strategy.get("mfgType")) == line_type
        ]
    return _strategy_id_labels(strategies)


def _line_lookup(lines: list[dict]) -> dict[str, dict]:
    return {line.get("lineId", ""): line for line in lines if line.get("lineId")}


def _strategy_payload(row: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    description = str(row.get("description", original.get("description", ""))).strip()
    return {
        "name": str(row.get("name", original.get("name", ""))).strip(),
        "description": description or None,
        "lineType": str(row.get("lineType", original.get("lineType", original.get("mfgType", "")))).strip(),
        "hasNugget": _parse_bool(row.get("hasNugget", original.get("hasNugget", False))),
        "beltSpeed": _require_float(row.get("beltSpeed", original.get("beltSpeed", 0.0)), "beltSpeed"),
        "parts": _parse_list_value(row.get("parts", original.get("parts", []))),
    }


def _line_rows(lines: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "lineId": line.get("lineId", ""),
            "friendlyName": line.get("friendlyName", ""),
            "lineType": line.get("lineType", ""),
            "plant": line.get("plant", ""),
            "hoursOfLaborAvailablePerShift": line.get("hoursOfLaborAvailablePerShift", 0.0),
            "unitsAvailable": line.get("unitsAvailable", 0),
            "lineThroughput": line.get("lineThroughput"),
            "permittedCutStrategyIds": line.get("permittedCutStrategyIds", []),
            "isActive": line.get("isActive", True),
        }
        for line in lines
    ]


def _line_payload(row: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    return {
        "friendlyName": str(row.get("friendlyName", original.get("friendlyName", ""))).strip(),
        "lineType": str(row.get("lineType", original.get("lineType", ""))).strip(),
        "plant": str(row.get("plant", original.get("plant", ""))).strip(),
        "hoursOfLaborAvailablePerShift": _require_float(
            row.get("hoursOfLaborAvailablePerShift", original.get("hoursOfLaborAvailablePerShift", 0.0)),
            "hoursOfLaborAvailablePerShift",
        ),
        "unitsAvailable": _require_int(row.get("unitsAvailable", original.get("unitsAvailable", 0)), "unitsAvailable"),
        "lineThroughput": _parse_optional_float(row.get("lineThroughput", original.get("lineThroughput"))),
        "permittedCutStrategyIds": _parse_list_value(
            row.get("permittedCutStrategyIds", original.get("permittedCutStrategyIds", []))
        ),
        "isActive": _parse_bool(row.get("isActive", original.get("isActive", True))),
    }


def _config_rows(configs: list[dict]) -> list[dict[str, Any]]:
    return [
        {
            "key": config.get("key", ""),
            "value": "" if config.get("value") is None else str(config.get("value")),
            "valueType": str(config.get("valueType", "")),
            "description": config.get("description", ""),
            "minValue": "" if config.get("minValue") is None else str(config.get("minValue")),
            "maxValue": "" if config.get("maxValue") is None else str(config.get("maxValue")),
            "updatedAt": config.get("updatedAt", ""),
        }
        for config in configs
    ]


def _config_payload(row: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    value_type = str(row.get("valueType", original.get("valueType", ""))).strip().lower()
    return {
        "value": _parse_config_value(row.get("value", original.get("value")), value_type),
        "valueType": value_type,
        "description": str(row.get("description", original.get("description", ""))).strip(),
        "minValue": _parse_optional_float(row.get("minValue", original.get("minValue"))),
        "maxValue": _parse_optional_float(row.get("maxValue", original.get("maxValue"))),
    }


def _line_column_config(plant_options: list[str], strategy_labels: dict[str, str], strategy_ids: list[str]) -> dict[str, Any]:
    return {
        "lineId": st.column_config.TextColumn("Line ID", disabled=True, pinned=True),
        "friendlyName": st.column_config.TextColumn("Friendly name"),
        "lineType": st.column_config.TextColumn("Line type"),
        "plant": st.column_config.SelectboxColumn(
            "Plant",
            options=plant_options,
            required=True,
        ),
        "hoursOfLaborAvailablePerShift": st.column_config.NumberColumn(
            "Hours of labor / shift",
            min_value=0.1,
            step=0.5,
        ),
        "unitsAvailable": st.column_config.NumberColumn("Units available", min_value=0, step=1),
        "lineThroughput": st.column_config.NumberColumn("Line throughput", min_value=0.0, step=1.0),
        "permittedCutStrategyIds": st.column_config.MultiselectColumn(
            "Permitted cut strategies",
            options=strategy_ids,
            format_func=lambda strategy_id: strategy_labels.get(strategy_id, strategy_id),
            help="Choose one or more cut strategies from the Enumeration API.",
        ),
        "isActive": st.column_config.CheckboxColumn("Active"),
    }


def render():
    st.markdown("---")

    with st.expander("Buckets", expanded=False):
        buckets = _safe_load(lambda: search_buckets({}))
        if buckets:
            editor = st.data_editor(
                _build_editor_frame(_bucket_rows(buckets)),
                key="advanced_settings_buckets_editor",
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                disabled=["bucketId"],
            )
            if st.button(
                "Save Bucket Changes",
                key="save_bucket_changes",
                type="primary",
                use_container_width=True,
            ):
                _save_updates(
                    editor.to_dict(orient="records"),
                    buckets,
                    "bucketId",
                    update_bucket,
                    _bucket_payload,
                    "bucket",
                )
        else:
            st.info("No buckets found.")
        with st.form("create_bucket_form"):
            min_w = st.number_input("minWeight", value=0.0)
            target_w = st.number_input("targetWeight", value=0.5)
            max_w = st.number_input("maxWeight", value=1.0)
            submit = st.form_submit_button(
                "Create Bucket",
                type="primary",
                use_container_width=True,
            )
            if submit:
                try:
                    create_bucket({"minWeight": min_w, "targetWeight": target_w, "maxWeight": max_w})
                    st.success("Bucket created. Refresh the page to see it in the grid.")
                except Exception as exc:
                    st.error(str(exc))

    with st.expander("SKUs", expanded=False):
        skus = _safe_load(lambda: search_skus({}))
        if skus:
            editor = st.data_editor(
                _build_editor_frame(_sku_rows(skus)),
                key="advanced_settings_skus_editor",
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                disabled=["tradeNumber"],
            )
            if st.button(
                "Save SKU Changes",
                key="save_sku_changes",
                type="primary",
                use_container_width=True,
            ):
                _save_updates(
                    editor.to_dict(orient="records"),
                    skus,
                    "tradeNumber",
                    update_sku,
                    _sku_payload,
                    "sku",
                )
        else:
            st.info("No SKUs found.")

        st.markdown("**Bulk Import SKUs**")
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        if uploaded is not None:
            try:
                content = uploaded.read().decode("utf-8")
                if uploaded.name.lower().endswith(".json"):
                    data = json.loads(content)
                else:
                    reader = csv.DictReader(io.StringIO(content))
                    data = list(reader)
                result = batch_import_skus(data)
                st.success(f"Imported: {result}")
            except Exception as exc:
                st.error(str(exc))

    with st.expander("Cut Strategies", expanded=False):
        strategies = _safe_load(lambda: search_cut_strategies({}))
        if strategies:
            editor = st.data_editor(
                _build_editor_frame(_strategy_rows(strategies)),
                key="advanced_settings_strategies_editor",
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                disabled=["strategyId"],
            )
            if st.button(
                "Save Cut Strategy Changes",
                key="save_strategy_changes",
                type="primary",
                use_container_width=True,
            ):
                _save_updates(
                    editor.to_dict(orient="records"),
                    strategies,
                    "strategyId",
                    update_cut_strategy,
                    _strategy_payload,
                    "cut strategy",
                )
        else:
            st.info("No cut strategies found.")

    with st.expander("Lines", expanded=False):
        configs = _safe_load(get_all_configs)
        plant_options = _plant_options(configs)
        strategies = _safe_load(lambda: search_cut_strategies({}))
        strategy_ids, strategy_labels = _strategy_id_labels(strategies)

        if not plant_options:
            st.warning("No plants found in the global config (mix.availablePlants).")
        if not strategy_ids:
            st.warning("No cut strategies are currently available from the Enumeration API.")

        lines = _safe_load(list_lines)
        if lines:
            st.dataframe(
                _build_editor_frame(_line_rows(lines)),
                hide_index=True,
                use_container_width=True,
            )
            line_options = _line_lookup(lines)
            selected_line_id = st.selectbox(
                "Select line",
                options=list(line_options.keys()),
                format_func=lambda line_id: f"{line_id} - {line_options[line_id].get('friendlyName', line_id)}",
            )
            selected_line = line_options[selected_line_id]
            current_line_type = selected_line.get("lineType", "DB20")
            current_line_type = current_line_type if current_line_type in ["DB20", "DSI884", "DSI888"] else "DB20"

            with st.container(border=True):
                edit_friendly_name = st.text_input(
                    "Friendly name *",
                    value=selected_line.get("friendlyName", ""),
                    key=f"advanced_settings_line_friendly_name_{selected_line_id}",
                )
                edit_line_type = st.selectbox(
                    "Line type *",
                    options=["DB20", "DSI884", "DSI888"],
                    index=["DB20", "DSI884", "DSI888"].index(current_line_type),
                    key=f"advanced_settings_line_type_{selected_line_id}",
                )
                selected_strategy_ids, selected_strategy_labels = _strategy_options_for_line(
                    strategies,
                    edit_line_type,
                )
                selected_strategy_ids_default = [
                    strategy_id
                    for strategy_id in selected_line.get("permittedCutStrategyIds", [])
                    if strategy_id in selected_strategy_ids
                ]
                edit_plant = (
                    st.selectbox(
                        "Plant *",
                        options=plant_options,
                        index=plant_options.index(selected_line.get("plant", "")) if selected_line.get("plant", "") in plant_options else 0,
                        key=f"advanced_settings_line_plant_{selected_line_id}",
                    )
                    if plant_options
                    else st.text_input(
                        "Plant *",
                        value=selected_line.get("plant", ""),
                        key=f"advanced_settings_line_plant_{selected_line_id}",
                    )
                )
                edit_hours_of_labor = st.number_input(
                    "Hours of labor available per shift *",
                    min_value=0.1,
                    value=float(selected_line.get("hoursOfLaborAvailablePerShift", 8.0)),
                    step=0.5,
                    key=f"advanced_settings_line_hours_{selected_line_id}",
                )
                edit_units_available = st.number_input(
                    "Units available *",
                    min_value=0,
                    value=int(selected_line.get("unitsAvailable", 0)),
                    step=1,
                    key=f"advanced_settings_line_units_{selected_line_id}",
                )
                edit_line_throughput = st.number_input(
                    "Line throughput",
                    min_value=0.0,
                    value=float(selected_line.get("lineThroughput") or 0.0),
                    step=1.0,
                    key=f"advanced_settings_line_throughput_{selected_line_id}",
                )
                edit_is_active = st.checkbox(
                    "Active",
                    value=bool(selected_line.get("isActive", True)),
                    key=f"advanced_settings_line_active_{selected_line_id}",
                )
                if selected_strategy_ids:
                    edit_cut_strategy_ids = st.multiselect(
                        "Permitted cut strategies",
                        options=selected_strategy_ids,
                        default=selected_strategy_ids_default,
                        format_func=lambda strategy_id: selected_strategy_labels.get(strategy_id, strategy_id),
                        key=f"advanced_settings_line_strategies_{selected_line_id}",
                    )
                else:
                    st.multiselect(
                        "Permitted cut strategies",
                        options=[],
                        default=[],
                        key=f"advanced_settings_line_strategies_{selected_line_id}",
                        disabled=True,
                    )
                    edit_cut_strategy_ids = []

                if edit_line_type != current_line_type:
                    st.info("Changing line type will refresh the permitted cut strategy options on the next run.")

                save_clicked = st.button(
                    "Save Line Changes",
                    key=f"save_line_changes_{selected_line_id}",
                    type="primary",
                    use_container_width=True,
                )

            if save_clicked:
                payload = {
                    "friendlyName": edit_friendly_name.strip(),
                    "lineType": edit_line_type,
                    "plant": edit_plant.strip(),
                    "hoursOfLaborAvailablePerShift": float(edit_hours_of_labor),
                    "unitsAvailable": int(edit_units_available),
                    "lineThroughput": float(edit_line_throughput),
                    "permittedCutStrategyIds": edit_cut_strategy_ids,
                    "isActive": edit_is_active,
                }
                try:
                    update_line(selected_line_id, payload)
                    st.success("Line updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

            if st.button(
                "Delete Selected Line",
                key=f"delete_line_{selected_line_id}",
                type="secondary",
            ):
                try:
                    # Import lazily to avoid expanding the module surface unless needed.
                    from api_client import delete_line as delete_line_api

                    delete_line_api(selected_line_id)
                    st.success("Line deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("No lines configured.")

    with st.expander("Global Config", expanded=False):
        configs = _safe_load(get_all_configs)
        if configs:
            editor = st.data_editor(
                _build_editor_frame(_config_rows(configs)),
                key="advanced_settings_configs_editor",
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                disabled=["key", "updatedAt"],
            )
            if st.button(
                "Save Config Changes",
                key="save_config_changes",
                type="primary",
                use_container_width=True,
            ):
                _save_updates(
                    editor.to_dict(orient="records"),
                    configs,
                    "key",
                    update_config,
                    _config_payload,
                    "config",
                )
        else:
            st.info("No configs available.")

    st.markdown("---")
    st.caption("Changes here affect the core services. Use caution and confirm before saving high-impact settings.")
