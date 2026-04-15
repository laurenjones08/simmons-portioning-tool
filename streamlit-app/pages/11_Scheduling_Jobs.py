"""
Scheduling Jobs page - create and track plant-scoped scheduling jobs.
"""

from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_client import (  # noqa: E402
    APIError,
    cancel_scheduling_job,
    get_all_configs,
    get_scheduling_job,
    get_scheduling_job_artifacts,
    list_scheduling_jobs,
    search_skus,
    submit_scheduling_job,
)


def _handle_api_error(error: APIError, action: str) -> None:
    if error.status_code == 0:
        st.error("Could not reach the Scheduling APIs. Check your connection.")
    elif error.status_code == 404:
        st.warning("Resource not found.")
    elif error.status_code == 409:
        st.warning(error.detail)
    else:
        st.error(f"Failed to {action}: {error.detail}")


def _load_plants() -> list[str]:
    try:
        configs = get_all_configs()
        plant_config = next((cfg for cfg in configs if cfg.get("key") == "mix.availablePlants"), None)
        plant_value = str(plant_config.get("value", "")) if plant_config else ""
        return [part.strip() for part in plant_value.split(",") if part.strip()]
    except APIError as error:
        _handle_api_error(error, "load plant reference data")
        return []


def _load_skus() -> list[dict]:
    try:
        return search_skus({})
    except APIError as error:
        _handle_api_error(error, "load SKU reference data")
        return []


def _sku_matches_filter(sku: dict, search_text: str) -> bool:
    if not search_text.strip():
        return True
    needle = search_text.strip().lower()
    haystack = " ".join(
        [
            str(sku.get("tradeNumber", "")),
            str(sku.get("customerName", "")),
            str(sku.get("productType", "")),
            str(sku.get("birdSize", "")),
        ]
    ).lower()
    return needle in haystack


def _filter_skus_for_plant(skus: list[dict], plant_id: str, search_text: str) -> list[dict]:
    filtered: list[dict] = []
    for sku in skus:
        sku_plant = str(sku.get("prodPlant", "")).strip()
        trade_number = str(sku.get("tradeNumber", "")).strip()
        if not trade_number:
            continue
        if plant_id.strip() and sku_plant != plant_id.strip():
            continue
        if not _sku_matches_filter(sku, search_text):
            continue
        filtered.append(sku)
    return filtered


def _sku_label(sku: dict) -> str:
    trade_number = str(sku.get("tradeNumber", "")).strip()
    customer = str(sku.get("customerName", "")).strip()
    product_type = str(sku.get("productType", "")).strip()
    bird_size = str(sku.get("birdSize", "")).strip()
    pieces = [piece for piece in [customer, product_type, bird_size] if piece]
    suffix = " / ".join(pieces)
    return trade_number if not suffix else f"{trade_number} - {suffix}"


def _job_label(job: dict) -> str:
    run_id = str(job.get("runId", "")).strip() or "Unnamed run"
    plant_id = str(job.get("plantId", "")).strip()
    status = str(job.get("status", "")).strip()
    created_at = str(job.get("createdAt", "")).strip()
    created_date = created_at[:10] if created_at else "unknown date"
    return f"{run_id} | {plant_id} | {status} | {created_date}"


def _job_sort_key(job: dict) -> str:
    created_at = str(job.get("createdAt", "")).strip()
    return created_at


def _load_jobs(status_filter: str | None = None) -> list[dict]:
    try:
        jobs = list_scheduling_jobs(status_filter=status_filter)
        jobs.sort(key=_job_sort_key, reverse=True)
        return jobs
    except APIError as error:
        _handle_api_error(error, "load scheduling jobs")
        return []


def _load_job_detail(job_id: str) -> dict | None:
    try:
        return get_scheduling_job(job_id)
    except APIError as error:
        _handle_api_error(error, "load job detail")
        return None


def _load_job_artifacts(job_id: str) -> list[dict]:
    try:
        return get_scheduling_job_artifacts(job_id)
    except APIError as error:
        _handle_api_error(error, "load job artifacts")
        return []


def _default_run_id() -> str:
    return f"schedule-{date.today().isoformat()}"


def _init_state() -> None:
    defaults = {
        "scheduling_job_plants": [],
        "scheduling_job_skus": [],
        "scheduling_job_rows": None,
        "scheduling_job_rows_filter": None,
        "scheduling_job_status_filter": "",
        "scheduling_job_selected_id": "",
        "scheduling_job_detail": None,
        "scheduling_job_artifacts": [],
        "scheduling_job_sku_search": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_artifacts(artifacts: list[dict]) -> None:
    if not artifacts:
        st.info("This job does not have any downloadable artifacts yet.")
        return

    artifact_rows = [
        {
            "artifactName": artifact.get("artifactName", ""),
            "fileName": artifact.get("fileName", ""),
            "bucket": artifact.get("bucket", ""),
            "key": artifact.get("key", ""),
            "downloadUrl": artifact.get("downloadUrl", ""),
        }
        for artifact in artifacts
    ]
    st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)

    st.markdown("**Downloads**")
    for artifact in artifacts:
        artifact_name = artifact.get("artifactName", "artifact")
        download_url = artifact.get("downloadUrl", "")
        if download_url:
            st.markdown(f"- [{artifact_name}]({download_url})")
        else:
            st.markdown(f"- {artifact_name}")


st.set_page_config(page_title="Scheduling Jobs", layout="wide")
st.title("Scheduling Jobs")
st.caption("Create and track plant-scoped scheduling runs backed by the scheduling worker API.")

_init_state()

if not st.session_state.scheduling_job_plants or not st.session_state.scheduling_job_skus:
    st.session_state.scheduling_job_plants = _load_plants()
    st.session_state.scheduling_job_skus = _load_skus()

plant_options = st.session_state.scheduling_job_plants or []
all_skus = st.session_state.scheduling_job_skus or []

st.subheader("Plant Context")
if plant_options:
    selected_plant = st.selectbox(
        "Plant ID",
        options=plant_options,
        key="scheduling_job_plant_id",
    )
else:
    selected_plant = st.text_input("Plant ID", value="", key="scheduling_job_plant_id")

sku_search = st.text_input(
    "SKU search",
    value=st.session_state.scheduling_job_sku_search,
    placeholder="trade number, customer, product type, or bird size",
)
st.session_state.scheduling_job_sku_search = sku_search

filtered_skus = _filter_skus_for_plant(all_skus, selected_plant, sku_search)
sku_ids = [str(sku.get("tradeNumber", "")).strip() for sku in filtered_skus if str(sku.get("tradeNumber", "")).strip()]
sku_lookup = {str(sku.get("tradeNumber", "")).strip(): sku for sku in filtered_skus if str(sku.get("tradeNumber", "")).strip()}

with st.expander("SKU Reference from Enumeration API", expanded=False):
    if filtered_skus:
        st.write(f"Loaded {len(filtered_skus)} SKU record(s) for plant `{selected_plant}`.")
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
                for sku in filtered_skus
            ]
        )
        st.dataframe(sku_df, width="stretch", hide_index=True)
    else:
        st.info("No SKUs match the selected plant and search text.")

st.subheader("Create Scheduling Job")
with st.form("create_scheduling_job_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        run_id = st.text_input("runId *", value=_default_run_id(), placeholder="schedule-2026-04-12")
        plan_start_date = st.date_input("Plan Start Date", value=date.today())
        horizon_days = st.number_input("Horizon Days", min_value=1, value=12, step=1)
    with col2:
        short_term_file = st.text_input(
            "Short Term File (optional)",
            value="",
            placeholder="C:/path/to/short_term_demand.xlsx",
        )
        output_dir = st.text_input("Output Dir", value="outputs", placeholder="outputs")
        save_csv = st.checkbox("Save CSV artifacts", value=True)
        tee = st.checkbox("Show solver output (tee)", value=False)

    selected_sku_ids = st.multiselect(
        "SKU IDs *",
        options=sku_ids,
        format_func=lambda sku_id: _sku_label(sku_lookup.get(sku_id, {"tradeNumber": sku_id})),
        help="Pick the SKUs that should be included in the scheduling job.",
    )
    submit_clicked = st.form_submit_button("Submit Scheduling Job")

if submit_clicked:
    if not run_id.strip():
        st.warning("runId is required.")
    elif not selected_plant.strip():
        st.warning("Plant ID is required.")
    elif not selected_sku_ids:
        st.warning("Select at least one SKU for the scheduling job.")
    else:
        payload = {
            "runId": run_id.strip(),
            "plantId": selected_plant.strip(),
            "skuIds": selected_sku_ids,
            "shortTermFile": short_term_file.strip() or None,
            "saveCsv": bool(save_csv),
            "outputDir": output_dir.strip() or "outputs",
            "tee": bool(tee),
            "planStartDate": plan_start_date.isoformat(),
            "horizonDays": int(horizon_days),
        }
        try:
            result = submit_scheduling_job(payload)
            st.success(
                f"Scheduling job submitted. Status: **{result.get('status', 'unknown')}** "
                f"jobId: `{result.get('jobId', '')}`"
            )
            st.session_state.scheduling_job_rows = _load_jobs(st.session_state.scheduling_job_status_filter or None)
            st.session_state.scheduling_job_selected_id = result.get("jobId", "")
            st.session_state.scheduling_job_detail = None
            st.session_state.scheduling_job_artifacts = []
            st.rerun()
        except APIError as error:
            _handle_api_error(error, "submit scheduling job")

st.divider()
st.subheader("Track Scheduling Jobs")

tracking_col1, tracking_col2 = st.columns([1, 1])
with tracking_col1:
    status_filter = st.selectbox(
        "Status Filter",
        options=["", "pending", "running", "completed", "failed", "cancelled"],
        index=["", "pending", "running", "completed", "failed", "cancelled"].index(st.session_state.scheduling_job_status_filter),
    )
with tracking_col2:
    if st.button("Refresh Jobs"):
        st.session_state.scheduling_job_rows = _load_jobs(status_filter or None)
        st.session_state.scheduling_job_rows_filter = status_filter or None

st.session_state.scheduling_job_status_filter = status_filter

if (
    st.session_state.scheduling_job_rows is None
    or st.session_state.scheduling_job_rows_filter != (status_filter or None)
):
    st.session_state.scheduling_job_rows = _load_jobs(status_filter or None)
    st.session_state.scheduling_job_rows_filter = status_filter or None

job_rows = st.session_state.scheduling_job_rows or []
job_rows = [
    job
    for job in job_rows
    if not selected_plant.strip() or str(job.get("plantId", "")).strip() == selected_plant.strip()
]

if job_rows:
    job_df = pd.DataFrame(
        [
            {
                "jobId": job.get("jobId", ""),
                "runId": job.get("runId", ""),
                "plantId": job.get("plantId", ""),
                "status": job.get("status", ""),
                "createdAt": job.get("createdAt", ""),
                "updatedAt": job.get("updatedAt", ""),
                "skuCount": len(job.get("skuIds", []) or []),
                "saveCsv": job.get("saveCsv", False),
                "outputDir": job.get("outputDir", ""),
            }
            for job in job_rows
        ]
    )
    st.dataframe(job_df, width="stretch", hide_index=True)

    job_lookup = {job.get("jobId", ""): job for job in job_rows if job.get("jobId")}
    selected_job_id = st.selectbox(
        "Select a job to view details",
        options=[""] + list(job_lookup.keys()),
        index=([""] + list(job_lookup.keys())).index(st.session_state.scheduling_job_selected_id)
        if st.session_state.scheduling_job_selected_id in job_lookup
        else 0,
        format_func=lambda job_id: "Select a job..." if not job_id else _job_label(job_lookup[job_id]),
    )
    if selected_job_id:
        st.session_state.scheduling_job_selected_id = selected_job_id
        detail = _load_job_detail(selected_job_id)
        artifacts = _load_job_artifacts(selected_job_id)
        st.session_state.scheduling_job_detail = detail
        st.session_state.scheduling_job_artifacts = artifacts
    else:
        st.session_state.scheduling_job_selected_id = ""
        st.session_state.scheduling_job_detail = None
        st.session_state.scheduling_job_artifacts = []

    detail = st.session_state.scheduling_job_detail
    artifacts = st.session_state.scheduling_job_artifacts or []

    if detail:
        st.subheader(f"Job Detail - `{detail.get('jobId', '')}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**runId:** {detail.get('runId', 'N/A')}")
            st.markdown(f"**plantId:** {detail.get('plantId', 'N/A')}")
            st.markdown(f"**status:** {detail.get('status', 'N/A')}")
            st.markdown(f"**createdAt:** {detail.get('createdAt', 'N/A')}")
            st.markdown(f"**updatedAt:** {detail.get('updatedAt', 'N/A')}")
        with col2:
            st.markdown(f"**planStartDate:** {detail.get('planStartDate', 'N/A')}")
            st.markdown(f"**horizonDays:** {detail.get('horizonDays', 'N/A')}")
            st.markdown(f"**saveCsv:** {detail.get('saveCsv', False)}")
            st.markdown(f"**outputDir:** {detail.get('outputDir', 'N/A')}")
            st.markdown(f"**tee:** {detail.get('tee', False)}")

        st.markdown(f"**skuIds:** {', '.join(detail.get('skuIds', []) or []) or '-'}")
        if detail.get("shortTermFile"):
            st.markdown(f"**shortTermFile:** {detail.get('shortTermFile')}")
        if detail.get("errorMessage"):
            st.error(detail.get("errorMessage"))

        if detail.get("artifactBucket") or detail.get("artifactPrefix"):
            st.markdown(
                f"**artifactBucket:** {detail.get('artifactBucket') or '-'}  \n"
                f"**artifactPrefix:** {detail.get('artifactPrefix') or '-'}"
            )

        if detail.get("status") == "running":
            st.info("The job is still running.")
        elif detail.get("status") == "completed":
            st.success("The job completed successfully.")
        elif detail.get("status") == "failed":
            st.error("The job failed.")
        elif detail.get("status") == "cancelled":
            st.warning("The job was cancelled.")

        st.divider()
        st.subheader("Artifacts")
        _render_artifacts(artifacts)

        if detail.get("status") in {"pending", "running"}:
            st.divider()
            if st.button(f"Cancel Job `{detail.get('jobId', '')}`", type="primary"):
                try:
                    cancel_scheduling_job(detail["jobId"])
                    st.success("Cancellation requested.")
                    st.session_state.scheduling_job_rows = _load_jobs(status_filter or None)
                    st.session_state.scheduling_job_detail = _load_job_detail(detail["jobId"])
                    st.session_state.scheduling_job_artifacts = _load_job_artifacts(detail["jobId"])
                    st.rerun()
                except APIError as error:
                    _handle_api_error(error, "cancel scheduling job")
else:
    st.info("No scheduling jobs found for the current plant context.")
