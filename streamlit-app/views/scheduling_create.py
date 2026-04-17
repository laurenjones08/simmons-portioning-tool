"""Scheduling create view."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from api_client import APIError, cancel_scheduling_job, get_all_configs, list_scheduling_jobs, search_skus, submit_scheduling_job, upload_scheduling_short_term_file
from views.scheduling_shared import format_timestamp, job_label, sort_records_by_date, status_badge

_MAX_ERROR_PREVIEW_CHARS = 2000
_MAX_SKU_REFERENCE_ROWS = 200
_MAX_SKU_SELECTOR_OPTIONS = 300


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


@st.cache_data(show_spinner=False)
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


def _limit_sku_selector_options(sku_ids: list[str], selected_ids: list[str]) -> list[str]:
    if len(sku_ids) <= _MAX_SKU_SELECTOR_OPTIONS:
        return sku_ids

    capped = sku_ids[:_MAX_SKU_SELECTOR_OPTIONS]
    for sku_id in selected_ids:
        if sku_id and sku_id not in capped:
            capped.append(sku_id)
    return capped
def _sku_label(sku: dict) -> str:
    trade_number = str(sku.get("tradeNumber", "")).strip()
    customer = str(sku.get("customerName", "")).strip()
    product_type = str(sku.get("productType", "")).strip()
    bird_size = str(sku.get("birdSize", "")).strip()
    pieces = [piece for piece in [customer, product_type, bird_size] if piece]
    suffix = " / ".join(pieces)
    return trade_number if not suffix else f"{trade_number} - {suffix}"


def _job_sort_key(job: dict) -> str:
    return str(job.get("createdAt", "")).strip()


def _job_status(job: dict) -> str:
    return str(job.get("status", "")).strip().lower()


def _load_jobs(status_filter: str | None = None) -> list[dict]:
    try:
        jobs = list_scheduling_jobs(status_filter=status_filter)
        jobs.sort(key=_job_sort_key, reverse=True)
        return jobs
    except APIError as error:
        _handle_api_error(error, "load scheduling jobs")
        return []


def _upsert_job_in_rows(rows: list[dict] | None, job: dict) -> list[dict]:
    job_id = str(job.get("jobId") or job.get("_id", "")).strip()
    deduped = [row for row in (rows or []) if str(row.get("jobId") or row.get("_id", "")).strip() != job_id]
    return [job, *deduped]


def _truncate_text(text: str, limit: int = _MAX_ERROR_PREVIEW_CHARS) -> tuple[str, bool]:
    cleaned = str(text or "")
    if len(cleaned) <= limit:
        return cleaned, False
    return cleaned[:limit].rstrip() + "\n\n[traceback truncated]", True


def _default_run_id() -> str:
    return f"schedule-{date.today().isoformat()}"


def _upload_short_term_file_to_minio(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    return upload_scheduling_short_term_file(
        uploaded_file.getbuffer().tobytes(),
        uploaded_file.name,
    )


def _init_state() -> None:
    defaults = {
        "scheduling_job_plants": [],
        "scheduling_job_skus": [],
        "scheduling_job_rows": None,
        "scheduling_job_rows_filter": None,
        "scheduling_job_status_filter": "",
        "scheduling_job_cancel_target": "",
        "scheduling_job_sku_search": "",
        "scheduling_job_plant_id": "",
        "scheduling_job_selected_skus": [],
        "scheduling_job_last_submitted": None,
        "scheduling_job_submit_in_flight": False,
        "scheduling_job_show_recent_jobs": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_failed_job_details(jobs: list[dict]) -> None:
    failed_jobs = [job for job in jobs if _job_status(job) == "failed"]
    if not failed_jobs:
        return

    st.markdown("#### Failed Job Details")
    st.caption("The worker now stores both the failure message and the traceback for each failed job.")

    for job in failed_jobs[:3]:
        job_name = job_label(job)
        error_message = str(job.get("errorMessage", "")).strip()
        error_traceback = str(job.get("errorTraceback", "")).strip()

        with st.expander(f"{job_name} - failed", expanded=False):
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.write(f"**Job ID:** `{job.get('jobId') or job.get('_id', '')}`")
                st.write(f"**Run ID:** `{job.get('runId', 'â€”')}`")
                st.write(f"**Plant:** `{job.get('plantId', 'â€”')}`")
            with detail_cols[1]:
                st.write(f"**Started:** {format_timestamp(job.get('startedAt'))}")
                st.write(f"**Finished:** {format_timestamp(job.get('finishedAt'))}")
                st.write(f"**Status:** {job.get('status', 'â€”')}")

            if error_message:
                st.error(error_message)
            else:
                st.warning("The worker did not return a failure message for this job.")

            if error_traceback:
                preview, was_truncated = _truncate_text(error_traceback)
                st.code(preview, language="text")
                if was_truncated:
                    with st.expander("Show full traceback", expanded=False):
                        st.code(error_traceback, language="text")
            else:
                st.info("No traceback was stored for this failure.")


def _render_sku_reference(filtered_skus: list[dict]) -> None:
    show_reference = st.checkbox(
        "Show SKU reference",
        value=False,
        help="Load the SKU table only when you want to inspect the currently filtered rows.",
        key="scheduling_job_show_sku_reference",
    )
    if not show_reference:
        return

    if not filtered_skus:
        st.info("No SKUs match the selected plant and search text.")
        return

    preview_rows = filtered_skus[:_MAX_SKU_REFERENCE_ROWS]
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
            for sku in preview_rows
        ]
    )
    st.caption(
        f"Showing {len(preview_rows)} of {len(filtered_skus)} matching SKUs."
        if len(filtered_skus) > len(preview_rows)
        else f"{len(filtered_skus)} SKU records match the current plant and search filters."
    )
    st.dataframe(sku_df, use_container_width=True, hide_index=True, height=260)


def _render_recent_jobs(jobs: list[dict]) -> None:
    if not jobs:
        st.info("No scheduling jobs found yet. Submit a run above to get started.")
        return

    rows = []
    for job in jobs[:10]:
        rows.append(
            {
                "Run ID": job.get("runId", "—"),
                "Plant": job.get("plantId", "—"),
                "Status": job.get("status", "—"),
                "Plan Start": job.get("planStartDate", "—"),
                "Horizon": job.get("horizonDays", "—"),
                "Submitted": format_timestamp(job.get("createdAt")),
                "Finished": format_timestamp(job.get("finishedAt")),
                "_jobId": job.get("jobId") or job.get("_id", ""),
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df.drop(columns=["_jobId"]), use_container_width=True, hide_index=True)
    _render_failed_job_details(jobs)

    active_ids = [row["_jobId"] for row in rows if row["Status"] in ("running", "pending") and row["_jobId"]]
    if active_ids:
        cancel_cols = st.columns([2, 1])
        with cancel_cols[0]:
            previous_target = st.session_state.get("scheduling_job_cancel_target", "")
            cancel_target = st.selectbox(
                "Cancel a running or pending job",
                options=[""] + active_ids,
                index=([""] + active_ids).index(previous_target) if previous_target in active_ids else 0,
                format_func=lambda job_id: "Select a job..." if not job_id else job_id,
                key="scheduling_job_cancel_target",
            )
        with cancel_cols[1]:
            st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
            if st.button("Cancel Selected Job", type="secondary", use_container_width=True) and cancel_target:
                try:
                    cancel_scheduling_job(cancel_target)
                    st.success(f"Cancellation requested for `{cancel_target}`.")
                    st.session_state.scheduling_job_rows = _load_jobs(st.session_state.scheduling_job_status_filter or None)
                    st.rerun()
                except APIError as error:
                    _handle_api_error(error, "cancel scheduling job")


def render():
    st.markdown(
        "<div class='simmons-card' style='margin-top:12px;margin-bottom:16px'>"
        "<div style='font-size:28px;font-weight:800;color:#00264F'>Create a schedule</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    _init_state()
    if not st.session_state.scheduling_job_plants:
        st.session_state.scheduling_job_plants = _load_plants()

    plant_options = st.session_state.scheduling_job_plants or []
    all_skus = st.session_state.scheduling_job_skus or []

    if not all_skus:
        st.session_state.scheduling_job_skus = _load_skus()
        all_skus = st.session_state.scheduling_job_skus or []

    st.markdown("### Create Scheduling Run")
    plant_cols = st.columns(2)
    with plant_cols[0]:
        if plant_options:
            selected_plant = st.selectbox(
                "Plant ID",
                options=plant_options,
                index=plant_options.index(st.session_state.scheduling_job_plant_id)
                if st.session_state.scheduling_job_plant_id in plant_options
                else 0,
                help="Pick the plant that owns the SKUs you want to schedule.",
                key="scheduling_job_plant_selector",
            )
        else:
            selected_plant = st.text_input(
                "Plant ID",
                value=st.session_state.scheduling_job_plant_id,
                help="Pick the plant that owns the SKUs you want to schedule.",
                key="scheduling_job_plant_selector",
            )
        sku_search = st.text_input(
            "SKU search",
            value=st.session_state.scheduling_job_sku_search,
            placeholder="trade number, customer, product type, or bird size",
            help="Filter the SKU list before selecting the items to include in the run.",
            key="scheduling_job_sku_search_input",
        )

    st.session_state.scheduling_job_plant_id = str(selected_plant).strip()
    st.session_state.scheduling_job_sku_search = str(sku_search).strip()

    filtered_skus = _filter_skus_for_plant(all_skus, selected_plant, sku_search)
    sku_ids = [str(sku.get("tradeNumber", "")).strip() for sku in filtered_skus if str(sku.get("tradeNumber", "")).strip()]
    sku_id_set = set(sku_ids)
    sku_lookup = {
        str(sku.get("tradeNumber", "")).strip(): sku
        for sku in filtered_skus
        if str(sku.get("tradeNumber", "")).strip()
    }

    if len(sku_ids) > _MAX_SKU_SELECTOR_OPTIONS:
        st.info(
            f"This plant currently has {len(sku_ids)} matching SKUs. "
            f"To keep the page responsive, the selector shows the first {_MAX_SKU_SELECTOR_OPTIONS} plus any already selected SKUs. "
            "Narrow the SKU search to work with a smaller set."
        )

    with st.expander("SKU reference", expanded=False):
        _render_sku_reference(filtered_skus)

    with st.form("create_scheduling_job_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            run_id = st.text_input(
                "Run ID *",
                value=_default_run_id(),
                placeholder="schedule-2026-04-16",
                help="A friendly label for the run. It will show up in job history and the analytics page.",
            )
            plan_start_date = st.date_input(
                "Plan Start Date",
                value=date.today(),
                help="The first date the scheduler should consider.",
            )
            horizon_days = st.number_input(
                "Horizon Days",
                min_value=1,
                value=12,
                step=1,
                help="How many production days to include in the schedule.",
            )
        with col2:
            uploaded_short_term_file = st.file_uploader(
                "Short Term File (optional)",
                type=["csv", "tsv", "txt", "xlsx", "xls"],
                help="Upload a short-term demand file. The file is copied into shared storage so the worker can read it during the run.",
            )
            output_dir = st.text_input(
                "Output Dir",
                value="outputs",
                placeholder="outputs",
                help="Folder name used for CSV artifact output.",
            )
            save_csv = st.checkbox(
                "Save CSV artifacts",
                value=True,
                help="Store CSV artifacts for download after the run finishes.",
            )
            tee = st.checkbox("Show solver output (tee)", value=False, help="Stream solver logs during the run.")

            default_selected_skus = [sku for sku in st.session_state.scheduling_job_selected_skus if sku in sku_id_set]
            selectable_skus = _limit_sku_selector_options(sku_ids, default_selected_skus)
            selected_sku_ids = st.multiselect(
                "SKU IDs *",
                options=selectable_skus,
                default=default_selected_skus,
                format_func=lambda sku_id: _sku_label(sku_lookup.get(sku_id, {"tradeNumber": sku_id})),
                help="Pick the SKUs that should be included in the scheduling job.",
            )
        submit_clicked = st.form_submit_button(
            "Submit Scheduling Job",
            type="primary",
            use_container_width=True,
            disabled=bool(st.session_state.get("scheduling_job_submit_in_flight")),
        )

    st.session_state.scheduling_job_plant_id = str(selected_plant).strip()
    st.session_state.scheduling_job_sku_search = str(sku_search).strip()
    st.session_state.scheduling_job_selected_skus = selected_sku_ids

    if submit_clicked:
        if not run_id.strip():
            st.warning("Run ID is required.")
        elif not str(selected_plant).strip():
            st.warning("Plant ID is required.")
        elif not selected_sku_ids:
            st.warning("Select at least one SKU for the scheduling job.")
        else:
            short_term_file_path = _upload_short_term_file_to_minio(uploaded_short_term_file)
            payload = {
                "runId": run_id.strip(),
                "plantId": str(selected_plant).strip(),
                "skuIds": selected_sku_ids,
                "shortTermFile": short_term_file_path,
                "saveCsv": bool(save_csv),
                "outputDir": output_dir.strip() or "outputs",
                "tee": bool(tee),
                "planStartDate": plan_start_date.isoformat(),
                "horizonDays": int(horizon_days),
            }
            try:
                st.session_state.scheduling_job_submit_in_flight = True
                with st.spinner("Submitting scheduling job..."):
                    result = submit_scheduling_job(payload)
                st.session_state.scheduling_job_last_submitted = result
                st.success(
                    f"Scheduling job submitted. Status: **{result.get('status', 'unknown')}** "
                    f"jobId: `{result.get('jobId', '')}`"
                )
                st.session_state.scheduling_job_rows = _upsert_job_in_rows(
                    st.session_state.scheduling_job_rows,
                    result,
                )
                st.session_state.scheduling_job_rows_filter = st.session_state.scheduling_job_status_filter or None
                st.session_state.scheduling_job_selected_id = result.get("jobId", "")
            except APIError as error:
                _handle_api_error(error, "submit scheduling job")
            finally:
                st.session_state.scheduling_job_submit_in_flight = False

    last_submitted = st.session_state.get("scheduling_job_last_submitted")
    if last_submitted:
        st.markdown(
            f"<div class='simmons-card' style='margin-top:10px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap'>"
            f"<div style='font-weight:700'>Last submitted run</div>"
            f"<div class='simmons-small'>Run ID: <strong>{last_submitted.get('runId', '—')}</strong></div>"
            f"<div>{status_badge(last_submitted.get('status', 'pending'))}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Recent Scheduling Jobs")
    show_recent_jobs = st.checkbox(
        "Show recent jobs",
        value=bool(st.session_state.get("scheduling_job_show_recent_jobs", False)),
        help="Enable this only when you need to inspect job history. Keeping it hidden makes the page much lighter.",
        key="scheduling_job_show_recent_jobs_toggle",
    )
    st.session_state.scheduling_job_show_recent_jobs = show_recent_jobs

    if show_recent_jobs:
        tracking_col1, tracking_col2 = st.columns([1, 1])
        with tracking_col1:
            status_filter = st.selectbox(
                "Status Filter",
                options=["", "pending", "running", "completed", "failed", "cancelled"],
                index=["", "pending", "running", "completed", "failed", "cancelled"].index(
                    st.session_state.scheduling_job_status_filter
                ),
            )
        with tracking_col2:
            st.markdown("<div style='height:1.85rem'></div>", unsafe_allow_html=True)
            if st.button("Refresh Jobs", key="refresh_schedule_jobs", use_container_width=True):
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
        if selected_plant.strip():
            job_rows = [job for job in job_rows if str(job.get("plantId", "")).strip() == str(selected_plant).strip()]

        _render_recent_jobs(sort_records_by_date(job_rows, ("createdAt",)))
    else:
        st.caption("Recent jobs are hidden by default to keep the page responsive while you work on a new run.")
