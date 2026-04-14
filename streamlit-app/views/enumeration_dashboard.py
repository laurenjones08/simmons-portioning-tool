"""Enumeration Dashboard view for single-page router."""
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from api_client import (
    APIError,
    search_skus,
    search_buckets,
    search_mixes,
    search_mix_metrics,
    search_cut_strategies,
    submit_job,
    list_jobs,
    get_job,
    cancel_job,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_badge(status: str) -> str:
    colors = {
        "completed": ("#4CAF50", "#e8f5e9"),
        "running": ("#0046AD", "#e3f0ff"),
        "pending": ("#FFB74D", "#fff8e1"),
        "failed": ("#D9534F", "#fdecea"),
        "cancelled": ("#6b7280", "#f3f4f6"),
    }
    fg, bg = colors.get(status, ("#333", "#eee"))
    return (
        f"<span style='background:{bg};color:{fg};padding:2px 8px;"
        f"border-radius:12px;font-size:12px;font-weight:600'>{status.upper()}</span>"
    )


def _render_job_status_panel(jobs: list[dict]) -> None:
    if not jobs:
        st.info("No enumeration jobs found.")
        return
    recent = sorted(jobs, key=lambda j: j.get("createdAt", ""), reverse=True)[:5]
    for j in recent:
        jid = j.get("jobId") or j.get("_id", "—")
        run_id = j.get("runId", "—")
        status = j.get("status", "unknown")
        created = j.get("createdAt", "")
        if isinstance(created, str) and created:
            try:
                created = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                pass
        stages = j.get("stages", [])
        sku_count = j.get("skuCount", 0)
        badge = _status_badge(status)
        stages_html = ""
        if stages:
            stage_parts = ", ".join(
                f"{s.get('stage')}:{s.get('status')}" for s in stages if isinstance(s, dict)
            )
            stages_html = f"<div class='simmons-small'>Stages: {stage_parts}</div>"
        st.markdown(
            f"<div class='simmons-card' style='margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<strong>{run_id}</strong>{badge}</div>"
            f"<div class='simmons-small' style='margin-top:4px'>Job ID: {jid} &nbsp;|&nbsp; SKUs: {sku_count} &nbsp;|&nbsp; {created}</div>"
            + stages_html
            + "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    # ── 1. CURRENT SKU INPUT DATA ──────────────────────────────────────────
    try:
        skus = search_skus({})
    except Exception:
        skus = []

    # Precompute filter options for use in the run form below
    if skus:
        df_skus = pd.DataFrame(skus)
        plant_options = sorted(df_skus["prodPlant"].dropna().unique().tolist()) if "prodPlant" in df_skus.columns else []
    else:
        df_skus = pd.DataFrame()
        plant_options = []

    with st.expander("SKU Input Data", expanded=False):
        if not skus:
            st.info("No SKUs returned from Enumeration API. Ensure the API is running and SKUs are loaded.")
        else:
            display_cols = [
                "tradeNumber", "customerName", "customerType", "productType",
                "targetWeight", "minWeight", "maxWeight", "allowedParts",
                "birdSize", "prodPlant",
            ]
            present_cols = [c for c in display_cols if c in df_skus.columns]

            fc1, fc2, fc3 = st.columns([3, 2, 2])
            q = fc1.text_input("Search trade # or customer", value="")
            plant_filter_sku = fc2.selectbox("Filter by Plant", options=["All"] + plant_options, key="sku_plant_filter")
            bird_opts = sorted(df_skus["birdSize"].dropna().unique().tolist()) if "birdSize" in df_skus.columns else []
            bird_filter_sku = fc3.selectbox("Filter by Bird Size", options=["All"] + bird_opts, key="sku_bird_filter")

            df_view = df_skus.copy()
            if q:
                qlow = q.lower()
                cols_to_search = [c for c in ["tradeNumber", "customerName"] if c in df_view.columns]
                mask = pd.Series([False] * len(df_view), index=df_view.index)
                for c in cols_to_search:
                    mask = mask | df_view[c].astype(str).str.lower().str.contains(qlow)
                df_view = df_view[mask]
            if plant_filter_sku != "All" and "prodPlant" in df_view.columns:
                df_view = df_view[df_view["prodPlant"] == plant_filter_sku]
            if bird_filter_sku != "All" and "birdSize" in df_view.columns:
                df_view = df_view[df_view["birdSize"] == bird_filter_sku]

            st.dataframe(df_view[present_cols], use_container_width=True, height=260)
            st.caption(f"{len(df_view)} of {len(df_skus)} SKUs shown")

    st.markdown("---")

    # ── 2. RUN ENUMERATION ─────────────────────────────────────────────────
    st.subheader("Run Enumeration")
    st.caption("Submits an enumeration job to the Worker API. The job evaluates all feasible portioning combinations for the selected plant and bird size, then ranks results by upgrade % and economic value.")

    with st.form("enumeration_run_form"):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("**Run Identity**")
            default_run_id = f"enum-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
            run_id_input = st.text_input("Run ID (label)", value=default_run_id,
                                         help="Human-readable identifier stored with the job record.")
        with rc2:
            st.markdown("**Data Filters**")
            plant_choice = st.selectbox("Plant Filter", options=["(all plants)"] + plant_options,
                                         help="Limit enumeration to SKUs belonging to this plant.")
            bird_choice = st.selectbox("Bird Size Filter", options=["(all sizes)", "SB", "BB"],
                                        help="SB = Small Bird, BB = Big Bird. Leave blank to include all.")
        with rc3:
            st.markdown("**Engine Settings**")
            max_combo = st.selectbox("Max Combination Size", options=[1, 2, 3, 4], index=3,
                                      help="Maximum number of SKUs evaluated per combination. 4 is standard.")
            batch_size_input = st.number_input("Batch Size", min_value=100, max_value=10000, value=1000, step=100,
                                                help="Combinations processed per DB write batch. 1000 is recommended.")

        submitted = st.form_submit_button("▶ Submit Enumeration Job", type="primary")

    if submitted:
        payload: dict = {"runId": run_id_input, "maxCombinationSize": max_combo, "batchSize": batch_size_input}
        if plant_choice and plant_choice != "(all plants)":
            payload["plantFilter"] = plant_choice
        if bird_choice and bird_choice != "(all sizes)":
            payload["birdSizeFilter"] = bird_choice

        try:
            result = submit_job(payload)
            job_id_created = result.get("jobId") or result.get("_id", "")
            st.session_state["enum_last_submitted_job"] = result
            st.success(f"Job submitted — Run ID: **{run_id_input}** &nbsp; Job ID: `{job_id_created}`")
        except APIError as exc:
            if exc.status_code == 409:
                st.error("Another enumeration job is already running. Cancel it or wait for it to complete before submitting a new run.")
            else:
                st.error(f"API error {exc.status_code}: {exc.detail}")
        except Exception as exc:
            st.error(f"Could not submit job: {exc}")

    # Last submission preview
    if "enum_last_submitted_job" in st.session_state:
        last = st.session_state["enum_last_submitted_job"]
        status_val = last.get("status", "pending")
        badge = _status_badge(status_val)
        st.markdown(
            f"<div class='simmons-card' style='margin-top:8px'>"
            f"<strong>Last Submitted:</strong> {last.get('runId', '—')} &nbsp; {badge}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 3. JOB STATUS ─────────────────────────────────────────────────────
    st.subheader("Enumeration Job Status")
    col_jobs, col_cancel = st.columns([5, 1])
    with col_cancel:
        refresh_jobs = st.button("↻ Refresh", key="enum_refresh_jobs")

    try:
        jobs = list_jobs()
    except Exception:
        jobs = []

    if jobs:
        # Active/recent jobs table
        jobs_sorted = sorted(jobs, key=lambda j: j.get("createdAt", ""), reverse=True)

        def _fmt_dt(val: str) -> str:
            if not val:
                return "—"
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).strftime("%m/%d %H:%M")
            except Exception:
                return val

        rows = []
        for j in jobs_sorted[:10]:
            rows.append({
                "Run ID": j.get("runId", "—"),
                "Status": j.get("status", "—"),
                "SKUs": j.get("skuCount", 0),
                "Plant": j.get("plantFilter") or "all",
                "Bird Size": j.get("birdSizeFilter") or "all",
                "Submitted": _fmt_dt(j.get("createdAt", "")),
                "Finished": _fmt_dt(j.get("finishedAt", "")),
                "_jobId": j.get("jobId") or j.get("_id", ""),
            })
        df_jobs = pd.DataFrame(rows)
        st.dataframe(df_jobs.drop(columns=["_jobId"]), use_container_width=True, hide_index=True)

        # Cancel control
        running_ids = [j["_jobId"] for j in rows if j["Status"] in ("running", "pending") and j["_jobId"]]
        if running_ids:
            cancel_target = st.selectbox("Cancel a running/pending job", options=["—"] + running_ids)
            if st.button("✕ Cancel Selected Job", type="secondary") and cancel_target != "—":
                try:
                    cancel_job(cancel_target)
                    st.success(f"Cancellation requested for job `{cancel_target}`.")
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.info("No enumeration jobs found. Submit a run above to get started.")

    st.markdown("---")

    # ── 4. ENUMERATION RESULTS / SNAPSHOTS ────────────────────────────────
    st.subheader("Portioning Snapshots")
    st.caption("Each completed enumeration job produces a ranked mix snapshot. Select a snapshot to inspect portioning metrics.")

    try:
        mixes = search_mixes({})
    except Exception:
        mixes = []

    if not mixes:
        st.info("No enumeration snapshots available. Run an enumeration job above to generate results.")
    else:
        mix_options = {m.get("_id", f"mix-{i}"): m for i, m in enumerate(mixes)}
        mix_labels = {
            mid: f"{m.get('reqPlant','?')} | {m.get('reqBirdSize','?')} | {mid}"
            for mid, m in mix_options.items()
        }
        selected_mix_id = st.selectbox(
            "Select Snapshot",
            options=[""] + list(mix_options.keys()),
            format_func=lambda x: mix_labels.get(x, x) if x else "— select a snapshot —",
        )

        if selected_mix_id:
            mix = mix_options[selected_mix_id]
            mc1, mc2 = st.columns([3, 1])

            with mc1:
                try:
                    metrics = search_mix_metrics({"mixId": selected_mix_id})
                except Exception:
                    metrics = []

                if metrics:
                    df_m = pd.DataFrame(metrics)
                    metric_display = [
                        c for c in [
                            "bucketId", "upgradePercentage", "value",
                            "trimPercentage", "totalProductProducedGrams",
                        ] if c in df_m.columns
                    ]
                    # Sort by upgrade descending
                    if "upgradePercentage" in df_m.columns:
                        df_m = df_m.sort_values("upgradePercentage", ascending=False)

                    st.markdown("**Ranked Portioning Decisions**")
                    st.dataframe(df_m[metric_display], use_container_width=True, height=380)

                    # Export
                    csv_bytes = df_m[metric_display].to_csv(index=False).encode()
                    st.download_button(
                        "⬇ Export Snapshot CSV",
                        data=csv_bytes,
                        file_name=f"snapshot-{selected_mix_id[:8]}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No metrics found for this snapshot.")

            with mc2:
                st.markdown("**Snapshot KPIs**")
                try:
                    metrics_kpi = metrics if metrics else search_mix_metrics({"mixId": selected_mix_id})
                    upgrades = [m.get("upgradePercentage") for m in metrics_kpi if m.get("upgradePercentage") is not None]
                    trims = [m.get("trimPercentage") for m in metrics_kpi if m.get("trimPercentage") is not None]
                    avg_up = sum(upgrades) / len(upgrades) if upgrades else None
                    avg_tr = sum(trims) / len(trims) if trims else None
                    max_up = max(upgrades) if upgrades else None
                except Exception:
                    avg_up = avg_tr = max_up = None

                st.markdown(
                    f"<div class='simmons-card'>"
                    f"<div class='simmons-kpi'>{f'{avg_up:.1f}%' if avg_up is not None else '—'}</div>"
                    f"<div class='simmons-kpi-label'>Avg Upgrade %</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='simmons-card' style='margin-top:8px'>"
                    f"<div class='simmons-kpi'>{f'{max_up:.1f}%' if max_up is not None else '—'}</div>"
                    f"<div class='simmons-kpi-label'>Best Upgrade %</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='simmons-card' style='margin-top:8px'>"
                    f"<div class='simmons-kpi'>{f'{avg_tr:.1f}%' if avg_tr is not None else '—'}</div>"
                    f"<div class='simmons-kpi-label'>Avg Trim %</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='simmons-card' style='margin-top:8px'>"
                    f"<div class='simmons-kpi'>{len(metrics) if metrics else '—'}</div>"
                    f"<div class='simmons-kpi-label'>Candidate Combinations</div></div>",
                    unsafe_allow_html=True,
                )

                # Send to scheduling
                st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
                if st.button("→ Send to Scheduling", type="primary", help="Open Scheduling Dashboard with this snapshot pre-selected"):
                    st.session_state["sched_selected_mix_id"] = selected_mix_id
                    st.session_state.ui_sidebar_nav = "Scheduling Dashboard"
                    st.session_state.ui_selected_page = "Scheduling Dashboard"
                    try:
                        st.experimental_set_query_params(page="Scheduling Dashboard")
                    except Exception:
                        pass

    st.markdown("---")

    # ── 5. ANALYTICS ──────────────────────────────────────────────────────
    st.subheader("Enumeration Analytics")
    if mixes:
        # Load metrics for first available mix to populate charts
        try:
            sample_metrics = search_mix_metrics({"mixId": mixes[0].get("_id")})
        except Exception:
            sample_metrics = []

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.markdown("**Upgrade % Distribution**")
            if sample_metrics:
                upgrades = [m.get("upgradePercentage") for m in sample_metrics if m.get("upgradePercentage") is not None]
                if upgrades:
                    st.bar_chart(pd.Series(upgrades, name="Upgrade %"))
                else:
                    st.info("No upgrade data.")
            else:
                st.info("Load a snapshot to see distribution.")

        with ac2:
            st.markdown("**Trim % Distribution**")
            if sample_metrics:
                trims = [m.get("trimPercentage") for m in sample_metrics if m.get("trimPercentage") is not None]
                if trims:
                    st.bar_chart(pd.Series(trims, name="Trim %"))
                else:
                    st.info("No trim data.")
            else:
                st.info("Load a snapshot to see distribution.")

        with ac3:
            st.markdown("**Snapshot Count by Plant**")
            try:
                df_mix_all = pd.DataFrame(mixes)
                if "reqPlant" in df_mix_all.columns:
                    plant_counts = df_mix_all["reqPlant"].value_counts()
                    st.bar_chart(plant_counts)
                else:
                    st.info("No plant data in snapshots.")
            except Exception:
                st.info("No data.")
    else:
        st.info("No snapshots available for analytics.")

    st.caption("Workflow: Review Input Data → Configure & Submit Enumeration Run → Review Job Status → Inspect Ranked Snapshot → Send to Scheduling")

