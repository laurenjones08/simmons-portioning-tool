"""Enumeration Dashboard view for single-page router."""
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from api_client import (
    APIError,
    search_skus,
    search_mixes,
    search_mix_metrics,
    search_cut_strategies,
    submit_job,
    list_jobs,
    cancel_job,
)


# ---------------------------------------------------------------------------
# Cached data loaders (TTL=60 seconds, cleared on manual refresh)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def _load_all_mixes():
    try:
        return search_mixes({})
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def _load_all_metrics():
    try:
        return search_mix_metrics({})
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def _load_cut_strategies():
    try:
        strats = search_cut_strategies({})
        return {s["_id"]: s for s in strats if "_id" in s}
    except Exception:
        return {}


def _clear_results_cache():
    _load_all_mixes.clear()
    _load_all_metrics.clear()
    _load_cut_strategies.clear()


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


def _fmt_dt(val: str) -> str:
    if not val:
        return "—"
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00")).strftime("%m/%d %H:%M")
    except Exception:
        return val


def _fmt_skus(skus: dict) -> str:
    if not skus:
        return "—"
    return ", ".join(f"{k}→{v}" for k, v in sorted(skus.items()))


def _build_mixes_df(mixes: list, strategies: dict, metrics_by_mix: dict) -> pd.DataFrame:
    rows = []
    for m in mixes:
        mid = m.get("_id", "")
        strat = strategies.get(m.get("cutStrategyID", ""), {})
        parts = ", ".join(strat.get("parts", [])) or "—"
        line_type = m.get("mfgType") or strat.get("lineType", "—")
        mix_metrics = metrics_by_mix.get(mid, [])
        upgrades = [mm.get("upgradePercentage") for mm in mix_metrics if mm.get("upgradePercentage") is not None]
        trims = [mm.get("trimPercentage") for mm in mix_metrics if mm.get("trimPercentage") is not None]
        best_up = round(max(upgrades), 2) if upgrades else None
        avg_up = round(sum(upgrades) / len(upgrades), 2) if upgrades else None
        avg_tr = round(sum(trims) / len(trims), 2) if trims else None
        flags = []
        if m.get("includesFDS"):
            flags.append("FDS")
        if m.get("includesRTL"):
            flags.append("RTL")
        if m.get("includesNug"):
            flags.append("NUG")
        rows.append({
            "_id": mid,
            "SKUs": _fmt_skus(m.get("skus", {})),
            "# SKUs": len(m.get("skus", {})),
            "Line": line_type,
            "Parts": parts,
            "Plant": m.get("reqPlant", "—"),
            "Bird Size": m.get("reqBirdSize", "—"),
            "Fillets": m.get("numFillets", 0),
            "Belt Speed": m.get("beltSpeed"),
            "Flags": " ".join(flags) if flags else "—",
            "Best Upgrade %": best_up,
            "Avg Upgrade %": avg_up,
            "Avg Trim %": avg_tr,
            "Buckets": len(mix_metrics),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    # ── 1. CURRENT SKU INPUT DATA ──────────────────────────────────────────
    try:
        skus = search_skus({})
    except Exception:
        skus = []

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
    st.markdown(
        "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
        "border-left:4px solid #0046AD;padding-left:12px;'>Run Enumeration</h3>",
        unsafe_allow_html=True,
    )
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
            _clear_results_cache()
        except APIError as exc:
            if exc.status_code == 409:
                st.error("Another enumeration job is already running. Cancel it or wait for it to complete before submitting a new run.")
            else:
                st.error(f"API error {exc.status_code}: {exc.detail}")
        except Exception as exc:
            st.error(f"Could not submit job: {exc}")

    if "enum_last_submitted_job" in st.session_state:
        last = st.session_state["enum_last_submitted_job"]
        badge = _status_badge(last.get("status", "pending"))
        st.markdown(
            f"<div class='simmons-card' style='margin-top:8px'>"
            f"<strong>Last Submitted:</strong> {last.get('runId', '—')} &nbsp; {badge}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 3. JOB STATUS ─────────────────────────────────────────────────────
    hdr_col, refresh_col = st.columns([5, 1])
    with hdr_col:
        st.markdown(
            "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
            "border-left:4px solid #0046AD;padding-left:12px;'>Enumeration Job Status</h3>",
            unsafe_allow_html=True,
        )
    with refresh_col:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("↻ Refresh", key="enum_refresh_jobs"):
            st.rerun()

    try:
        jobs = list_jobs()
    except Exception:
        jobs = []

    if jobs:
        jobs_sorted = sorted(jobs, key=lambda j: j.get("createdAt", ""), reverse=True)

        rows = []
        for j in jobs_sorted[:10]:
            stages = j.get("stages", [])
            total_combos = sum(s.get("totalCombinations", 0) for s in stages if isinstance(s, dict))
            rows.append({
                "Run ID": j.get("runId", "—"),
                "Status": j.get("status", "—"),
                "SKUs": j.get("skuCount", 0),
                "Plant": j.get("plantFilter") or "all",
                "Bird": j.get("birdSizeFilter") or "all",
                "Max Combo": j.get("maxCombinationSize", "—"),
                "Total Combos": total_combos if total_combos else "—",
                "Submitted": _fmt_dt(j.get("createdAt", "")),
                "Finished": _fmt_dt(j.get("finishedAt", "")),
                "_jobId": j.get("jobId") or j.get("_id", ""),
            })

        df_jobs = pd.DataFrame(rows)
        st.dataframe(df_jobs.drop(columns=["_jobId"]), use_container_width=True, hide_index=True)

        # Stage detail expander for the most recent completed job
        completed = [j for j in jobs_sorted if j.get("status") == "completed"]
        if completed:
            latest = completed[0]
            with st.expander(f"Stage Detail — {latest.get('runId', '—')}", expanded=False):
                stages = latest.get("stages", [])
                if stages:
                    sc_cols = st.columns(len(stages))
                    for i, s in enumerate(stages):
                        processed = s.get("processedCombinations", 0)
                        total = s.get("totalCombinations", 0)
                        pct = int(100 * processed / total) if total else 0
                        with sc_cols[i]:
                            st.markdown(
                                f"<div class='simmons-card' style='text-align:center'>"
                                f"<div style='font-size:13px;font-weight:600;color:#0046AD'>Stage {s.get('stage')}</div>"
                                f"<div class='simmons-kpi' style='font-size:20px'>{processed:,}</div>"
                                f"<div class='simmons-kpi-label'>of {total:,} combos</div>"
                                f"<div style='margin-top:6px;font-size:11px;color:#4CAF50'>{pct}% complete</div>"
                                f"<div style='margin-top:2px;font-size:10px;color:#888'>"
                                f"{_fmt_dt(s.get('startedAt',''))} → {_fmt_dt(s.get('finishedAt',''))}"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )

        # Cancel control
        running_ids = [r["_jobId"] for r in rows if r["Status"] in ("running", "pending") and r["_jobId"]]
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

    # ── 4. ENUMERATION RESULTS ────────────────────────────────────────────
    res_hdr, res_refresh = st.columns([5, 1])
    with res_hdr:
        st.markdown(
            "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
            "border-left:4px solid #0046AD;padding-left:12px;'>Enumeration Results</h3>",
            unsafe_allow_html=True,
        )
        st.caption("Ranked portioning mixes generated by the engine. Select any row to inspect bucket-level metrics and unit plans.")
    with res_refresh:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("↻ Refresh Results", key="enum_refresh_results"):
            _clear_results_cache()
            st.rerun()

    with st.spinner("Loading mixes and metrics…"):
        all_mixes = _load_all_mixes()
        all_metrics = _load_all_metrics()
        all_strategies = _load_cut_strategies()

    if not all_mixes:
        st.info("No enumeration results available yet. Submit a job above to generate portioning mixes.")
    else:
        # Build lookups
        metrics_by_mix: dict[str, list] = {}
        for mm in all_metrics:
            mid = mm.get("mixId", "")
            metrics_by_mix.setdefault(mid, []).append(mm)

        df_mixes = _build_mixes_df(all_mixes, all_strategies, metrics_by_mix)

        # ── KPI summary row ──────────────────────────────────────────────
        all_upgrades = [mm.get("upgradePercentage") for mm in all_metrics if mm.get("upgradePercentage") is not None]
        all_trims = [mm.get("trimPercentage") for mm in all_metrics if mm.get("trimPercentage") is not None]
        best_upgrade = max(all_upgrades) if all_upgrades else None
        avg_upgrade = sum(all_upgrades) / len(all_upgrades) if all_upgrades else None
        avg_trim = sum(all_trims) / len(all_trims) if all_trims else None

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Mixes", len(all_mixes))
        k2.metric("Total Metrics", len(all_metrics))
        k3.metric("Best Upgrade %", f"{best_upgrade:.1f}%" if best_upgrade is not None else "—")
        k4.metric("Avg Upgrade %", f"{avg_upgrade:.1f}%" if avg_upgrade is not None else "—")

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

        # ── Filter controls ──────────────────────────────────────────────
        with st.expander("Filter Results", expanded=False):
            filt1, filt2, filt3, filt4 = st.columns(4)
            plant_vals = sorted(df_mixes["Plant"].dropna().unique().tolist())
            bird_vals = sorted(df_mixes["Bird Size"].dropna().unique().tolist())
            line_vals = sorted(df_mixes["Line"].dropna().unique().tolist())
            filt_plant = filt1.selectbox("Plant", ["All"] + plant_vals, key="res_filt_plant")
            filt_bird = filt2.selectbox("Bird Size", ["All"] + bird_vals, key="res_filt_bird")
            filt_line = filt3.selectbox("Line Type", ["All"] + line_vals, key="res_filt_line")
            filt_min_upgrade = filt4.slider("Min Best Upgrade %", 0.0, 100.0, 0.0, 0.5, key="res_filt_upgrade")

        df_filtered = df_mixes.copy()
        if filt_plant != "All":
            df_filtered = df_filtered[df_filtered["Plant"] == filt_plant]
        if filt_bird != "All":
            df_filtered = df_filtered[df_filtered["Bird Size"] == filt_bird]
        if filt_line != "All":
            df_filtered = df_filtered[df_filtered["Line"] == filt_line]
        if filt_min_upgrade > 0:
            df_filtered = df_filtered[df_filtered["Best Upgrade %"].fillna(0) >= filt_min_upgrade]
        df_filtered = df_filtered.sort_values("Best Upgrade %", ascending=False, na_position="last")

        st.caption(f"Showing {len(df_filtered)} of {len(df_mixes)} mixes")

        # ── Ranked mixes table (row-selectable) ──────────────────────────
        display_cols = ["SKUs", "# SKUs", "Line", "Parts", "Plant", "Bird Size",
                        "Fillets", "Belt Speed", "Flags", "Best Upgrade %", "Avg Upgrade %", "Avg Trim %", "Buckets"]
        df_display = df_filtered[display_cols].reset_index(drop=True)

        event = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=340,
            on_select="rerun",
            selection_mode="single-row",
            key="mixes_table",
            column_config={
                "Best Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Avg Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Avg Trim %": st.column_config.NumberColumn(format="%.2f%%"),
                "Belt Speed": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        selected_rows = event.selection.rows if event and event.selection else []
        if selected_rows:
            row_idx = selected_rows[0]
            selected_mix_id = df_filtered.iloc[row_idx]["_id"]
        else:
            selected_mix_id = None

        # ── Mix detail panel ──────────────────────────────────────────────
        if selected_mix_id:
            mix_obj = next((m for m in all_mixes if m.get("_id") == selected_mix_id), None)
            mix_metrics = metrics_by_mix.get(selected_mix_id, [])
            strat = all_strategies.get(mix_obj.get("cutStrategyID", ""), {}) if mix_obj else {}

            st.markdown("---")
            st.markdown("#### Mix Detail")

            det1, det2, det3 = st.columns([2, 2, 1])

            with det1:
                st.markdown("**Mix Configuration**")
                if mix_obj:
                    skus_str = _fmt_skus(mix_obj.get("skus", {}))
                    flags = []
                    if mix_obj.get("includesFDS"):
                        flags.append("FDS")
                    if mix_obj.get("includesRTL"):
                        flags.append("RTL")
                    if mix_obj.get("includesNug"):
                        flags.append("NUG")
                    st.markdown(
                        f"<div class='simmons-card'>"
                        f"<table style='width:100%;font-size:13px;border-collapse:collapse'>"
                        f"<tr><td style='color:#888;padding:2px 8px 2px 0'>SKUs</td><td><strong>{skus_str}</strong></td></tr>"
                        f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Line Type</td><td>{mix_obj.get('mfgType','—')}</td></tr>"
                        f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Parts</td><td>{', '.join(strat.get('parts',[]) or ['—'])}</td></tr>"
                        f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Plant</td><td>{mix_obj.get('reqPlant','—')}</td></tr>"
                        f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Bird Size</td><td>{mix_obj.get('reqBirdSize','—')}</td></tr>"
                        f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Fillets</td><td>{mix_obj.get('numFillets','—')} ({mix_obj.get('filletWeight','—')}g)</td></tr>"
                        f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Belt Speed</td><td>{mix_obj.get('beltSpeed','—')}</td></tr>"
                        f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Flags</td><td>{' '.join(flags) if flags else '—'}</td></tr>"
                        f"</table></div>",
                        unsafe_allow_html=True,
                    )

            with det2:
                st.markdown("**Bucket Performance**")
                if mix_metrics:
                    df_mm = pd.DataFrame(mix_metrics).sort_values("upgradePercentage", ascending=False)
                    show_cols = [c for c in ["bucketId", "upgradePercentage", "trimPercentage",
                                              "totalProductProducedGrams", "value"] if c in df_mm.columns]
                    st.dataframe(
                        df_mm[show_cols].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                        height=220,
                        column_config={
                            "upgradePercentage": st.column_config.NumberColumn("Upgrade %", format="%.2f%%"),
                            "trimPercentage": st.column_config.NumberColumn("Trim %", format="%.2f%%"),
                            "totalProductProducedGrams": st.column_config.NumberColumn("Total Wt (g)", format="%.1f"),
                            "value": st.column_config.NumberColumn("Value $", format="%.2f"),
                        },
                    )
                else:
                    st.info("No bucket metrics for this mix.")

            with det3:
                st.markdown("**KPIs**")
                if mix_metrics:
                    upgrades_m = [mm.get("upgradePercentage") for mm in mix_metrics if mm.get("upgradePercentage") is not None]
                    trims_m = [mm.get("trimPercentage") for mm in mix_metrics if mm.get("trimPercentage") is not None]
                    st.metric("Best Upgrade", f"{max(upgrades_m):.1f}%" if upgrades_m else "—")
                    st.metric("Avg Upgrade", f"{sum(upgrades_m)/len(upgrades_m):.1f}%" if upgrades_m else "—")
                    st.metric("Avg Trim", f"{sum(trims_m)/len(trims_m):.1f}%" if trims_m else "—")
                    st.metric("Buckets", len(mix_metrics))
                st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
                if st.button("→ Send to Scheduling", type="primary", key="send_to_sched"):
                    st.session_state["sched_selected_mix_id"] = selected_mix_id
                    try:
                        st.query_params["page"] = "Scheduling Dashboard"
                    except Exception:
                        st.session_state.ui_selected_page = "Scheduling Dashboard"
                        st.session_state.ui_sidebar_nav = "Scheduling Dashboard"

            # Best bucket unit plan
            if mix_metrics:
                best_metric = max(mix_metrics, key=lambda mm: mm.get("upgradePercentage") or 0)
                unit_plan = best_metric.get("unitPlan", [])
                if unit_plan:
                    st.markdown("**Unit Plan — Best Performing Bucket**")
                    st.caption(f"Bucket: `{best_metric.get('bucketId','—')}` · Upgrade: {best_metric.get('upgradePercentage',0):.2f}% · Trim: {best_metric.get('trimPercentage',0):.2f}%")
                    df_plan = pd.DataFrame(unit_plan)
                    plan_cols = [c for c in ["sku", "partCode", "unitsInPlan", "totalWeightInPlan", "pctOfTotal"] if c in df_plan.columns]
                    st.dataframe(
                        df_plan[plan_cols].reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "totalWeightInPlan": st.column_config.NumberColumn("Total Wt (g)", format="%.1f"),
                            "pctOfTotal": st.column_config.NumberColumn("% of Total", format="%.1f%%"),
                        },
                    )

            # Export
            export_col, _ = st.columns([2, 4])
            with export_col:
                if mix_metrics:
                    df_export = pd.DataFrame(mix_metrics).sort_values("upgradePercentage", ascending=False)
                    csv_bytes = df_export.to_csv(index=False).encode()
                    st.download_button(
                        "⬇ Export Metrics CSV",
                        data=csv_bytes,
                        file_name=f"mix-metrics-{selected_mix_id[:8]}.csv",
                        mime="text/csv",
                    )
        else:
            st.caption("Click any row above to inspect its bucket-level metrics and unit plan.")

    st.markdown("---")

    # ── 5. ANALYTICS ──────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
        "border-left:4px solid #0046AD;padding-left:12px;'>Enumeration Analytics</h3>",
        unsafe_allow_html=True,
    )
    if all_mixes and all_metrics:
        ac1, ac2, ac3 = st.columns(3)

        with ac1:
            st.markdown("**Best Upgrade % Distribution (all mixes)**")
            upgrades_per_mix = []
            for m in all_mixes:
                mid = m.get("_id", "")
                mix_ups = [mm.get("upgradePercentage") for mm in metrics_by_mix.get(mid, []) if mm.get("upgradePercentage") is not None]
                if mix_ups:
                    upgrades_per_mix.append(max(mix_ups))
            if upgrades_per_mix:
                buckets_hist = pd.cut(pd.Series(upgrades_per_mix), bins=10)
                hist_counts = buckets_hist.value_counts().sort_index()
                hist_df = pd.DataFrame({"Upgrade % Range": hist_counts.index.astype(str), "Count": hist_counts.values})
                st.bar_chart(hist_df.set_index("Upgrade % Range"))
            else:
                st.info("No upgrade data.")

        with ac2:
            st.markdown("**Top 10 Mixes by Best Upgrade %**")
            if not df_mixes.empty:
                top10 = df_mixes.nlargest(10, "Best Upgrade %")[["SKUs", "Line", "Plant", "Best Upgrade %"]]
                st.dataframe(
                    top10.reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                    height=280,
                    column_config={"Best Upgrade %": st.column_config.NumberColumn(format="%.2f%%")},
                )
            else:
                st.info("No results to display.")

        with ac3:
            st.markdown("**Mix Count by Line Type**")
            if not df_mixes.empty and "Line" in df_mixes.columns:
                line_counts = df_mixes["Line"].value_counts()
                st.bar_chart(line_counts)
            else:
                st.info("No line type data.")
    else:
        st.info("No results available for analytics. Submit an enumeration job to generate data.")

    st.caption("Workflow: Review Input Data → Configure & Submit Enumeration Run → Review Job Status → Inspect Results → Send to Scheduling")

