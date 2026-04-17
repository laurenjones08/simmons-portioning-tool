"""Enumeration Dashboard view for single-page router."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean

import pandas as pd
import streamlit as st

from api_client import (
    APIError,
    cancel_job,
    list_jobs,
    search_buckets,
    search_cut_strategies,
    search_mix_metrics,
    search_mixes,
    search_skus,
    submit_job,
)


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
        strategies = search_cut_strategies({})
        return {item["_id"]: item for item in strategies if "_id" in item}
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def _load_all_buckets():
    try:
        return search_buckets({})
    except Exception:
        return []


def _clear_results_cache():
    _load_all_mixes.clear()
    _load_all_metrics.clear()
    _load_cut_strategies.clear()
    _load_all_buckets.clear()


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
    return ", ".join(f"{k}->{v}" for k, v in sorted(skus.items()))


def _fmt_bucket_range(bucket: dict | None) -> str:
    if not bucket:
        return "[—, —]"
    min_weight = bucket.get("minWeight")
    max_weight = bucket.get("maxWeight")
    if min_weight is None or max_weight is None:
        return "[—, —]"
    return f"[{float(min_weight):g}, {float(max_weight):g}]"


def _bucket_label_map(buckets: list[dict]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for bucket in buckets:
        bucket_id = str(bucket.get("_id", ""))
        if bucket_id:
            labels[bucket_id] = _fmt_bucket_range(bucket)
    return labels


def _metrics_by_mix(all_metrics: list[dict]) -> dict[str, list[dict]]:
    metrics: dict[str, list[dict]] = {}
    for metric in all_metrics:
        mix_id = str(metric.get("mixId", ""))
        metrics.setdefault(mix_id, []).append(metric)
    return metrics


def _bucket_metric(metrics: list[dict], bucket_id: str | None) -> dict | None:
    if not bucket_id:
        return None
    return next((metric for metric in metrics if str(metric.get("bucketId", "")) == bucket_id), None)


def _mix_score(metrics: list[dict]) -> dict[str, float | None]:
    upgrades = [item.get("upgradePercentage") for item in metrics if item.get("upgradePercentage") is not None]
    trims = [item.get("trimPercentage") for item in metrics if item.get("trimPercentage") is not None]
    values = [item.get("value") for item in metrics if item.get("value") is not None]
    return {
        "best_upgrade": max(upgrades) if upgrades else None,
        "avg_upgrade": mean(upgrades) if upgrades else None,
        "avg_trim": mean(trims) if trims else None,
        "avg_value": mean(values) if values else None,
    }


def _build_mixes_df(
    mixes: list,
    strategies: dict,
    metrics_by_mix: dict[str, list],
    bucket_filter_id: str | None,
    bucket_labels: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for mix in mixes:
        mix_id = str(mix.get("_id", ""))
        strategy = strategies.get(mix.get("cutStrategyID", ""), {})
        mix_metrics = metrics_by_mix.get(mix_id, [])
        selected_metric = _bucket_metric(mix_metrics, bucket_filter_id)

        flags = []
        if mix.get("includesFDS"):
            flags.append("FDS")
        if mix.get("includesRTL"):
            flags.append("RTL")
        if mix.get("includesNug"):
            flags.append("NUG")

        row = {
            "_id": mix_id,
            "SKUs": _fmt_skus(mix.get("skus", {})),
            "# SKUs": len(mix.get("skus", {})),
            "Line": mix.get("mfgType") or strategy.get("lineType", "—"),
            "Parts": ", ".join(strategy.get("parts", [])) or "—",
            "Plant": mix.get("reqPlant", "—"),
            "Bird Size": mix.get("reqBirdSize", "—"),
            "# Fillets": mix.get("numFillets", 0),
            "Fillets": mix.get("numFillets", 0),
            "Belt Speed": mix.get("beltSpeed"),
            "Customer Type": " ".join(flags) if flags else "—",
            "Flags": " ".join(flags) if flags else "—",
            "Buckets": len(mix_metrics),
        }
        if selected_metric:
            row.update(
                {
                    "Bucket": bucket_labels.get(str(selected_metric.get("bucketId", "")), "—"),
                    "Bucket Upgrade %": selected_metric.get("upgradePercentage"),
                    "Bucket Trim %": selected_metric.get("trimPercentage"),
                    "Bucket Value": selected_metric.get("value"),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _build_sku_analytics_df(
    skus: list[dict],
    mixes: list[dict],
    metrics_by_mix: dict[str, list[dict]],
    bucket_filter_id: str | None,
) -> pd.DataFrame:
    sku_meta = {str(sku.get("tradeNumber", "")).strip(): sku for sku in skus if str(sku.get("tradeNumber", "")).strip()}
    sku_stats: dict[str, dict] = {}

    for mix in mixes:
        mix_id = str(mix.get("_id", ""))
        mix_metrics = metrics_by_mix.get(mix_id, [])
        if bucket_filter_id:
            mix_metrics = [metric for metric in mix_metrics if str(metric.get("bucketId", "")) == bucket_filter_id]
        if not mix_metrics:
            continue

        mix_score = _mix_score(mix_metrics)
        mix_avg_upgrade = mix_score["avg_upgrade"]
        mix_best_upgrade = mix_score["best_upgrade"]
        mix_avg_trim = mix_score["avg_trim"]

        for sku_trade in mix.get("skus", {}).keys():
            sku_trade = str(sku_trade).strip()
            if not sku_trade:
                continue
            stat = sku_stats.setdefault(
                sku_trade,
                {
                    "SKU": sku_trade,
                    "Customer": sku_meta.get(sku_trade, {}).get("customerName", "—"),
                    "Product Type": sku_meta.get(sku_trade, {}).get("productType", "—"),
                    "Plant": sku_meta.get(sku_trade, {}).get("prodPlant", "—"),
                    "Mixes": set(),
                    "Avg Upgrades": [],
                    "Best Upgrades": [],
                    "Avg Trims": [],
                },
            )
            stat["Mixes"].add(mix_id)
            if mix_avg_upgrade is not None:
                stat["Avg Upgrades"].append(float(mix_avg_upgrade))
            if mix_best_upgrade is not None:
                stat["Best Upgrades"].append(float(mix_best_upgrade))
            if mix_avg_trim is not None:
                stat["Avg Trims"].append(float(mix_avg_trim))

    rows = []
    for sku_trade, stat in sku_stats.items():
        avg_up = mean(stat["Avg Upgrades"]) if stat["Avg Upgrades"] else None
        best_up = max(stat["Best Upgrades"]) if stat["Best Upgrades"] else None
        worst_up = min(stat["Best Upgrades"]) if stat["Best Upgrades"] else None
        avg_trim = mean(stat["Avg Trims"]) if stat["Avg Trims"] else None
        rows.append(
            {
                "SKU": sku_trade,
                "Customer": stat["Customer"],
                "Product Type": stat["Product Type"],
                "Plant": stat["Plant"],
                "Mix Count": len(stat["Mixes"]),
                "Avg Mix Upgrade %": avg_up,
                "Best Mix Upgrade %": best_up,
                "Worst Mix Upgrade %": worst_up,
                "Avg Trim %": avg_trim,
            }
        )

    return pd.DataFrame(rows)


def _render_sku_input_section(skus: list, df_skus: pd.DataFrame, plant_options: list[str]):
    with st.expander("SKU Input Data", expanded=False):
        if not skus:
            st.info("No SKUs returned from Enumeration API. Ensure the API is running and SKUs are loaded.")
            return

        display_cols = [
            "tradeNumber",
            "customerName",
            "customerType",
            "productType",
            "targetWeight",
            "minWeight",
            "maxWeight",
            "allowedParts",
            "birdSize",
            "prodPlant",
        ]
        present_cols = [c for c in display_cols if c in df_skus.columns]

        fc1, fc2, fc3 = st.columns([3, 2, 2])
        query = fc1.text_input("Search trade # or customer", value="")
        plant_filter = fc2.selectbox("Filter by Plant", options=["All"] + plant_options, key="sku_plant_filter")
        bird_opts = sorted(df_skus["birdSize"].dropna().unique().tolist()) if "birdSize" in df_skus.columns else []
        bird_filter = fc3.selectbox("Filter by Bird Size", options=["All"] + bird_opts, key="sku_bird_filter")

        df_view = df_skus.copy()
        if query:
            qlow = query.lower()
            cols_to_search = [c for c in ["tradeNumber", "customerName"] if c in df_view.columns]
            mask = pd.Series([False] * len(df_view), index=df_view.index)
            for col in cols_to_search:
                mask = mask | df_view[col].astype(str).str.lower().str.contains(qlow)
            df_view = df_view[mask]
        if plant_filter != "All" and "prodPlant" in df_view.columns:
            df_view = df_view[df_view["prodPlant"] == plant_filter]
        if bird_filter != "All" and "birdSize" in df_view.columns:
            df_view = df_view[df_view["birdSize"] == bird_filter]

        st.dataframe(df_view[present_cols], use_container_width=True, height=260)
        st.caption(f"{len(df_view)} of {len(df_skus)} SKUs shown")


def _render_results_section(
    all_mixes: list,
    all_metrics: list,
    all_strategies: dict,
    all_buckets: list,
):
    bucket_labels = _bucket_label_map(all_buckets)
    bucket_ids = [str(bucket.get("_id", "")) for bucket in all_buckets if str(bucket.get("_id", ""))]
    metrics_by_mix = _metrics_by_mix(all_metrics)

    res_hdr, res_refresh = st.columns([5, 1])
    with res_hdr:
        st.markdown(
            "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
            "border-left:4px solid #0046AD;padding-left:12px;'>Enumeration Results</h3>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Ranked portioning mixes generated by the engine. Select any row to inspect bucket-level metrics and unit plans."
        )
    with res_refresh:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("↻ Refresh Results", key="enum_refresh_results"):
            _clear_results_cache()
            st.rerun()

    with st.spinner("Loading mixes and metrics..."):
        df_mixes = _build_mixes_df(all_mixes, all_strategies, metrics_by_mix, None, bucket_labels)

    if not all_mixes:
        st.info("No enumeration results available yet. Submit a job below to generate portioning mixes.")
        return df_mixes, metrics_by_mix, None, bucket_labels

    filt0 = st.text_input(
        "Search mixes",
        value="",
        key="res_filt_search",
        placeholder="Search SKUs, plant, bird size, parts, or flags",
    )
    filt1, filt2, filt3, filt4 = st.columns([2, 2, 2, 3])
    filt5, filt6, filt7, filt8 = st.columns([2, 2, 2, 2])
    plant_vals = sorted(df_mixes["Plant"].dropna().unique().tolist()) if not df_mixes.empty else []
    bird_vals = sorted(df_mixes["Bird Size"].dropna().unique().tolist()) if not df_mixes.empty else []
    line_vals = sorted(df_mixes["Line"].dropna().unique().tolist()) if not df_mixes.empty else []
    part_vals = sorted(
        {
            part
            for strategy in all_strategies.values()
            for part in (strategy.get("parts", []) or [])
            if str(part).strip()
        }
    )
    flag_options = ["FDS", "RTL", "NUG"]
    sku_min = int(df_mixes["# SKUs"].min()) if not df_mixes.empty else 1
    sku_max = int(df_mixes["# SKUs"].max()) if not df_mixes.empty else 1
    fillet_min = int(df_mixes["Fillets"].min()) if not df_mixes.empty else 0
    fillet_max = int(df_mixes["Fillets"].max()) if not df_mixes.empty else 0
    belt_min = float(df_mixes["Belt Speed"].min()) if not df_mixes.empty and df_mixes["Belt Speed"].notna().any() else 0.0
    belt_max = float(df_mixes["Belt Speed"].max()) if not df_mixes.empty and df_mixes["Belt Speed"].notna().any() else 0.0

    filt_plant = filt1.selectbox("Plant", ["All"] + plant_vals, key="res_filt_plant")
    filt_bird = filt2.selectbox("Bird Size", ["All"] + bird_vals, key="res_filt_bird")
    filt_line = filt3.selectbox("Line Type", ["All"] + line_vals, key="res_filt_line")
    filt_bucket = filt4.selectbox(
        "Bucket",
        ["All"] + bucket_ids,
        key="res_filt_bucket",
        format_func=lambda x: bucket_labels.get(x, "All") if x != "All" else "All",
    )
    filt_parts = filt5.multiselect("Parts", options=part_vals, key="res_filt_parts")
    filt_flags = filt6.multiselect("Flags", options=flag_options, key="res_filt_flags")
    filt_sku_range = filt7.slider(
        "SKU Count",
        min_value=sku_min,
        max_value=max(sku_min, sku_max),
        value=(sku_min, max(sku_min, sku_max)),
        key="res_filt_sku_range",
    )
    filt_fillet_range = filt8.slider(
        "Fillets",
        min_value=fillet_min,
        max_value=max(fillet_min, fillet_max),
        value=(fillet_min, max(fillet_min, fillet_max)),
        key="res_filt_fillet_range",
    )

    belt_row1, belt_row2 = st.columns([2, 2])
    with belt_row1:
        filt_belt_range = st.slider(
            "Belt Speed",
            min_value=belt_min,
            max_value=max(belt_min, belt_max),
            value=(belt_min, max(belt_min, belt_max)),
            key="res_filt_belt_range",
        )
    with belt_row2:
        st.caption("Use these filters together to narrow the mix list before inspecting details.")

    selected_bucket_id = None if filt_bucket == "All" else filt_bucket
    if selected_bucket_id:
        df_mixes = _build_mixes_df(all_mixes, all_strategies, metrics_by_mix, selected_bucket_id, bucket_labels)

    df_filtered = df_mixes.copy()
    if filt_plant != "All":
        df_filtered = df_filtered[df_filtered["Plant"] == filt_plant]
    if filt_bird != "All":
        df_filtered = df_filtered[df_filtered["Bird Size"] == filt_bird]
    if filt_line != "All":
        df_filtered = df_filtered[df_filtered["Line"] == filt_line]
    if filt_parts:
        df_filtered = df_filtered[df_filtered["Parts"].apply(lambda value: any(part in str(value) for part in filt_parts))]
    if filt_flags:
        df_filtered = df_filtered[df_filtered["Flags"].apply(lambda value: all(flag in str(value) for flag in filt_flags))]
    if filt_sku_range:
        sku_low, sku_high = filt_sku_range
        df_filtered = df_filtered[(df_filtered["# SKUs"] >= sku_low) & (df_filtered["# SKUs"] <= sku_high)]
    if filt_fillet_range:
        fillet_low, fillet_high = filt_fillet_range
        df_filtered = df_filtered[(df_filtered["Fillets"] >= fillet_low) & (df_filtered["Fillets"] <= fillet_high)]
    if filt_belt_range:
        belt_low, belt_high = filt_belt_range
        df_filtered = df_filtered[
            df_filtered["Belt Speed"].fillna(-1).between(belt_low, belt_high, inclusive="both")
        ]
    if filt0.strip():
        q = filt0.strip().lower()
        search_cols = ["SKUs", "Line", "Parts", "Plant", "Bird Size", "Flags"]
        search_frame = df_filtered[search_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        df_filtered = df_filtered[search_frame.str.contains(q, na=False)]

    if selected_bucket_id:
        df_filtered = df_filtered.sort_values("Bucket Upgrade %", ascending=False, na_position="last")
    else:
        df_filtered = df_filtered.sort_values(["# SKUs", "Buckets"], ascending=[False, False], na_position="last")

    st.caption(f"Showing {len(df_filtered)} of {len(df_mixes)} mixes")

    active_filters = []
    if filt0.strip():
        active_filters.append(f"Search: {filt0.strip()}")
    if filt_plant != "All":
        active_filters.append(f"Plant: {filt_plant}")
    if filt_bird != "All":
        active_filters.append(f"Bird Size: {filt_bird}")
    if filt_line != "All":
        active_filters.append(f"Line: {filt_line}")
    if filt_bucket != "All":
        active_filters.append(f"Bucket: {bucket_labels.get(filt_bucket, filt_bucket)}")
    if filt_parts:
        active_filters.append(f"Parts: {', '.join(filt_parts)}")
    if filt_flags:
        active_filters.append(f"Flags: {', '.join(filt_flags)}")
    if filt_sku_range and (filt_sku_range[0] != sku_min or filt_sku_range[1] != sku_max):
        active_filters.append(f"SKU Count: {filt_sku_range[0]}-{filt_sku_range[1]}")
    if filt_fillet_range and (filt_fillet_range[0] != fillet_min or filt_fillet_range[1] != fillet_max):
        active_filters.append(f"Fillets: {filt_fillet_range[0]}-{filt_fillet_range[1]}")
    if filt_belt_range and (filt_belt_range[0] != belt_min or filt_belt_range[1] != belt_max):
        active_filters.append(f"Belt Speed: {filt_belt_range[0]:g}-{filt_belt_range[1]:g}")

    if active_filters:
        chip_html = "".join(
            f"<span style='display:inline-block;background:#e3f0ff;color:#0046AD;"
            f"border:1px solid #b9d4ff;border-radius:999px;padding:4px 10px;margin:0 6px 6px 0;"
            f"font-size:12px;font-weight:600'>{item}</span>"
            for item in active_filters
        )
        st.markdown(
            f"<div style='margin:6px 0 10px 0'><strong>Active Filters</strong><div>{chip_html}</div></div>",
            unsafe_allow_html=True,
        )

    page_size = st.selectbox(
        "Rows per page",
        options=[25, 50, 100, 200],
        index=1,
        key="res_page_size",
        help="Larger pages render more rows at once and can feel slower on large result sets.",
    )
    total_rows = len(df_filtered)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    current_page = int(st.session_state.get("res_page", 1))
    current_page = max(1, min(current_page, total_pages))
    st.session_state["res_page"] = current_page

    page_nav_left, page_nav_mid, page_nav_right = st.columns([1, 2, 1])
    with page_nav_left:
        if st.button("◀ Prev", key="res_page_prev", disabled=current_page <= 1):
            st.session_state["res_page"] = current_page - 1
            st.rerun()
    with page_nav_mid:
        st.caption(f"Page {current_page} of {total_pages} | {total_rows} filtered mixes")
    with page_nav_right:
        if st.button("Next ▶", key="res_page_next", disabled=current_page >= total_pages):
            st.session_state["res_page"] = current_page + 1
            st.rerun()

    start_idx = (current_page - 1) * page_size
    end_idx = start_idx + page_size
    df_page = df_filtered.iloc[start_idx:end_idx].copy()

    display_cols = [
        "SKUs",
        "# SKUs",
        "Line",
        "Parts",
        "Plant",
        "Bird Size",
        "Fillets",
        "Belt Speed",
        "Flags",
        "Buckets",
    ]
    if selected_bucket_id:
        display_cols[display_cols.index("Buckets"):display_cols.index("Buckets")] = [
            "Bucket",
            "Bucket Upgrade %",
            "Bucket Trim %",
            "Bucket Value",
        ]

    df_display = df_page[display_cols].reset_index(drop=True)

    column_config = {
        "Belt Speed": st.column_config.NumberColumn(format="%.1f"),
    }
    if selected_bucket_id:
        column_config.update(
            {
                "Bucket Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Bucket Trim %": st.column_config.NumberColumn(format="%.2f%%"),
                "Bucket Value": st.column_config.NumberColumn(format="%.2f"),
            }
        )

    event = st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        height=340,
        on_select="rerun",
        selection_mode="single-row",
        key="mixes_table",
        column_config=column_config,
    )

    total_mix_count = len(all_mixes)
    total_metric_count = len(all_metrics)

    if selected_bucket_id:
        bucket_metrics = [metric for metric in all_metrics if str(metric.get("bucketId", "")) == selected_bucket_id]
        bucket_label = bucket_labels.get(selected_bucket_id, "[—, —]")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Mixes", total_mix_count)
        k2.metric("Total Metrics", total_metric_count)
        k3.metric("Bucket", bucket_label)

        bucket_upgrades = [m.get("upgradePercentage") for m in bucket_metrics if m.get("upgradePercentage") is not None]
        bucket_trims = [m.get("trimPercentage") for m in bucket_metrics if m.get("trimPercentage") is not None]
        k4.metric("Bucket Upgrade %", f"{mean(bucket_upgrades):.1f}%" if bucket_upgrades else "—")
        k5.metric("Bucket Trim %", f"{mean(bucket_trims):.1f}%" if bucket_trims else "—")
        st.caption("The KPI cards above are scoped to the selected bucket.")
    else:
        k1, k2 = st.columns(2)
        k1.metric("Total Mixes", total_mix_count)
        k2.metric("Total Metrics", total_metric_count)
        st.info("Select a bucket to surface bucket-scoped upgrade and trim metrics.")

    selected_rows = event.selection.rows if event and event.selection else []
    selected_mix_id = df_page.iloc[selected_rows[0]]["_id"] if selected_rows else None

    if selected_mix_id:
        mix_obj = next((mix for mix in all_mixes if str(mix.get("_id", "")) == selected_mix_id), None)
        mix_metrics = metrics_by_mix.get(selected_mix_id, [])
        if selected_bucket_id:
            mix_metrics = [metric for metric in mix_metrics if str(metric.get("bucketId", "")) == selected_bucket_id]
        strategy = all_strategies.get(mix_obj.get("cutStrategyID", ""), {}) if mix_obj else {}

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
                    f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Line Type</td><td>{mix_obj.get('mfgType', '—')}</td></tr>"
                    f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Parts</td><td>{', '.join(strategy.get('parts', []) or ['—'])}</td></tr>"
                    f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Plant</td><td>{mix_obj.get('reqPlant', '—')}</td></tr>"
                    f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Bird Size</td><td>{mix_obj.get('reqBirdSize', '—')}</td></tr>"
                    f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Fillets</td><td>{mix_obj.get('numFillets', '—')} ({mix_obj.get('filletWeight', '—')}g)</td></tr>"
                    f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Belt Speed</td><td>{mix_obj.get('beltSpeed', '—')}</td></tr>"
                    f"<tr><td style='color:#888;padding:2px 8px 2px 0'>Flags</td><td>{' '.join(flags) if flags else '—'}</td></tr>"
                    f"</table></div>",
                    unsafe_allow_html=True,
                )

        with det2:
            st.markdown("**Bucket Performance**")
            if mix_metrics:
                df_mm = pd.DataFrame(mix_metrics).sort_values("upgradePercentage", ascending=False)
                df_mm["Bucket"] = df_mm["bucketId"].map(lambda value: bucket_labels.get(str(value), "[—, —]"))
                show_cols = [
                    col
                    for col in [
                        "Bucket",
                        "upgradePercentage",
                        "trimPercentage",
                        "totalProductProducedGrams",
                        "value",
                    ]
                    if col in df_mm.columns
                ]
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
                upgrades_m = [m.get("upgradePercentage") for m in mix_metrics if m.get("upgradePercentage") is not None]
                trims_m = [m.get("trimPercentage") for m in mix_metrics if m.get("trimPercentage") is not None]
                st.metric("Buckets", len(mix_metrics))
                st.metric("Best Upgrade", f"{max(upgrades_m):.1f}%" if upgrades_m else "—")
                st.metric("Avg Upgrade", f"{mean(upgrades_m):.1f}%" if upgrades_m else "—")
                st.metric("Avg Trim", f"{mean(trims_m):.1f}%" if trims_m else "—")
            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            if st.button("→ Send to Scheduling", type="primary", key="send_to_sched"):
                st.session_state["sched_selected_mix_id"] = selected_mix_id
                try:
                    st.query_params["page"] = "Scheduling Dashboard"
                except Exception:
                    st.session_state.ui_selected_page = "Scheduling Dashboard"
                    st.session_state.ui_sidebar_nav = "Scheduling Dashboard"

        if mix_metrics:
            best_metric = max(mix_metrics, key=lambda item: item.get("upgradePercentage") or 0)
            unit_plan = best_metric.get("unitPlan", [])
            if unit_plan:
                best_bucket_label = bucket_labels.get(str(best_metric.get("bucketId", "")), "[—, —]")
                st.markdown("**Unit Plan - Best Performing Bucket**")
                st.caption(
                    f"Bucket {best_bucket_label} | Upgrade: {best_metric.get('upgradePercentage', 0):.2f}% | Trim: {best_metric.get('trimPercentage', 0):.2f}%"
                )
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

        export_col, _ = st.columns([2, 4])
        with export_col:
            if mix_metrics:
                df_export = pd.DataFrame(mix_metrics).sort_values("upgradePercentage", ascending=False)
                st.download_button(
                    "⬇ Export Metrics CSV",
                    data=df_export.to_csv(index=False).encode(),
                    file_name=f"mix-metrics-{selected_mix_id[:8]}.csv",
                    mime="text/csv",
                )
    else:
        st.caption("Click any row above to inspect its bucket-level metrics and unit plan.")

    return df_mixes, metrics_by_mix, selected_bucket_id, bucket_labels


def _render_analytics_section(df_mixes: pd.DataFrame, selected_bucket_id: str | None, bucket_labels: dict[str, str]):
    st.markdown(
        "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
        "border-left:4px solid #0046AD;padding-left:12px;'>Enumeration Analytics</h3>",
        unsafe_allow_html=True,
    )

    if not df_mixes.empty:
        a1, a2, a3 = st.columns(3)

        with a1:
            st.markdown("**Mix Count by Line Type**")
            st.bar_chart(df_mixes["Line"].value_counts())

        with a2:
            st.markdown("**Mix Count by Plant**")
            st.bar_chart(df_mixes["Plant"].value_counts())

        with a3:
            st.markdown("**Mix Count by SKU Count**")
            st.bar_chart(df_mixes["# SKUs"].value_counts().sort_index())

        if selected_bucket_id:
            st.caption(f"Bucket-scoped KPI cards above are filtered to {bucket_labels.get(selected_bucket_id, '[—, —]')}.")
        else:
            st.caption("Use the bucket filter in Results to surface bucket-scoped upgrade and trim metrics.")
    else:
        st.info("No results available for analytics. Submit a job below to generate data.")


def _render_sku_analytics_section(
    skus: list[dict],
    all_mixes: list[dict],
    metrics_by_mix: dict[str, list[dict]],
    selected_bucket_id: str | None,
    bucket_labels: dict[str, str],
):
    st.markdown(
        "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
        "border-left:4px solid #0046AD;padding-left:12px;'>SKU Analytics</h3>",
        unsafe_allow_html=True,
    )

    scope_text = "all buckets"
    if selected_bucket_id:
        scope_text = f"bucket {bucket_labels.get(selected_bucket_id, '[—, —]')}"
    st.caption(
        "This section highlights SKUs that repeatedly show lower upgrade performance or appear in very few mixes. "
        f"Statistics are calculated across {scope_text}."
    )

    df_sku = _build_sku_analytics_df(skus, all_mixes, metrics_by_mix, selected_bucket_id)
    if df_sku.empty:
        st.info("No SKU analytics available yet.")
        return

    worst_upgrade = df_sku.sort_values(["Avg Mix Upgrade %", "Mix Count"], ascending=[True, True]).head(10)
    fewest_mixes = df_sku.sort_values(["Mix Count", "Avg Mix Upgrade %"], ascending=[True, True]).head(10)
    best_upgrade = df_sku.sort_values(["Avg Mix Upgrade %", "Mix Count"], ascending=[False, False]).head(10)
    most_mixes = df_sku.sort_values(["Mix Count", "Avg Mix Upgrade %"], ascending=[False, False]).head(10)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Lowest Average Upgrade SKUs**")
        st.dataframe(
            worst_upgrade.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=280,
            column_config={
                "Avg Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Best Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Worst Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Avg Trim %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
    with cols[1]:
        st.markdown("**Fewest Mixes SKUs**")
        st.dataframe(
            fewest_mixes.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=280,
            column_config={
                "Avg Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Best Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Worst Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Avg Trim %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    cols2 = st.columns(2)
    with cols2[0]:
        st.markdown("**Highest Average Upgrade SKUs**")
        st.dataframe(
            best_upgrade.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=280,
            column_config={
                "Avg Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Best Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Worst Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Avg Trim %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
    with cols2[1]:
        st.markdown("**Most Mixes SKUs**")
        st.dataframe(
            most_mixes.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=280,
            column_config={
                "Avg Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Best Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Worst Mix Upgrade %": st.column_config.NumberColumn(format="%.2f%%"),
                "Avg Trim %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )


def _render_job_status_section():
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

    if not jobs:
        st.info("No enumeration jobs found. Submit a run below to get started.")
        return

    jobs_sorted = sorted(jobs, key=lambda job: job.get("createdAt", ""), reverse=True)

    rows = []
    for job in jobs_sorted[:10]:
        stages = job.get("stages", [])
        total_combos = sum(stage.get("totalCombinations", 0) for stage in stages if isinstance(stage, dict))
        rows.append(
            {
                "Run ID": job.get("runId", "—"),
                "Status": job.get("status", "—"),
                "SKUs": job.get("skuCount", 0),
                "Plant": job.get("plantFilter") or "all",
                "Bird": job.get("birdSizeFilter") or "all",
                "Max Combo": job.get("maxCombinationSize", "—"),
                "Total Combos": total_combos if total_combos else "—",
                "Submitted": _fmt_dt(job.get("createdAt", "")),
                "Finished": _fmt_dt(job.get("finishedAt", "")),
                "_jobId": job.get("jobId") or job.get("_id", ""),
            }
        )

    df_jobs = pd.DataFrame(rows)
    st.dataframe(df_jobs.drop(columns=["_jobId"]), use_container_width=True, hide_index=True)

    completed = [job for job in jobs_sorted if job.get("status") == "completed"]
    if completed:
        latest = completed[0]
        with st.expander(f"Stage Detail - {latest.get('runId', '—')}", expanded=False):
            stages = latest.get("stages", [])
            if stages:
                sc_cols = st.columns(len(stages))
                for index, stage in enumerate(stages):
                    processed = stage.get("processedCombinations", 0)
                    total = stage.get("totalCombinations", 0)
                    pct = int(100 * processed / total) if total else 0
                    with sc_cols[index]:
                        st.markdown(
                            f"<div class='simmons-card' style='text-align:center'>"
                            f"<div style='font-size:13px;font-weight:600;color:#0046AD'>Stage {stage.get('stage')}</div>"
                            f"<div class='simmons-kpi' style='font-size:20px'>{processed:,}</div>"
                            f"<div class='simmons-kpi-label'>of {total:,} combos</div>"
                            f"<div style='margin-top:6px;font-size:11px;color:#4CAF50'>{pct}% complete</div>"
                            f"<div style='margin-top:2px;font-size:10px;color:#888'>"
                            f"{_fmt_dt(stage.get('startedAt', ''))} -> {_fmt_dt(stage.get('finishedAt', ''))}"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

    running_ids = [row["_jobId"] for row in rows if row["Status"] in ("running", "pending") and row["_jobId"]]
    if running_ids:
        cancel_target = st.selectbox("Cancel a running/pending job", options=["—"] + running_ids)
        if st.button("✕ Cancel Selected Job", type="secondary") and cancel_target != "—":
            try:
                cancel_job(cancel_target)
                st.success(f"Cancellation requested for job `{cancel_target}`.")
            except Exception as exc:
                st.error(str(exc))


def _render_job_submission_section(plant_options: list[str]):
    st.markdown("---")
    st.markdown(
        "<h3 style='color:#0046AD;font-size:20px;font-weight:700;margin:36px 0 18px 0;"
        "border-left:4px solid #0046AD;padding-left:12px;'>Run Enumeration</h3>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Submits an enumeration job to the Worker API. The job evaluates all feasible portioning combinations for the selected plant and bird size, then ranks results by upgrade % and economic value."
    )
    st.warning(
        "Running a new enumeration job will recreate all mixes and mix_metrics records for the selected scope. Submit only when you want to refresh the live enumeration output."
    )

    with st.form("enumeration_run_form"):
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("**Run Identity**")
            default_run_id = f"enum-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
            run_id_input = st.text_input(
                "Run ID (label)",
                value=default_run_id,
                help="Human-readable identifier stored with the job record.",
            )
        with rc2:
            st.markdown("**Data Filters**")
            plant_choice = st.selectbox(
                "Plant Filter",
                options=["(all plants)"] + plant_options,
                help="Limit enumeration to SKUs belonging to this plant.",
            )
            bird_choice = st.selectbox(
                "Bird Size Filter",
                options=["(all sizes)", "SB", "BB"],
                help="SB = Small Bird, BB = Big Bird. Leave blank to include all.",
            )
        with rc3:
            st.markdown("**Engine Settings**")
            max_combo = st.selectbox(
                "Max Combination Size",
                options=[1, 2, 3, 4],
                index=3,
                help="Maximum number of SKUs evaluated per combination. 4 is standard.",
            )
            batch_size_input = st.number_input(
                "Batch Size",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Combinations processed per DB write batch. 1000 is recommended.",
            )

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
            st.success(f"Job submitted - Run ID: **{run_id_input}** &nbsp; Job ID: `{job_id_created}`")
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


def render():
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

    all_mixes = _load_all_mixes()
    all_metrics = _load_all_metrics()
    all_strategies = _load_cut_strategies()
    all_buckets = _load_all_buckets()

    df_mixes, metrics_by_mix, selected_bucket_id, bucket_labels = _render_results_section(
        all_mixes,
        all_metrics,
        all_strategies,
        all_buckets,
    )
    st.markdown("---")
    _render_analytics_section(df_mixes, selected_bucket_id, bucket_labels)
    st.markdown("---")
    _render_sku_input_section(skus, df_skus, plant_options)
    st.markdown("---")
    _render_sku_analytics_section(skus, all_mixes, metrics_by_mix, selected_bucket_id, bucket_labels)
    st.markdown("---")
    _render_job_status_section()
    _render_job_submission_section(plant_options)

    st.caption(
        "Workflow: Review Input Data -> Review Results -> Review Analytics -> Review SKU Analytics -> Review Job Status -> Configure & Submit Enumeration Run"
    )
