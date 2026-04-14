"""Enumeration Dashboard view for single-page router."""
import streamlit as st
import pandas as pd
from api_client import search_skus, search_buckets, search_mixes, search_mix_metrics
import time


def render():
    # Current SKU input data
    try:
        skus = search_skus({})
    except Exception:
        skus = []

    st.subheader("Current SKU Input Data")
    with st.expander("SKU Data Viewer", expanded=True):
        if not skus:
            st.info("No SKUs returned from Enumeration API.")
        else:
            df_skus = pd.DataFrame(skus)
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

            q = st.text_input("Search SKU or Customer", value="")
            plant_options = sorted(df_skus["prodPlant"].dropna().unique().tolist()) if "prodPlant" in df_skus.columns else []
            plant_filter = st.selectbox("Plant", options=[""] + plant_options)

            df_view = df_skus.copy()
            if q:
                qlow = q.lower()
                cols_to_search = [c for c in ["tradeNumber", "customerName"] if c in df_view.columns]
                mask = False
                for c in cols_to_search:
                    mask = mask | df_view[c].astype(str).str.lower().str.contains(qlow)
                df_view = df_view[mask]
            if plant_filter:
                df_view = df_view[df_view.get("prodPlant") == plant_filter]

            st.dataframe(df_view[present_cols], width="stretch", height=300)

    st.markdown("---")

    st.subheader("User Parameter Controls")
    cols = st.columns(3)

    with cols[0]:
        st.markdown("**Production Inputs**")
        try:
            buckets = search_buckets({})
            bucket_labels = [f"{b.get('minWeight')}–{b.get('maxWeight')}" for b in buckets]
        except Exception:
            buckets = []
            bucket_labels = []
        bucket_choice = st.selectbox("Bucket Size", options=[""] + bucket_labels)
        bird_choice = st.selectbox("Bird Size", options=["", "SB", "BB", "ALL"])
        plant_choice = st.selectbox("Plant", options=[""] + (plant_options if plant_options else [""]))

    with cols[1]:
        st.markdown("**Business Constraints**")
        min_fds = st.checkbox("Minimum Food Service (FDS)")
        nugget_method = st.selectbox("Nugget Handling", options=["auto", "manual", "ignore"])
        trim_threshold = st.slider("Trim Threshold (%)", min_value=0, max_value=100, value=10)
        value_weighting = st.select_slider("Value Weighting", options=["low", "medium", "high"], value="medium")

    with cols[2]:
        st.markdown("**Model Controls**")
        preview_mode = st.radio("Run Mode", options=["Preview", "Full"], horizontal=True)
        snapshot_name = st.text_input("Snapshot Name (optional)")
        run_button = st.button("Run Enumeration", type="primary")

    st.markdown("---")

    st.subheader("Ranked Portioning Decisions")
    results_col, analytics_col = st.columns([3, 1])

    with results_col:
        try:
            mixes = search_mixes({})
        except Exception:
            mixes = []

        if not mixes:
            st.info("No enumeration snapshots available.")
        else:
            mix_map = {m.get("_id"): m for m in mixes}
            selected_mix = st.selectbox("Select Snapshot", options=[""] + list(mix_map.keys()))
            if selected_mix:
                mix = mix_map[selected_mix]
                st.markdown(f"**Snapshot:** {selected_mix} — Plant: {mix.get('reqPlant')} — BirdSize: {mix.get('reqBirdSize')}")
                try:
                    metrics = search_mix_metrics({"mixId": selected_mix})
                except Exception:
                    metrics = []

                if metrics:
                    df_metrics = pd.DataFrame(metrics)
                    display_cols = [c for c in ["bucketId", "upgradePercentage", "value", "trimPercentage", "totalProductProducedGrams"] if c in df_metrics.columns]
                    st.dataframe(df_metrics[display_cols], width="stretch", height=360)
                else:
                    st.info("No metrics found for this snapshot.")

    with analytics_col:
        st.markdown("**KPIs**")
        kpis = {"candidate_count": len(mixes) if mixes else 0, "avg_upgrade": None}
        if mixes:
            try:
                first = mixes[0]
                m = search_mix_metrics({"mixId": first.get("_id")})
                ups = [x.get("upgradePercentage") for x in m if x.get("upgradePercentage") is not None]
                kpis["avg_upgrade"] = sum(ups) / len(ups) if ups else None
            except Exception:
                kpis["avg_upgrade"] = None

        st.markdown(f"**Candidate Mixes:** {kpis.get('candidate_count',0)}")
        st.markdown(f"**Avg Upgrade % (sample):** {f\"{kpis['avg_upgrade']:.2f}%\" if kpis.get('avg_upgrade') is not None else 'n/a'}")

    st.markdown("---")

    st.subheader("Enumeration Analytics")
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        st.markdown("**Upgrade Distribution**")
        if mixes:
            try:
                fm = search_mix_metrics({"mixId": mixes[0].get("_id")})
                upgrades = [m.get("upgradePercentage") for m in fm if m.get("upgradePercentage") is not None]
                if upgrades:
                    st.bar_chart(pd.Series(upgrades))
                else:
                    st.info("No upgrade data to chart.")
            except Exception:
                st.info("No data for chart.")
        else:
            st.info("No snapshots to analyze.")

    with chart_col2:
        st.markdown("**Value vs Trim (scatter)**")
        st.info("Interactive scatterplot placeholder — implement with Altair/Plotly")

    with chart_col3:
        st.markdown("**Candidate Count by Cut Strategy**")
        st.info("Bar chart placeholder")

    st.markdown("---")
    st.caption("Workflow: Review Input Data → Select Constraints → Run Enumeration → Review Ranked Decisions → Push to Scheduling")
