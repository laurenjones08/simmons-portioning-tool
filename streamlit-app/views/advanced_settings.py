"""Advanced Settings view for single-page router."""
import streamlit as st
import pandas as pd
from api_client import (
    search_buckets,
    create_bucket,
    search_skus,
    batch_import_skus,
    search_cut_strategies,
    list_lines,
    get_all_configs,
)


def render():
    st.markdown("---")
    with st.expander("Buckets", expanded=False):
        try:
            buckets = search_buckets({})
        except Exception:
            buckets = []
        if buckets:
            st.dataframe(pd.DataFrame(buckets))
        else:
            st.info("No buckets found.")
        with st.form("create_bucket_form"):
            min_w = st.number_input("minWeight", value=0.0)
            target_w = st.number_input("targetWeight", value=0.5)
            max_w = st.number_input("maxWeight", value=1.0)
            submit = st.form_submit_button("Create Bucket")
            if submit:
                try:
                    create_bucket({"minWeight": min_w, "targetWeight": target_w, "maxWeight": max_w})
                    st.success("Bucket created — refresh the page.")
                except Exception as e:
                    st.error(str(e))

    with st.expander("SKUs", expanded=False):
        try:
            skus = search_skus({})
        except Exception:
            skus = []
        if skus:
            st.dataframe(pd.DataFrame(skus))
        else:
            st.info("No SKUs found.")
        st.markdown("**Bulk Import SKUs**")
        uploaded = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
        if uploaded is not None:
            try:
                import json, io, csv
                content = uploaded.read().decode("utf-8")
                if uploaded.name.lower().endswith('.json'):
                    data = json.loads(content)
                else:
                    reader = csv.DictReader(io.StringIO(content))
                    data = list(reader)
                result = batch_import_skus(data)
                st.success(f"Imported: {result}")
            except Exception as e:
                st.error(str(e))

    with st.expander("Cut Strategies", expanded=False):
        try:
            strategies = search_cut_strategies({})
        except Exception:
            strategies = []
        if strategies:
            st.dataframe(pd.DataFrame(strategies))
        else:
            st.info("No cut strategies found.")

    with st.expander("Lines", expanded=False):
        try:
            lines = list_lines()
        except Exception:
            lines = []
        if lines:
            st.dataframe(pd.DataFrame(lines))
        else:
            st.info("No lines configured.")

    with st.expander("Global Config", expanded=False):
        try:
            configs = get_all_configs()
        except Exception:
            configs = []
        if configs:
            st.dataframe(pd.DataFrame(configs))
        else:
            st.info("No configs available.")

    st.markdown("---")
    st.caption("Changes here affect the core services. Use caution and confirm before saving high-impact settings.")
