"""Imports / Exports consolidated page."""
import os
import sys
import io
import csv
import json

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ui.theme import apply_theme
from ui.layout import render_sidebar, render_header
from api_client import search_mixes, search_mix_metrics, batch_import_skus


st.set_page_config(page_title="Imports & Exports", page_icon="📤", layout="wide")
apply_theme()
current = render_sidebar(selected="Exports / Reports")
render_header("Imports / Exports", "Data import and export tools")

st.markdown("---")

with st.expander("Export Mix Snapshots", expanded=True):
    try:
        mixes = search_mixes({})
    except Exception:
        mixes = []

    if mixes:
        df = pd.DataFrame(mixes)
        st.dataframe(df[["_id", "reqPlant", "reqBirdSize"]] if set(["_id","reqPlant","reqBirdSize"]).issubset(df.columns) else df)
        selected = st.selectbox("Select mix to export", options=[m.get("_id") for m in mixes])
        if st.button("Export selected mix as JSON") and selected:
            try:
                # naive export of mix metrics
                metrics = search_mix_metrics({"mixId": selected})
                b = json.dumps(metrics, default=str).encode("utf-8")
                st.download_button("Download JSON", data=b, file_name=f"mix-{selected}.json", mime="application/json")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("No mixes available to export.")

with st.expander("Bulk Import SKUs", expanded=False):
    uploaded = st.file_uploader("Upload CSV or JSON for SKUs", type=["csv","json"])
    if uploaded:
        try:
            content = uploaded.read().decode("utf-8")
            if uploaded.name.lower().endswith('.json'):
                data = json.loads(content)
            else:
                reader = csv.DictReader(io.StringIO(content))
                data = list(reader)
            if st.button("Import SKUs"):
                res = batch_import_skus(data)
                st.success(f"Import result: {res}")
        except Exception as e:
            st.error(str(e))
