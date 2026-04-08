from __future__ import annotations

import streamlit as st

from old.portioning import load_uploaded, list_excel_sheets
from old.portioning import sidebar_controls
from old.portioning.engines.base import EngineInput
from old.portioning.engines.enumeration_engine import EnumerationEngine
from old.portioning.transforms.normalize import normalize_results
from old.portioning import rank_results


st.set_page_config(page_title="Portioning Model", layout="wide")

st.title("Portioning Model")
st.caption("Interactive enumeration for portioning optimization.")

uploaded = st.file_uploader("Upload input file (CSV or Excel)", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.info("Upload a CSV/XLSX to begin.")
    st.stop()

excel_sheets = list_excel_sheets(uploaded)
sheet_for_preview = excel_sheets[0] if excel_sheets else None

# Peek at plants for enumeration dropdown if possible
plants = None
try:
    preview = load_uploaded(uploaded, sheet_name=sheet_for_preview).df
    if "ProdPlant" in preview.columns:
        plants = sorted(preview["ProdPlant"].dropna().astype(str).str.upper().str.strip().unique().tolist())
except Exception:
    plants = None

ui = sidebar_controls(plants=plants, excel_sheets=excel_sheets)

# Load the selected sheet / data
loaded = load_uploaded(uploaded, sheet_name=None)
df_in = loaded.df

with st.expander("Preview input data", expanded=False):
    st.write(f"File: **{loaded.filename}**")
    st.dataframe(df_in.head(50), width="stretch")

# Use enumeration engine
engine = EnumerationEngine()

run = st.button("Run model", type="primary")

if not run:
    st.stop()

with st.spinner("Running..."):
    inp = EngineInput(
        df=df_in,
        trim_cap=ui.trim_cap,
        bucket=ui.bucket,
        bird_size=ui.bird_size,
        min_nuggets=ui.min_nuggets,
        customer_constraint=ui.customer_constraint,
        plant=ui.plant
    )

    res = engine.run(inp)

# Show warnings/meta
for w in res.warnings:
    st.warning(w)

meta_cols = st.columns(2)
meta_cols[0].metric("Trim cap", f"{ui.trim_cap:.0f}%")
meta_cols[1].metric("Rows", len(res.results_df) if res.results_df is not None else 0)

if res.meta:
    with st.expander("Run metadata", expanded=False):
        st.json(res.meta)

# Normalize + rank for UI
norm = normalize_results(res.results_df)
ranked = rank_results(norm)

st.subheader("Ranked Results")
st.dataframe(ranked, width="stretch")

csv_bytes = ranked.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results as CSV",
    data=csv_bytes,
    file_name="portioning_results.csv",
    mime="text/csv",
)
