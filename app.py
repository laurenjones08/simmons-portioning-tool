from __future__ import annotations

import streamlit as st
import pandas as pd

from portioning.io import load_uploaded, list_excel_sheets
from portioning.ui import sidebar_controls
from portioning.engines.base import EngineInput
from portioning.engines.enumeration_engine import EnumerationEngine
from portioning.engines.two_stage_engine import TwoStageEngine
from portioning.transforms.normalize import normalize_results
from portioning.transforms.ranking import rank_results


st.set_page_config(page_title="Portioning Model", layout="wide")

st.title("Portioning Model")
st.caption("Interactive enumeration + Two-stage monthly optimizer (Bulk + Cleanup).")

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
loaded = load_uploaded(uploaded, sheet_name=(ui.sheet_name if excel_sheets else None))
df_in = loaded.df

with st.expander("Preview input data", expanded=False):
    st.write(f"File: **{loaded.filename}**" + (f" | Sheet: **{loaded.sheet_name}**" if loaded.sheet_name else ""))
    st.dataframe(df_in.head(50), use_container_width=True)

# Choose engine
engine = EnumerationEngine() if ui.engine == "enumeration" else TwoStageEngine()

run = st.button("Run model", type="primary")

if not run:
    st.stop()

with st.spinner("Running..."):
    inp = EngineInput(
        df=df_in,
        engine_name=ui.engine,
        trim_cap=ui.trim_cap,
        bucket=ui.bucket,
        bird_size=ui.bird_size,
        min_nuggets=ui.min_nuggets,
        customer_constraint=ui.customer_constraint,
        plant=ui.plant,
        sheet_name=ui.sheet_name,
        use_month_config=ui.use_month_config,
        time_limit_sec=ui.time_limit_sec,
        gap=ui.gap,
        chunk_size=ui.chunk_size,
        pieces_per_min=ui.pieces_per_min,
        line_eff=ui.line_eff,
    )

    res = engine.run(inp)

# Show warnings/meta
for w in res.warnings:
    st.warning(w)

meta_cols = st.columns(3)
meta_cols[0].metric("Engine", ui.engine)
meta_cols[1].metric("Trim cap", f"{ui.trim_cap:.0f}%")
meta_cols[2].metric("Rows", len(res.results_df) if res.results_df is not None else 0)

if res.meta:
    with st.expander("Run metadata", expanded=False):
        st.json(res.meta)

# Normalize + rank for UI
norm = normalize_results(res.results_df)
ranked = rank_results(norm)

st.subheader("Ranked Results")
st.dataframe(ranked, use_container_width=True)

csv_bytes = ranked.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results as CSV",
    data=csv_bytes,
    file_name="portioning_results.csv",
    mime="text/csv",
)
