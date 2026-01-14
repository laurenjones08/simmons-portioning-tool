import streamlit as st
import pandas as pd

st.set_page_config(page_title="Mini Portioning Model", layout="wide")

st.title("Mini Portioning Model (Streamlit Test)")
st.caption("This is a simple prototype to validate the UI + pairing logic.")

# --- Sidebar inputs (mimic your real controls) ---
st.sidebar.header("Inputs")
bucket = st.sidebar.selectbox("Bucket (WIP range)", ["(390, 480)", "(476, 550)", "(551, 625)"])
bird_size = st.sidebar.radio("Bird size", ["SB", "BB", "ALL"], horizontal=True)
trim_cap = st.sidebar.slider("Trim % allowed", min_value=0, max_value=40, value=15, step=1)
customer_constraint = st.sidebar.selectbox(
    "Customer constraint",
    ["NONE", "Must include RTL", "Must include FDS"]
)

st.divider()

# --- Mini sample data (replace later with your CSV load) ---
data = [
    {"SKU_IDs": "17191, 20918, 39771", "CustomerTypes": "FDS, FDS, RTL", "TargetSum_g": 511, "Upgrade_%": 99.61, "Trim_%": 0.39},
    {"SKU_IDs": "17732, 38130, 39771", "CustomerTypes": "FDS, RTL, RTL", "TargetSum_g": 510, "Upgrade_%": 99.42, "Trim_%": 0.58},
    {"SKU_IDs": "42340, 20918, 39771", "CustomerTypes": "RTL, FDS, RTL", "TargetSum_g": 509, "Upgrade_%": 99.22, "Trim_%": 0.78},
    {"SKU_IDs": "55100, 24227",        "CustomerTypes": "FDS, FDS",      "TargetSum_g": 410, "Upgrade_%": 80.00, "Trim_%": 20.00},
]
df = pd.DataFrame(data)

# --- Example filtering logic to simulate constraints ---
if customer_constraint == "Must include RTL":
    df = df[df["CustomerTypes"].str.contains("RTL")]
elif customer_constraint == "Must include FDS":
    df = df[df["CustomerTypes"].str.contains("FDS")]

df = df[df["Trim_%"] <= trim_cap]

# --- Results header ---
col1, col2, col3 = st.columns(3)
col1.metric("Selected bucket", bucket)
col2.metric("Bird size", bird_size)
col3.metric("Valid combinations", len(df))

# --- Results table ---
st.subheader("Ranked Results (demo)")
df_sorted = df.sort_values(["Upgrade_%", "Trim_%"], ascending=[False, True]).reset_index(drop=True)
df_sorted.insert(0, "Rank", range(1, len(df_sorted) + 1))
st.dataframe(df_sorted, use_container_width=True)

# --- Download button ---
csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results as CSV",
    data=csv_bytes,
    file_name="mini_model_results.csv",
    mime="text/csv",
)

st.info("Next step: replace the sample dataframe with your real optimizer output (pandas dataframe).")
