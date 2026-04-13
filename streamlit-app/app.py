import streamlit as st

st.set_page_config(
    page_title="Portioning Tool",
    layout="wide",
)
from ui.theme import apply_theme
from ui.components import header


apply_theme()

st.title("Portioning Tool")
header("Portioning Tool", "Management interface for enumeration and scheduling")

st.markdown(
    """
Welcome to the **Portioning Tool** management interface.

Use the sidebar to navigate between pages:

- **Overview** — Operational snapshot and quick actions.
- **Buckets** — Manage weight bucket definitions used for enumeration bucketing.
- **SKUs** — Create, search, update, and delete Stock Keeping Units.
- **Cut Strategies** — Manage manufacturing cut configurations.
- **Mix Visualization** — Browse and filter enumeration mix results.
- **Mix Generation** — Submit and monitor enumeration jobs.
- **Scheduling Visualizer** — Preview and commit scheduling runs.
- **Global Config** — View and edit system configuration parameters.
"""
)
