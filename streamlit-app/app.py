import streamlit as st

st.set_page_config(
    page_title="Portioning Tool",
    layout="wide",
)

st.title("Portioning Tool")
st.markdown(
    """
Welcome to the **Portioning Tool** management interface.

Use the sidebar to navigate between pages:

- **Buckets** — Manage weight bucket definitions used for enumeration bucketing.
- **SKUs** — Create, search, update, and delete Stock Keeping Units.
- **Cut Strategies** — Manage manufacturing cut configurations.
- **Mix Visualization** — Browse and filter enumeration mix results.
- **Mix Generation** — Submit and monitor enumeration jobs.
- **Global Config** — View and edit system configuration parameters.
"""
)
