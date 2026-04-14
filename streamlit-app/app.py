import streamlit as st
import importlib

st.set_page_config(page_title="Portioning Tool", layout="wide")
from ui.theme import apply_theme
from ui.layout import render_header, render_sidebar


apply_theme()

# render the sidebar first so it can set session state via its widget
try:
    render_sidebar()
except Exception:
    pass

# Determine the active page: prefer the top-nav widget, then session_state, then query param, else default to Home
params = {}
try:
    params = st.experimental_get_query_params()
except Exception:
    params = {}

# priority: sidebar widget, session_state, query param
page = None
if "ui_sidebar_nav" in st.session_state and st.session_state.ui_sidebar_nav:
    page = st.session_state.ui_sidebar_nav
elif "ui_selected_page" in st.session_state and st.session_state.ui_selected_page:
    page = st.session_state.ui_selected_page
else:
    page = params.get("page", ["Home"])[0]

if not page:
    page = "Home"

# synchronize both session keys and query params to the chosen page
try:
    st.session_state.ui_selected_page = page
except Exception:
    pass
try:
    st.session_state.ui_sidebar_nav = page
except Exception:
    pass
try:
    st.experimental_set_query_params(page=page)
except Exception:
    pass

# mapping of page display names to view modules and subtitles
page_map = {
    "Home": ("views.home", "Simmons Portioning Decision Support — Overview"),
    "Enumeration Dashboard": ("views.enumeration_dashboard", "Portioning Decision Support System"),
    "Scheduling Dashboard": ("views.scheduling_dashboard", "Scheduling Decision Support"),
    "Advanced Settings": ("views.advanced_settings", "Configuration and master data"),
    "Snapshot Comparison": ("views.snapshot_comparison", "Compare two enumeration snapshots side-by-side"),
    "Exports / Reports": ("views.imports_exports", "Data import and export tools"),
}

subtitle = page_map.get(page, (None, ""))[1] if page in page_map else ""

render_header(page, subtitle)

# dynamically import and render the selected view
module_path = page_map.get(page, (None, ""))[0]
if module_path:
    try:
        mod = importlib.import_module(module_path)
        if hasattr(mod, "render"):
            mod.render()
        else:
            st.error(f"View {module_path} has no render() function.")
    except Exception as e:
        st.exception(e)
else:
    st.info("Select a page from the top navigation.")
