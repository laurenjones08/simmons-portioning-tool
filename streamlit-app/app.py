import streamlit as st
import importlib

st.set_page_config(page_title="Portioning Tool", layout="wide")
from ui.theme import apply_theme
from ui.layout import render_header

apply_theme()

# ── Determine active page from URL query params ────────────────────────────
try:
    page = st.query_params.get("page") or "Home"
except Exception:
    try:
        params = st.experimental_get_query_params()
        page = (params.get("page") or ["Home"])[0] or "Home"
    except Exception:
        page = "Home"

# Sync session state so views that reference it still work
st.session_state.ui_selected_page = page
st.session_state.ui_sidebar_nav = page

# ── Page registry ──────────────────────────────────────────────────────────
page_map = {
    "Home":                  ("views.home",                  "Simmons Portioning Decision Support — Overview"),
    "Enumeration Dashboard": ("views.enumeration_dashboard", "Portioning Decision Support System"),
    "Scheduling Dashboard":  ("views.scheduling_dashboard",  "Scheduling control center"),
    "Scheduling Create":     ("views.scheduling_create",     "Create a scheduling run"),
    "Scheduling Insights":   ("views.scheduling_insights",   "Scheduling analytics and tables"),
    "Advanced Settings":     ("views.advanced_settings",     "Configuration and master data"),
    "Snapshot Comparison":   ("views.snapshot_comparison",   "Compare two enumeration snapshots side-by-side"),
    "Exports / Reports":     ("views.imports_exports",       "Data import and export tools"),
}

subtitle = page_map.get(page, (None, ""))[1]
render_header(page, subtitle)

# ── Dynamically import and render the active view ─────────────────────────
module_path = page_map.get(page, (None, ""))[0]
if module_path:
    try:
        mod = importlib.import_module(module_path)
        if hasattr(mod, "render"):
            mod.render()
        else:
            st.error(f"View module {module_path!r} has no render() function.")
    except Exception as exc:
        st.exception(exc)
else:
    st.info("Select a page from the navigation bar.")
