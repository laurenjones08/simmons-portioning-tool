"""Streamlit UI layout — fixed top navbar with Simmons branding."""
import base64
import os
from functools import lru_cache
from urllib.parse import quote

import streamlit as st


# Navigation items: (display label, full page key matching page_map in app.py)
_NAV_ITEMS = [
    ("Home",         "Home"),
    ("Enumeration",  "Enumeration Dashboard"),
    ("Scheduling",   "Scheduling Dashboard"),
    ("Settings",     "Advanced Settings"),
    ("Compare",      "Snapshot Comparison"),
    ("Reports",      "Exports / Reports"),
]


@lru_cache(maxsize=1)
def _get_logo_data_uri() -> str | None:
    """Load and base64-encode the Simmons logo. Cached after first call."""
    _ui_dir = os.path.dirname(os.path.abspath(__file__))
    _app_dir = os.path.dirname(_ui_dir)  # streamlit-app/
    candidates = [
        os.path.join(_app_dir, "simmons_logo.png"),
        os.path.join(_app_dir, "static", "simmons_logo.png"),
        os.path.join(_app_dir, "assets", "simmons_logo.png"),
    ]
    logo_dir = os.path.join(_app_dir, "simmons_logo")
    if os.path.isdir(logo_dir):
        for fname in sorted(os.listdir(logo_dir)):
            fpath = os.path.join(logo_dir, fname)
            if os.path.isfile(fpath) and any(
                fname.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")
            ):
                candidates.append(fpath)
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                ext = p.rsplit(".", 1)[-1].lower()
                mime = {
                    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "gif": "image/gif", "webp": "image/webp",
                }.get(ext, "image/png")
                return f"data:{mime};base64,{data}"
            except Exception:
                continue
    return None


def render_sidebar(selected: str | None = None) -> str:
    """No-op: sidebar replaced by top navbar. Ensures session state keys exist."""
    if "ui_selected_page" not in st.session_state:
        st.session_state.ui_selected_page = selected or "Home"
    if "ui_sidebar_nav" not in st.session_state:
        st.session_state.ui_sidebar_nav = st.session_state.ui_selected_page
    return st.session_state.ui_selected_page


def render_header(title: str = "Home", subtitle: str | None = None, **_kwargs) -> None:
    """Render the fixed top navbar and (for non-Home pages) a page header strip."""
    logo_uri = _get_logo_data_uri()
    logo_html = (
        f'<img src="{logo_uri}" class="sfy-nav-logo" alt="Simmons Prepared Foods" />'
        if logo_uri
        else '<span class="sfy-nav-wordmark">SIMMONS</span>'
    )

    links_html = "\n".join(
        f'<a href="#" onclick="window.parent.location.href=\'?page={quote(page_key)}\'; return false;" '
        f'class="sfy-nav-link{"  sfy-nav-active" if title == page_key else ""}">{label}</a>'
        for label, page_key in _NAV_ITEMS
    )

    page_header_html = ""
    if title and title != "Home":
        sub_text = f"<p>{subtitle}</p>" if subtitle else ""
        page_header_html = (
            f'<div class="sfy-page-header"><h2>{title}</h2>{sub_text}</div>'
        )

    st.markdown(
        f"""
        <div class="sfy-navbar">
          <div class="sfy-navbar-inner">
            <div class="sfy-nav-brand">{logo_html}</div>
            <nav class="sfy-nav-links">{links_html}</nav>
          </div>
        </div>
        {page_header_html}
        """,
        unsafe_allow_html=True,
    )