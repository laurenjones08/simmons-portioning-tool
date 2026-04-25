"""Streamlit UI layout with a sticky top navbar."""

import base64
import os
from functools import lru_cache

import streamlit as st

from ui.theme import init_theme_mode


_NAV_ITEMS = [
    ("Home", "Home"),
    ("Enumeration", "Enumeration Dashboard"),
    ("Scheduling", "Scheduling Dashboard"),
    ("Settings", "Advanced Settings"),
]


@lru_cache(maxsize=1)
def _get_logo_data_uri() -> str | None:
    """Load and base64-encode the Simmons logo. Cached after first call."""
    ui_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(ui_dir)
    candidates = [
        os.path.join(app_dir, "simmons_logo.png"),
        os.path.join(app_dir, "static", "simmons_logo.png"),
        os.path.join(app_dir, "assets", "simmons_logo.png"),
    ]
    logo_dir = os.path.join(app_dir, "simmons_logo")
    if os.path.isdir(logo_dir):
        for fname in sorted(os.listdir(logo_dir)):
            fpath = os.path.join(logo_dir, fname)
            if os.path.isfile(fpath) and any(
                fname.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")
            ):
                candidates.append(fpath)
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "rb") as file_handle:
                    data = base64.b64encode(file_handle.read()).decode()
                ext = path.rsplit(".", 1)[-1].lower()
                mime = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "gif": "image/gif",
                    "webp": "image/webp",
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
    """Render the sticky top navbar using real Streamlit buttons and logo."""
    init_theme_mode()
    logo_uri = _get_logo_data_uri()
    logo_img_html = (
        f'<div class="sfy-logo-wrap"><img src="{logo_uri}" alt="Simmons Prepared Foods" class="sfy-logo-img"/></div>'
        if logo_uri
        else '<div class="sfy-logo-wrap"><span class="sfy-logo-text">SIMMONS</span></div>'
    )

    with st.container():
        st.markdown('<div class="sfy-navbar-shell"></div>', unsafe_allow_html=True)

        logo_col, *nav_cols = st.columns([2.4] + [1] * len(_NAV_ITEMS))

        with logo_col:
            st.markdown(logo_img_html, unsafe_allow_html=True)

        for index, (label, page_key) in enumerate(_NAV_ITEMS):
            with nav_cols[index]:
                is_active = title == page_key
                if st.button(
                    label,
                    key=f"_nav_{page_key}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    try:
                        st.query_params["page"] = page_key
                    except Exception:
                        st.session_state.ui_selected_page = page_key
                        st.session_state.ui_sidebar_nav = page_key
                    st.rerun()


def render_page_header(title: str = "Home", subtitle: str | None = None) -> None:
    """Render a page title row below the sticky navbar."""
    if not title or title == "Home":
        return

    sub_text = f"<p>{subtitle}</p>" if subtitle else ""
    header_col = st.columns([1])[0]
    with header_col:
        st.markdown(
            f'<div class="sfy-page-header-shell"><div class="sfy-page-header"><h2>{title}</h2>{sub_text}</div></div>',
            unsafe_allow_html=True,
        )
