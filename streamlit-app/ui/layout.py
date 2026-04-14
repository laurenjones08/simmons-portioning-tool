import streamlit as st
import os


def render_sidebar(selected: str | None = None) -> str:
    """Render a persistent left sidebar navigation and return selected page key.

    This is lightweight and works with Streamlit pages; it sets a session_state
    key so pages can reflect the current selection.
    """
    # Render a Simmons-branded sidebar navigation and keep session_state in sync.
    nav_links = [
        ("Home", "Home"),
        ("Enumeration Dashboard", "Enumeration Dashboard"),
        ("Scheduling Dashboard", "Scheduling Dashboard"),
        ("Advanced Settings", "Advanced Settings"),
        ("Snapshot Comparison", "Snapshot Comparison"),
        ("Exports / Reports", "Exports / Reports"),
    ]

    options = [n for (_, n) in nav_links]

    # initialize session keys
    if "ui_selected_page" not in st.session_state:
        st.session_state.ui_selected_page = selected or "Home"
    if "ui_sidebar_nav" not in st.session_state:
        st.session_state.ui_sidebar_nav = st.session_state.ui_selected_page

    # keep query params in sync if present
    try:
        params = st.experimental_get_query_params()
        if "page" in params and params["page"]:
            st.session_state.ui_selected_page = params["page"][0]
            st.session_state.ui_sidebar_nav = st.session_state.ui_selected_page
    except Exception:
        pass

    def _on_sidebar_change():
        sel = st.session_state.get("ui_sidebar_nav")
        try:
            st.session_state.ui_selected_page = sel
        except Exception:
            pass
        try:
            st.experimental_set_query_params(page=sel)
        except Exception:
            pass
        # No explicit rerun needed; widget change will trigger a rerun automatically

    # Sidebar logo + nav
    with st.sidebar:
        # Prefer a provided logo image if present in common asset locations
        logo_paths = [
            os.path.join("streamlit-app", "static", "simmons_logo.png"),
            os.path.join("streamlit-app", "assets", "simmons_logo.png"),
            os.path.join("streamlit-app", "simmons_logo.png"),
            os.path.join("assets", "simmons_logo.png"),
        ]
        # also support a directory named streamlit-app/simmons_logo with arbitrary image files
        logo_dir = os.path.join("streamlit-app", "simmons_logo")
        logo_rendered = False
        # try explicit paths first
        for p in logo_paths:
            if os.path.exists(p):
                try:
                    st.image(p, width=300)
                    logo_rendered = True
                    break
                except Exception:
                    logo_rendered = False
        # then try any file inside streamlit-app/simmons_logo/*
        if not logo_rendered and os.path.isdir(logo_dir):
            try:
                for fname in os.listdir(logo_dir):
                    fpath = os.path.join(logo_dir, fname)
                    if os.path.isfile(fpath):
                        # basic image extension check
                        if any(fname.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                            try:
                                st.image(fpath, width=300)
                                logo_rendered = True
                                break
                            except Exception:
                                continue
            except Exception:
                logo_rendered = False
        if not logo_rendered:
            st.markdown("<div class='simmons-sidebar-logo'><h2 style='margin:0;color:white'>Simmons</h2></div>", unsafe_allow_html=True)
        # Use a single radio widget for reliable single-click navigation
        try:
            current_idx = options.index(st.session_state.ui_sidebar_nav) if st.session_state.ui_sidebar_nav in options else 0
        except Exception:
            current_idx = 0

        st.radio("", options=options, index=current_idx, key="ui_sidebar_nav", on_change=_on_sidebar_change)

    return st.session_state.ui_selected_page


def render_header(title: str, subtitle: str | None = None, plant: str | None = None, snapshot: str | None = None, last_run: str | None = None) -> None:
    # Top banner with site name and simple navigation links
    nav_links = [
        ("Home", "Home"),
        ("Enumeration Dashboard", "Enumeration Dashboard"),
        ("Scheduling Dashboard", "Scheduling Dashboard"),
        ("Advanced Settings", "Advanced Settings"),
        ("Snapshot Comparison", "Snapshot Comparison"),
        ("Exports / Reports", "Exports / Reports"),
    ]
    # Banner with hoverable tabs and dropdown sections
    # Build HTML for nav tabs with submenus (sections)
    # Only Advanced Settings exposes dropdown sections; other tabs navigate directly
    page_sections = {
        "Home": [],
        "Enumeration Dashboard": [],
        "Scheduling Dashboard": [],
        "Advanced Settings": ["Buckets", "SKUs", "Cut Strategies", "Lines", "Config"],
        "Snapshot Comparison": [],
        "Exports / Reports": [],
    }

    # Render top banner with title and subtitle inside the blue panel
    subtitle_html = f"<div class='simmons-small' style='margin-top:6px'>{subtitle}</div>" if subtitle else ""
    # left side contains title + subtitle; right side can show metadata if provided
    meta_html = ""
    if plant or snapshot or last_run:
        meta_items = []
        if plant:
            meta_items.append(f"<div><strong>Plant</strong><div style='font-size:12px'>{plant}</div></div>")
        if snapshot:
            meta_items.append(f"<div><strong>Snapshot</strong><div style='font-size:12px'>{snapshot}</div></div>")
        if last_run:
            meta_items.append(f"<div><strong>Last Run</strong><div style='font-size:12px'>{last_run}</div></div>")
        meta_html = "<div class='banner-meta'>" + "".join(meta_items) + "</div>"

    st.markdown(
        f"<div class='simmons-top-banner'><div class='banner-inner'><div class='banner-left'><h1 style='margin:0'>{title}</h1>{subtitle_html}</div>{meta_html}</div></div>",
        unsafe_allow_html=True,
    )
