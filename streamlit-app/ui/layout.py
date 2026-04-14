import streamlit as st


def render_sidebar(selected: str | None = None) -> str:
    """Render a persistent left sidebar navigation and return selected page key.

    This is lightweight and works with Streamlit pages; it sets a session_state
    key so pages can reflect the current selection.
    """
    if "ui_selected_page" not in st.session_state:
        st.session_state.ui_selected_page = selected or "Enumeration Dashboard"

    st.sidebar.markdown("<div class='simmons-sidebar-logo'><h3 style='color:var(--simmons-blue);margin:0'>Simmons</h3><div style='color:var(--simmons-muted);font-size:12px'>Portioning</div></div>", unsafe_allow_html=True)

    pages = [
        ("Enumeration Dashboard", "🔬"),
        ("Scheduling Dashboard", "🗓️"),
        ("Advanced Settings", "⚙️"),
        ("Snapshot Comparison", "🗂️"),
        ("Exports / Reports", "📤"),
    ]

    for title, icon in pages:
        if st.sidebar.button(f"{icon}  {title}"):
            st.session_state.ui_selected_page = title

    # quick spacer
    st.sidebar.markdown("---")

    return st.session_state.ui_selected_page


def render_header(title: str, subtitle: str | None = None, plant: str | None = None, snapshot: str | None = None, last_run: str | None = None) -> None:
    cols = st.columns([6, 2, 2, 2])
    with cols[0]:
        st.markdown(f"# {title}")
        if subtitle:
            st.markdown(f"<div class='simmons-small'>{subtitle}</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"**Plant**  \n {plant or '—'}")
    with cols[2]:
        st.markdown(f"**Snapshot**  \n {snapshot or '—'}")
    with cols[3]:
        st.markdown(f"**Last Run**  \n {last_run or '—'}")
