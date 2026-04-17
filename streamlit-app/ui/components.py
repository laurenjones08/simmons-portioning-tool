import streamlit as st


def header(title: str, subtitle: str | None = None) -> None:
    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown(f"## {title}")
        if subtitle:
            st.markdown(f"<div class='simmons-small'>{subtitle}</div>", unsafe_allow_html=True)
    with col2:
        st.write("")


def kpi_card(label: str, value: str | int | float, help_text: str | None = None) -> None:
    """Render a small KPI card used on the overview page."""
    st.markdown("<div class='simmons-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='simmons-kpi'>{value}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='simmons-kpi-label'>{label}</div>", unsafe_allow_html=True)
    if help_text:
        st.markdown(f"<div class='simmons-small'>{help_text}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def job_card(job: dict) -> None:
    """Compact card showing job summary."""
    status = job.get("status", "unknown")
    run_id = job.get("runId", "-")
    created = job.get("createdAt", "-")
    st.markdown("<div class='simmons-card'>", unsafe_allow_html=True)
    st.markdown(f"**{run_id}**  ")
    st.markdown(f"Status: **{status}**  ")
    st.markdown(f"Submitted: {created}")
    st.markdown("</div>", unsafe_allow_html=True)
