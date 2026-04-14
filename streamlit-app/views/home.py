"""Home view for single-page router.

This implements the executive overview / landing page according to the design brief.
"""
import streamlit as st


def render():
    # The page title and subtitle are rendered in the blue banner by the header.
    # Below the banner, present a clean vertical dashboard layout with cards.

    # Intro / description
    st.markdown("""
    <div class='simmons-card'>
      <h2 style='margin:0'>Simmons Portioning Optimization Platform</h2>
      <div style='margin-top:8px' class='simmons-small'>
        This platform provides data-driven support for poultry portioning and production scheduling decisions. The tool evaluates feasible portioning combinations, ranks strategies by estimated value and upgrade, and connects those outputs to scheduling models that optimize production feasibility and demand fulfillment.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Workflow overview
    st.subheader("Workflow Overview")
    wf_cols = st.columns([1, 1, 1, 1])
    wf_items = [
        ("📥", "Input Data", "Raw SKU and production constraints uploaded from Simmons data systems."),
        ("🧭", "Enumeration", "Generates and ranks feasible portioning decisions."),
        ("📅", "Scheduling", "Allocates portioning strategies across time and capacity constraints."),
        ("📊", "Results", "Operational recommendations and production insights.")
    ]

    for col, item in zip(wf_cols, wf_items):
        icon, title, desc = item
        col.markdown(f"<div class='simmons-card' style='text-align:left'><h3 style='margin:0'>{icon} {title}</h3><div class='simmons-small' style='margin-top:6px'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Documentation / quick links
    st.subheader("Documentation & Quick Links")
    # Wire real links: documentation portal, GitHub repo, DB guide, and API gateway
    docs = [
        ("📖", "Documentation Portal", "Full MkDocs site (local)", "http://localhost:3000"),
        ("🔗", "Git Repository", "Source code and CI", "https://github.com/laurenjones08/simmons-portioning-tool"),
        ("🚀", "Database Quick Start", "Common DB operations and quick start", "https://github.com/laurenjones08/simmons-portioning-tool/blob/main/QUICK_START_DB.md"),
        ("🧭", "API Gateway", "Local gateway and service endpoints", "http://localhost:8080"),
    ]
    doc_cols = st.columns(4)
    for (icon, title, desc, link), col in zip(docs, doc_cols):
        col.markdown(
            f"<div class='simmons-card' style='text-align:left'><div style='font-size:20px'>{icon} <strong>{title}</strong></div><div class='simmons-small' style='margin-top:6px'>{desc}</div><div style='margin-top:8px'><a href='{link}' target='_blank'><button style='padding:8px 12px;border-radius:6px;background:#0046AD;color:#fff;border:none'>Open</button></a></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Recent activity / snapshots
    st.subheader("Recent Activity")
    ra_cols = st.columns(3)
    # Placeholder content — if integration exists, populate with API calls
    ra_cols[0].markdown("<div class='simmons-card'><strong>Last Enumeration Run</strong><div class='simmons-small'>No runs yet</div></div>", unsafe_allow_html=True)
    ra_cols[1].markdown("<div class='simmons-card'><strong>Last Scheduling Run</strong><div class='simmons-small'>No runs yet</div></div>", unsafe_allow_html=True)
    ra_cols[2].markdown("<div class='simmons-card'><strong>Active Snapshot</strong><div class='simmons-small'>—</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Quick actions
    st.subheader("Quick Actions")
    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("Run Enumeration"):
        try:
            st.session_state.ui_sidebar_nav = "Enumeration Dashboard"
            st.session_state.ui_selected_page = "Enumeration Dashboard"
            st.experimental_set_query_params(page="Enumeration Dashboard")
        except Exception:
            pass
    if c2.button("Open Scheduling"):
        try:
            st.session_state.ui_sidebar_nav = "Scheduling Dashboard"
            st.session_state.ui_selected_page = "Scheduling Dashboard"
            st.experimental_set_query_params(page="Scheduling Dashboard")
        except Exception:
            pass
    if c3.button("View Documentation"):
        try:
            st.experimental_set_query_params(page="Advanced Settings")
        except Exception:
            pass

    st.caption("Designed for schedulers, operators, planners, and leadership — use the navigation to access dashboards and run models.")
