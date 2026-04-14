"""Home view for single-page router."""
import streamlit as st


def render():
    st.header("What this tool does")
    st.markdown(
        """
- Centralized UI for running enumeration and scheduling models.
- Use the top navigation to access dashboards and settings.
- Run Enumeration to produce immutable snapshots, review ranked portioning decisions, and push results to scheduling.
- Visualize mix metrics and scheduling assignments before committing.

This front-end calls into the existing Enumeration and Scheduling APIs — ensure the services are running and the environment variables `ENUMERATION_API_URL` and `WORKER_API_URL` are set when running locally.
        """
    )

    st.markdown("---")

    st.subheader("Quick Links")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Enumeration**\n\nOpen the Enumeration Dashboard to review SKUs, set constraints, and run enumeration.")
    with cols[1]:
        st.markdown("**Scheduling**\n\nPreview schedules and visualize line assignments before committing to production.")
    with cols[2]:
        st.markdown("**Settings & Exports**\n\nManage buckets, cut strategies, and export snapshots or reports.")

    st.markdown("---")
    st.caption("Fonts, colors, and layout follow the Simmons brand: white background, consistent typography, and Simmons blue accents.")
