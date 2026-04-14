import streamlit as st


def apply_theme() -> None:
    """Inject a simple Simmons-inspired theme (blue/gray/white) via CSS.

    This keeps styling centralized so other pages can call `apply_theme()`.
    """
    css = """
    <style>
    /* Base colors */
    :root {
      --simmons-blue: #0046AD; /* Primary Blue */
      --simmons-secondary: #003478; /* Secondary Blue */
      --simmons-dark: #00264F; /* Dark Accent */
      --simmons-accent: #D9534F; /* Danger */
      --simmons-warning: #FFB74D; /* Amber */
      --simmons-success: #4CAF50; /* Soft green */
      --simmons-muted: #6b7280;
      --card-bg: #ffffff;
      --page-bg: #f4f6f9;
    }

    /* Page background */
    .stApp {
      background: var(--page-bg);
      color: var(--simmons-dark);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }

    /* Cards */
    .simmons-card {
      background: var(--card-bg);
      border-radius: 8px;
      padding: 14px 18px;
      box-shadow: 0 2px 6px rgba(16,24,40,0.06);
      border: 1px solid rgba(16,24,40,0.04);
    }

    .simmons-kpi {
      font-weight: 700;
      color: var(--simmons-dark);
      font-size: 20px;
    }

    .simmons-kpi-label {
      color: var(--simmons-muted);
      font-size: 12px;
    }

    .simmons-small {
      color: var(--simmons-muted);
      font-size: 13px;
    }

    /* Buttons and primary accents */
    .stButton>button {
      background-color: var(--simmons-blue) !important;
      color: white !important;
      border-radius: 6px !important;
      padding: 6px 10px !important;
      border: none !important;
    }

    .stButton>button.secondary {
      background-color: white !important;
      color: var(--simmons-blue) !important;
      border: 1px solid var(--simmons-blue) !important;
    }

    /* Sidebar logo */
    .simmons-sidebar-logo {
      padding: 8px 0px 12px 0px;
    }

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
