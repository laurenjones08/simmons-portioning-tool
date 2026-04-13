import streamlit as st


def apply_theme() -> None:
    """Inject a simple Simmons-inspired theme (blue/gray/white) via CSS.

    This keeps styling centralized so other pages can call `apply_theme()`.
    """
    css = """
    <style>
    /* Base colors */
    :root {
      --simmons-blue: #0b5fa5;
      --simmons-dark: #263238;
      --simmons-accent: #d32f2f;
      --simmons-muted: #6b7280;
      --card-bg: #ffffff;
      --page-bg: #f6f8fa;
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
      padding: 12px 16px;
      box-shadow: 0 1px 4px rgba(16,24,40,0.06);
      border: 1px solid rgba(16,24,40,0.04);
    }

    .simmons-kpi {
      font-weight: 700;
      color: var(--simmons-blue);
      font-size: 22px;
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
      background-color: var(--simmons-blue);
      color: white;
      border-radius: 6px;
      padding: 6px 10px;
    }

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
