import streamlit as st


def apply_theme() -> None:
    """Inject Simmons-branded CSS: fixed top navbar, cards, typography, component overrides."""
    css = """
    <style>
    /* ═══════════════════════════════════════════════════════
       SIMMONS PORTIONING TOOL — DESIGN SYSTEM
       Primary:    #0046AD  (Simmons Blue)
       Navy:       #003478  (Dark Navy)
       Dark:       #00264F
       ═══════════════════════════════════════════════════════ */
    :root {
      --simmons-blue:    #0046AD;
      --simmons-navy:    #003478;
      --simmons-secondary: #003478;
      --simmons-dark:    #00264F;
      --simmons-accent:  #D9534F;
      --simmons-warning: #FFB74D;
      --simmons-success: #4CAF50;
      --simmons-muted:   #6b7280;
      --card-bg:         #ffffff;
      --page-bg:         #f8f9fb;
    }

    /* Page base */
    .stApp {
      background: var(--page-bg);
      color: var(--simmons-dark);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
      line-height: 1.5;
    }

    /* ── Hide Streamlit chrome ──────────────────────────── */
    [data-testid="stHeader"]  { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    #stDecoration             { display: none !important; }
    #MainMenu                 { visibility: hidden !important; }
    footer                    { visibility: hidden !important; }

    /* ── Hide sidebar entirely ──────────────────────────── */
    [data-testid="stSidebar"]         { display: none !important; }
    section[data-testid="stMain"]     { margin-left: 0 !important; }

    /* ── Content offset for fixed navbar ─────────────────── */
    .block-container {
      padding-top: 84px !important;
      padding-left: 2rem !important;
      padding-right: 2rem !important;
      max-width: 100% !important;
    }

    /* ════════════════════════════════════════════════════════
       FIXED TOP NAVBAR
       ════════════════════════════════════════════════════════ */
    .sfy-navbar {
      position: fixed;
      top: 0; left: 0; right: 0;
      height: 68px;
      background: linear-gradient(90deg, var(--simmons-dark) 0%, var(--simmons-navy) 60%, var(--simmons-blue) 100%);
      z-index: 99999;
      box-shadow: 0 2px 18px rgba(0,0,0,0.28);
    }

    .sfy-navbar-inner {
      display: flex;
      align-items: center;
      height: 100%;
      padding: 0 32px;
      gap: 16px;
    }

    /* Logo — white pill so the Simmons logo is legible on any background */
    .sfy-nav-brand {
      display: flex;
      align-items: center;
      flex-shrink: 0;
      background: white;
      border-radius: 6px;
      padding: 5px 12px;
      margin-right: 16px;
    }

    .sfy-nav-logo {
      height: 44px;
      width: auto;
      display: block;
    }

    .sfy-nav-wordmark {
      font-size: 18px;
      font-weight: 800;
      color: var(--simmons-dark);
      letter-spacing: 0.04em;
    }

    /* Nav links */
    .sfy-nav-links {
      display: flex;
      align-items: stretch;
      gap: 0;
      flex: 1;
    }

    .sfy-nav-link {
      color: rgba(255,255,255,0.78) !important;
      text-decoration: none !important;
      padding: 0 20px;
      font-size: 13.5px;
      font-weight: 600;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      border-bottom: 3px solid transparent;
      display: flex;
      align-items: center;
      white-space: nowrap;
      transition: color 0.12s, border-color 0.12s, background 0.12s;
    }

    .sfy-nav-link:hover {
      color: white !important;
      background: rgba(255,255,255,0.08);
      border-bottom-color: rgba(255,255,255,0.45);
    }

    .sfy-nav-active {
      color: white !important;
      background: rgba(255,255,255,0.12);
      border-bottom-color: white;
    }

    /* ── Page header strip ──────────────────────────────── */
    .sfy-page-header {
      padding: 4px 0 20px 0;
      border-bottom: 2px solid rgba(0,52,120,0.10);
      margin-bottom: 24px;
    }

    .sfy-page-header h2 {
      font-size: 22px;
      font-weight: 700;
      color: var(--simmons-dark);
      margin: 0 0 3px 0;
    }

    .sfy-page-header p {
      color: var(--simmons-muted);
      font-size: 13px;
      margin: 0;
    }


    /* ════════════════════════════════════════════════════════
       CARDS & KPI
       ════════════════════════════════════════════════════════ */
    .simmons-card {
      background: var(--card-bg);
      border-radius: 8px;
      padding: 14px 18px;
      box-shadow: 0 2px 6px rgba(16,24,40,0.06);
      border: 1px solid rgba(16,24,40,0.05);
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

    /* ════════════════════════════════════════════════════════
       BUTTONS
       ════════════════════════════════════════════════════════ */
    .stButton > button {
      background-color: var(--simmons-blue) !important;
      color: white !important;
      border-radius: 6px !important;
      padding: 6px 14px !important;
      border: none !important;
      font-weight: 600 !important;
    }

    .stButton > button:hover {
      background-color: var(--simmons-navy) !important;
    }

    /* ════════════════════════════════════════════════════════
       STATUS BADGES
       ════════════════════════════════════════════════════════ */
    .status-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
    }

    /* ════════════════════════════════════════════════════════
       TABS
       ════════════════════════════════════════════════════════ */
    [data-testid="stTabs"] button {
      font-weight: 600;
      color: var(--simmons-dark) !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
      border-bottom: 3px solid var(--simmons-blue) !important;
      color: var(--simmons-blue) !important;
    }

    /* Form submit button emphasis */
    [data-testid="stFormSubmitButton"] button {
      background: linear-gradient(90deg, var(--simmons-blue), var(--simmons-secondary)) !important;
      font-weight: 700 !important;
      font-size: 14px !important;
      padding: 8px 20px !important;
    }

    /* Dataframe / table container */
    [data-testid="stDataFrame"] {
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid rgba(16,24,40,0.06);
    }

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
