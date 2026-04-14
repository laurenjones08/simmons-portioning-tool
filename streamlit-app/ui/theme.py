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

    /* ── Block container: no top padding (navbar is first element) ───── */
    .block-container {
      padding-top: 0 !important;
      padding-left: 2rem !important;
      padding-right: 2rem !important;
      max-width: 100% !important;
    }

    /* ════════════════════════════════════════════════════════
       STICKY TOP NAVBAR
       Built from real st.button widgets + an HTML logo.
       The .sfy-navbar-start marker is injected just before the
       st.columns row so we can target it precisely with :has().
       ════════════════════════════════════════════════════════ */

    /* Target the stHorizontalBlock immediately after our .sfy-navbar-start marker */
    div:has(.sfy-navbar-start) + div [data-testid="stHorizontalBlock"] {
      background: linear-gradient(90deg, var(--simmons-dark) 0%, var(--simmons-navy) 55%, var(--simmons-blue) 100%) !important;
      position: sticky !important;
      top: 0 !important;
      z-index: 9999 !important;
      margin-left:  -2rem !important;
      margin-right: -2rem !important;
      width: calc(100% + 4rem) !important;
      min-height: 68px !important;
      align-items: center !important;
      box-shadow: 0 2px 18px rgba(0,0,0,0.26) !important;
      padding: 0 12px !important;
      gap: 0 !important;
    }

    /* Logo wrapper */
    .sfy-logo-wrap {
      display: flex;
      align-items: center;
      justify-content: flex-start;
      height: 68px;
      padding-left: 8px;
    }
    .sfy-logo-img {
      height: 58px;
      max-width: 260px;
      width: auto;
      background: white;
      border-radius: 6px;
      padding: 5px 14px;
      margin-top: -10px;
    }
    .sfy-logo-text {
      color: white;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: 0.04em;
    }

    /* Nav buttons: transparent on the blue background */
    div:has(.sfy-navbar-start) + div [data-testid="stHorizontalBlock"] button {
      background: transparent !important;
      color: rgba(255,255,255,0.80) !important;
      border: none !important;
      border-bottom: 3px solid transparent !important;
      border-radius: 0 !important;
      box-shadow: none !important;
      font-size: 12px !important;
      font-weight: 700 !important;
      letter-spacing: 0.07em !important;
      text-transform: uppercase !important;
      height: 68px !important;
      padding: 0 4px !important;
      width: 100% !important;
      transition: color 0.12s, border-color 0.12s, background 0.12s !important;
    }
    div:has(.sfy-navbar-start) + div [data-testid="stHorizontalBlock"] button:hover {
      color: white !important;
      background: rgba(255,255,255,0.09) !important;
      border-bottom-color: rgba(255,255,255,0.55) !important;
    }
    /* Active page button (type=primary) */
    div:has(.sfy-navbar-start) + div [data-testid="stHorizontalBlock"] [data-testid="baseButton-primary"] {
      color: white !important;
      background: rgba(255,255,255,0.13) !important;
      border-bottom-color: white !important;
    }
    /* Suppress any focus ring on nav buttons */
    div:has(.sfy-navbar-start) + div [data-testid="stHorizontalBlock"] button:focus {
      outline: none !important;
      box-shadow: none !important;
    }

    /* ── Page header strip ──────────────────────────────── */
    .sfy-page-header {
      padding: 20px 0 18px 0;
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
