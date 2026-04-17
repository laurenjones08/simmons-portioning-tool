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
      min-height: 82px !important;
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
      height: 82px;
      padding-left: 8px;
    }
    .sfy-logo-img {
      height: 62px;
      max-width: 280px;
      width: auto;
      background: white;
      border-radius: 6px;
      padding: 5px 14px;
      margin-top: -18px;
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
      height: 82px !important;
      padding: 0 6px !important;
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

    /* ════════════════════════════════════════════════════════
       WIDGET LIGHT-MODE OVERRIDES
       (config.toml sets base=light; these add Simmons polish)
       ════════════════════════════════════════════════════════ */

    /* ── Input focus ring ─────────────────────────────────── */
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
      border-color: #0046AD !important;
      box-shadow: 0 0 0 3px rgba(0,70,173,0.12) !important;
      outline: none !important;
    }

    /* ── Selectbox & MultiSelect borders ──────────────────── */
    [data-baseweb="select"] > div:first-child {
      background: #ffffff !important;
      border-color: #d1d5db !important;
    }
    [data-baseweb="select"] > div:first-child:focus-within {
      border-color: #0046AD !important;
      box-shadow: 0 0 0 3px rgba(0,70,173,0.12) !important;
    }

    /* ── Dropdown menu popup ──────────────────────────────── */
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] ul {
      background: #ffffff !important;
      border: 1px solid #d1d5db !important;
      border-radius: 6px !important;
      box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
    }
    [data-baseweb="option"]:hover,
    [data-baseweb="option"][aria-selected="true"] {
      background: #e8f0fb !important;
      color: #0046AD !important;
    }

    /* ── MultiSelect tags ─────────────────────────────────── */
    [data-baseweb="tag"] {
      background: #dde9f8 !important;
      color: #003478 !important;
      border-radius: 4px !important;
    }
    [data-baseweb="tag"] span { color: #003478 !important; }

    /* ── Slider track & thumb ─────────────────────────────── */
    [data-testid="stSlider"] [role="slider"] {
      background: #0046AD !important;
      border-color: #0046AD !important;
    }

    /* ── Expander ─────────────────────────────────────────── */
    [data-testid="stExpander"] {
      background: #ffffff !important;
      border: 1px solid rgba(16,24,40,0.08) !important;
      border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary {
      background: #ffffff !important;
    }

    /* ── Metric tiles ─────────────────────────────────────── */
    [data-testid="stMetric"] {
      background: #ffffff !important;
      border-radius: 8px !important;
      padding: 14px 18px !important;
      border: 1px solid rgba(16,24,40,0.06) !important;
      box-shadow: 0 2px 6px rgba(16,24,40,0.06) !important;
    }

    /* ── DataFrames ───────────────────────────────────────── */
    [data-testid="stDataFrame"] > div {
      border-radius: 8px !important;
      overflow: hidden !important;
    }

    /* ── Info / warning / error banners ───────────────────── */
    [data-testid="stAlert"] {
      border-radius: 6px !important;
    }

    /* ── Native chart wrappers (bar/line/area) ────────────── */
    [data-testid="stArrowVegaLiteChart"],
    [data-testid="stVegaLiteChart"] {
      background: #ffffff !important;
      border-radius: 8px !important;
      padding: 12px !important;
      border: 1px solid rgba(16,24,40,0.06) !important;
      box-shadow: 0 2px 6px rgba(16,24,40,0.06) !important;
    }

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
