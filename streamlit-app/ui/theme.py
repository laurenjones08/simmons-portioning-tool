import streamlit as st


def _state_get(key: str, default=None):
    session_state = st.session_state
    if isinstance(session_state, dict):
        return session_state.get(key, default)
    return session_state.get(key, default)


def _state_set(key: str, value) -> None:
    session_state = st.session_state
    if isinstance(session_state, dict):
        session_state[key] = value
    else:
        session_state[key] = value


def init_theme_mode() -> str:
    if _state_get("ui_theme_mode") is None:
        _state_set("ui_theme_mode", "light")
    return str(_state_get("ui_theme_mode", "light"))


def is_dark_mode() -> bool:
    return init_theme_mode() == "dark"


def apply_theme() -> None:
    """Inject Simmons-branded CSS for the active light or dark theme."""
    theme_mode = init_theme_mode()
    dark_mode = theme_mode == "dark"
    theme_variables = """
      --simmons-blue:    #6EA8FF;
      --simmons-navy:    #1E4F9B;
      --simmons-secondary: #2D6FD2;
      --simmons-dark:    #E8EEF8;
      --simmons-accent:  #FF8F8A;
      --simmons-warning: #FFCD73;
      --simmons-success: #72D18C;
      --simmons-muted:   #A9B6CB;
      --card-bg:         #142033;
      --page-bg:         #0B1220;
      --border-soft:     rgba(167, 189, 230, 0.18);
      --shadow-soft:     0 10px 30px rgba(0,0,0,0.32);
      --input-bg:        #111B2D;
      --input-border:    #31425F;
      --hover-bg:        rgba(110, 168, 255, 0.14);
      --table-bg:        #0F1A2B;
      --dialog-bg:       #122033;
      --logo-bg:         #F6F8FC;
      --navbar-bg:       rgba(12, 20, 34, 0.94);
      --navbar-border:   rgba(167, 189, 230, 0.16);
      --navbar-shadow:   0 12px 30px rgba(0,0,0,0.28);
      --navbar-text:     #DCE7F8;
      --navbar-active:   rgba(110, 168, 255, 0.16);
      --navbar-hover:    rgba(110, 168, 255, 0.12);
      --theme-icon-bg:   linear-gradient(135deg, #1C3355, #28456E);
      --theme-icon-text: #F8FBFF;
    """ if dark_mode else """
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
      --border-soft:     rgba(16,24,40,0.08);
      --shadow-soft:     0 2px 6px rgba(16,24,40,0.06);
      --input-bg:        #ffffff;
      --input-border:    #d1d5db;
      --hover-bg:        #e8f0fb;
      --table-bg:        #ffffff;
      --dialog-bg:       #ffffff;
      --logo-bg:         #ffffff;
      --navbar-bg:       rgba(255,255,255,0.92);
      --navbar-border:   rgba(16,24,40,0.08);
      --navbar-shadow:   0 10px 30px rgba(15, 23, 42, 0.08);
      --navbar-text:     #5F6F86;
      --navbar-active:   #e8f0fb;
      --navbar-hover:    #eef4ff;
      --theme-icon-bg:   linear-gradient(135deg, #ffffff, #edf4ff);
      --theme-icon-text: #00264F;
    """

    css = f"""
    <style>
    :root {{
{theme_variables}
    }}

    .stApp {{
      background: var(--page-bg);
      color: var(--simmons-dark);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
      line-height: 1.5;
    }}

    [data-testid="stHeader"] {{ display: none !important; }}
    [data-testid="stToolbar"] {{ display: none !important; }}
    #stDecoration {{ display: none !important; }}
    #MainMenu {{ visibility: hidden !important; }}
    footer {{ visibility: hidden !important; }}
    [data-testid="stSidebar"] {{ display: none !important; }}
    section[data-testid="stMain"] {{ margin-left: 0 !important; }}

    .block-container {{
      padding-top: 0 !important;
      padding-left: 2rem !important;
      padding-right: 2rem !important;
      max-width: 100% !important;
    }}

    .sfy-navbar-shell {{
      display: block;
      height: 0 !important;
      overflow: hidden;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) {{
      margin: 0 !important;
      padding: 0 !important;
    }}

    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] {{
      background: var(--navbar-bg) !important;
      backdrop-filter: blur(16px) saturate(1.15) !important;
      position: sticky !important;
      top: 0 !important;
      z-index: 9999 !important;
      margin-left: -2rem !important;
      margin-right: -2rem !important;
      width: calc(100% + 4rem) !important;
      min-height: 56px !important;
      align-items: center !important;
      box-shadow: var(--navbar-shadow) !important;
      border-bottom: 1px solid var(--navbar-border) !important;
      padding: 8px 12px !important;
      gap: 8px !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] > div {{
      display: flex !important;
      align-items: center !important;
      min-height: 48px !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] [data-testid="column"] {{
      display: flex !important;
      align-items: center !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] [data-testid="element-container"] {{
      margin: 0 !important;
      width: 100% !important;
    }}

    .sfy-logo-wrap {{
      display: flex;
      align-items: center;
      justify-content: flex-start;
      height: 44px;
      padding-left: 8px;
      width: 100%;
    }}
    .sfy-logo-img {{
      display: block;
      height: 28px;
      max-width: 160px;
      width: auto;
      background: transparent;
      border-radius: 0;
      padding: 0;
      margin-top: 0;
    }}
    .sfy-logo-text {{
      color: var(--simmons-dark);
      font-size: 15px;
      font-weight: 800;
      letter-spacing: 0.04em;
    }}

    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] button {{
      background: transparent !important;
      color: var(--navbar-text) !important;
      border: none !important;
      border-bottom: none !important;
      border-radius: 999px !important;
      box-shadow: none !important;
      font-size: 11px !important;
      font-weight: 700 !important;
      letter-spacing: 0.08em !important;
      text-transform: uppercase !important;
      height: 40px !important;
      min-height: 40px !important;
      padding: 0 10px !important;
      width: 100% !important;
      transition: color 0.18s ease, background 0.18s ease, transform 0.18s ease !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] button:hover {{
      color: var(--simmons-dark) !important;
      background: var(--navbar-hover) !important;
      transform: translateY(-1px) !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] [data-testid="baseButton-primary"] {{
      color: var(--simmons-dark) !important;
      background: var(--navbar-active) !important;
      box-shadow: inset 0 0 0 1px rgba(0, 70, 173, 0.12) !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] button:focus {{
      outline: none !important;
      box-shadow: none !important;
    }}

    .sfy-page-header-shell {{
      display: block;
      width: 100%;
      clear: both;
    }}
    .sfy-page-header {{
      display: block;
      width: 100%;
      max-width: 100%;
      text-align: left;
      padding: 10px 0 12px 0;
      border-bottom: 2px solid var(--border-soft);
      margin-bottom: 18px;
    }}
    [data-testid="element-container"]:has(.sfy-page-header-shell) {{
      width: 100% !important;
      margin: 0 !important;
      padding-top: 14px !important;
      padding-bottom: 0 !important;
      display: block !important;
      flex: 1 1 100% !important;
    }}
    .sfy-page-header h2 {{
      font-size: 22px;
      font-weight: 700;
      color: var(--simmons-dark);
      margin: 0 0 3px 0;
    }}
    .sfy-page-header p {{
      color: var(--simmons-muted);
      font-size: 13px;
      margin: 0;
    }}

    .simmons-card {{
      background: var(--card-bg);
      border-radius: 8px;
      padding: 14px 18px;
      margin: 0 0 14px 0;
      box-shadow: var(--shadow-soft);
      border: 1px solid var(--border-soft);
    }}
    .simmons-kpi {{
      font-weight: 700;
      color: var(--simmons-dark);
      font-size: 20px;
    }}
    .simmons-kpi-label,
    .simmons-small,
    .stCaption {{
      color: var(--simmons-muted) !important;
      font-size: 13px;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, div {{
      color: inherit;
    }}

    .stButton > button {{
      background-color: var(--simmons-blue) !important;
      color: white !important;
      border-radius: 6px !important;
      padding: 6px 14px !important;
      border: none !important;
      font-weight: 600 !important;
    }}
    .stButton > button:hover {{
      background-color: var(--simmons-navy) !important;
    }}

    [data-testid="stTabs"] button {{
      font-weight: 600;
      color: var(--simmons-dark) !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
      border-bottom: 3px solid var(--simmons-blue) !important;
      color: var(--simmons-blue) !important;
    }}

    [data-testid="stFormSubmitButton"] button {{
      background: linear-gradient(90deg, var(--simmons-blue), var(--simmons-secondary)) !important;
      font-weight: 700 !important;
      font-size: 14px !important;
      padding: 8px 20px !important;
    }}

    [data-testid="stDataFrame"] {{
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border-soft);
      background: var(--table-bg) !important;
      margin-bottom: 14px !important;
    }}
    [data-testid="stDataFrame"] > div {{
      border-radius: 8px !important;
      overflow: hidden !important;
      background: var(--table-bg) !important;
    }}

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {{
      background: var(--input-bg) !important;
      color: var(--simmons-dark) !important;
      border-color: var(--input-border) !important;
    }}
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {{
      border-color: var(--simmons-blue) !important;
      box-shadow: 0 0 0 3px rgba(0,70,173,0.12) !important;
      outline: none !important;
    }}

    [data-baseweb="select"] > div:first-child {{
      background: var(--input-bg) !important;
      border-color: var(--input-border) !important;
      color: var(--simmons-dark) !important;
    }}
    [data-baseweb="select"] > div:first-child:focus-within {{
      border-color: var(--simmons-blue) !important;
      box-shadow: 0 0 0 3px rgba(0,70,173,0.12) !important;
    }}
    [data-baseweb="select"] *,
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] * {{
      color: var(--simmons-dark) !important;
    }}
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] ul {{
      background: var(--input-bg) !important;
      border: 1px solid var(--input-border) !important;
      border-radius: 6px !important;
      box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important;
    }}
    [data-baseweb="option"]:hover,
    [data-baseweb="option"][aria-selected="true"] {{
      background: var(--hover-bg) !important;
      color: var(--simmons-blue) !important;
    }}
    [data-baseweb="tag"] {{
      background: var(--hover-bg) !important;
      color: var(--simmons-blue) !important;
      border-radius: 4px !important;
    }}
    [data-baseweb="tag"] span {{
      color: var(--simmons-blue) !important;
    }}

    [data-testid="stSlider"] [role="slider"] {{
      background: var(--simmons-blue) !important;
      border-color: var(--simmons-blue) !important;
    }}

    [data-testid="stExpander"] {{
      background: var(--card-bg) !important;
      border: 1px solid var(--border-soft) !important;
      border-radius: 8px !important;
    }}
    [data-testid="stExpander"] summary {{
      background: var(--card-bg) !important;
      color: var(--simmons-dark) !important;
    }}

    [data-testid="stMetric"] {{
      background: var(--card-bg) !important;
      border-radius: 8px !important;
      padding: 14px 18px !important;
      border: 1px solid var(--border-soft) !important;
      box-shadow: var(--shadow-soft) !important;
      margin-bottom: 14px !important;
    }}

    [data-testid="stAlert"] {{
      border-radius: 6px !important;
      margin-bottom: 14px !important;
    }}

    [data-testid="stArrowVegaLiteChart"],
    [data-testid="stVegaLiteChart"] {{
      background: var(--card-bg) !important;
      border-radius: 8px !important;
      padding: 12px !important;
      border: 1px solid var(--border-soft) !important;
      box-shadow: var(--shadow-soft) !important;
      margin-bottom: 14px !important;
    }}

    [data-testid="stDialog"],
    [data-testid="stDialog"] > div {{
      background: transparent !important;
    }}
    [data-testid="stDialog"] [role="dialog"] {{
      background: var(--dialog-bg) !important;
      border-radius: 16px !important;
      border: 1px solid var(--border-soft) !important;
      box-shadow: 0 24px 80px rgba(0,38,79,0.22) !important;
      max-height: calc(100vh - 24px) !important;
      overflow-y: auto !important;
    }}
    [data-testid="stDialog"] [role="dialog"] > div,
    [data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stDialog"] [data-testid="element-container"],
    [data-testid="stDialog"] [data-testid="stMarkdownContainer"] {{
      background: transparent !important;
      box-shadow: none !important;
    }}
    [data-testid="stDialog"] [data-testid="stHorizontalBlock"] {{
      background: transparent !important;
      position: static !important;
      top: auto !important;
      z-index: auto !important;
      margin-left: 0 !important;
      margin-right: 0 !important;
      width: 100% !important;
      min-height: 0 !important;
      align-items: stretch !important;
      box-shadow: none !important;
      padding: 0 !important;
      gap: 1rem !important;
    }}
    [data-testid="stDialog"] [data-testid="stHorizontalBlock"] > div {{
      min-width: 0 !important;
    }}
    [data-testid="stDialog"] [data-testid="stHorizontalBlock"] button {{
      height: auto !important;
      padding: 6px 14px !important;
      border-radius: 6px !important;
      border-bottom: none !important;
      letter-spacing: normal !important;
      text-transform: none !important;
    }}
    [data-testid="stDialog"] [data-testid="stVerticalBlock"] {{
      background: transparent !important;
    }}
    [data-testid="stDialog"] [data-testid="stDataFrame"] {{
      background: var(--card-bg) !important;
      border-radius: 12px !important;
      border: 1px solid var(--border-soft) !important;
      box-shadow: var(--shadow-soft) !important;
    }}
    [data-testid="stDialog"] [data-testid="stDataFrame"] > div {{
      background: var(--card-bg) !important;
    }}
    .simmons-detail-config-card {{
      margin-bottom: 16px !important;
    }}
    .simmons-dialog-footer-divider {{
      margin: 10px 0 12px 0;
      border-top: 1px solid var(--border-soft);
      width: 100%;
    }}

    .simmons-kpi-flex-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 0 0 16px 0;
    }}
    .simmons-kpi-flex-card {{
      flex: 1 1 150px;
      min-width: 150px;
      background: var(--card-bg);
      border-radius: 12px;
      padding: 14px 18px;
      margin: 0;
      border: 1px solid var(--border-soft);
      box-shadow: var(--shadow-soft);
    }}
    .simmons-kpi-flex-label {{
      color: var(--simmons-muted);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .simmons-kpi-flex-value {{
      color: var(--simmons-dark);
      font-size: 28px;
      font-weight: 700;
      line-height: 1.1;
      margin-top: 8px;
    }}

    .simmons-link-button {{
      display: inline-block;
      margin-top: 12px;
      padding: 8px 14px;
      border-radius: 6px;
      background: var(--simmons-blue);
      color: #ffffff !important;
      text-decoration: none !important;
      font-weight: 600;
    }}
    .simmons-link-button:hover {{
      background: var(--simmons-navy);
      color: #ffffff !important;
    }}

    .simmons-skeleton-shell {{
      display: grid;
      gap: 16px;
      margin: 4px 0 8px 0;
    }}
    .simmons-skeleton-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .simmons-skeleton-card {{
      background: linear-gradient(180deg, var(--card-bg) 0%, var(--input-bg) 100%);
      border: 1px solid var(--border-soft);
      border-radius: 14px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(0,70,173,0.08);
    }}
    .simmons-skeleton-line {{
      height: 12px;
      border-radius: 999px;
      margin-bottom: 12px;
      background: linear-gradient(90deg, #dbe7f8 0%, #f5f8fd 50%, #dbe7f8 100%);
      background-size: 200% 100%;
      animation: simmons-shimmer 1.25s ease-in-out infinite;
    }}
    .simmons-skeleton-line-title {{
      height: 16px;
      width: 42%;
      margin-bottom: 18px;
    }}
    .simmons-skeleton-line:last-child {{
      margin-bottom: 0;
      width: 76%;
    }}
    @keyframes simmons-shimmer {{
      0% {{ background-position: 200% 0; }}
      100% {{ background-position: -200% 0; }}
    }}

    .simmons-theme-toggle-shell {{
      display: flex;
      align-items: center;
      justify-content: flex-end;
      min-height: 44px;
      width: 100%;
      padding-right: 0;
    }}
    .simmons-theme-icon-anchor {{
      display: block;
      width: 0;
      height: 0;
      overflow: hidden;
    }}
    [data-testid="element-container"]:has(.simmons-theme-icon-anchor) {{
      margin: 0 !important;
      padding: 0 !important;
    }}
    div:has(.simmons-theme-icon-anchor) + div button {{
      width: 36px !important;
      min-width: 36px !important;
      height: 36px !important;
      min-height: 36px !important;
      padding: 0 !important;
      border-radius: 999px !important;
      margin-left: auto !important;
      background: var(--theme-icon-bg) !important;
      color: var(--theme-icon-text) !important;
      border: 1px solid var(--navbar-border) !important;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.14) !important;
      font-size: 18px !important;
      line-height: 1 !important;
      transition: transform 0.24s ease, box-shadow 0.24s ease, background 0.24s ease !important;
      animation: simmons-theme-pop 0.28s ease;
    }}
    div:has(.simmons-theme-icon-anchor) + div {{
      display: flex !important;
      align-items: center !important;
      justify-content: flex-end !important;
      min-height: 40px !important;
    }}
    div:has(.simmons-theme-icon-anchor) + div button:hover {{
      transform: translateY(-1px) rotate(10deg) scale(1.04) !important;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16) !important;
    }}
    div:has(.simmons-theme-icon-anchor) + div button p {{
      color: var(--theme-icon-text) !important;
      font-size: 18px !important;
      font-weight: 700 !important;
      margin: 0 !important;
    }}
    @keyframes simmons-theme-pop {{
      0% {{ transform: scale(0.8) rotate(-18deg); opacity: 0.4; }}
      100% {{ transform: scale(1) rotate(0deg); opacity: 1; }}
    }}

    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] .stButton {{
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      min-height: 44px !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] .stButton > button {{
      background: transparent !important;
      color: var(--navbar-text) !important;
      border: 1px solid transparent !important;
      border-radius: 999px !important;
      box-shadow: none !important;
      font-size: 12px !important;
      font-weight: 700 !important;
      letter-spacing: 0.04em !important;
      text-transform: none !important;
      height: 38px !important;
      min-height: 38px !important;
      padding: 0 12px !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] .stButton > button:hover {{
      background: var(--navbar-hover) !important;
      color: var(--simmons-dark) !important;
      border-color: var(--navbar-border) !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] [data-testid="baseButton-primary"] {{
      background: var(--navbar-active) !important;
      color: var(--simmons-dark) !important;
      border-color: rgba(0, 70, 173, 0.16) !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] > div:first-child [data-testid="column"] {{
      justify-content: flex-start !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] > div:last-child [data-testid="column"] {{
      display: flex !important;
      align-items: center !important;
      justify-content: flex-end !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] > div:last-child .stButton {{
      display: flex !important;
      align-items: center !important;
      justify-content: flex-end !important;
      width: 100% !important;
      min-height: 44px !important;
    }}
    [data-testid="element-container"]:has(.sfy-navbar-shell) + div [data-testid="stHorizontalBlock"] > div:last-child .stButton > button {{
      width: 38px !important;
      min-width: 38px !important;
      height: 38px !important;
      min-height: 38px !important;
      padding: 0 !important;
      margin-left: auto !important;
      margin-right: 0 !important;
      display: inline-flex !important;
      align-items: center !important;
      justify-content: center !important;
      background: var(--theme-icon-bg) !important;
      color: var(--theme-icon-text) !important;
      border: 1px solid var(--navbar-border) !important;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.14) !important;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
