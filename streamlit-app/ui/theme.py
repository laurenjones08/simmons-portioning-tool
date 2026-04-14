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
      --page-bg: #ffffff; /* White background per design request */
    }

    /* Page background and typography */
    .stApp {
      background: var(--page-bg);
      color: var(--simmons-dark);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
      line-height: 1.4;
    }

    /* Top banner */
    .simmons-top-banner {
      background: linear-gradient(90deg, var(--simmons-blue), var(--simmons-secondary));
      color: white;
      padding: 12px 18px;
      border-radius: 6px;
      margin-bottom: 18px;
    }

    .simmons-top-banner .banner-inner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .simmons-top-banner .banner-left {
      display: flex;
      flex-direction: column;
      justify-content: center;
    }

    .simmons-top-banner .banner-meta {
      display: flex;
      gap: 18px;
      color: rgba(255,255,255,0.95);
      font-size: 13px;
    }
    /* Ensure subtitle and small banner text are readable (white on blue) */
    .simmons-top-banner .simmons-small {
      color: #ffffff !important;
      opacity: 0.95 !important;
    }

    .simmons-top-banner h1 { margin: 0; font-size: 22px; }

    .simmons-top-nav a { color: rgba(255,255,255,0.95); text-decoration: none; margin-left: 12px; font-weight:600 }
    /* Top nav tabs and dropdowns */
    .simmons-top-nav-tabs { display:flex; gap:12px; align-items:center }
    .simmons-top-nav-tabs .tab { position:relative; padding:8px 12px; border-radius:6px; cursor:pointer; color:rgba(255,255,255,0.95); font-weight:600 }
    .simmons-top-nav-tabs .tab:hover { background: rgba(255,255,255,0.06) }
    .simmons-top-nav-tabs .tab .dropdown { display:none; position:absolute; top:110%; left:0; background: var(--card-bg); color: var(--simmons-dark); min-width:200px; box-shadow: 0 6px 18px rgba(2,6,23,0.12); border-radius:6px; padding:6px 0; z-index:9999 }
    .simmons-top-nav-tabs .tab:hover .dropdown { display:block }
    .simmons-top-nav-tabs .tab .dropdown a { display:block; padding:8px 12px; color:var(--simmons-dark); text-decoration:none }
    .simmons-top-nav-tabs .tab .dropdown a:hover { background: #f3f4f6 }

    /* Style selectbox in banner to match Simmons blue */
    div[data-testid="stSelectbox"] > div {
      background: transparent !important;
    }

    /* Target the inner select when Streamlit renders a native select element */
    .simmons-top-banner select, div[data-testid="stSelectbox"] select {
      background: var(--simmons-blue) !important;
      color: white !important;
      border: none !important;
      padding: 6px 10px !important;
      border-radius: 6px !important;
      font-weight: 600 !important;
    }

    /* Fallback: style Streamlit's selectbox container only when inside the top banner */
    .simmons-top-banner div[data-testid="stSelectbox"] {
      /* Make selectbox container visually part of the top banner */
      background: linear-gradient(90deg, var(--simmons-blue), var(--simmons-secondary)) !important;
      color: white !important;
      padding: 6px !important;
      border-radius: 8px !important;
      margin-top: -10px !important;
      display: inline-block !important;
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

    /* Style Streamlit's default left sidebar as a white navigation pane */
    [data-testid="stSidebar"] {
      background: var(--page-bg) !important;
      color: var(--simmons-dark) !important;
      padding: 12px !important;
      border-radius: 0 8px 8px 0 !important;
      box-shadow: 2px 0 12px rgba(2,6,23,0.04) !important;
      border-right: 1px solid rgba(16,24,40,0.04) !important;
      /* Fixed sidebar width to prevent user resizing */
      width: 320px !important;
      min-width: 320px !important;
      max-width: 320px !important;
    }

    /* Sidebar content adjustments */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h1 {
      color: var(--simmons-dark) !important;
    }
    [data-testid="stSidebar"] .stRadio, [data-testid="stSidebar"] .stSelectbox {
      background: transparent !important;
      color: var(--simmons-dark) !important;
    }

    .simmons-sidebar-logo {
      padding: 8px 0px 12px 0px;
      text-align: left;
      background: white;
    }
    /* Ensure any logo image in the sidebar has a non-transparent white background for visibility */
    [data-testid="stSidebar"] img {
      background: white !important;
      padding: 6px !important;
      border-radius: 6px !important;
      display: block !important;
      margin: 8px auto !important;
      /* make logo span most of the fixed sidebar width */
      width: 300px !important;
      height: auto !important;
      opacity: 1 !important;
    }
    /* (removed global :empty hide — it hid img elements in some browsers) */
    /* Sidebar button styles */
    .simmons-sidebar-btn {
      background: transparent;
      color: var(--simmons-dark);
      padding: 8px 12px;
      border-radius: 6px;
      margin: 6px 0;
      font-weight: 600;
    }
    .simmons-sidebar-btn.active {
      background: linear-gradient(90deg, rgba(0,70,173,0.08), rgba(0,52,120,0.06));
      border-left: 4px solid rgba(0,70,173,0.18);
    }

    /* Render radio choices as blue rounded boxes for readability */
    [data-testid="stSidebar"] .stRadio label {
      display: block;
      padding: 10px 14px;
      border-radius: 8px;
      margin: 8px 0;
      font-weight: 600;
      color: white !important;
      cursor: pointer;
      transition: background 120ms ease, transform 120ms ease;
      /* stronger, visible blue even before selection */
      background: linear-gradient(90deg, var(--simmons-blue), var(--simmons-secondary));
      box-shadow: 0 1px 0 rgba(16,24,40,0.02) inset;
      opacity: 0.95;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
      transform: translateY(-1px);
      background: linear-gradient(90deg, var(--simmons-secondary), var(--simmons-blue));
    }
    /* Show the native radio dot on the right and keep it usable */
    [data-testid="stSidebar"] .stRadio input[type="radio"] {
      opacity: 1 !important;
      margin-left: 12px !important;
      width: 14px !important;
      height: 14px !important;
      accent-color: white !important;
    }
    /* Stronger active appearance for the selected choice */
    [data-testid="stSidebar"] .stRadio label:has(input[type="radio"]:checked) {
      background: linear-gradient(90deg, var(--simmons-dark), var(--simmons-blue));
      border-left: 6px solid rgba(255,255,255,0.22);
      color: white !important;
      box-shadow: 0 8px 24px rgba(2,6,23,0.12);
      transform: translateY(-2px);
    }

    /* Layout: place label text left and radio dot on the right */
    [data-testid="stSidebar"] .stRadio label {
      display: flex !important;
      align-items: center !important;
      justify-content: space-between !important;
      white-space: nowrap !important;
    }

    /* Hide empty radio group label above the choices (prevents stray empty blue box) */
    [data-testid="stSidebar"] .stRadio > label:empty {
      display: none !important;
      height: 0 !important;
      margin: 0 !important;
      padding: 0 !important;
    }
    /* Fallback: if label is not empty (contains whitespace or non-breaking chars), hide the first label in the radio group */
    [data-testid="stSidebar"] .stRadio > label:first-child {
      display: none !important;
      height: 0 !important;
      margin: 0 !important;
      padding: 0 !important;
    }

    /* Hide Streamlit's main menu and footer chrome for a cleaner UI */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Status badges (rendered via inline HTML throughout views) */
    .status-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
    }

    /* Tab styling */
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
