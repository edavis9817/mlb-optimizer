"""MLB Toolbox — Plotly theme helper + global CSS / nav bar."""

import streamlit as st


def plotly_theme(**overrides) -> dict:
    """Return a base Plotly layout dict — dark slate + deep blue theme.

    Pass keyword overrides to customise per-chart (e.g. title, height, showlegend).
    Nested dict overrides are shallow-merged with the base dicts.
    """
    base: dict = dict(
        paper_bgcolor="#141d2e",   # match main bg
        plot_bgcolor="#1c2a42",    # slightly lifted card surface
        font=dict(color="#7a9ebc", size=11),
        title=dict(font=dict(color="#d6e8f8", size=13), x=0.02),
        xaxis=dict(
            gridcolor="#1e3250", linecolor="#1e3250",
            zerolinecolor="#253d58", zerolinewidth=1,
            tickfont=dict(color="#7a9ebc"), title_font=dict(color="#a8c8e8"),
        ),
        yaxis=dict(
            gridcolor="#1e3250", linecolor="#1e3250",
            zerolinecolor="#253d58", zerolinewidth=1,
            tickfont=dict(color="#7a9ebc"), title_font=dict(color="#a8c8e8"),
        ),
        legend=dict(
            bgcolor="#1c2a42", bordercolor="#253d58", borderwidth=1,
            font=dict(color="#7a9ebc"),
        ),
        margin=dict(l=50, r=20, t=45, b=50),
        showlegend=False,
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# Meta tags injection
# ---------------------------------------------------------------------------

def inject_meta_tags():
    """Inject viewport and og:title meta tags for link previews."""
    st.markdown(
        """<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
<meta property="og:title" content="MLB Toolbox" />
<meta property="og:site_name" content="MLB Toolbox" />
<meta property="og:description" content="Data-driven baseball analysis" />
<meta name="apple-mobile-web-app-title" content="MLB Toolbox" />""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Query-param page routing
# ---------------------------------------------------------------------------

_VALID_PAGES = {"home", "league", "simulator", "roster_optimizer", "rankings", "glossary", "team", "feedback"}


def get_current_page() -> str:
    """Read ?page= from query params, set session state, and return the page key."""
    if "page" not in st.session_state:
        qp = st.query_params.get("page", "home")
        st.session_state["page"] = qp if qp in _VALID_PAGES else "home"

    # Keep URL bar in sync with current page (no extra rerun)
    try:
        st.query_params["page"] = st.session_state["page"]
    except Exception:
        pass

    return st.session_state.get("page", "league")


# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------

def render_nav_bar():
    """Render the top navigation bar using pure HTML links (query-param routing).

    IMPORTANT: all HTML is built as a single flat string with NO newlines so
    that Streamlit's markdown parser cannot mistake indented lines for code blocks.
    """
    page = st.session_state.get("page", "home")

    def _a(p: str, label: str) -> str:
        """Return a single-line <a> tag for a nav item."""
        if p == page:
            return (
                f'<a href="?page={p}" target="_self" style="color:#3b82f6;text-decoration:none;'
                f'font-weight:700;font-size:0.9rem;padding:0.4rem 0.9rem;'
                f'border-bottom:2px solid #3b6fd4;white-space:nowrap;">{label}</a>'
            )
        return (
            f'<a href="?page={p}" target="_self" style="color:#4a687e;text-decoration:none;'
            f'font-weight:600;font-size:0.9rem;padding:0.4rem 0.9rem;'
            f'border-bottom:2px solid transparent;white-space:nowrap;">{label}</a>'
        )

    bc = "#3b82f6"

    # ── Call 1: comprehensive global theme CSS ────────────────────────────────
    st.markdown("""<style>
/* ══════════════════════════════════════════════════════════════════════
   MLB TOOLBOX — Global Theme: Dark Slate + Deep Blue + Muted Teal
   ══════════════════════════════════════════════════════════════════════ */

/* ── Backgrounds ──────────────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"] {
  background: #141d2e !important;
}
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  background: #141d2e !important;
}
/* === Hide Streamlit default sidebar (we use custom nav bar) === */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
  display: none !important;
}
[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
.stMainBlockContainer {
  margin-left: 0 !important;
}
/* === ISSUE 1 — Hide Streamlit default toolbar/header/footer === */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }

.block-container {
  padding-top: 0rem !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  animation: fadeIn 0.15s ease-out;
}
[data-testid="stMainBlockContainer"] {
  padding-top: 0.5rem !important;
}
[data-testid="stMain"] > div:first-child {
  padding-top: 0 !important;
}

/* === GLOBAL POLISH — Animations & UX === */

/* Page load fade-in — fast to mask rerun flash */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* Dim content while Streamlit is processing a rerun */
.stApp[data-test-script-state="running"] .block-container {
  opacity: 0.85;
  transition: opacity 0.08s ease;
}

/* Prevent jarring scroll-to-top on rerun */
html { scroll-behavior: auto !important; }

/* Custom scrollbar — dark theme */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0e1720; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #253d58; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3b6fd4; }

/* Card hover lift — applies to expanders, metric cards, rk-answer boxes */
[data-testid="stExpander"],
[data-testid="stMetric"],
.rk-answer {
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stExpander"]:hover,
[data-testid="stMetric"]:hover,
.rk-answer:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}

/* Nav active page — sliding underline */
.mlb-nav a {
  position: relative;
  transition: color 0.2s ease;
}
.mlb-nav a::after {
  content: '';
  position: absolute;
  bottom: 0; left: 50%; width: 0;
  height: 2px; background: #3b6fd4;
  transition: width 0.3s ease, left 0.3s ease;
}
.mlb-nav a:hover::after { width: 80%; left: 10%; }

/* Alternating table rows + hover */
[data-testid="stDataFrame"] .dvn-scroller tbody tr:nth-child(even) td {
  background: rgba(255,255,255,0.015) !important;
}
[data-testid="stDataFrame"] .dvn-scroller tbody tr:hover td {
  background: rgba(255,255,255,0.05) !important;
  transition: background-color 0.15s ease;
}

/* Input focus glow */
[data-baseweb="input"]:focus-within > div,
[data-baseweb="base-input"]:focus-within > input,
[data-baseweb="textarea"]:focus-within > textarea {
  border-color: #4da6ff !important;
  box-shadow: 0 0 0 2px rgba(77, 166, 255, 0.25) !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

/* Team detail card slide-in */
@keyframes slideUp {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Team logo hover */
.tpick-logo-card {
  transition: transform 0.2s ease, filter 0.2s ease;
}
.tpick-logo-card:hover {
  transform: scale(1.08);
  filter: drop-shadow(0 0 8px rgba(255,255,255,0.15));
}

/* === ISSUE 3 — Loading state UI === */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
  text-align: center;
  gap: 16px;
}
.loading-icon { font-size: 48px; }
.loading-title {
  font-size: 1.1rem; font-weight: 700; color: #d6e8f8;
}
.loading-sub {
  font-size: 0.78rem; color: #4a687e;
}
.loading-bar {
  width: 200px; height: 3px;
  background: #1e3250;
  border-radius: 2px;
  overflow: hidden;
}
.loading-bar::after {
  content: '';
  display: block;
  height: 100%;
  width: 40%;
  background: #4a9eff;
  border-radius: 2px;
  animation: loadpulse 1.5s ease-in-out infinite;
}
@keyframes loadpulse {
  0%   { transform: translateX(-100%); }
  100% { transform: translateX(350%); }
}

/* Style Streamlit's native spinner/loading widget */
[data-testid="stSpinner"] {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  min-height: 40vh;
  text-align: center !important;
}
[data-testid="stSpinner"] > div {
  color: #93c5fd !important;
  font-size: 0.88rem !important;
}

/* ── Cards / containers ───────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: #1c2a42 !important;
  border: 1px solid #1e3250 !important;
  border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
  color: #d6e8f8 !important;
}
[data-testid="stExpander"] summary:hover {
  background: #1d2f47 !important;
}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
  background: transparent;
}

/* ── Metrics ──────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: #18243a;
  border: 1px solid #1e3250;
  border-radius: 8px;
  padding: 0.5rem 0.75rem;
}
[data-testid="stMetricLabel"] { color: #7a9ebc !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: #d6e8f8 !important; font-size: 1.25rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.72rem !important; }
[data-testid="stMetricDelta"][data-direction="up"]   > div { color: #22c55e !important; }
[data-testid="stMetricDelta"][data-direction="down"] > div { color: #dc2626 !important; }

/* ── Tabs ─────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px !important;
  border-bottom: 1px solid #1e3250 !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
  font-size: 0.82rem !important;
  padding: 5px 14px !important;
  color: #4a687e !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
  color: #3b82f6 !important;
  border-bottom: 2px solid #3b6fd4 !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  color: #7a9ebc !important;
  background: rgba(30,50,80,0.3) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────── */
[data-testid="stButton"] > button[kind="primary"] {
  background: #2b5cc8 !important;
  border: 1px solid #3b6fd4 !important;
  color: #ffffff !important;
  border-radius: 6px !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
  background: #3b6fd4 !important;
  border-color: #5080e4 !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid #253d58 !important;
  color: #7a9ebc !important;
  border-radius: 6px !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
  background: #1d2f47 !important;
  border-color: #3b6fd4 !important;
  color: #d6e8f8 !important;
}
[data-testid="stButton"] > button:disabled {
  background: #151f2e !important;
  border-color: #1a2a3a !important;
  color: #3a5068 !important;
  opacity: 0.65 !important;
}
/* Download button — treat as secondary */
[data-testid="stDownloadButton"] > button {
  background: transparent !important;
  border: 1px solid #253d58 !important;
  color: #7a9ebc !important;
  border-radius: 6px !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background: #1d2f47 !important;
  border-color: #3b6fd4 !important;
  color: #d6e8f8 !important;
}

/* ── Inputs (text, number, select) ───────────────────────────────── */
[data-baseweb="input"] > div,
[data-baseweb="base-input"] > input,
[data-baseweb="textarea"] > textarea {
  background: #12202e !important;
  border-color: #1e3250 !important;
  color: #d6e8f8 !important;
  border-radius: 6px !important;
}
[data-baseweb="input"]:focus-within > div,
[data-baseweb="base-input"]:focus-within > input {
  border-color: #3b6fd4 !important;
}
[data-baseweb="select"] > div {
  background: #12202e !important;
  border-color: #1e3250 !important;
  color: #d6e8f8 !important;
  border-radius: 6px !important;
}
[data-baseweb="select"] > div:hover {
  border-color: #3b6fd4 !important;
}
[data-baseweb="popover"] [role="listbox"] {
  background: #1c2a42 !important;
  border: 1px solid #253d58 !important;
}
[data-baseweb="option"]:hover {
  background: #1d2f47 !important;
}
[data-baseweb="option"][aria-selected="true"] {
  background: #1e3a6a !important;
}
/* Number input */
[data-testid="stNumberInput"] input {
  background: #12202e !important;
  border-color: #1e3250 !important;
  color: #d6e8f8 !important;
}

/* ── Sliders ──────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
  background: #3b6fd4 !important;
  border-color: #3b6fd4 !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div:first-child {
  background: #1e3250 !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div:nth-child(2) {
  background: #3b6fd4 !important;
}

/* ── Multiselect chips ────────────────────────────────────────────── */
[data-baseweb="tag"] {
  background-color: #1e3a6a !important;
  border: 1px solid #2b5cc8 !important;
  border-radius: 4px !important;
}
[data-baseweb="tag"] span { color: #93c5fd !important; }
[data-baseweb="tag"] svg  { opacity: 0.6; }

/* ── Checkboxes ───────────────────────────────────────────────────── */
[data-testid="stCheckbox"] label { color: #7a9ebc !important; }
[data-baseweb="checkbox"] svg    { fill: #3b6fd4 !important; }

/* ── Alert / info boxes ───────────────────────────────────────────── */
[data-testid="stInfo"] {
  background: rgba(30,50,90,0.35) !important;
  border-left: 3px solid #3b6fd4 !important;
  border-radius: 6px !important;
  color: #93c5fd !important;
}
[data-testid="stSuccess"] {
  background: rgba(16,60,30,0.35) !important;
  border-left: 3px solid #22c55e !important;
  border-radius: 6px !important;
  color: #86efac !important;
}
[data-testid="stWarning"] {
  background: rgba(70,45,10,0.35) !important;
  border-left: 3px solid #d97706 !important;
  border-radius: 6px !important;
  color: #fcd34d !important;
}
[data-testid="stError"] {
  background: rgba(70,15,15,0.35) !important;
  border-left: 3px solid #dc2626 !important;
  border-radius: 6px !important;
  color: #fca5a5 !important;
}

/* ── Dataframe / table ────────────────────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
  border: 1px solid #1e3250 !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
/* Header row — centered */
[data-testid="stDataFrame"]    .dvn-scroller thead th,
[data-testid="stDataEditor"]   .dvn-scroller thead th {
  background: #1d2f47 !important;
  color: #93b8d8 !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  border-bottom: 1px solid #253d58 !important;
  text-align: center !important;
}
/* All cells — centered */
[data-testid="stDataFrame"]    .dvn-scroller tbody td,
[data-testid="stDataEditor"]   .dvn-scroller tbody td {
  text-align: center !important;
}
/* Cell hover */
[data-testid="stDataFrame"]    .dvn-scroller tbody tr:hover td,
[data-testid="stDataEditor"]   .dvn-scroller tbody tr:hover td {
  background: rgba(30,60,110,0.18) !important;
}

/* ── Caption / markdown ───────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p,
[data-testid="stCaption"] {
  color: #7a9ebc !important;
  font-size: 0.82rem !important;
}
/* ── General paragraph / body text brighter ──────────────────────── */
.stApp p, .stApp span, .stApp label, .stApp div {
  color: inherit;
}
[data-testid="stMarkdownContainer"] p {
  color: #93b8d8 !important;
  font-size: 0.88rem !important;
  line-height: 1.65 !important;
}
h1,h2,h3,h4,h5,h6 { color: #d6e8f8 !important; }

/* ── Progress column fill ─────────────────────────────────────────── */
[data-testid="stDataFrame"] [data-testid="stProgressBar"] > div,
[data-testid="stDataEditor"] [data-testid="stProgressBar"] > div {
  background: #2b5cc8 !important;
}

/* ── Plotly chart wrapper ─────────────────────────────────────────── */
[data-testid="stImage"] {
  border-radius: 10px !important; overflow: hidden !important;
  border: 1px solid #1e3250 !important; background: #18243a !important;
}
[data-testid="stImage"] img { border-radius: 10px !important; display: block; }
[data-testid="stPlotlyChart"] {
  border-radius: 10px !important; overflow: hidden !important;
  border: 1px solid #1e3250 !important;
}

/* ── Hide multiselect clear-all button ───────────────────────────── */
button[data-testid="stMultiSelectClearButton"] { display: none !important; }
[data-baseweb="clear-icon"]  { display: none !important; }
[aria-label="Clear all"]     { display: none !important; }

/* === MOBILE RESPONSIVE FIXES === */

/* ══════════════════════════════════════════════════════════════════════
   RESPONSIVE — Tablet (≤1024px)
   ══════════════════════════════════════════════════════════════════════ */
@media (max-width: 1024px) {
  .block-container { padding: 0.5rem 0.8rem !important; }
  [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
  }
  [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    min-width: 45% !important;
  }
}

/* ══════════════════════════════════════════════════════════════════════
   RESPONSIVE — Mobile (≤768px)
   ══════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {

  /* ── Issue 5 — Global spacing & padding ────────────────────────── */
  .block-container {
    padding: 0.3rem 0.75rem !important;
    max-width: 100% !important;
  }

  /* ── Issue 5 — Typography scale-down ───────────────────────────── */
  h1 { font-size: 1.4rem !important; }
  h2 { font-size: 1.2rem !important; }
  h3 { font-size: 1.1rem !important; }
  h4 { font-size: 0.9rem !important; }

  /* ── Issue 5 — Streamlit columns stack vertically ──────────────── */
  [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    gap: 0.4rem !important;
  }
  [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
  [data-testid="column"] {
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }

  /* ── Issue 5 — Reduce vertical gaps ────────────────────────────── */
  [data-testid="stVerticalBlock"] > div {
    gap: 0.5rem !important;
  }

  /* ── Issue 5 — Tables scroll horizontally ──────────────────────── */
  [data-testid="stDataFrame"],
  [data-testid="stDataEditor"] {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
  }

  /* ── Issue 5 — Plotly charts fill width + disable touch zoom ──── */
  .js-plotly-plot { width: 100% !important; }
  .js-plotly-plot .draglayer,
  .js-plotly-plot .scrollbox { touch-action: pan-y !important; }
  [data-testid="stPlotlyChart"] {
    min-height: 280px !important;
    touch-action: pan-y !important;
  }

  /* ── Issue 5 — Metrics compact ─────────────────────────────────── */
  [data-testid="stMetric"] {
    padding: 0.3rem 0.5rem !important;
  }
  [data-testid="stMetricLabel"] { font-size: 0.62rem !important; }
  [data-testid="stMetricValue"] { font-size: 1rem !important; }

  /* ── Issue 5 — Tabs scrollable ─────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
    flex-wrap: nowrap !important;
  }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
  .stTabs [data-baseweb="tab"] {
    font-size: 0.72rem !important;
    padding: 4px 10px !important;
    flex-shrink: 0 !important;
  }

  /* ── Issue 5 — Expanders compact ───────────────────────────────── */
  [data-testid="stExpander"] summary {
    font-size: 0.78rem !important;
    padding: 0.4rem 0.6rem !important;
  }

  /* ── Issue 5 — Buttons full width ──────────────────────────────── */
  [data-testid="stButton"] > button,
  [data-testid="stDownloadButton"] > button {
    width: 100% !important;
    font-size: 0.78rem !important;
  }

  /* ── Issue 5 — Sidebar full-width overlay ──────────────────────── */
  [data-testid="stSidebar"] {
    min-width: 100vw !important;
    max-width: 100vw !important;
  }

  /* ── Issue 5 — Select / multiselect full width ─────────────────── */
  [data-baseweb="select"],
  [data-baseweb="input"] {
    min-width: 0 !important;
  }

  /* ── Issue 2 — Nav bar: scrollable pill row on mobile ──────────── */
  .mlb-nav {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
    flex-wrap: nowrap !important;
    gap: 0.25rem !important;
    padding: 0.3rem 0.5rem !important;
    position: relative;
  }
  .mlb-nav::-webkit-scrollbar { display: none; }
  .mlb-nav::after {
    content: '';
    position: sticky;
    right: 0;
    top: 0;
    bottom: 0;
    min-width: 32px;
    flex-shrink: 0;
    background: linear-gradient(to right, transparent, #111927 85%);
    pointer-events: none;
    z-index: 2;
  }
  .mlb-nav a {
    font-size: 13px !important;
    padding: 0.45rem 0.65rem !important;
    min-height: 44px !important;
    display: inline-flex !important;
    align-items: center !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    border-radius: 8px !important;
  }
  /* Hide the separator and subtitle on mobile */
  .mlb-nav > span:nth-child(2),
  .mlb-nav > span:nth-child(3),
  .mlb-nav > span:nth-child(4) {
    display: none !important;
  }

  /* ── League Analysis — header and summary cards on mobile ────── */
  .ef-hdr {
    flex-direction: column !important;
    align-items: stretch !important;
    padding: 10px 12px !important;
  }
  .ef-hdr-stats {
    justify-content: center !important;
    gap: 6px !important;
  }
  .ef-hdr-stats > div {
    flex: 1 1 auto !important;
    min-width: 70px !important;
    padding: 6px 8px !important;
  }
  .ef-summary-grid {
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 8px !important;
  }

  /* ── Issue 6 — Sticky bar safe area ────────────────────────────── */
  .mlb-sbar {
    padding-bottom: max(8px, env(safe-area-inset-bottom)) !important;
    z-index: 999 !important;
    gap: 10px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    font-size: 0.64rem !important;
  }
  .mlb-sbar .sb-t { font-size: 12px !important; }
  .mlb-sbar .sb-v,
  .mlb-sbar .sb-val { font-size: 11px !important; }
  .mlb-sbar .sb-l,
  .mlb-sbar .sb-lbl { font-size: 7px !important; }
  .mlb-sbar-pad { height: max(44px, calc(44px + env(safe-area-inset-bottom))) !important; }
}

/* ══════════════════════════════════════════════════════════════════════
   RESPONSIVE — Small phone (≤480px)
   ══════════════════════════════════════════════════════════════════════ */
@media (max-width: 480px) {
  .block-container { padding: 0.2rem 0.4rem !important; }
  h1 { font-size: 1.2rem !important; }
  h2 { font-size: 1rem !important; }
  h3 { font-size: 0.88rem !important; }
  [data-testid="stMetricValue"] { font-size: 0.88rem !important; }
  .mlb-sbar { gap: 6px !important; padding-left: 8px !important; padding-right: 8px !important; }
  .mlb-sbar .sb-t { font-size: 10px !important; }
  .mlb-sbar .sb-v,
  .mlb-sbar .sb-val { font-size: 10px !important; }
}
</style>""", unsafe_allow_html=True)

    # ── Call 2: nav HTML only — no <style> block, no newlines, no indented lines ──
    nav = (
        '<div class="mlb-nav" style="display:flex;align-items:center;padding:0.5rem 0 0;flex-wrap:wrap;">'
        f'<a href="?page=home" target="_self" style="color:{bc};text-decoration:none;font-weight:900;'
        f'font-size:1.1rem;padding:0.3rem 0.8rem 0.3rem 0;white-space:nowrap;">&#9918; MLB Toolbox</a>'
        '<span style="color:#253d58;padding:0 0.5rem;line-height:1;">|</span>'
        '<span style="font-size:0.65rem;color:#2e4a62;text-transform:uppercase;'
        'letter-spacing:0.06em;margin-right:1.5rem;">Data-driven baseball analysis</span>'
        '<span style="flex:1;"></span>'
        + _a('rankings', '🏆 Rankings')
        + _a('team', '🏟️ Team Analysis')
        + _a('league', '📊 Player Analysis')
        + _a('simulator', '🎮 Roster Simulator')
        + _a('glossary', '📖 Methodology')
        + _a('feedback', '💬 Feedback')
        + '</div>'
        + '<hr style="margin:0.4rem 0 0.6rem;border:none;border-top:1px solid #1e3250;">'
    )
    st.markdown(nav, unsafe_allow_html=True)

    # ── Dark / Light mode toggle (compact, right-aligned under nav) ──────
    st.markdown(
        "<style>"
        "[data-testid='stMainBlockContainer'] > div:first-child "
        ".toggle-row { display:flex; justify-content:flex-end; margin-top:-0.6rem; margin-bottom:0.3rem; }"
        "</style>",
        unsafe_allow_html=True,
    )
    _tc1, _tc2 = st.columns([11, 1])
    with _tc2:
        _light = st.toggle("☀️", key="light_mode", value=True)
    if _light:
        st.markdown("""<style>
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
        [data-testid="stBottom"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
            background: #f0f2f6 !important; }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background: #e8ebf0 !important; }
        h1,h2,h3,h4,h5,h6 { color: #1a1a2e !important; }
        [data-testid="stMarkdownContainer"] p { color: #2d3748 !important; }
        [data-testid="stCaptionContainer"] p, [data-testid="stCaption"] { color: #4a5568 !important; }
        [data-testid="stExpander"] { background: #ffffff !important; border-color: #e2e8f0 !important; }
        [data-testid="stExpander"] summary { color: #2d3748 !important; }
        [data-testid="stMetric"] { background: #ffffff; border-color: #e2e8f0; }
        [data-testid="stMetricLabel"] { color: #4a5568 !important; }
        [data-testid="stMetricValue"] { color: #1a1a2e !important; }
        .stTabs [data-baseweb="tab"] { color: #4a5568 !important; }
        .stTabs [aria-selected="true"] { color: #2b5cc8 !important; border-bottom-color: #2b5cc8 !important; }
        [data-testid="stDataFrame"] .dvn-scroller thead th { background: #edf2f7 !important; color: #2d3748 !important; }
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] { border-color: #e2e8f0 !important; }
        .rk-answer { background: #ffffff !important; border-color: #e2e8f0 !important; }
        .rk-answer .rk-team { color: #1a1a2e !important; }
        .rk-answer .rk-q { color: #4a5568 !important; }
        .rk-answer .rk-val { color: #4a5568 !important; }
        .mlb-nav a { color: #2d3748 !important; }
        </style>""", unsafe_allow_html=True)
