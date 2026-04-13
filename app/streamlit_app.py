"""
streamlit_app.py
----------------
MLB Toolbox — Interactive Streamlit UI.

Run with:
  streamlit run app/streamlit_app.py
  (from the mlb_optimizer/ root directory)
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_APP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.projections  import make_projections
from src.archetypes   import build_archetype_definitions, assign_archetypes
from src.optimizer    import run_optimizer
from src.simulation   import monte_carlo
from src.backtest     import run_backtest
from src.diagnostics  import budget_frontier, marginal_analysis
from src.artifacts    import write_run_artifacts
from src.team_mode    import (
    get_all_teams,
    get_team_payroll_history,
    build_offseason_scenario,
)
from src.depth_chart  import (
    get_depth_chart_dir,
)
from utils.constants import (
    PAYROLL_TEAM_MAP as _PAYROLL_2026_TEAM_MAP_CONST,
    CBT_TIERS as _CBT_TIERS_CONST,
    POS_GROUP_MAP as _POS_GROUP_MAP_CONST,
    ELIGIBLE_SLOTS_MAP as _ELIGIBLE_SLOTS_MAP_CONST,
    OPTIONAL_SLOTS as _OPTIONAL_SLOTS_CONST,
    ROSTER_TEMPLATE as _ROSTER_TEMPLATE_CONST,
    PG_CHART_COLORS as _PG_CHART_COLORS_CONST,
    ABBR_TO_FULL as _ABBR_TO_FULL_CONST,
    TEAM_CITIES as _TEAM_CITIES_CONST,
    TEAM_COLORS as _TEAM_COLORS_CONST,
    LOGO_FILE_NAMES as _LOGO_FILE_NAMES_CONST,
    MLB_TEAM_ID_MAP as _MLB_TEAM_ID_MAP_CONST,
    AL_DIVISIONS, NL_DIVISIONS,
    REPLACEMENT_LEVEL_WINS,
    FWAR_CONTENDER_FLOOR,
    GOOGLE_SHEET_WEBHOOK,
    STAGE_DISPLAY_MAP,
    STAGE_COLORS as STAGE_COLORS_CONST,
    STAGE_BG_COLORS,
)
from utils.player_utils import (
    fix_player_name as _fix_player_name,
    fix_player_col as _fix_player_col,
    headshot_url as _headshot_url,
    hover_img_tag as _hover_img_tag,
)
from utils.team_utils import cbt_info as _cbt_info, ordinal as _ordinal
from utils.theme import plotly_theme as _pt
from utils.components import (
    render_feedback_widget as _render_feedback_widget,
    render_glossary as _render_glossary,
    loading_placeholder as _loading_placeholder,
)
from utils.data_loading import (
    R2_BASE_URL,
    R2_MODE as _R2_MODE,
    ROOT_DIR as _DL_ROOT_DIR,
    DEFAULT_CONFIG as _DEFAULT_CONFIG,
    HEADSHOTS_DIR as _HEADSHOTS_DIR,
    RAZZBALL_PATH as _RAZZBALL_PATH,
    init_etag_metadata as _init_etag_metadata,
    save_etag_metadata as _save_etag_metadata,
    compute_cache_key as _compute_cache_key,
    get_cached_file_path as _get_cached_file_path,
    read_csv as _read_csv,
    read_excel as _read_excel,
    r2_image as _r2_image,
    data_url as _data_url,
    load_base_config as _load_base_config,
    resolve_data_path as _resolve_data_path,
    file_hash as _file_hash,
    dir_hash as _dir_hash,
    parse_payroll_val as _parse_payroll_val,
    team_logo_url as _team_logo_url,
    cached_projections as _cached_projections,
    cached_archetypes as _cached_archetypes,
    cached_wins as _cached_wins,
    cached_payroll_history as _cached_payroll_history,
    cached_team_scenario as _cached_team_scenario,
    cached_simulator_data as _cached_simulator_data,
    cached_2026_payroll as _cached_2026_payroll,
    cached_player_history as _cached_player_history,
    cached_war_reliability as _cached_war_reliability,
    cached_razzball as _cached_razzball,
    cached_mlbam_lookup as _cached_mlbam_lookup,
    cached_40man_roster as _cached_40man_roster,
    load_enriched_roster as _load_enriched_roster,
    build_carousel_players as _build_carousel_players,
    cached_carousel_images as _cached_carousel_images,
    fetch_2026_standings as _fetch_2026_standings,
    fetch_2026_standings_full as _fetch_2026_standings_full,
    fetch_2026_team_stats as _fetch_2026_team_stats,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLB Toolbox",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="auto",
)

# Override Streamlit's default meta tags for link previews (iOS, iMessage, etc.)
st.markdown(
    """<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
<meta property="og:title" content="MLB Toolbox" />
<meta property="og:site_name" content="MLB Toolbox" />
<meta property="og:description" content="Data-driven baseball analysis" />
<meta name="apple-mobile-web-app-title" content="MLB Toolbox" />""",
    unsafe_allow_html=True,
)

try:
    import requests as _requests
    _requests_available = True
except ImportError:
    _requests_available = False

import io as _io
import unicodedata as _unicodedata
from pathlib import Path as _Path

# Aliases for constants (used by inline render functions and legacy page calls)
_ROSTER_TEMPLATE = _ROSTER_TEMPLATE_CONST
_ELIGIBLE_SLOTS_MAP = _ELIGIBLE_SLOTS_MAP_CONST
_OPTIONAL_SLOTS = _OPTIONAL_SLOTS_CONST
_PG_CHART_COLORS = _PG_CHART_COLORS_CONST


def _arch_label(arch_id: str) -> str:
    """Convert 'SP_FA_Elite' -> 'Elite Starter (Free Agent)'."""
    _pos = {
        "SP": "Starter",  "RP": "Reliever",  "C": "Catcher",
        "1B": "1st Base", "2B": "2nd Base",  "3B": "3rd Base", "SS": "Shortstop",
        "CI": "Corner IF","MI": "Middle IF",  "CF": "Center Field",
        "OF": "Corner OF","DH": "DH",
    }
    _stage = {"FA": "Free Agent", "Arb": "Arb-Eligible", "Pre-Arb": "Pre-Arb"}
    parts = arch_id.split("_", 2)          # "SP_Pre-Arb_Solid" -> ["SP","Pre-Arb","Solid"]
    if len(parts) != 3:
        return arch_id
    pos_str   = _pos.get(parts[0], parts[0])
    stage_str = _stage.get(parts[1], parts[1])
    tier_str  = parts[2]
    return f"{tier_str} {pos_str} ({stage_str})"

# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------

def _render_nav_bar():
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
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
  background: #111a24 !important;
  border-right: 1px solid #1e3250 !important;
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
  animation: fadeIn 0.4s ease-out;
}

/* === GLOBAL POLISH — Animations & UX === */

/* Page load fade-in */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Smooth scrolling */
html { scroll-behavior: smooth; }

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
        + '<hr style="margin:0.4rem 0 1rem;border:none;border-top:1px solid #1e3250;">'
    )
    st.markdown(nav, unsafe_allow_html=True)

    # ── Dark / Light mode toggle ─────────────────────────────────────────
    _tc1, _tc2 = st.columns([10, 1])
    with _tc2:
        _light = st.toggle("☀️", key="light_mode", value=False)
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


# _render_player_comparison → moved to pages/roster_simulator.py


# _render_trade_analyzer, _render_position_coverage, _render_roster_summary → moved to pages/roster_simulator.py


# ---------------------------------------------------------------------------
# Player card  → moved to pages/roster_simulator.py
# ---------------------------------------------------------------------------

# _render_player_card → moved to pages/roster_simulator.py


# ---------------------------------------------------------------------------
# Landing / Home page
# ---------------------------------------------------------------------------

def _render_home_page():
    """Landing page: Stadium Tunnel + Starting Lineup concept."""
    from pages.home import render as _home_render
    _home_render(
        data_url=_data_url,
        read_csv=_read_csv,
        cached_mlbam_lookup=_cached_mlbam_lookup,
        razzball_path=_RAZZBALL_PATH,
    )



# _render_best_fits → moved to pages/roster_simulator.py


# ---------------------------------------------------------------------------
# Roster Simulator — main page (delegated to pages/roster_simulator.py)
# ---------------------------------------------------------------------------

def _render_simulator_page():
    from pages.roster_simulator import render
    render({
        'data_url': _data_url, 'read_csv': _read_csv, 'r2_mode': _R2_MODE,
        'r2_base_url': R2_BASE_URL,
        'DEFAULT_CONFIG': _DEFAULT_CONFIG, 'ROOT_DIR': _ROOT_DIR,
        'HEADSHOTS_DIR': _HEADSHOTS_DIR, 'RAZZBALL_PATH': _RAZZBALL_PATH,
        'load_base_config': _load_base_config,
        'resolve_data_path': _resolve_data_path,
        'file_hash': _file_hash, 'dir_hash': _dir_hash,
        'cached_simulator_data': _cached_simulator_data,
        'cached_2026_payroll': _cached_2026_payroll,
        'cached_40man_roster': _cached_40man_roster,
        'cached_war_reliability': _cached_war_reliability,
        'cached_player_history': _cached_player_history,
        'cached_razzball': _cached_razzball,
        'cached_mlbam_lookup': _cached_mlbam_lookup,
        'ROSTER_TEMPLATE': _ROSTER_TEMPLATE,
        'ELIGIBLE_SLOTS_MAP': _ELIGIBLE_SLOTS_MAP,
        'OPTIONAL_SLOTS': _OPTIONAL_SLOTS,
        'PG_CHART_COLORS': _PG_CHART_COLORS,
    })


# ---------------------------------------------------------------------------
# Optimizer page helpers — inline settings
# ---------------------------------------------------------------------------

def _build_inline_settings(base_cfg: dict) -> tuple[dict, bool]:
    """Render inline optimizer settings (expander) and return (cfg, run_clicked)."""
    cfg = copy.deepcopy(base_cfg)

    with st.expander("⚙️ Optimizer Settings", expanded=False):
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown("**Budget & Market**")
            cfg["budget_M"] = st.slider(
                "Budget ($M)", min_value=40, max_value=400,
                value=int(cfg.get("budget_M", 130)), step=5,
                key="opt_budget",
            )
            cfg["market_dpw_M"] = st.slider(
                "Market $/WAR ($M)", min_value=3.0, max_value=12.0,
                value=float(cfg.get("market_dpw_M", 5.5)), step=0.5,
                key="opt_mkt",
            )
            st.markdown("**Player Filters**")
            cfg["min_war_threshold"] = st.slider(
                "Min WAR threshold", -2.0, 4.0,
                float(cfg.get("min_war_threshold", -99)), 0.5,
                key="opt_minwar",
            )

        with s2:
            st.markdown("**Roster Structure**")
            cfg["roster_slots"]["SP"] = st.slider(
                "SP slots", min_value=3, max_value=6,
                value=int(cfg["roster_slots"].get("SP", 5)),
                key="opt_sp",
            )
            if st.checkbox(
                "Include DH slot",
                value=("DH" in cfg["roster_slots"] and cfg["roster_slots"]["DH"] > 0),
                key="opt_dh",
            ):
                cfg["roster_slots"]["DH"] = 1
            else:
                cfg["roster_slots"].pop("DH", None)

            st.markdown("**Optimizer Mode**")
            cfg["optimizer_mode"] = st.selectbox(
                "Mode", ["archetype", "player"],
                index=0 if cfg.get("optimizer_mode", "archetype") == "archetype" else 1,
                key="opt_mode",
            )

        with s3:
            st.markdown("**Objective Weights**")
            ow = cfg.get("objective_weights", {"wins": 1.0, "surplus": 0.0, "risk_penalty": 0.0})
            ow["wins"]         = st.slider("Wins weight",          0.0, 1.0, float(ow.get("wins",         1.0)), 0.05, key="opt_ow_wins")
            ow["surplus"]      = st.slider("Surplus value weight", 0.0, 1.0, float(ow.get("surplus",      0.0)), 0.05, key="opt_ow_surp")
            ow["risk_penalty"] = st.slider("Risk penalty",         0.0, 1.0, float(ow.get("risk_penalty", 0.0)), 0.05, key="opt_ow_risk")
            cfg["objective_weights"] = ow

            st.markdown("**Simulation**")
            cfg["mc_simulations"] = st.selectbox(
                "MC simulations", [100, 500, 1000, 5000, 10000], index=2,
                key="opt_mc",
            )
            cfg["mc_seed"] = st.number_input(
                "Random seed", value=int(cfg.get("mc_seed", 42)), step=1,
                key="opt_seed",
            )

    run_clicked = st.button("▶ Run Optimizer", type="primary", key="opt_run_btn")
    return cfg, run_clicked


# ---------------------------------------------------------------------------
# Optimizer page — full 6-tab layout
# ---------------------------------------------------------------------------

def _render_optimizer_page():
    """Render the Roster Optimizer (original app logic, 6 tabs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io as _io
    import json as _json

    st.markdown("### 🔧 Roster Optimizer")

    if not os.path.exists(_DEFAULT_CONFIG):
        st.error(f"Config file not found: {_DEFAULT_CONFIG}\nRun from mlb_optimizer/ root.")
        return

    base_cfg = _load_base_config(_DEFAULT_CONFIG)

    salary_path = _resolve_data_path(base_cfg["raw_salary_war_path"], _DEFAULT_CONFIG)
    wins_path   = _resolve_data_path(base_cfg["raw_wins_path"],       _DEFAULT_CONFIG)

    if not os.path.exists(salary_path):
        st.error(
            f"Salary/WAR data file not found:\n`{salary_path}`\n\n"
            "Update `raw_salary_war_path` in `configs/default_config.json`."
        )
        return

    cfg, run_btn = _build_inline_settings(base_cfg)

    salary_hash       = _file_hash(salary_path)
    proj_weights_json = _json.dumps(base_cfg["projection_weights"], sort_keys=True)

    proj_df, raw_df = _cached_projections(
        salary_path, salary_hash, proj_weights_json,
        int(base_cfg["season"]),
        bool(base_cfg.get("clip_negative_war", True)),
        float(base_cfg.get("min_war_threshold", -99)),
        int(base_cfg.get("max_contract_years", 99)),
    )

    min_war_live = float(cfg.get("min_war_threshold", -99))
    proj_df_live = proj_df[proj_df["proj_WAR"] >= min_war_live].copy()

    proj_hash      = hashlib.md5(proj_df_live.to_csv(index=False).encode()).hexdigest()
    proj_json      = proj_df_live.to_json(orient="records")
    arch_df, proj_with_arch = _cached_archetypes(proj_hash, proj_json)

    wins_hash = _file_hash(wins_path)
    wins_df   = _cached_wins(wins_path, wins_hash)

    bt_cfg    = {"wins_intercept": float(cfg.get("wins_intercept", 48.0))}
    bt_result = run_backtest(raw_df, wins_df, bt_cfg)
    adj       = bt_result.adjustment_factor

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_cfg" not in st.session_state:
        st.session_state["last_cfg"] = None

    if run_btn or st.session_state["last_result"] is None:
        with st.spinner("Optimizing ..."):
            opt_result = run_optimizer(arch_df, cfg, projected_df=proj_with_arch)
        st.session_state["last_result"] = opt_result
        st.session_state["last_cfg"]    = copy.deepcopy(cfg)

    opt_result = st.session_state["last_result"]
    run_cfg    = st.session_state["last_cfg"] or cfg

    sim_result = monte_carlo(opt_result.roster_df, run_cfg, backtest_adjustment=adj)

    # --- Tabs ---
    tab_roster, tab_arch, tab_wins, tab_frontier, tab_diag, tab_export = st.tabs([
        "🏟 Roster", "📊 Archetypes", "🎲 Win Dist", "📈 Frontier", "🔧 Diagnostics", "💾 Export",
    ])

    # ===========================================================
    # Tab 1 — Roster
    # ===========================================================
    with tab_roster:
        st.subheader("Optimal Roster")
        if opt_result.roster_df.empty:
            st.warning(f"No feasible roster found. Solver status: {opt_result.status}")
        else:
            rdf        = opt_result.roster_df.copy()
            total_war  = rdf["war_mean"].sum()
            total_cost = rdf["cost_mean"].sum()
            budget_val = run_cfg["budget_M"]
            remaining  = budget_val - total_cost

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total proj. WAR",      f"{total_war:.1f}")
            col2.metric("Payroll used ($M)",     f"${total_cost:.1f}M")
            col3.metric("Budget remaining ($M)", f"${remaining:.1f}M")
            col4.metric("Solver",                opt_result.status)

            _rdf_disp = rdf.copy()
            _rdf_disp.insert(1, "Role", _rdf_disp["archetype_id"].apply(_arch_label))
            display_cols = [c for c in ["slot", "Role", "pos_group", "war_mean", "war_sd", "cost_mean", "n_players"]
                            if c in _rdf_disp.columns]
            st.dataframe(
                _rdf_disp[display_cols].rename(columns={
                    "slot": "Slot", "pos_group": "Pos Group",
                    "war_mean": "Proj WAR", "war_sd": "WAR SD",
                    "cost_mean": "Est Cost", "n_players": "# Players",
                }).style.format({
                    "Proj WAR": "{:.2f}",
                    "WAR SD":   "{:.2f}",
                    "Est Cost": "${:.1f}M",
                }),
                use_container_width=True,
            )

            try:
                slot_vals  = rdf["war_mean"].values.tolist()
                slot_labels = [
                    f"{r['slot']} · {_arch_label(r['archetype_id'])}"
                    for _, r in rdf.iterrows()
                ]
                vmin, vmax = min(slot_vals), max(slot_vals)
                norm  = [(v - vmin) / max(vmax - vmin, 0.01) for v in slot_vals]
                def _ryg(t):
                    if t < 0.5:
                        r, g = 1.0, t * 2
                    else:
                        r, g = 1.0 - (t - 0.5) * 2, 1.0
                    return f"rgb({int(r*220)},{int(g*200)},60)"
                slot_colors = [_ryg(n) for n in norm]
                y_max = max(slot_vals) * 1.25 if slot_vals else 5
                fig = go.Figure(data=[go.Bar(
                    x=slot_labels, y=slot_vals,
                    marker_color=slot_colors, marker_line_width=0,
                    text=[f"{v:.2f}" for v in slot_vals],
                    textposition="outside", textfont=dict(color="#dbeafe", size=9),
                    hovertemplate="%{x}<br>WAR: %{y:.2f}<extra></extra>",
                )])
                fig.update_layout(**_pt(
                    title=f"Roster WAR by Slot  (${budget_val:.0f}M budget)",
                    yaxis=dict(title="Projected WAR", range=[0, y_max]),
                    xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
                    height=380,
                    margin=dict(l=50, r=20, t=45, b=110),
                ))
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})
            except Exception as e:
                st.info(f"Chart unavailable: {e}")

    # ===========================================================
    # Tab 2 — Archetypes
    # ===========================================================
    with tab_arch:
        st.subheader("Archetype Mix")
        if opt_result.archetype_mix:
            mix_df = pd.DataFrame(
                [{"Archetype": k, "Slots": v} for k, v in opt_result.archetype_mix.items()]
            ).sort_values("Slots", ascending=False)
            st.dataframe(mix_df, width="stretch")

        st.markdown("---")
        st.subheader("Available Archetypes (gold inputs)")
        disp_arch = arch_df.copy()
        if "eligible_slots" in disp_arch.columns:
            disp_arch["eligible_slots"] = disp_arch["eligible_slots"].apply(
                lambda v: ", ".join(v) if isinstance(v, list) else str(v)
            )
        fmt = {"war_mean": "{:.2f}", "war_sd": "{:.2f}", "cost_mean": "${:.1f}M", "cost_sd": "${:.1f}M"}
        st.dataframe(
            disp_arch.style.format({k: v for k, v in fmt.items() if k in disp_arch.columns}),
            use_container_width=True,
        )

    # ===========================================================
    # Tab 3 — Win Distribution
    # ===========================================================
    with tab_wins:
        st.subheader("Win Distribution (Monte Carlo)")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean wins",    f"{sim_result.mean_wins:.1f}")
        c2.metric("Median (P50)", f"{sim_result.median_wins:.1f}")
        c3.metric("P10",          f"{sim_result.p10:.1f}")
        c4.metric("P90",          f"{sim_result.p90:.1f}")
        c5.metric("Playoff odds", f"{sim_result.playoff_odds:.1%}")

        try:
            threshold = float(run_cfg.get("playoff_threshold_wins", 88))
            wins_arr  = sim_result.wins_array
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=wins_arr, nbinsx=50,
                marker_color="#2563eb", marker_line_width=0, opacity=0.85,
                name="Simulations",
                hovertemplate="Wins: %{x}<br>Count: %{y}<extra></extra>",
            ))
            # Vertical reference lines via shapes + annotations
            x_max = float(np.max(wins_arr)) + 5
            for x_val, color, dash, lbl in [
                (sim_result.p10,         "#ef4444", "dash",  f"P10 = {sim_result.p10:.0f}"),
                (sim_result.median_wins, "#dbeafe", "solid", f"P50 = {sim_result.median_wins:.0f}"),
                (sim_result.p90,         "#22c55e", "dash",  f"P90 = {sim_result.p90:.0f}"),
                (threshold,              "#f59e0b", "dot",   f"Playoff ≥ {threshold:.0f}"),
            ]:
                fig.add_vline(x=x_val, line_color=color, line_dash=dash,
                              line_width=1.8, opacity=0.9,
                              annotation_text=lbl,
                              annotation_font_color=color,
                              annotation_position="top right")
            fig.add_vrect(x0=threshold, x1=x_max,
                          fillcolor="#22c55e", opacity=0.06, line_width=0)
            fig.update_layout(**_pt(
                title=(f"Win Distribution — {run_cfg['mc_simulations']:,} sims "
                       f"(playoff odds: {sim_result.playoff_odds:.1%})"),
                xaxis=dict(title="Season Wins"),
                yaxis=dict(title="Simulations"),
                height=380,
                showlegend=False,
            ))
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
        except Exception as e:
            st.info(f"Chart unavailable: {e}")

        if bt_result.n_team_years > 0:
            st.caption(
                f"Backtest: RMSE = {bt_result.rmse:.1f} wins over {bt_result.n_team_years} team-seasons  "
                f"| Bias adjustment applied: {bt_result.adjustment_factor:+.1f} wins"
            )

    # ===========================================================
    # Tab 4 — Budget Frontier
    # ===========================================================
    with tab_frontier:
        st.subheader("Budget Frontier")

        with st.spinner("Computing frontier (10 budget points) ..."):
            front_df = budget_frontier(arch_df, run_cfg)

        if not front_df.empty and "expected_wins" in front_df.columns:
            fdf = front_df.dropna(subset=["expected_wins"])

            try:
                curr_b    = run_cfg["budget_M"]
                threshold = float(run_cfg.get("playoff_threshold_wins", 88))
                bx = fdf["budget_M"].tolist()
                ew = fdf["expected_wins"].tolist()

                traces = []
                if "p10" in fdf.columns and "p90" in fdf.columns:
                    traces.append(go.Scatter(
                        x=bx + bx[::-1],
                        y=fdf["p90"].tolist() + fdf["p10"].tolist()[::-1],
                        fill="toself", fillcolor="rgba(59,130,246,0.12)",
                        line=dict(width=0), name="P10–P90 band",
                        hoverinfo="skip",
                    ))
                traces.append(go.Scatter(
                    x=bx, y=ew, mode="lines+markers",
                    line=dict(color="#3b82f6", width=2.5),
                    marker=dict(color="#3b82f6", size=7),
                    name="Expected wins",
                    hovertemplate="$%{x:.0f}M → %{y:.1f} wins<extra></extra>",
                ))
                fig = go.Figure(data=traces)
                fig.add_vline(x=curr_b, line_color="#f59e0b", line_dash="dash",
                              line_width=1.5,
                              annotation_text=f"Current ${curr_b:.0f}M",
                              annotation_font_color="#f59e0b")
                fig.add_hline(y=threshold, line_color="#22c55e", line_dash="dot",
                              line_width=1.5,
                              annotation_text=f"Playoff ≥ {threshold:.0f}",
                              annotation_font_color="#22c55e")
                fig.update_layout(**_pt(
                    title="Wins vs Budget",
                    xaxis=dict(title="Budget ($M)"),
                    yaxis=dict(title="Expected wins"),
                    showlegend=True,
                    height=380,
                    margin=dict(l=50, r=20, t=45, b=50),
                ))
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})
            except Exception as e:
                st.info(f"Chart unavailable: {e}")

            st.dataframe(
                fdf[["budget_M","expected_wins","p10","p90","playoff_odds","total_cost_M","status"]]
                  .style.format({
                      "budget_M":      "${:.0f}M",
                      "expected_wins": "{:.1f}",
                      "p10":           "{:.1f}",
                      "p90":           "{:.1f}",
                      "playoff_odds":  "{:.1%}",
                      "total_cost_M":  "${:.1f}M",
                  }),
                use_container_width=True,
            )
        else:
            st.warning("Frontier computation returned no results.")

    # ===========================================================
    # Tab 5 — Diagnostics
    # ===========================================================
    with tab_diag:
        st.subheader("Marginal Analysis")

        with st.spinner("Running marginal analysis ..."):
            upgrades_df, cuts_df = marginal_analysis(opt_result.roster_df, arch_df, run_cfg)

        col_u, col_c = st.columns(2)
        with col_u:
            st.markdown("**Top Upgrades** (best delta-wins / delta-cost)")
            if not upgrades_df.empty:
                st.dataframe(
                    upgrades_df.style.format({
                        "delta_war":        "{:+.2f}",
                        "delta_cost_M":     "${:+.1f}M",
                        "delta_wins":       "{:+.2f}",
                        "delta_wins_per_M": "{:.3f}",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("No upgrades available.")

        with col_c:
            st.markdown("**Top Cuts** (best savings / win cost)")
            if not cuts_df.empty:
                st.dataframe(
                    cuts_df.style.format({
                        "delta_war":    "{:+.2f}",
                        "delta_cost_M": "${:+.1f}M",
                        "savings_M":    "${:.1f}M",
                        "delta_wins":   "{:+.2f}",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("No cuts available.")

        st.markdown("---")
        st.subheader("Backtest Summary")
        if bt_result.n_team_years > 0:
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("RMSE (wins)",         f"{bt_result.rmse:.2f}")
            bc2.metric("Bias (mean residual)", f"{bt_result.bias:+.2f}")
            bc3.metric("Team-seasons",         str(bt_result.n_team_years))
        else:
            st.info("No wins data loaded -- backtest skipped. Check `raw_wins_path` in config.")

    # ===========================================================
    # Tab 6 — Export
    # ===========================================================
    with tab_export:
        st.subheader("Export Run")

        cfg_str = _json.dumps(run_cfg, indent=2)
        st.download_button(
            "Download config.json",
            data=cfg_str,
            file_name="config.json",
            mime="application/json",
        )

        if not opt_result.roster_df.empty:
            csv_buf = _io.StringIO()
            opt_result.roster_df.to_csv(csv_buf, index=False)
            st.download_button(
                "Download roster.csv",
                data=csv_buf.getvalue(),
                file_name="roster.csv",
                mime="text/csv",
            )

        st.markdown("---")
        if st.button("Save full run to runs/ folder", key="export_save_btn"):
            with st.spinner("Writing artifacts ..."):
                from datetime import datetime
                ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = os.path.join(_ROOT_DIR, "runs", f"run_{ts}")
                os.makedirs(run_dir, exist_ok=True)
                diag_bundle = {
                    "frontier_df": front_df if "front_df" in dir() else pd.DataFrame(),
                    "upgrades_df": upgrades_df if "upgrades_df" in dir() else pd.DataFrame(),
                    "cuts_df":     cuts_df     if "cuts_df"     in dir() else pd.DataFrame(),
                    "backtest":    bt_result,
                    "sim_result":  sim_result,
                }
                write_run_artifacts(run_dir, run_cfg, arch_df, opt_result, diag_bundle, {})
            st.success(f"Run saved to: `{run_dir}`")

        st.markdown("---")
        st.subheader("Run Metadata")
        st.json({
            "solver_status":   opt_result.status,
            "objective_value": round(opt_result.objective_value, 4)
                               if opt_result.objective_value == opt_result.objective_value else None,
            "n_archetypes":    len(arch_df),
            "n_players_pool":  len(proj_df_live),
            "budget_M":        run_cfg["budget_M"],
            "optimizer_mode":  run_cfg["optimizer_mode"],
            "mc_simulations":  run_cfg["mc_simulations"],
            "mean_wins":       round(sim_result.mean_wins, 1),
            "playoff_odds":    round(sim_result.playoff_odds, 3),
            "backtest_adj":    round(adj, 2),
        })


# ---------------------------------------------------------------------------
# Team Moves recommendation panel (primary Team Planner section)
# ---------------------------------------------------------------------------

def _render_team_moves(
    scenario: dict,
    player_pool: pd.DataFrame,
    selected_team: str,
    available_M: float,
    mkt_rate: float,
) -> None:
    """Primary section: recommended FA signings, trade targets, and cost cuts."""
    st.markdown("---")
    st.markdown("### Recommended Offseason Moves")
    st.caption(
        f"Budget available after committed payroll: **${available_M:.1f}M**  "
        "· Free agents can be signed directly. "
        "Arb & Pre-Arb players are under team control — **acquisition requires a trade**."
    )

    if player_pool.empty:
        st.info("2026 payroll data not available — cannot generate move recommendations.")
        return

    # Players already on this team (exclude from pool)
    _on_team: set[str] = set()
    for _df in [
        scenario.get("locked_df",   pd.DataFrame()),
        scenario.get("arb_df",      pd.DataFrame()),
        scenario.get("expiring_df", pd.DataFrame()),
    ]:
        if not _df.empty and "Player" in _df.columns:
            _on_team.update(_df["Player"].tolist())

    pool = player_pool[~player_pool["Player"].isin(_on_team)].copy()

    # Which pos_groups fill open slots
    remaining = scenario.get("remaining_slots", {})
    _open_pgs: set[str] = set()
    for _slot in remaining:
        for _pg, _slots in _ELIGIBLE_SLOTS_MAP.items():
            if _slot in _slots:
                _open_pgs.add(_pg)

    # Value metrics
    pool["_market_val_M"] = (pool["WAR_Total"].fillna(0) * mkt_rate).round(2)
    pool["_surplus_M"]    = (pool["_market_val_M"] - pool["Salary_M"].fillna(0)).round(2)
    pool["_fills_need"]   = pool["pos_group"].isin(_open_pgs)

    fa_pool    = pool[pool["Stage_Clean"] == "FA"].copy()
    trade_pool = pool[pool["Stage_Clean"].isin(["Arb", "Pre-Arb"])].copy()

    _acq_col   = "Acquisition"
    _show_base = ["Player", "Team", "Position", "Stage_Clean", "Age",
                  "WAR_Total", "Salary_M", "W_per_M", "_surplus_M", _acq_col]
    _fmt_mv = {
        "WAR_Total": "{:.1f}", "Salary_M": "${:.1f}M",
        "W_per_M": "{:.2f}", "_surplus_M": "${:+.1f}M", "Age": "{:.0f}",
    }
    _rename_mv = {
        "WAR_Total": "WAR (2025)", "Salary_M": "2026 Sal $M",
        "W_per_M": "W/$M", "_surplus_M": "Surplus vs Mkt", "Stage_Clean": "Stage",
    }

    tm_t1, tm_t2, tm_t3 = st.tabs([
        "Free Agent Signings",
        "Trade Targets",
        "Cost Cut Candidates",
    ])

    # ── Tab 1: FA Signings ────────────────────────────────────────────────────
    with tm_t1:
        fa_affordable = fa_pool[
            fa_pool["Salary_M"].notna() & (fa_pool["Salary_M"] <= available_M + 0.01)
        ].copy()
        fa_affordable[_acq_col] = "Sign (FA)"

        fa_priority = fa_affordable[fa_affordable["_fills_need"]].sort_values(
            ["W_per_M", "WAR_Total"], ascending=[False, False], na_position="last"
        )
        fa_other = fa_affordable[~fa_affordable["_fills_need"]].sort_values(
            ["W_per_M", "WAR_Total"], ascending=[False, False], na_position="last"
        )
        fa_sorted = pd.concat([fa_priority, fa_other]).drop_duplicates("Player")

        if fa_sorted.empty:
            st.info(f"No free agents found within the ${available_M:.0f}M available budget.")
        else:
            _fa_cols = [c for c in _show_base if c in fa_sorted.columns]
            st.caption(
                f"⭐ = fills an open roster position  ·  {len(fa_sorted)} players affordable  "
                f"·  Sorted by efficiency (W/$M), needs first."
            )
            _fa_disp = fa_sorted[_fa_cols].rename(columns=_rename_mv).reset_index(drop=True)
            _fa_disp.insert(0, "Need", ["⭐" if v else "" for v in fa_sorted["_fills_need"].tolist()])
            st.dataframe(
                _fa_disp.style.format({k: v for k, v in {
                    **_fmt_mv,
                    "WAR (2025)": "{:.1f}", "2026 Sal $M": "${:.1f}M",
                    "W/$M": "{:.2f}", "Surplus vs Mkt": "${:+.1f}M", "Age": "{:.0f}",
                }.items() if k in _fa_disp.columns}),
                hide_index=True, use_container_width=True, height=500,
            )

    # ── Tab 2: Trade Targets ──────────────────────────────────────────────────
    with tm_t2:
        st.caption(
            "Arb & Pre-Arb players are locked to their teams — they **cannot** be signed as FAs. "
            "Any acquisition requires a trade. Sorted by projected WAR, needs first."
        )
        trade_priority = trade_pool[trade_pool["_fills_need"]].sort_values(
            "WAR_Total", ascending=False, na_position="last"
        )
        trade_other = trade_pool[~trade_pool["_fills_need"]].sort_values(
            "WAR_Total", ascending=False, na_position="last"
        )
        trade_sorted = pd.concat([trade_priority, trade_other]).drop_duplicates("Player")
        trade_sorted[_acq_col] = "Trade Required"

        if trade_sorted.empty:
            st.info("No Arb/Pre-Arb trade targets found in pool.")
        else:
            _tr_cols = [c for c in _show_base if c in trade_sorted.columns]
            _tr_disp = trade_sorted[_tr_cols].rename(columns=_rename_mv).reset_index(drop=True)
            _tr_disp.insert(0, "Need", ["⭐" if v else "" for v in trade_sorted["_fills_need"].tolist()])
            st.dataframe(
                _tr_disp.style.format({k: v for k, v in {
                    **_fmt_mv,
                    "WAR (2025)": "{:.1f}", "2026 Sal $M": "${:.1f}M",
                    "W/$M": "{:.2f}", "Surplus vs Mkt": "${:+.1f}M", "Age": "{:.0f}",
                }.items() if k in _tr_disp.columns}),
                hide_index=True, use_container_width=True, height=500,
            )

    # ── Tab 3: Cost Cut Candidates ────────────────────────────────────────────
    with tm_t3:
        st.caption(
            "Current roster players delivering below-market value. "
            "Releasing or trading them frees budget for higher-impact additions."
        )
        _roster_rows: list[pd.DataFrame] = []
        for _df, _src in [
            (scenario.get("locked_df", pd.DataFrame()), "Under Contract"),
            (scenario.get("arb_df",    pd.DataFrame()), "Arb-Eligible"),
        ]:
            if _df.empty:
                continue
            _tmp = _df.copy()
            _tmp["_source"] = _src
            if "sal_2026_M" in _tmp.columns:
                _tmp["Salary_M"] = pd.to_numeric(_tmp["sal_2026_M"], errors="coerce")
            elif "sal_2025_M" in _tmp.columns:
                _tmp["Salary_M"] = pd.to_numeric(_tmp["sal_2025_M"], errors="coerce")
            if "war_2025" in _tmp.columns:
                _tmp["WAR_Total"] = pd.to_numeric(_tmp["war_2025"], errors="coerce")
            _roster_rows.append(_tmp)

        if not _roster_rows:
            st.info("No current roster data available.")
        else:
            _rc_df = pd.concat(_roster_rows, ignore_index=True)
            _rc_df["WAR_Total"] = pd.to_numeric(_rc_df.get("WAR_Total", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            _rc_df["Salary_M"]  = pd.to_numeric(_rc_df.get("Salary_M",  pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            _rc_df["_market_val_M"] = (_rc_df["WAR_Total"] * mkt_rate).round(2)
            _rc_df["_surplus_M"]    = (_rc_df["_market_val_M"] - _rc_df["Salary_M"]).round(2)
            _rc_df["W_per_M"]       = (_rc_df["WAR_Total"] / _rc_df["Salary_M"].clip(lower=0.1)).round(3)

            _cuts = _rc_df.sort_values("_surplus_M").reset_index(drop=True)
            _cut_cols = [c for c in ["Player", "pos_group", "age", "WAR_Total", "Salary_M",
                                     "W_per_M", "_surplus_M", "_source"]
                         if c in _cuts.columns]
            _cut_disp = _cuts[_cut_cols].rename(columns={
                "pos_group": "Position", "age": "Age",
                "WAR_Total": "WAR (2025)", "Salary_M": "2026 Sal $M",
                "W_per_M": "W/$M", "_surplus_M": "Surplus vs Mkt", "_source": "Status",
            }).reset_index(drop=True)

            def _clr_surplus(v):
                try:
                    return "background-color:#4a1a1a;color:white" if float(v) < 0 else ""
                except Exception:
                    return ""

            st.dataframe(
                _cut_disp.style
                    .format({k: v for k, v in {
                        "WAR (2025)": "{:.1f}", "2026 Sal $M": "${:.1f}M",
                        "W/$M": "{:.2f}", "Surplus vs Mkt": "${:+.1f}M", "Age": "{:.0f}",
                    }.items() if k in _cut_disp.columns})
                    .map(_clr_surplus, subset=["Surplus vs Mkt"] if "Surplus vs Mkt" in _cut_disp.columns else []),
                hide_index=True, use_container_width=True, height=500,
            )


# ---------------------------------------------------------------------------
# Team Planner — UI helpers
# ---------------------------------------------------------------------------

def _render_hero_panel(
    selected_team: str,
    budget_M: float,
    committed_M: float,
    available_M: float,
    total_war: float,
    open_slots: int,
    surplus_M: float,
) -> None:
    """Render styled hero metric cards at the top of the Team Planner."""
    est_wins    = 48 + total_war / 1.5
    budget_pct  = committed_M / max(budget_M, 1) * 100
    avail_bdr   = "#22c55e" if available_M >= 20 else "#f59e0b" if available_M >= 10 else "#ef4444"
    surp_bdr    = "#22c55e" if surplus_M >= 0 else "#ef4444"
    risk_lvl    = "Low" if open_slots <= 3 and available_M >= 20 else ("High" if open_slots >= 7 or available_M < 10 else "Mid")
    risk_color  = {"Low": "#22c55e", "Mid": "#f59e0b", "High": "#ef4444"}[risk_lvl]

    def _card(label, val, sub, border="#1e3250", vcolor="#d6e8f8"):
        return (
            f'<div style="background:#18243a;border:1px solid {border};'
            f'border-radius:10px;padding:14px 8px;text-align:center;">'
            f'<div style="font-size:10px;color:#4a687e;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">{label}</div>'
            f'<div style="font-size:26px;font-weight:800;color:{vcolor};line-height:1">{val}</div>'
            f'<div style="font-size:10px;color:#2e4a62;margin-top:4px">{sub}</div></div>'
        )

    html = (
        '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:16px 0 20px 0;">'
        + _card("Est. Wins",  f"{est_wins:.0f}",      "2026 Projection",                  "#253d58",  "#93c5fd")
        + _card("Total WAR",  f"{total_war:.1f}",     "Current Roster")
        + _card("Committed",  f"${committed_M:.0f}M", f"{budget_pct:.0f}% of ${budget_M:.0f}M")
        + _card("Available",  f"${available_M:.0f}M", "To Spend",                          avail_bdr, avail_bdr)
        + _card("Surplus",    f"${surplus_M:+.0f}M",  "vs Market Rate",                   surp_bdr,  surp_bdr)
        + _card("Risk",       risk_lvl,               f"{open_slots} open slot{'s' if open_slots != 1 else ''}", risk_color, risk_color)
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _inject_sticky_bar(
    selected_team: str,
    budget_M: float,
    committed_M: float,
    available_M: float,
    total_war: float,
    open_slots: int,
) -> None:
    """Inject a CSS fixed summary bar visible while the user scrolls."""
    est_wins    = 48 + total_war / 1.5
    avail_color = "#22c55e" if available_M >= 20 else "#f59e0b" if available_M >= 10 else "#ef4444"

    html = f"""<style>
.mlb-sbar{{position:fixed;bottom:0;top:auto;left:0;right:0;z-index:9998;
background:rgba(10,15,24,0.97);border-top:1px solid #1e3250;
padding:5px 24px;display:flex;align-items:center;gap:18px;
backdrop-filter:blur(6px);box-shadow:0 -2px 16px rgba(0,0,0,.4);}}
.mlb-sbar .sb-t{{font-size:14px;font-weight:800;color:#3b82f6;white-space:nowrap}}
.mlb-sbar .sb-i{{display:flex;flex-direction:column;align-items:center;gap:1px}}
.mlb-sbar .sb-l{{font-size:9px;color:#4a687e;text-transform:uppercase;letter-spacing:.7px}}
.mlb-sbar .sb-v{{font-size:13px;font-weight:700;color:#d6e8f8}}
</style>
<div class="mlb-sbar">
<span class="sb-t">⚾ {selected_team}</span>
<div class="sb-i"><span class="sb-l">Budget</span><span class="sb-v">${budget_M}M</span></div>
<div class="sb-i"><span class="sb-l">Committed</span><span class="sb-v">${committed_M:.0f}M</span></div>
<div class="sb-i"><span class="sb-l">Available</span><span class="sb-v" style="color:{avail_color}">${available_M:.0f}M</span></div>
<div class="sb-i"><span class="sb-l">WAR</span><span class="sb-v">{total_war:.1f}</span></div>
<div class="sb-i"><span class="sb-l">Est Wins</span><span class="sb-v">{est_wins:.0f}</span></div>
<div class="sb-i"><span class="sb-l">Open Slots</span><span class="sb-v">{open_slots}</span></div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_payroll_charts(
    depth_df: "pd.DataFrame | None",
    scenario: dict,
    budget_M: float,
    mkt_rate: float,
) -> None:
    """Payroll donut, WAR-by-position bar, and committed-vs-available budget bar."""
    col_l, col_r = st.columns(2)

    with col_l:
        if depth_df is not None and not depth_df.empty and "sal_2026_M" in depth_df.columns:
            def _broad(pg):
                if pg == "SP":                        return "Starting Pitching"
                if pg == "RP":                        return "Bullpen"
                if pg in ("C","1B","2B","3B","SS"):   return "Infield"
                if pg in ("OF","CF","LF","RF"):       return "Outfield"
                if pg == "DH":                        return "DH / Bench"
                return "Other"

            _dc = depth_df.copy()
            _dc["_cat"] = _dc["pos_group"].apply(_broad)
            _by_cat = (
                _dc.groupby("_cat")["sal_2026_M"].sum().reset_index()
                .rename(columns={"_cat": "Category"})
            )
            _by_cat = _by_cat[_by_cat["sal_2026_M"] > 0].sort_values("sal_2026_M", ascending=False)
            _pal = {
                "Starting Pitching": "#4a9eff", "Bullpen": "#7ee8a2",
                "Infield": "#f59e0b", "Outfield": "#a78bfa",
                "DH / Bench": "#fb923c", "Other": "#6b7280",
            }
            _total_sal = float(_dc["sal_2026_M"].sum())

            fig_d = go.Figure(go.Pie(
                labels=_by_cat["Category"],
                values=_by_cat["sal_2026_M"],
                hole=0.52,
                marker_colors=[_pal.get(c, "#6b7280") for c in _by_cat["Category"]],
                textinfo="label+percent",
                textfont=dict(size=9),
                hovertemplate="%{label}<br>$%{value:.1f}M (%{percent})<extra></extra>",
            ))
            fig_d.add_annotation(
                text=f"${_total_sal:.0f}M<br><span style='font-size:9px'>26-man</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="#e8f4ff"),
            )
            fig_d.update_layout(**_pt(
                title="Payroll Allocation (26-man)",
                height=300, showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
            ))
            st.plotly_chart(fig_d, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Depth chart not available — payroll chart skipped.")

    with col_r:
        if depth_df is not None and not depth_df.empty and "proj_WAR" in depth_df.columns:
            _war_pg = (
                depth_df.groupby("pos_group")["proj_WAR"].sum().reset_index()
                .rename(columns={"proj_WAR": "WAR"})
            )
            _war_pg = _war_pg[_war_pg["WAR"].notna() & (_war_pg["WAR"].abs() > 0)]
            _war_pg = _war_pg.sort_values("WAR", ascending=True)

            fig_w = go.Figure(go.Bar(
                x=_war_pg["WAR"], y=_war_pg["pos_group"],
                orientation="h",
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in _war_pg["WAR"]],
                text=_war_pg["WAR"].apply(lambda v: f"{v:.1f}"),
                textposition="outside",
                textfont=dict(size=9, color="#a8c8e8"),
                hovertemplate="%{y}: %{x:.1f} WAR<extra></extra>",
            ))
            fig_w.update_layout(**_pt(
                title="Projected WAR by Position Group",
                xaxis=dict(title="Proj WAR"),
                height=300,
                margin=dict(l=50, r=60, t=40, b=30),
            ))
            st.plotly_chart(fig_w, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Depth chart not available — WAR chart skipped.")

    # Budget bar
    _comm  = scenario.get("committed_payroll_M", 0.0)
    _avail = max(budget_M - _comm, 0.0)

    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(
        name="Committed", x=["2026 Budget"], y=[_comm],
        marker_color="#4a9eff",
        text=[f"${_comm:.0f}M committed"], textposition="inside",
        hovertemplate="Committed: $%{y:.1f}M<extra></extra>",
    ))
    fig_b.add_trace(go.Bar(
        name="Available", x=["2026 Budget"], y=[_avail],
        marker_color="#1a3a5c",
        text=[f"${_avail:.0f}M available"], textposition="inside",
        hovertemplate="Available: $%{y:.1f}M<extra></extra>",
    ))
    fig_b.add_shape(type="line", x0=-0.5, x1=0.5, y0=budget_M, y1=budget_M,
                    line=dict(color="#ef4444", width=2, dash="dash"))
    fig_b.add_annotation(x=0.45, y=budget_M, text=f"  Budget cap: ${budget_M}M",
                         showarrow=False, font=dict(color="#ef4444", size=10), xanchor="left")
    fig_b.update_layout(**_pt(
        title="Committed vs Available Budget",
        yaxis=dict(title="$M"), barmode="stack",
        height=200, showlegend=True,
        margin=dict(l=50, r=120, t=40, b=30),
    ))
    st.plotly_chart(fig_b, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Team Planner page
# ---------------------------------------------------------------------------

def _render_team_planner(base_cfg: dict | None = None):
    """Render the Team Offseason Planner."""
    import json as _json

    st.markdown("### 🗓 Team Offseason Planner")

    if base_cfg is None:
        if not os.path.exists(_DEFAULT_CONFIG):
            st.error(f"Config file not found: {_DEFAULT_CONFIG}")
            return
        base_cfg = _load_base_config(_DEFAULT_CONFIG)

    # ── Data paths ────────────────────────────────────────────────────────────
    salary_abs    = _resolve_data_path(base_cfg["raw_salary_war_path"], _DEFAULT_CONFIG)
    data_dir      = os.path.dirname(salary_abs)
    combined_path = os.path.join(data_dir, "mlb_combined_2021_2025.csv")
    if not os.path.exists(combined_path):
        st.error(f"Combined data file not found:\n`{combined_path}`")
        return

    try:
        team_list = get_all_teams(data_dir)
    except Exception as e:
        st.error(f"Could not load team list: {e}")
        return

    # ── Controls row ──────────────────────────────────────────────────────────
    col_sel, col_opts, col_dc = st.columns([2, 2, 2])
    with col_sel:
        selected_team = st.selectbox("Select a team", team_list, key="team_planner_team")
    with col_opts:
        include_arb = st.checkbox(
            "Include arb-eligible players as retained",
            value=True,
            help="Arbitration players are team-controlled for 2026.",
            key="team_planner_include_arb",
        )
    with col_dc:
        include_minors = st.checkbox(
            "Include AAA/AA depth players",
            value=False,
            help="Adds minor league players from 2026 depth charts. Unmatched players assumed at league min ($0.74M).",
            key="team_planner_include_minors",
        )

    _dc_dir = get_depth_chart_dir(data_dir)

    # ── Payroll history ───────────────────────────────────────────────────────
    pay_hist  = _cached_payroll_history(data_dir)
    team_hist = pay_hist[pay_hist["Team"] == selected_team].sort_values("Year")

    if not team_hist.empty:
        hist_cols = st.columns(len(team_hist))
        for i, (_, row) in enumerate(team_hist.iterrows()):
            hist_cols[i].metric(f"{int(row['Year'])} payroll", f"${row['payroll_M']:.0f}M")

    recent_3    = team_hist.sort_values("Year").tail(3)["payroll_M"]
    hist_avg    = float(recent_3.mean()) if not recent_3.empty else 130.0
    default_bgt = min(int(hist_avg * 1.05 / 5) * 5, 400)

    budget_M = st.slider(
        "2026 total budget ($M)", min_value=40, max_value=400,
        value=default_bgt, step=5,
        help="Default is ~5% above the team's recent average payroll.",
        key="team_planner_budget",
    )

    # ── Build scenario ────────────────────────────────────────────────────────
    roster_slots_json = _json.dumps(base_cfg["roster_slots"], sort_keys=True)
    combined_hash     = _file_hash(combined_path)

    with st.spinner(f"Loading {selected_team} roster ..."):
        try:
            scenario = _cached_team_scenario(
                data_dir=data_dir, team=selected_team,
                combined_hash=combined_hash,
                roster_slots_json=roster_slots_json,
                market_dpw_M=float(base_cfg.get("market_dpw_M", 5.5)),
                include_arb=include_arb,
                budget_override_M=float(budget_M),
                depth_chart_dir=_dc_dir,
                include_minors=include_minors,
            )
        except Exception as e:
            st.error(f"Error building scenario: {e}")
            st.exception(e)
            return

    committed_M         = scenario["committed_payroll_M"]
    available_M         = scenario["available_budget_M"]
    remaining           = scenario["remaining_slots"]
    depth_df            = scenario.get("depth_chart_df", None)
    depth_total         = scenario.get("depth_total_M", 0.0)
    depth_avail         = scenario.get("depth_available_M", available_M)
    minors_df           = scenario.get("minors_df", None)
    minors_40man_df     = scenario.get("minors_40man_df", pd.DataFrame())
    minors_40man_cost_M = float(scenario.get("minors_40man_cost_M", 0.0))
    _mkt_rate           = float(base_cfg.get("market_dpw_M", 5.5))

    # ── Hero panel values ─────────────────────────────────────────────────────
    _has_dc = _dc_dir and depth_df is not None and not depth_df.empty
    if _has_dc:
        _hero_war    = float(depth_df["proj_WAR"].fillna(0).sum()) if "proj_WAR" in depth_df.columns else 0.0
        _hero_comm   = depth_total - minors_40man_cost_M
        _hero_avail  = depth_avail
    else:
        _hero_war    = (
            scenario["locked_df"]["war_2025"].clip(lower=0).sum()
            + (scenario["arb_df"]["war_2025"].clip(lower=0).sum() if include_arb else 0.0)
        )
        _hero_comm   = committed_M
        _hero_avail  = available_M

    _locked_rows: list[pd.DataFrame] = []
    for _df, _ in [
        (scenario.get("locked_df", pd.DataFrame()), "locked"),
        (scenario.get("arb_df",    pd.DataFrame()), "arb"),
    ]:
        if _df.empty:
            continue
        _tmp = _df.copy()
        _tmp["_sal_M"] = pd.to_numeric(
            _tmp.get("sal_2026_M", _tmp.get("sal_2025_M", pd.Series(dtype=float))), errors="coerce"
        ).fillna(0)
        _tmp["_war"]   = pd.to_numeric(_tmp.get("war_2025", pd.Series(dtype=float)), errors="coerce").fillna(0)
        _locked_rows.append(_tmp)

    _hero_surplus = 0.0
    if _locked_rows:
        _all_locked   = pd.concat(_locked_rows, ignore_index=True)
        _hero_surplus = float(
            (_all_locked["_war"].clip(lower=0) * _mkt_rate - _all_locked["_sal_M"]).sum()
        )

    _open_slots = sum(remaining.values())

    # ── Hero panel ────────────────────────────────────────────────────────────
    _render_hero_panel(
        selected_team, budget_M, _hero_comm, _hero_avail,
        _hero_war, _open_slots, _hero_surplus,
    )

    # ── Sticky summary bar ────────────────────────────────────────────────────
    _inject_sticky_bar(selected_team, budget_M, _hero_comm, _hero_avail, _hero_war, _open_slots)

    # ── Main workflow tabs ────────────────────────────────────────────────────
    tab_overview, tab_moves, tab_contracts, tab_optimizer, tab_history = st.tabs([
        "📋 Roster Overview",
        "🎯 Offseason Moves",
        "📑 Contracts",
        "⚙️ Optimizer",
        "📊 Roster History",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Roster Overview
    # ══════════════════════════════════════════════════════════════════════════
    with tab_overview:
        if _has_dc:
            _render_payroll_charts(depth_df, scenario, budget_M, _mkt_rate)
            st.markdown("---")

            with st.expander("2026 Projected 26-Man Roster (Depth Chart)", expanded=True):
                dc_show = depth_df[[
                    c for c in ["Player","pos_group","age","proj_WAR","sal_2026_M","contract_status","salary_source"]
                    if c in depth_df.columns
                ]].copy().rename(columns={
                    "pos_group":       "Position",
                    "proj_WAR":        "Proj WAR",
                    "sal_2026_M":      "2026 Salary ($M)",
                    "contract_status": "Status",
                    "salary_source":   "Source",
                })
                st.dataframe(
                    dc_show.style.format(
                        {"Proj WAR": "{:.1f}", "2026 Salary ($M)": "${:.2f}M", "age": "{:.0f}"},
                        na_rep="—",
                    ),
                    use_container_width=True, hide_index=True,
                )
                lg_min_count = (depth_df["salary_source"] == "League Min").sum() if "salary_source" in depth_df.columns else 0
                st.caption(
                    f"26-man payroll: **${_hero_comm:.1f}M**  "
                    f"({lg_min_count} player(s) at league minimum $0.74M)"
                )

            if not minors_40man_df.empty:
                with st.expander(
                    f"40-Man Optioned to Minors — {len(minors_40man_df)} players  "
                    f"(${minors_40man_cost_M:.1f}M at league min, counted in total spend)",
                    expanded=False,
                ):
                    st.caption(
                        "On the 40-man but assigned to AAA/AA. Salary counted in total committed payroll, "
                        "**excluded from charts**."
                    )
                    _40m_cols = [c for c in ["Player","pos_group","age","proj_level","max_level"] if c in minors_40man_df.columns]
                    st.dataframe(
                        minors_40man_df[_40m_cols].rename(columns={
                            "pos_group": "Position", "proj_level": "Proj Level", "max_level": "Max Level",
                        }).style.format({"age": "{:.0f}"}, na_rep="—"),
                        use_container_width=True, hide_index=True,
                    )

            if include_minors and minors_df is not None and not minors_df.empty:
                with st.expander(f"Prospects (not on 40-man) — {len(minors_df)} players", expanded=False):
                    m_cols = [c for c in ["Player","pos_group","age","proj_level","max_level"] if c in minors_df.columns]
                    st.dataframe(
                        minors_df[m_cols].rename(columns={
                            "pos_group": "Position", "proj_level": "Proj Level", "max_level": "Max Level",
                        }),
                        use_container_width=True, hide_index=True,
                    )
        else:
            _fc1, _fc2, _fc3, _fc4 = st.columns(4)
            _fc1.metric("2026 budget",        f"${budget_M}M")
            _fc2.metric("Committed payroll",  f"${committed_M:.1f}M")
            _fc3.metric("Available to spend", f"${available_M:.1f}M")
            _fc4.metric("Open roster slots",  str(_open_slots))
            if _dc_dir is None:
                st.caption("Depth chart directory not found — showing payroll-only view.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Offseason Moves
    # ══════════════════════════════════════════════════════════════════════════
    with tab_moves:
        _payroll_dir_26 = os.path.join(_ROOT_DIR, "2026 Payroll")
        _player_pool_26 = pd.DataFrame()
        if os.path.exists(_payroll_dir_26):
            try:
                _player_pool_26 = _cached_2026_payroll(
                    _payroll_dir_26, combined_path, _dir_hash(_payroll_dir_26)
                )
            except Exception:
                pass
        _render_team_moves(scenario, _player_pool_26, selected_team, _hero_avail, _mkt_rate)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Contracts
    # ══════════════════════════════════════════════════════════════════════════
    with tab_contracts:
        def _fmt_roster_df(df: pd.DataFrame, extra_cols=None) -> pd.DataFrame:
            base_cols = ["Player", "pos_group", "age", "war_2025", "sal_2025_M", "Contract"]
            if extra_cols:
                base_cols = base_cols + extra_cols
            show = [c for c in base_cols if c in df.columns]
            return df[show].rename(columns={
                "pos_group":  "Position",
                "war_2025":   "2025 WAR",
                "sal_2025_M": "2025 Salary ($M)",
            })

        with st.expander(f"Under Contract for 2026 — {len(scenario['locked_df'])} players", expanded=True):
            if scenario["locked_df"].empty:
                st.info("No players with committed 2026 contracts found.")
            else:
                ldf     = scenario["locked_df"].copy()
                display = _fmt_roster_df(ldf, extra_cols=["sal_2026_M"])
                display = display.rename(columns={"sal_2026_M": "2026 Salary ($M)"})
                st.dataframe(
                    display.style.format({
                        "2025 WAR": "{:.1f}", "2025 Salary ($M)": "${:.2f}M",
                        "2026 Salary ($M)": "${:.2f}M", "age": "{:.0f}",
                    }),
                    use_container_width=True,
                )
                st.caption(f"Total locked 2026 payroll: **${ldf['sal_2026_M'].sum():.1f}M**")

        with st.expander(f"Arbitration-Eligible — {len(scenario['arb_df'])} players", expanded=True):
            if scenario["arb_df"].empty:
                st.info("No arbitration-eligible players found.")
            else:
                adf     = scenario["arb_df"].copy()
                display = _fmt_roster_df(adf, extra_cols=["arb_tier", "proj_arb_cost_M", "recommendation"])
                display = display.rename(columns={
                    "proj_arb_cost_M": "Est. 2026 Arb Cost ($M)",
                    "arb_tier":        "Arb Year",
                    "recommendation":  "Recommendation",
                })

                def _color_rec(val):
                    if val == "Keep":       return "background-color: #1a4731; color: white"
                    if val == "Non-tender": return "background-color: #4a1a1a; color: white"
                    return ""

                styled = display.style.format({
                    "2025 WAR": "{:.1f}", "2025 Salary ($M)": "${:.2f}M",
                    "Est. 2026 Arb Cost ($M)": "${:.2f}M", "age": "{:.0f}",
                })
                if "Recommendation" in display.columns:
                    styled = styled.applymap(_color_rec, subset=["Recommendation"])
                st.dataframe(styled, width="stretch")
                keep_count = (adf.get("recommendation", pd.Series()) == "Keep").sum()
                st.caption(
                    f"Est. arb payroll if all retained: **${adf['proj_arb_cost_M'].sum():.1f}M** | "
                    f"Suggested keeps: **{keep_count}** of {len(adf)}"
                )
                if include_arb:
                    st.info("Arb players counted as retained. Non-tender candidates free up salary.")

        with st.expander(f"Expiring / Becoming Free Agent — {len(scenario['expiring_df'])} players", expanded=False):
            if scenario["expiring_df"].empty:
                st.info("No expiring players found.")
            else:
                edf     = scenario["expiring_df"].copy()
                display = _fmt_roster_df(edf)
                st.dataframe(
                    display.style.format({
                        "2025 WAR": "{:.1f}", "2025 Salary ($M)": "${:.2f}M", "age": "{:.0f}",
                    }),
                    use_container_width=True,
                )
                st.caption("High-WAR expiring players may be worth re-signing.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Optimizer
    # ══════════════════════════════════════════════════════════════════════════
    with tab_optimizer:
        if not remaining:
            st.success("All roster slots are filled by locked/arb players — nothing to optimize.")
        else:
            _oc1, _oc2 = st.columns([1, 2])
            with _oc1:
                st.markdown(f"**{_open_slots} Open Slots to Fill**")
                slot_df = pd.DataFrame([{"Slot": k, "Count": v} for k, v in sorted(remaining.items())])
                st.dataframe(slot_df, width="stretch", hide_index=True)
            with _oc2:
                st.markdown(
                    f"Budget remaining after committed payroll: **${_hero_avail:.1f}M**  \n"
                    "The optimizer will suggest the best available FA/Arb additions from the full "
                    "player pool to fill open slots within budget."
                )

            run_team_opt = st.button("Run Offseason Optimizer", type="primary", key="team_planner_run_opt")

            if "team_opt_result" not in st.session_state:
                st.session_state["team_opt_result"] = {}

            if run_team_opt:
                team_cfg = dict(base_cfg)
                team_cfg["budget_M"]       = float(_hero_avail)
                team_cfg["roster_slots"]   = dict(remaining)
                team_cfg["market_mode"]    = "all"
                team_cfg["optimizer_mode"] = "archetype"

                team_players     = set(scenario["roster_status_df"]["Player"].tolist())
                combined_df_full = _read_csv(combined_path, low_memory=False)
                filtered_raw     = combined_df_full[~combined_df_full["Player"].isin(team_players)].copy()

                with st.spinner("Building projections & running optimizer ..."):
                    try:
                        team_proj      = make_projections(filtered_raw, team_cfg)
                        team_proj_live = team_proj[
                            team_proj["proj_WAR"] >= float(team_cfg.get("min_war_threshold", 0.0))
                        ].copy()
                        team_arch      = build_archetype_definitions(team_proj_live)
                        team_proj_arch = assign_archetypes(team_proj_live)
                        opt            = run_optimizer(team_arch, team_cfg, projected_df=team_proj_arch)
                        st.session_state["team_opt_result"] = {
                            "opt": opt, "proj": team_proj_live, "arch": team_arch,
                            "cfg": team_cfg, "team": selected_team,
                        }
                    except Exception as e:
                        st.error(f"Optimizer error: {e}")
                        st.exception(e)

            team_opt_data = st.session_state.get("team_opt_result", {})
            opt           = team_opt_data.get("opt")

            if opt is not None and team_opt_data.get("team") == selected_team:
                rdf = opt.roster_df
                if rdf.empty:
                    st.warning(f"Optimizer returned no roster. Status: {opt.status}")
                else:
                    total_w  = rdf["war_mean"].sum()
                    total_c  = rdf["cost_mean"].sum()
                    full_war = total_w + (
                        scenario["locked_df"]["war_2025"].clip(lower=0).sum()
                        + (scenario["arb_df"]["war_2025"].clip(lower=0).sum() if include_arb else 0.0)
                    )

                    res1, res2, res3, res4 = st.columns(4)
                    res1.metric("Additions WAR",             f"{total_w:.1f}")
                    res2.metric("Additions cost",            f"${total_c:.1f}M")
                    res3.metric("Projected full-roster WAR", f"{full_war:.1f}")
                    res4.metric("Solver status",             opt.status)

                    st.markdown("#### Recommended Additions")
                    _tp_rdf  = rdf.copy()
                    _tp_rdf.insert(1, "Role", _tp_rdf["archetype_id"].apply(_arch_label))
                    _tp_cols = [c for c in ["slot","Role","pos_group","war_mean","war_sd","cost_mean","n_players"]
                                if c in _tp_rdf.columns]
                    st.dataframe(
                        _tp_rdf[_tp_cols].rename(columns={
                            "slot": "Slot", "pos_group": "Pos Group",
                            "war_mean": "Proj WAR", "war_sd": "WAR SD",
                            "cost_mean": "Est Cost", "n_players": "# Players",
                        }).style.format({"Proj WAR": "{:.2f}", "WAR SD": "{:.2f}", "Est Cost": "${:.1f}M"}),
                        use_container_width=True,
                    )

                    st.markdown("#### Full 2026 Projected Roster")
                    locked_summary = scenario["locked_df"][
                        ["Player", "pos_group", "war_2025", "sal_2026_M"]
                    ].copy().rename(columns={
                        "war_2025": "proj_WAR", "sal_2026_M": "cost_M", "pos_group": "Position",
                    })
                    locked_summary["source"]        = "Under Contract"
                    locked_summary["assigned_slot"] = locked_summary.apply(
                        lambda r: scenario["assignments_df"].set_index("Player")
                        .get("assigned_slot", pd.Series(dtype=str)).get(r["Player"], "-"),
                        axis=1,
                    )

                    if include_arb and not scenario["arb_df"].empty:
                        arb_keep = scenario["arb_df"][
                            scenario["arb_df"].get("recommendation", pd.Series()) != "Non-tender"
                        ][["Player", "pos_group", "war_2025", "proj_arb_cost_M"]].rename(columns={
                            "war_2025": "proj_WAR", "proj_arb_cost_M": "cost_M", "pos_group": "Position",
                        })
                        arb_keep["source"]        = "Arb (Retained)"
                        arb_keep["assigned_slot"] = "-"
                        locked_summary = pd.concat([locked_summary, arb_keep], ignore_index=True)

                    st.dataframe(
                        locked_summary[["Player","Position","assigned_slot","proj_WAR","cost_M","source"]]
                        .rename(columns={"assigned_slot": "Slot", "proj_WAR": "Proj WAR", "cost_M": "Cost ($M)"})
                        .style.format({"Proj WAR": "{:.1f}", "Cost ($M)": "${:.2f}M"}),
                        use_container_width=True,
                    )
                    st.caption(
                        f"Total projected 2026 payroll: **${committed_M + total_c:.1f}M** "
                        f"(committed ${committed_M:.1f}M + additions ${total_c:.1f}M)"
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — Roster History (value analysis)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_history:
        _hist_all = _cached_player_history(combined_path, _file_hash(combined_path))
        _th = _hist_all[_hist_all["Team"] == selected_team].copy()
        _th["WAR_Total"] = pd.to_numeric(_th["WAR_Total"], errors="coerce").fillna(0.0)
        _th["Year"]      = pd.to_numeric(_th["Year"],      errors="coerce")
        _th["Salary_M"]  = pd.to_numeric(_th.get("Salary_M", pd.Series(dtype=float)), errors="coerce").fillna(0.72)

        _psumm = (
            _th.groupby("Player").agg(
                Seasons    =("Year",      "count"),
                Total_WAR  =("WAR_Total", "sum"),
                Peak_WAR   =("WAR_Total", "max"),
                Total_Sal_M=("Salary_M",  "sum"),
            ).reset_index()
        )
        _psumm["Market_Val_M"] = (_psumm["Total_WAR"] * _mkt_rate).round(2)
        _psumm["Surplus_M"]    = (_psumm["Market_Val_M"] - _psumm["Total_Sal_M"]).round(2)

        _cur_players = set(scenario["locked_df"]["Player"].tolist())
        if include_arb:
            _cur_players |= set(scenario["arb_df"]["Player"].tolist())

        _cur_val = _psumm[_psumm["Player"].isin(_cur_players)].copy()
        _dep_val = _psumm[~_psumm["Player"].isin(_cur_players)].copy()

        # Q1: Current roster value
        with st.expander(
            f"Value Delivered by Current Roster (2021–2025) — {len(_cur_val)} players tracked",
            expanded=True,
        ):
            if _cur_val.empty:
                st.info("No historical data found for current players on this team.")
            else:
                _c1, _c2, _c3, _c4 = st.columns(4)
                _c1.metric("Total WAR Delivered",  f"{_cur_val['Total_WAR'].sum():.1f}")
                _c2.metric("Market Value of WAR",  f"${_cur_val['Market_Val_M'].sum():.0f}M",
                           help=f"Total WAR × ${_mkt_rate}M/WAR market rate.")
                _c3.metric("Total Salary Paid",    f"${_cur_val['Total_Sal_M'].sum():.0f}M")
                _surplus = _cur_val["Surplus_M"].sum()
                _c4.metric("Total Surplus Value",  f"${_surplus:+.0f}M",
                           delta="above market" if _surplus >= 0 else "below market",
                           delta_color="normal" if _surplus >= 0 else "inverse")

                _cv     = _cur_val.sort_values("Total_WAR", ascending=True)
                _cv_clr = ["#22c55e" if s >= 0 else "#ef4444" for s in _cv["Surplus_M"]]
                _cv_txt = [
                    f"{r['Total_WAR']:.1f} WAR  |  paid ${r['Total_Sal_M']:.0f}M  |  surplus ${r['Surplus_M']:+.0f}M"
                    for _, r in _cv.iterrows()
                ]
                _fig_cur = go.Figure(go.Bar(
                    x=_cv["Total_WAR"], y=_cv["Player"], orientation="h",
                    marker_color=_cv_clr,
                    text=_cv_txt, textposition="outside",
                    textfont=dict(size=9, color="#7aa2c0"),
                    hovertemplate="%{y}<br>Total WAR: %{x:.1f}<extra></extra>",
                ))
                _fig_cur.update_layout(**_pt(
                    title=f"{selected_team} — Current Players' Total WAR While on Team (2021–2025)",
                    xaxis=dict(title="Total WAR (cumulative)"),
                    height=max(320, len(_cv) * 28),
                    margin=dict(l=160, r=280, t=45, b=40),
                ))
                st.plotly_chart(_fig_cur, width="stretch", config={"displayModeBar": False})

                _cv_tbl = _cur_val.sort_values("Total_WAR", ascending=False)[
                    ["Player","Seasons","Total_WAR","Peak_WAR","Total_Sal_M","Market_Val_M","Surplus_M"]
                ].reset_index(drop=True)
                _cv_tbl.columns = ["Player","Seasons","Total WAR","Peak WAR","Salary Paid $M","Market Val $M","Surplus $M"]
                st.dataframe(
                    _cv_tbl.style.format({
                        "Total WAR": "{:.1f}", "Peak WAR": "{:.1f}",
                        "Salary Paid $M": "${:.1f}M", "Market Val $M": "${:.1f}M",
                        "Surplus $M": "${:+.1f}M",
                    }, na_rep="—"),
                    hide_index=True, use_container_width=True,
                )

        # Q2: Departed player value
        _dep_pos = _dep_val[_dep_val["Total_WAR"] > 0].sort_values("Total_WAR", ascending=False)
        with st.expander(
            f"Value From Departed Players (2021–2025) — {len(_dep_pos)} tracked",
            expanded=False,
        ):
            if _dep_pos.empty:
                st.info("No departed players with positive WAR found.")
            else:
                _d1, _d2, _d3 = st.columns(3)
                _d1.metric("Total WAR From Departed", f"{_dep_pos['Total_WAR'].sum():.1f}")
                _d2.metric("Total Salary Paid",        f"${_dep_pos['Total_Sal_M'].sum():.0f}M")
                _d3.metric("Market Value Delivered",   f"${_dep_pos['Market_Val_M'].sum():.0f}M",
                           help="WAR × market rate — how much those seasons were worth.")

                _dv     = _dep_pos.head(20).sort_values("Total_WAR", ascending=True)
                _dv_clr = ["#22c55e" if s >= 0 else "#ef4444" for s in _dv["Surplus_M"]]
                _dv_txt = [
                    f"{r['Total_WAR']:.1f} WAR  |  paid ${r['Total_Sal_M']:.0f}M  |  surplus ${r['Surplus_M']:+.0f}M"
                    for _, r in _dv.iterrows()
                ]
                _fig_dep = go.Figure(go.Bar(
                    x=_dv["Total_WAR"], y=_dv["Player"], orientation="h",
                    marker_color=_dv_clr,
                    text=_dv_txt, textposition="outside",
                    textfont=dict(size=9, color="#7aa2c0"),
                    hovertemplate="%{y}<br>Total WAR on team: %{x:.1f}<extra></extra>",
                ))
                _fig_dep.update_layout(**_pt(
                    title=f"{selected_team} — Departed Players' Total WAR While on Team (Top 20)",
                    xaxis=dict(title="Total WAR (while on this team)"),
                    height=max(320, len(_dv) * 28),
                    margin=dict(l=160, r=280, t=45, b=40),
                ))
                st.plotly_chart(_fig_dep, width="stretch", config={"displayModeBar": False})

                _dv_tbl = _dep_pos.head(20)[
                    ["Player","Seasons","Total_WAR","Peak_WAR","Total_Sal_M","Market_Val_M","Surplus_M"]
                ].reset_index(drop=True)
                _dv_tbl.columns = ["Player","Seasons","Total WAR","Peak WAR","Salary Paid $M","Market Val $M","Surplus $M"]
                st.dataframe(
                    _dv_tbl.style.format({
                        "Total WAR": "{:.1f}", "Peak WAR": "{:.1f}",
                        "Salary Paid $M": "${:.1f}M", "Market Val $M": "${:.1f}M",
                        "Surplus $M": "${:+.1f}M",
                    }, na_rep="—"),
                    hide_index=True, use_container_width=True,
                )


# ---------------------------------------------------------------------------
# Roster Optimizer page (combines Optimizer + Team Planner as sub-tabs)
# ---------------------------------------------------------------------------

def _render_roster_optimizer_page(base_cfg: dict | None = None):
    """Roster Optimizer: sub-tabs for Team Planner and the full Optimizer."""
    st.markdown("### 🔧 Roster Optimizer")

    t_planner, t_optimizer = st.tabs(["🗓 Team Planner", "⚙️ General Optimizer"])

    with t_planner:
        _render_team_planner(base_cfg)

    with t_optimizer:
        _render_optimizer_page()


# ---------------------------------------------------------------------------
# Interactive efficiency frontier (mirrors R Shiny app-2-2.R)
# ---------------------------------------------------------------------------

def _render_efficiency_frontier():
    """WAR vs Salary efficiency frontier — interactive Streamlit port of app-2-2.R."""
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess

    # ── data path ────────────────────────────────────────────────────────────
    if _R2_MODE:
        _cpath = _data_url("data/mlb_combined_2021_2025.csv")
    else:
        if not os.path.exists(_DEFAULT_CONFIG):
            st.warning("Config file not found — cannot load player data for frontier analysis.")
            return
        _bcfg     = _load_base_config(_DEFAULT_CONFIG)
        _sal_abs  = _resolve_data_path(_bcfg["raw_salary_war_path"], _DEFAULT_CONFIG)
        _data_dir = os.path.dirname(_sal_abs)
        _cpath    = os.path.join(_data_dir, "mlb_combined_2021_2025.csv")
        if not os.path.exists(_cpath):
            st.warning(f"Combined data file not found: `{_cpath}`")
            return

    # ── cached load ───────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False, ttl=86400)
    def _load_frontier(path: str, fhash: str) -> pd.DataFrame:
        _AL = {"BAL","BOS","CHW","CLE","DET","HOU","KCR","LAA","MIN","NYY","ATH","SEA","TBR","TEX","TOR"}
        _NL = {"ARI","ATL","CHC","CIN","COL","LAD","MIA","MIL","NYM","PHI","PIT","SDP","SFG","STL","WSN"}
        _PITCH = {"SP","RP","TWP","P"}
        df = _read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["Salary_M"]  = pd.to_numeric(df.get("Salary", pd.Series(dtype=float)), errors="coerce") / 1_000_000
        df["WAR_Total"] = pd.to_numeric(df.get("WAR_Total", 0), errors="coerce").fillna(0.0)
        # 2023 data has WAR_Total=0 for many players where the WAR column has the real value
        _war_raw = pd.to_numeric(df.get("WAR", np.nan), errors="coerce")
        _fix_mask = (df["WAR_Total"] == 0) & _war_raw.notna() & (_war_raw != 0)
        df.loc[_fix_mask, "WAR_Total"] = _war_raw[_fix_mask]
        df["Age"]       = pd.to_numeric(df.get("Age", np.nan), errors="coerce")
        df["Year"]      = pd.to_numeric(df.get("Year", np.nan), errors="coerce").astype("Int64")
        for col in ("Stage_Clean", "Team", "Position"):
            df[col] = df.get(col, pd.Series("Unknown", index=df.index)).fillna("Unknown").replace("", "Unknown")
        df["League"]      = df["Team"].apply(lambda t: "AL" if t in _AL else ("NL" if t in _NL else "Unknown"))
        df["Player_Type"] = df["Position"].apply(lambda p: "Pitcher" if str(p).strip().upper() in _PITCH else "Position Player")
        df = df[df["Salary_M"].notna() & (df["Salary_M"] > 0)].copy()
        # Sort by salary desc, then WAR_Total desc as tiebreaker so artifact duplicate rows
        # (same salary, WAR_Total=0) lose to the real row with actual WAR data
        df = (df.sort_values(["Salary_M", "WAR_Total"], ascending=[False, False], kind="mergesort")
               .drop_duplicates(["Player", "Year"], keep="first")
               .reset_index(drop=True))
        return df

    raw = _load_frontier(_cpath, _file_hash(_cpath))

    # ── Data ranges ───────────────────────────────────────────────────────────
    _all_teams  = ["All Teams"] + sorted(raw["Team"].dropna().unique())
    _all_stages = sorted(raw["Stage_Clean"].dropna().unique())
    _all_pos    = sorted(raw["Position"].dropna().unique())
    _all_years  = sorted(raw["Year"].dropna().unique())
    _age_bounds = (int(raw["Age"].min(skipna=True)), int(raw["Age"].max(skipna=True))) if raw["Age"].notna().any() else (18, 45)
    _war_bounds = (float(raw["WAR_Total"].min()), float(raw["WAR_Total"].max()))
    _sal_bounds = (float(raw["Salary_M"].min()), float(raw["Salary_M"].max()))

    # ── CSS: compact filter card + polished tabs ──────────────────────────────
    st.markdown("""<style>
/* Compact filter column labels (EF page) */
div[data-testid='column']:first-of-type label{
  font-size:0.70rem!important;margin-bottom:0!important;
  line-height:1.2!important;color:#7a9ebc!important;}
div[data-testid='column']:first-of-type .stSelectbox>div,
div[data-testid='column']:first-of-type .stMultiSelect>div{font-size:0.70rem!important;}
div[data-testid='column']:first-of-type p{font-size:0.70rem!important;margin:0.05rem 0!important;}
div[data-testid='column']:first-of-type hr{margin:0.2rem 0!important;border-color:#1e3250!important;}
div[data-testid='column']:first-of-type .stMultiSelect [data-baseweb='select']>div{
  padding:1px 4px!important;min-height:26px!important;}
div[data-testid='column']:first-of-type .stSelectbox [data-baseweb='select']>div{
  padding:2px 6px!important;min-height:26px!important;}
div[data-testid='column']:first-of-type .stSlider{
  padding-top:0!important;padding-bottom:0.05rem!important;}
</style>""", unsafe_allow_html=True)

    col_f, col_m = st.columns([0.82, 4.5])

    # ── Filter card ───────────────────────────────────────────────────────────
    with col_f:
        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:#4a687e;'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Filters</div>',
            unsafe_allow_html=True,
        )
        _league_sel = st.selectbox("League", ["Both Leagues", "AL", "NL"], key="ef_league")
        _year_sel   = st.multiselect(
            "Year(s)", [str(y) for y in _all_years],
            default=[str(y) for y in _all_years], key="ef_years",
        )
        _team_sel   = st.multiselect("Team(s)", _all_teams, default=["All Teams"], key="ef_teams")
        _stage_sel  = st.multiselect("Stage", _all_stages, default=_all_stages, key="ef_stages")

        with st.expander("Position / Player type", expanded=False):
            _pos_sel   = st.multiselect("Position(s)", _all_pos, default=_all_pos, key="ef_pos")
            _ptype_sel = st.selectbox("Player Type", ["All", "Position Player", "Pitcher"], key="ef_ptype")

        with st.expander("Range filters", expanded=False):
            _age_sel   = st.slider("Age", _age_bounds[0], _age_bounds[1], _age_bounds, key="ef_age")
            _excl_zero = st.checkbox("Excl. 0 WAR", value=False, key="ef_excl_zero")
            _war_sel   = st.slider("WAR", float(_war_bounds[0]), float(_war_bounds[1]), _war_bounds, step=0.1, key="ef_war")
            _sal_sel   = st.slider("Salary ($M)", float(_sal_bounds[0]), float(_sal_bounds[1]), _sal_bounds, step=0.5, key="ef_sal")

        st.markdown("<hr style='border-color:#1e3250;margin:8px 0;'>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:#4a687e;'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Display</div>',
            unsafe_allow_html=True,
        )
        _color_by   = st.selectbox("Color by", ["Career Stage", "Pay-Performance Ratio", "Cost/WAR vs Market"], key="ef_color")
        _reg_method = st.radio("Regression", ["OLS", "LOESS", "Poly"], horizontal=True, key="ef_reg")
        _show_reg   = st.checkbox("Trendline", value=True, key="ef_showreg")
        _shade_ovuv = st.checkbox("OV/UV zones", value=False, key="ef_shade")
        _hi_eff     = st.checkbox("Highlight efficient", value=False, key="ef_hi")
        _eff_thresh = st.slider("Efficiency %", 5, 30, 15, key="ef_thresh")

        st.markdown("<hr style='border-color:#1e3250;margin:8px 0;'>", unsafe_allow_html=True)
        if st.button("↺  Reset Filters", use_container_width=True, key="ef_reset_btn"):
            _keep = {"ef_reset_btn"}
            for _k in [k for k in st.session_state if k.startswith("ef_") and k not in _keep]:
                del st.session_state[_k]
            st.rerun()
        if st.button("✕  Clear All Filters", use_container_width=True, key="ef_clear_btn"):
            for _k in ("ef_years", "ef_teams", "ef_stages"):
                if _k in st.session_state:
                    del st.session_state[_k]
            st.rerun()

    with col_m:
        # ── Apply filters ─────────────────────────────────────────────────
        df = raw.copy()
        if _league_sel != "Both Leagues":
            df = df[df["League"] == _league_sel]
        if "All Teams" not in _team_sel and _team_sel:
            df = df[df["Team"].isin(_team_sel)]
        if _stage_sel:
            df = df[df["Stage_Clean"].isin(_stage_sel)]
        if _pos_sel:
            df = df[df["Position"].isin(_pos_sel)]
        if _year_sel:
            df = df[df["Year"].isin([int(y) for y in _year_sel])]
        if _ptype_sel != "All":
            df = df[df["Player_Type"] == _ptype_sel]
        df = df[(df["Age"] >= _age_sel[0]) & (df["Age"] <= _age_sel[1])]
        if _excl_zero:
            df = df[df["WAR_Total"] != 0]
        df = df[(df["WAR_Total"] >= _war_sel[0]) & (df["WAR_Total"] <= _war_sel[1])]
        df = df[(df["Salary_M"] >= _sal_sel[0]) & (df["Salary_M"] <= _sal_sel[1])]
        df = df.reset_index(drop=True)

        if len(df) < 5:
            st.warning(f"Only {len(df)} players match the filters. Relax some filters to see results.")
            return

        # ── Fit regression ────────────────────────────────────────────────
        x = df["WAR_Total"].values.astype(float)
        y = df["Salary_M"].values.astype(float)

        if _reg_method == "OLS":
            import statsmodels.api as _sm_api
            _Xc  = _sm_api.add_constant(x)
            _mod = _sm_api.OLS(y, _Xc).fit()
            predicted = _mod.predict(_Xc)
            _xseq = np.linspace(x.min(), x.max(), 200)
            _yseq = _mod.params[0] + _mod.params[1] * _xseq
            _reg_lbl = f"PPEL (OLS · R²={_mod.rsquared:.3f})"
        elif _reg_method == "LOESS":
            _sm = _sm_lowess(y, x, frac=0.75, return_sorted=True)
            predicted = np.interp(x, _sm[:, 0], _sm[:, 1])
            _xseq, _yseq = _sm[:, 0], _sm[:, 1]
            _reg_lbl = "PPEL (LOESS)"
        else:
            _coeffs = np.polyfit(x, y, 2)
            _pfn    = np.poly1d(_coeffs)
            predicted = _pfn(x)
            _xseq = np.linspace(x.min(), x.max(), 200)
            _yseq = _pfn(_xseq)
            _reg_lbl = "PPEL (Poly deg 2)"

        df = df.copy()
        df["predicted"]          = predicted
        df["residual"]           = df["Salary_M"] - df["predicted"]
        df["residual_pct"]       = df["residual"].rank(pct=True) * 100
        df["PPR"]                = df["Salary_M"] / df["predicted"].clip(lower=0.01)
        df["cost_per_WAR"]       = np.where(df["WAR_Total"] > 0, df["Salary_M"] / df["WAR_Total"], np.nan)
        df["mkt_cost_per_WAR"]   = np.where(df["WAR_Total"] > 0, df["predicted"] / df["WAR_Total"], np.nan)
        df["cost_per_WAR_ratio"] = (df["cost_per_WAR"] / df["mkt_cost_per_WAR"].clip(lower=0.01)).fillna(1.0)
        df["_efficient"]         = df["residual_pct"] <= _eff_thresh

        _STAGE_COLORS = {
            "Free Agent":      "#06d6a0",
            "Arbitration":     "#fbbf24",
            "Pre-Arbitration": "#38bdf8",
        }

        # ── KPI values for page header ────────────────────────────────────
        _n_shown  = len(df)
        _med_dpw  = float((df["Salary_M"] / df["WAR_Total"].replace(0, np.nan)).median())
        _flt_tms  = "All teams" if "All Teams" in _team_sel else f"{len(_team_sel)} team(s)"
        _flt_yrs  = f"{min(_year_sel)}–{max(_year_sel)}" if _year_sel else "All"
        _r2_txt   = f"R²={_mod.rsquared:.3f}" if _reg_method == "OLS" else _reg_lbl

        # ── Page header card ──────────────────────────────────────────────
        st.markdown(f"""<div class="ef-hdr" style="background:linear-gradient(135deg,#0f2035,#0d1b2a);border:1px solid #1e3a5c;
border-radius:10px;padding:16px 20px;margin-bottom:14px;display:flex;align-items:center;
justify-content:space-between;gap:16px;flex-wrap:wrap;">
  <div style="flex:1;min-width:200px;">
    <div style="font-size:20px;font-weight:800;color:#e8f4ff;margin-bottom:5px;">Player Analysis</div>
    <div style="font-size:13px;color:#93b8d8;line-height:1.6;">
      Explore how every MLB player's salary compares to their on-field production (fWAR).
      Use the tabs below to view the market value regression, multi-year efficiency, age curves,
      team breakdowns, and player stability ratings. Use the filters on the left to narrow
      by team, year, position, or career stage.
    </div>
  </div>
  <div class="ef-hdr-stats" style="display:flex;gap:10px;flex-shrink:0;flex-wrap:wrap;">
    <div style="background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;">
      <div style="font-size:10px;color:#7a9ebc;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Players</div>
      <div style="font-size:18px;font-weight:700;color:#e8f4ff;">{_n_shown:,}</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;">
      <div style="font-size:10px;color:#7a9ebc;letter-spacing:1px;margin-bottom:2px;">MEDIAN $/fWAR</div>
      <div style="font-size:18px;font-weight:700;color:#e8f4ff;">${_med_dpw:.1f}M</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;">
      <div style="font-size:10px;color:#7a9ebc;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Active Filters</div>
      <div style="font-size:12px;font-weight:600;color:#a8c8e8;">{_flt_tms} · {_flt_yrs}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── MLBAM ID map for headshot hover images ────────────────────────
        _mlb_ids = _cached_mlbam_lookup(_RAZZBALL_PATH)

        # ── Tabs ──────────────────────────────────────────────────────────
        t1, t2, t3, t5, t6, t7, t8 = st.tabs([
            "Cost Effective Line",
            "PPEL",
            "Age Trajectory",
            "Efficient Players",
            "Residual Analysis",
            "Pre-Arb Explorer",
            "WAR Stability",
        ])

        # ── Tab 1 — Cost Effective Line (PPEL 1/3/5-year views) ──────────
        with t1:
            st.markdown(
                "<div style='background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;"
                "padding:14px 18px;margin-bottom:12px;'>"
                "<div style='font-size:1rem;font-weight:700;color:#d6e8f8;margin-bottom:6px;'>Cost Effective Line</div>"
                "<div style='font-size:0.85rem;color:#93b8d8;line-height:1.7;'>"
                "This scatter plot compares every player's <b>actual salary</b> (Y-axis) against their "
                "<b>fWAR production</b> (X-axis). The <span style='color:#f4a261;font-weight:600;'>orange trendline</span> "
                "is the market average — it shows what teams typically pay for a given level of production.<br><br>"
                "<b>How to read it:</b> Dots <span style='color:#22c55e;font-weight:600;'>below</span> the line "
                "are bargains (the player produces more than their salary suggests). "
                "Dots <span style='color:#ef4444;font-weight:600;'>above</span> the line are overpaid relative to output. "
                "The further from the line, the bigger the gap between actual and market salary.<br><br>"
                "<b>What is fWAR?</b> WAR estimates a player's value in terms of extra wins they provide compared to a "
                "replacement-level player — a low-cost minor leaguer easily found to fill the same position. "
                "In 2025, the average player had an fWAR of 0.7, All-Stars averaged 3.2+, and MVP candidates reached 4.0+.<br><br>"
                "<b>Key terms:</b> <b>PPR</b> = Pay-to-Performance Ratio (career fWAR ÷ total contract $M; "
                "below 1.0 = underpaid, above 1.0 = overpaid). <b>Residual</b> = actual salary minus the trendline's "
                "predicted salary (negative = team is getting a deal)."
                "</div></div>",
                unsafe_allow_html=True,
            )
            # ── Frontier Summary card ──────────────────────────────────────
            _stage_ppr     = df.groupby("Stage_Clean")["PPR"].median().sort_values()
            _most_eff_stg  = _stage_ppr.index[0]  if len(_stage_ppr) > 0 else "—"
            _least_eff_stg = _stage_ppr.index[-1] if len(_stage_ppr) > 0 else "—"
            _most_eff_ppr  = float(_stage_ppr.iloc[0])  if len(_stage_ppr) > 0 else 1.0
            _least_eff_ppr = float(_stage_ppr.iloc[-1]) if len(_stage_ppr) > 0 else 1.0
            _underpaid_n   = int((df["PPR"] < 1.0).sum())
            _overpaid_n    = int((df["PPR"] > 1.0).sum())

            st.markdown(f"""<div style="background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;padding:14px 16px;margin-bottom:12px;">
  <div style="font-size:10px;font-weight:700;color:#5a8aaa;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Career Stages Explained</div>
  <div class="ef-summary-grid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">
    <div style="background:#0d1b2a;border:1px solid #14532d;border-top:3px solid #22c55e;border-radius:8px;padding:10px 12px;">
      <div style="font-size:13px;font-weight:700;color:#22c55e;margin-bottom:4px;">Pre-Arbitration</div>
      <div style="font-size:10px;color:#7a9ebc;line-height:1.5;">0–3 years service time. Salary near league minimum (~$740K). Teams control rights — often the best value in baseball.</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #14b8a6;border-top:3px solid #14b8a6;border-radius:8px;padding:10px 12px;">
      <div style="font-size:13px;font-weight:700;color:#14b8a6;margin-bottom:4px;">Arbitration</div>
      <div style="font-size:10px;color:#7a9ebc;line-height:1.5;">3–6 years service time. Salary negotiated or set by arbitration hearing. Pay rises based on prior performance.</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #3b6fd4;border-top:3px solid #60a5fa;border-radius:8px;padding:10px 12px;">
      <div style="font-size:13px;font-weight:700;color:#60a5fa;margin-bottom:4px;">Free Agent</div>
      <div style="font-size:10px;color:#7a9ebc;line-height:1.5;">6+ years service time. Player signs on the open market. Full market-rate salary — highest cost per WAR.</div>
    </div>
  </div>
  <div style="display:flex;gap:16px;margin-top:10px;">
    <div style="font-size:10px;color:#7a9ebc;"><span style="color:#22c55e;font-weight:700;">{_underpaid_n}</span> underpaid / <span style="color:#ef4444;font-weight:700;">{_overpaid_n}</span> overpaid (PPR vs 1.0)</div>
    <div style="font-size:10px;color:#7a9ebc;">{_r2_txt} · {_reg_method} · N={_n_shown:,}</div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Chart ─────────────────────────────────────────────────────
            _hover = df.apply(lambda r: (
                f"<b>{r['Player']}</b><br>"
                + f"{r['Team']}  {r['Year']}<br>"
                + f"WAR: {r['WAR_Total']:.1f}  |  Salary: ${r['Salary_M']:.2f}M<br>"
                + f"Expected: ${r['predicted']:.2f}M<br>"
                + f"PPR: {r['PPR']:.3f}  |  {r['Stage_Clean']}"
            ), axis=1)
            _sizes = np.where(df["_efficient"] & _hi_eff, 13, 7).tolist()

            fig1 = go.Figure()

            if _shade_ovuv and _show_reg and len(_xseq) > 1:
                _std = float(df["residual"].std())
                fig1.add_trace(go.Scatter(
                    x=np.concatenate([_xseq, _xseq[::-1]]),
                    y=np.concatenate([_yseq + _std, _yseq[::-1]]),
                    fill="toself", fillcolor="rgba(239,68,68,0.07)",
                    line=dict(color="rgba(0,0,0,0)"), name="Overpaid zone", hoverinfo="skip",
                ))
                fig1.add_trace(go.Scatter(
                    x=np.concatenate([_xseq, _xseq[::-1]]),
                    y=np.concatenate([_yseq, (_yseq - _std)[::-1]]),
                    fill="toself", fillcolor="rgba(34,197,94,0.07)",
                    line=dict(color="rgba(0,0,0,0)"), name="Underpaid zone", hoverinfo="skip",
                ))

            if _color_by == "Career Stage":
                for _stg, _grp in df.groupby("Stage_Clean"):
                    _c   = _STAGE_COLORS.get(_stg, "#94a3b8")
                    _idx = _grp.index
                    fig1.add_trace(go.Scatter(
                        x=_grp["WAR_Total"], y=_grp["Salary_M"],
                        mode="markers", name=_stg,
                        marker=dict(
                            color=_c, size=[_sizes[i] for i in _idx], opacity=0.78,
                            line=dict(color="#22c55e" if _hi_eff else "rgba(0,0,0,0)", width=1.2),
                        ),
                        text=_hover[_idx], hovertemplate="%{text}<extra></extra>",
                        customdata=_grp["Player"].values,
                    ))
            else:
                _col_vals = df["PPR"] if _color_by == "Pay-Performance Ratio" else df["cost_per_WAR_ratio"]
                _cscale   = [
                    [0.0, "#22c55e"], [0.33, "#86efac"], [0.5, "#ffffff"],
                    [0.67, "#fca5a5"], [1.0, "#ef4444"],
                ]
                fig1.add_trace(go.Scatter(
                    x=df["WAR_Total"], y=df["Salary_M"],
                    mode="markers", name="Players", showlegend=False,
                    marker=dict(
                        color=_col_vals, colorscale=_cscale, cmid=1.0,
                        size=_sizes, opacity=0.78,
                        colorbar=dict(title=_color_by, thickness=12, tickfont=dict(color="#7aa2c0")),
                        showscale=True,
                        line=dict(color="#22c55e" if _hi_eff else "rgba(0,0,0,0)", width=1.2),
                    ),
                    text=_hover, hovertemplate="%{text}<extra></extra>",
                    customdata=df["Player"].values,
                ))

            if _show_reg:
                fig1.add_trace(go.Scatter(
                    x=_xseq, y=_yseq, mode="lines",
                    line=dict(color="#f4a261", width=2),
                    name=_reg_lbl,
                ))

            # Quadrant callout annotations
            _xa = float(np.percentile(x[x > 0], 70)) if (x > 0).any() else float(x.mean())
            _yb = float(np.percentile(y, 22))
            _xb = float(np.percentile(x, 20))
            _ya = float(np.percentile(y, 78))
            fig1.add_annotation(
                x=_xa, y=_yb, text="High WAR, low cost — Efficient",
                showarrow=False, font=dict(color="#22c55e", size=9),
                opacity=0.55,
            )
            fig1.add_annotation(
                x=_xb, y=_ya, text="Low WAR, high cost — Overpaid",
                showarrow=False, font=dict(color="#ef4444", size=9),
                opacity=0.55,
            )

            fig1.update_layout(**_pt(
                title="",
                xaxis=dict(
                    title=dict(text="WAR (Wins Above Replacement)", font=dict(size=12, color="#c0d8f0")),
                    tickfont=dict(color="#c0d8f0", size=10),
                    gridcolor="#18293c", linecolor="#18293c", zerolinecolor="#18293c",
                ),
                yaxis=dict(
                    title=dict(text="Salary ($M)", font=dict(size=12, color="#c0d8f0")),
                    tickfont=dict(color="#c0d8f0", size=10),
                    gridcolor="#18293c", linecolor="#18293c", zerolinecolor="#18293c",
                ),
                height=580, showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#a8c8e8"),
                    itemsizing="constant",
                ),
                margin=dict(l=50, r=20, t=16, b=50),
                hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                font=dict(color="#dbeafe", size=12), namelength=-1),
            ))
            _ef_sel = st.plotly_chart(
                fig1, use_container_width=True, key="ef_frontier_scatter",
                on_select="rerun", selection_mode="points",
                config={"modeBarButtonsToKeep": ["zoom2d", "pan2d", "resetScale2d", "toImage"],
                        "displaylogo": False},
            )

            # ── Selected player card (click a dot to reveal) ───────────────
            _ef_pts = []
            try:
                _sel_obj = getattr(_ef_sel, "selection", None)
                if _sel_obj:
                    _ef_pts = getattr(_sel_obj, "points", []) or []
            except Exception:
                pass
            if _ef_pts:
                _clicked_name = _ef_pts[0].get("customdata", None)
                if isinstance(_clicked_name, list):
                    _clicked_name = _clicked_name[0] if _clicked_name else None
                if _clicked_name:
                    _prow_df = df[df["Player"] == _clicked_name]
                    if not _prow_df.empty:
                        _pr = _prow_df.iloc[0]
                        _hs_bytes = None
                        try:
                            if _R2_MODE and _requests_available:
                                _hs_r = _requests.get(
                                    f"{R2_BASE_URL}/data/headshots/{_pr['Player']}.png",
                                    timeout=5,
                                )
                                if _hs_r.status_code == 200:
                                    _hs_bytes = _hs_r.content
                            else:
                                _hs_local = os.path.join(_HEADSHOTS_DIR, f"{_pr['Player']}.png")
                                if os.path.exists(_hs_local):
                                    with open(_hs_local, "rb") as _hf:
                                        _hs_bytes = _hf.read()
                            if _hs_bytes is None and _requests_available:
                                _rzb = _cached_razzball(_RAZZBALL_PATH)
                                _mlbam_id = None
                                if not _rzb.empty and "Name" in _rzb.columns and "MLBAMID" in _rzb.columns:
                                    _rzb_r = _rzb[
                                        _rzb["Name"].str.strip().str.lower()
                                        == _pr["Player"].strip().lower()
                                    ]
                                    if not _rzb_r.empty:
                                        _mlbam_id = _rzb_r["MLBAMID"].iloc[0]
                                if _mlbam_id and str(_mlbam_id).isdigit():
                                    _hs_url = (
                                        "https://img.mlbstatic.com/mlb-photos/image/upload/"
                                        "d_people:generic:headshot:67:current.png/w_213,q_auto:best"
                                        f"/v1/people/{_mlbam_id}/headshot/67/current"
                                    )
                                    _hs_r2 = _requests.get(_hs_url, timeout=6)
                                    if _hs_r2.status_code == 200:
                                        _hs_bytes = _hs_r2.content
                        except Exception:
                            pass
                        _ppr_clr = "#22c55e" if _pr["PPR"] < 1.0 else "#ef4444"
                        _res_clr = "#22c55e" if _pr["residual"] <= 0 else "#ef4444"
                        _res_sgn = "−" if _pr["residual"] <= 0 else "+"
                        _card_a, _card_b = st.columns([1, 5])
                        with _card_a:
                            if _hs_bytes:
                                st.image(_hs_bytes, width=120)
                            else:
                                st.markdown(
                                    '<div style="width:120px;height:120px;background:#1a2f4a;'
                                    'border:1px solid #2a4a6a;border-radius:8px;display:flex;'
                                    'align-items:center;justify-content:center;'
                                    'color:#3a6080;font-size:32px;">👤</div>',
                                    unsafe_allow_html=True,
                                )
                        with _card_b:
                            _age_disp = int(_pr["Age"]) if pd.notna(_pr["Age"]) else "—"
                            st.markdown(f"""<div style="background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;padding:12px 16px;">
  <div style="font-size:17px;font-weight:800;color:#e8f4ff;margin-bottom:3px;">{_pr['Player']}</div>
  <div style="font-size:12px;color:#6a9ab8;margin-bottom:8px;">{_pr['Team']} &middot; {int(_pr['Year'])} &middot; {_pr['Stage_Clean']} &middot; Age {_age_disp}</div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
    <div style="background:#0d1b2a;border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">WAR</div>
      <div style="font-size:15px;font-weight:700;color:#e8f4ff;">{_pr['WAR_Total']:.1f}</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">Salary</div>
      <div style="font-size:15px;font-weight:700;color:#e8f4ff;">${_pr['Salary_M']:.1f}M</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">Expected</div>
      <div style="font-size:15px;font-weight:700;color:#e8f4ff;">${_pr['predicted']:.1f}M</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">PPR</div>
      <div style="font-size:15px;font-weight:700;color:{_ppr_clr};">{_pr['PPR']:.3f}</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">vs Market</div>
      <div style="font-size:15px;font-weight:700;color:{_res_clr};">{_res_sgn}${abs(_pr['residual']):.1f}M</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Footer explanation ─────────────────────────────────────────
            st.markdown(f"""<div style="background:#090f1a;border:1px solid #1a3050;border-radius:8px;
padding:12px 16px;margin-top:4px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;font-size:11px;color:#5a8aaa;">
    <div>
      <span style="color:#f4a261;font-weight:700;">Trendline ({_reg_method})</span><br>
      The orange line shows the market-predicted salary at each WAR level.
      Points below are underpaid relative to market; points above are overpaid.
    </div>
    <div>
      <span style="font-weight:700;color:#a8c8e8;">Dot colours (Career Stage)</span><br>
      <span style="color:#06d6a0;">●</span> Free Agent<br>
      <span style="color:#fbbf24;">●</span> Arbitration<br>
      <span style="color:#38bdf8;">●</span> Pre-Arb.<br>
      Switch colour mode in the Display panel.
    </div>
    <div>
      <span style="font-weight:700;color:#a8c8e8;">PPR (Pay-Performance Ratio)</span><br>
      Actual salary ÷ model-predicted salary.<br>
      PPR &lt; 1 = underpaid<br>
      PPR &gt; 1 = overpaid<br>
      PPR = 1 = fair market.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Selected sample summary ────────────────────────────────────
            _yrs_d  = f"{min(_year_sel)}–{max(_year_sel)}" if _year_sel else "All"
            _tms_d  = "All teams" if "All Teams" in _team_sel else (
                ", ".join(_team_sel[:3]) + ("…" if len(_team_sel) > 3 else "")
            )
            _pos_d  = "All" if len(_pos_sel) == len(_all_pos) else f"{len(_pos_sel)} pos"
            _stg_d  = "All" if len(_stage_sel) == len(_all_stages) else ", ".join(_stage_sel)
            st.markdown(f"""<div style="background:#090f1a;border:1px solid #1a3050;border-radius:8px;
padding:9px 16px;margin-top:6px;display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
  <span style="font-size:9px;color:#3a6080;text-transform:uppercase;letter-spacing:1px;white-space:nowrap;">Selected sample</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">Years</b> {_yrs_d}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">Teams</b> {_tms_d}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">League</b> {_league_sel}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">Stages</b> {_stg_d}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">Positions</b> {_pos_d}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">N</b> {_n_shown:,} player-seasons</span>
</div>""", unsafe_allow_html=True)

            with st.expander("Model summary"):
                if _reg_method == "OLS":
                    st.text(
                        f"Method: OLS\nFormula: Salary_M ~ WAR_Total\n"
                        f"R²: {_mod.rsquared:.4f}  Adj R²: {_mod.rsquared_adj:.4f}\n"
                        f"Intercept: {_mod.params[0]:.3f}  Slope (WAR): {_mod.params[1]:.3f}\n"
                        f"Residual Std Error: {_mod.mse_resid**0.5:.3f}"
                    )
                elif _reg_method == "LOESS":
                    st.text(f"Method: LOESS  span=0.75  N={len(df)}")
                else:
                    st.text(f"Method: Polynomial (deg 2)\n"
                            f"Coefficients (highest→lowest): {[f'{c:.4f}' for c in _coeffs]}")

            # ── Top 25 Most Underpaid Players (auto-adjusts with filters) ──
            st.markdown(
                "<div style='margin-top:1rem;font-size:0.92rem;font-weight:700;color:#d6e8f8;'>"
                "Top 25 Most Underpaid Players</div>"
                "<div style='font-size:0.76rem;color:#7a9ebc;margin-bottom:0.4rem;'>"
                "Ranked by PPR (lowest = most underpaid). Adjusts with all filters above. Excludes Pre-Arbitration players and players under 1 fWAR.</div>",
                unsafe_allow_html=True,
            )
            _top25_pool = df[~df["Stage_Clean"].str.contains("Pre.Arb|Pre-Arb|Pre_Arb|Pre Arb|Pre-Arbitration|PreArb", case=False, na=False, regex=True) & (df["WAR_Total"] >= 1.0)]
            _top25 = _top25_pool.nsmallest(25, "PPR")[["Player","Team","Year","WAR_Total","Salary_M","predicted","PPR","Stage_Clean"]].copy()
            _top25.insert(0, "#", range(1, len(_top25) + 1))
            _top25.columns = ["#", "Player", "Team", "Year", "fWAR", "Salary $M", "Expected $M", "PPR", "Stage"]
            _top25["Year"] = _top25["Year"].astype(int)
            st.dataframe(
                _top25.style.format({
                    "fWAR": "{:.1f}", "Salary $M": "{:.1f}",
                    "Expected $M": "{:.1f}", "PPR": "{:.3f}",
                }).apply(lambda row: ["background-color:#0c221866"] * len(row) if row["#"] <= 5 else [""] * len(row), axis=1),
                hide_index=True, use_container_width=True, height=385,
            )

        # ── Tab 3 — Age Trajectory ────────────────────────────────────────
        with t3:
            st.markdown(
                "<div style='background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;"
                "padding:14px 18px;margin-bottom:12px;'>"
                "<div style='font-size:1rem;font-weight:700;color:#d6e8f8;margin-bottom:6px;'>Age Trajectory</div>"
                "<div style='font-size:0.85rem;color:#93b8d8;line-height:1.7;'>"
                "See how player production changes as they age. Most players peak in their late 20s "
                "and decline into their 30s — but the shape varies by position and player type.<br><br>"
                "<b>How to read it:</b> The line shows the <b>average</b> metric at each age across all "
                "filtered players. Choose fWAR (production), Salary (cost), or Efficiency (production per $) "
                "to see different age curves. Split by Player Type or Position to compare groups."
                "</div></div>",
                unsafe_allow_html=True,
            )
            _c1, _c2, _c3 = st.columns([2, 2, 1])
            with _c1:
                _traj_m = st.selectbox("Metric", ["WAR", "Salary ($M)", "Efficiency (WAR/$M)"], key="ef_traj_m")
            with _c2:
                _traj_s = st.selectbox("Split by", ["None", "Player Type", "Position"], key="ef_traj_s")
            with _c3:
                _show_ind = st.checkbox("Individual points", value=False, key="ef_traj_ind")

            _mcol = {"WAR": "WAR_Total", "Salary ($M)": "Salary_M", "Efficiency (WAR/$M)": "_eff"}[_traj_m]
            df2 = df.copy()
            df2["_eff"] = df2["WAR_Total"] / df2["Salary_M"].clip(lower=0.01)
            df2["_age_i"] = df2["Age"].round().astype("Int64")
            df2 = df2.dropna(subset=["Age", _mcol])

            fig2 = go.Figure()
            _split_col = None if _traj_s == "None" else ("Player_Type" if _traj_s == "Player Type" else "Position")

            def _add_age_trace(grp, name, color=None):
                _agg = grp.groupby("_age_i")[_mcol].agg(["mean","count"]).reset_index()
                _kw = dict(color=color) if color else {}
                fig2.add_trace(go.Scatter(
                    x=_agg["_age_i"].astype(int), y=_agg["mean"],
                    mode="lines+markers", name=name,
                    marker=dict(size=(_agg["count"] / _agg["count"].max() * 14 + 5).clip(5, 20).tolist(), **_kw),
                    line=dict(**_kw),
                    hovertemplate=f"{name} age %{{x}}: avg=%{{y:.2f}}<extra></extra>",
                ))
                if _show_ind:
                    fig2.add_trace(go.Scatter(
                        x=grp["_age_i"].astype(int), y=grp[_mcol],
                        mode="markers", name=f"{name} (pts)", showlegend=False,
                        marker=dict(size=4, opacity=0.3, **_kw),
                        text=grp["Player"], hovertemplate="%{text}: %{y:.2f}<extra></extra>",
                    ))

            if _split_col is None:
                _add_age_trace(df2, "Average", "#2a9d8f")
            else:
                for _gn, _gg in df2.groupby(_split_col):
                    if len(_gg) >= 3:
                        _add_age_trace(_gg, str(_gn))

            fig2.update_layout(**_pt(
                title=f"Average {_traj_m} by Age",
                xaxis=dict(title="Age"), yaxis=dict(title=_traj_m),
                height=640, showlegend=True,
            ))
            st.plotly_chart(fig2, width="stretch")

            # ── Top 25 Standouts by age efficiency ──────────────────────
            st.markdown(
                "<div style='margin-top:1rem;font-size:0.92rem;font-weight:700;color:#d6e8f8;'>"
                "Top 25 Standouts — Best Value by Age</div>"
                "<div style='font-size:0.76rem;color:#7a9ebc;margin-bottom:0.4rem;'>"
                "Highest fWAR per $M among filtered players. Adjusts with all filters above.</div>",
                unsafe_allow_html=True,
            )
            _age_eff = df.copy()
            _age_eff["fWAR/$M"] = _age_eff["WAR_Total"] / _age_eff["Salary_M"].clip(lower=0.01)
            _age_top = _age_eff.nlargest(25, "fWAR/$M")[["Player","Team","Year","Age","WAR_Total","Salary_M","Stage_Clean"]].copy()
            _age_top["fWAR/$M"] = (_age_top["WAR_Total"] / _age_top["Salary_M"].clip(lower=0.01)).round(2)
            _age_top.insert(0, "#", range(1, len(_age_top) + 1))
            _age_top.columns = ["#", "Player", "Team", "Year", "Age", "fWAR", "Salary $M", "Stage", "fWAR/$M"]
            _age_top["Year"] = _age_top["Year"].astype(int)
            _age_top["Age"] = _age_top["Age"].round(0).astype(int)
            st.dataframe(
                _age_top.style.format({
                    "fWAR": "{:.1f}", "Salary $M": "{:.1f}", "fWAR/$M": "{:.2f}",
                }).apply(lambda row: ["background-color:#0c221866"] * len(row) if row["#"] <= 5 else [""] * len(row), axis=1),
                hide_index=True, use_container_width=True, height=385,
            )

        # ── Tab 5 — Efficient Players ─────────────────────────────────────
        with t5:
            _top100 = (
                df.nsmallest(100, "PPR")
                  [["Player","Team","League","Year","Age","Position","Stage_Clean",
                    "WAR_Total","Salary_M","predicted","residual","PPR","cost_per_WAR"]]
                  .rename(columns={
                      "Stage_Clean":"Stage","WAR_Total":"WAR","Salary_M":"Actual $M",
                      "predicted":"Expected $M","residual":"Residual $M","cost_per_WAR":"$/WAR",
                  }).reset_index(drop=True)
            )

            def _clr_ppr(val):
                try:
                    v = float(val)
                    if v < 0.75:  return "background-color:#14532d;color:white"
                    if v < 1.0:   return "background-color:#166534;color:white"
                    if v > 1.5:   return "background-color:#7f1d1d;color:white"
                    if v > 1.0:   return "background-color:#991b1b;color:white"
                except Exception:
                    pass
                return ""

            st.dataframe(
                _top100.style
                    .format({"WAR":"{:.1f}","Actual $M":"${:.2f}M","Expected $M":"${:.2f}M",
                             "Residual $M":"${:+.2f}M","PPR":"{:.3f}","$/WAR":"${:.2f}M","Age":"{:.0f}"}, na_rep="—")
                    .map(_clr_ppr, subset=["PPR"]),
                use_container_width=True, hide_index=True, height=480,
            )

        # ── Tab 6 — Residual Analysis ─────────────────────────────────────
        with t6:
            _r1, _r2 = st.columns(2)
            with _r1:
                fR1 = go.Figure()
                fR1.add_hline(y=0, line_color="#94a3b8", line_dash="dash", opacity=0.5)
                fR1.add_trace(go.Scatter(
                    x=df["predicted"], y=df["residual"], mode="markers",
                    marker=dict(color="#2a9d8f", size=5, opacity=0.65),
                    text=df["Player"],
                    hovertemplate="%{text}<br>Fitted: $%{x:.2f}M  Residual: $%{y:.2f}M<extra></extra>",
                ))
                fR1.update_layout(**_pt(title="Residuals vs Fitted",
                    xaxis=dict(title="Predicted Salary ($M)"), yaxis=dict(title="Residual ($M)"), height=400))
                st.plotly_chart(fR1, width="stretch")

            with _r2:
                fR2 = go.Figure(go.Histogram(
                    x=df["residual"], nbinsx=30,
                    marker=dict(color="#2a9d8f", line=dict(color="#0e1117", width=0.5)),
                ))
                fR2.add_vline(x=0, line_color="#f4a261", line_dash="dash")
                fR2.update_layout(**_pt(title="Residual Distribution",
                    xaxis=dict(title="Residual ($M)"), yaxis=dict(title="Count"), height=400))
                st.plotly_chart(fR2, width="stretch")

            fR3 = go.Figure()
            fR3.add_hline(y=0, line_color="#94a3b8", line_dash="dash", opacity=0.5)
            fR3.add_trace(go.Scatter(
                x=df["WAR_Total"], y=df["residual"], mode="markers",
                marker=dict(color="#a8dadc", size=5, opacity=0.6),
                text=df["Player"],
                hovertemplate="%{text}<br>WAR: %{x:.1f}  Residual: $%{y:.2f}M<extra></extra>",
            ))
            if len(df) >= 10:
                _rlo = _sm_lowess(df["residual"].values.astype(float),
                                  df["WAR_Total"].values.astype(float), frac=0.5, return_sorted=True)
                fR3.add_trace(go.Scatter(x=_rlo[:, 0], y=_rlo[:, 1], mode="lines",
                                         line=dict(color="#ef4444", width=2), name="LOESS trend"))
            fR3.update_layout(**_pt(title="Residuals vs WAR",
                xaxis=dict(title="WAR"), yaxis=dict(title="Residual ($M)"), height=440, showlegend=True))
            st.plotly_chart(fR3, width="stretch")

        # ── Tab 2 — PPEL (Pay-Performance Efficiency Line) ───────────────
        with t2:
            st.markdown(
                "<div style='background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;"
                "padding:14px 18px;margin-bottom:12px;'>"
                "<div style='font-size:1rem;font-weight:700;color:#d6e8f8;margin-bottom:6px;'>PPEL — Multi-Year Value Analysis</div>"
                "<div style='font-size:0.85rem;color:#93b8d8;line-height:1.7;'>"
                "This view extends the Cost Effective Line across <b>multiple seasons</b> to separate "
                "one-year flukes from sustained performance. Use the toggle to switch between 1-year, "
                "3-year, and 5-year windows.<br><br>"
                "<b>Why it matters:</b> A single great season can make a player look like a bargain, but "
                "PPEL3 and PPEL5 show whether that value held up over a full contract. Players who "
                "consistently sit below the line are the truly elite values in baseball.<br><br>"
                "<b>fWAR/$M</b> = fWAR per million dollars spent. Higher is better. "
                "Pre-Arb players often dominate this metric because they produce at near league-minimum salary (~$740K)."
                "</div></div>",
                unsafe_allow_html=True,
            )
            # Controls in a compact horizontal row above the chart
            _tc1, _tc2, _tc3, _tc4 = st.columns([1.4, 1, 1.2, 1])
            with _tc1:
                _pv_mode = st.radio("View", ["PPEL (1-Year)", "PPEL3 (3-Year)", "PPEL5 (5-Year)"],
                                    horizontal=True, key="ef_pvp_mode")
            with _tc2:
                _pv_n = st.slider("Top N players", 10, 50, 25, key="ef_pvp_n")
            with _tc3:
                _pv_sort = st.radio("Sort", ["Best value (high)", "Worst value (low)"],
                                    key="ef_pvp_sort")
            with _tc4:
                _pv_min_war = st.slider("Min WAR", 0.0, 5.0, 0.5, step=0.1, key="ef_pvp_minwar")

            if _pv_mode == "PPEL (1-Year)":
                # Scatter of filtered single-season data with CEL regression line
                _pvp_df = df[df["WAR_Total"] >= _pv_min_war].copy()
                _pvp_df["PVP"] = _pvp_df["WAR_Total"] / _pvp_df["Salary_M"].clip(lower=0.01)
                _pvp_df["_above"] = _pvp_df["residual"] <= 0  # below regression = underpaid

                _pvp_hover = _pvp_df.apply(lambda r: (
                    f"<b>{r['Player']}</b><br>"
                    + f"{r['Team']}  {r['Year']}<br>"
                    + f"WAR: {r['WAR_Total']:.1f}  Salary: ${r['Salary_M']:.2f}M<br>"
                    + f"PVP: {r['PVP']:.3f}  |  {r['Stage_Clean']}"
                ), axis=1)

                _pvp_clrs = ["#22c55e" if v else "#ef4444" for v in _pvp_df["_above"]]
                fig_cel = go.Figure()
                fig_cel.add_trace(go.Scatter(
                    x=_pvp_df["WAR_Total"], y=_pvp_df["Salary_M"],
                    mode="markers",
                    marker=dict(color=_pvp_clrs, size=7, opacity=0.75,
                                line=dict(color="rgba(0,0,0,0)", width=0)),
                    text=_pvp_hover, hovertemplate="%{text}<extra></extra>",
                    name="Players", showlegend=False,
                ))
                fig_cel.add_trace(go.Scatter(
                    x=_xseq, y=_yseq, mode="lines",
                    line=dict(color="#f4a261", width=2.5),
                    name="Cost Effective Line",
                ))
                fig_cel.update_layout(**_pt(
                    title="Cost Effective Line — WAR vs Salary (1-Year)",
                    xaxis=dict(title="WAR"), yaxis=dict(title="Salary ($M)"),
                    height=640, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                    font=dict(color="#dbeafe", size=12), namelength=-1),
                ))
                st.plotly_chart(fig_cel, width="stretch")

                # Ranked bar below
                _pvp_ranked = (
                    _pvp_df.nlargest(_pv_n, "PVP") if _pv_sort.startswith("Best")
                    else _pvp_df.nsmallest(_pv_n, "PVP")
                ).sort_values("PVP", ascending=_pv_sort.startswith("Worst"))
                _pvp_ranked["_lbl"] = (
                    _pvp_ranked["Player"] + "  (" + _pvp_ranked["Team"]
                    + ", " + _pvp_ranked["Year"].astype(str) + ")"
                )
                fig_bar1 = go.Figure(go.Bar(
                    y=_pvp_ranked["_lbl"], x=_pvp_ranked["PVP"], orientation="h",
                    marker=dict(color=["#22c55e" if v else "#ef4444"
                                       for v in (_pvp_ranked["residual"] <= 0)]),
                    text=_pvp_ranked["PVP"].round(3).astype(str),
                    textposition="outside", textfont=dict(color="#dbeafe", size=9),
                    hovertemplate="%{y}<br>PVP: %{x:.3f}<extra></extra>",
                ))
                fig_bar1.update_layout(**_pt(
                    title="PVP Ranking (WAR / Salary $M)",
                    xaxis=dict(title="PVP"), yaxis=dict(autorange="reversed"),
                    height=max(350, _pv_n * 22), margin=dict(l=230, r=60, t=40, b=40),
                ))
                st.plotly_chart(fig_bar1, width="stretch")
                _league_pvp = df["WAR_Total"].sum() / df["Salary_M"].sum()
                st.caption(f"League-average PVP (filtered data): **{_league_pvp:.3f}** WAR per $M")

            elif _pv_mode == "PPEL3 (3-Year)":  # PPEL3 — last 3 years, actual contract data only
                _all_yrs_sorted = sorted(raw["Year"].dropna().unique())
                _last3 = _all_yrs_sorted[-3:] if len(_all_yrs_sorted) >= 3 else _all_yrs_sorted

                _raw3 = raw[raw["Year"].isin(_last3) & (raw["Salary_M"] > 0)].copy()
                if _league_sel != "Both Leagues":
                    _raw3 = _raw3[_raw3["League"] == _league_sel]
                if "All Teams" not in _team_sel and _team_sel:
                    _raw3 = _raw3[_raw3["Team"].isin(_team_sel)]
                if _stage_sel:
                    _raw3 = _raw3[_raw3["Stage_Clean"].isin(_stage_sel)]
                if _pos_sel:
                    _raw3 = _raw3[_raw3["Position"].isin(_pos_sel)]
                if _ptype_sel != "All":
                    _raw3 = _raw3[_raw3["Player_Type"] == _ptype_sel]

                _p3 = (
                    _raw3.groupby("Player")
                    .agg(
                        WAR3    =("WAR_Total",   "sum"),
                        Sal3    =("Salary_M",    "sum"),
                        Seasons =("Year",        "nunique"),
                        Team    =("Team",        "last"),
                        Stage   =("Stage_Clean", "last"),
                        Position=("Position",    "last"),
                        Age     =("Age",         "last"),
                    )
                    .reset_index()
                )
                _p3 = _p3[(_p3["Sal3"] > 0) & (_p3["WAR3"] >= _pv_min_war)].copy()
                _p3["PVP3"] = _p3["WAR3"] / _p3["Sal3"].clip(lower=0.01)

                if len(_p3) < 5:
                    st.warning("Not enough players with 3-year data under current filters.")
                else:
                    _x3 = _p3["WAR3"].values.astype(float)
                    _y3 = _p3["Sal3"].values.astype(float)
                    if _reg_method == "OLS":
                        import statsmodels.api as _sm3
                        _m3  = _sm3.OLS(_y3, _sm3.add_constant(_x3)).fit()
                        _p3["_pred3"] = _m3.predict(_sm3.add_constant(_x3))
                        _x3seq = np.linspace(_x3.min(), _x3.max(), 200)
                        _y3seq = _m3.params[0] + _m3.params[1] * _x3seq
                        _cel_lbl = f"Cost Effective Line  R²={_m3.rsquared:.3f}"
                    elif _reg_method == "LOESS":
                        _lo3 = _sm_lowess(_y3, _x3, frac=0.75, return_sorted=True)
                        _p3["_pred3"] = np.interp(_x3, _lo3[:, 0], _lo3[:, 1])
                        _x3seq, _y3seq = _lo3[:, 0], _lo3[:, 1]
                        _cel_lbl = "Cost Effective Line (LOESS)"
                    else:
                        _cf3 = np.polyfit(_x3, _y3, 2)
                        _pf3 = np.poly1d(_cf3)
                        _p3["_pred3"] = _pf3(_x3)
                        _x3seq = np.linspace(_x3.min(), _x3.max(), 200)
                        _y3seq = _pf3(_x3seq)
                        _cel_lbl = "Cost Effective Line (Poly)"

                    _p3["_resid3"] = _p3["Sal3"] - _p3["_pred3"]
                    _p3["_above3"] = _p3["_resid3"] <= 0

                    _p3_hover = _p3.apply(lambda r: (
                        f"<b>{r['Player']}</b><br>"
                        + f"{r['Team']}  ({int(r['Seasons'])} seasons)<br>"
                        + f"3yr WAR: {r['WAR3']:.1f}  |  3yr Salary: ${r['Sal3']:.1f}M<br>"
                        + f"PVP3: {r['PVP3']:.3f}  |  {r['Stage']}"
                    ), axis=1)

                    _clrs3 = ["#22c55e" if v else "#ef4444" for v in _p3["_above3"]]
                    fig_p3 = go.Figure()
                    fig_p3.add_trace(go.Scatter(
                        x=_p3["WAR3"], y=_p3["Sal3"],
                        mode="markers",
                        marker=dict(color=_clrs3, size=8, opacity=0.8,
                                    line=dict(color="rgba(0,0,0,0)", width=0)),
                        text=_p3_hover, hovertemplate="%{text}<extra></extra>",
                        name="Players", showlegend=False,
                    ))
                    fig_p3.add_trace(go.Scatter(
                        x=_x3seq, y=_y3seq, mode="lines",
                        line=dict(color="#f4a261", width=2.5),
                        name=_cel_lbl,
                    ))
                    fig_p3.update_layout(**_pt(
                        title=f"Cost Effective Line — 3-Year Cumulative WAR vs Salary  ({_last3[0]}–{_last3[-1]})",
                        xaxis=dict(title="3-Year Total WAR"),
                        yaxis=dict(title="3-Year Total Salary ($M)"),
                        height=640, showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                        font=dict(color="#dbeafe", size=12), namelength=-1),
                    ))
                    st.plotly_chart(fig_p3, width="stretch")

                    _p3_ranked = (
                        _p3.nlargest(_pv_n, "PVP3") if _pv_sort.startswith("Best")
                        else _p3.nsmallest(_pv_n, "PVP3")
                    ).sort_values("PVP3", ascending=_pv_sort.startswith("Worst"))
                    _p3_ranked["_lbl"] = (
                        _p3_ranked["Player"] + "  (" + _p3_ranked["Team"] + ")"
                    )
                    fig_bar3 = go.Figure(go.Bar(
                        y=_p3_ranked["_lbl"], x=_p3_ranked["PVP3"], orientation="h",
                        marker=dict(color=["#22c55e" if v else "#ef4444"
                                           for v in (_p3_ranked["_above3"])]),
                        text=_p3_ranked["PVP3"].round(3).astype(str),
                        textposition="outside", textfont=dict(color="#dbeafe", size=9),
                        hovertemplate=(
                            "%{y}<br>PVP3: %{x:.3f}<br>"
                            "3yr WAR: " + _p3_ranked["WAR3"].round(1).astype(str)
                            + "  3yr Salary: $" + _p3_ranked["Sal3"].round(1).astype(str) + "M"
                            + "<extra></extra>"
                        ).tolist(),
                    ))
                    fig_bar3.update_layout(**_pt(
                        title="PVP3 Ranking (3-Year WAR / 3-Year Salary $M)",
                        xaxis=dict(title="PVP3"), yaxis=dict(autorange="reversed"),
                        height=max(350, _pv_n * 22), margin=dict(l=230, r=60, t=40, b=40),
                    ))
                    st.plotly_chart(fig_bar3, width="stretch")
                    _lg_pvp3 = _p3["WAR3"].sum() / _p3["Sal3"].sum()
                    st.caption(
                        f"Seasons used: **{', '.join(str(y) for y in _last3)}** "
                        f"({len(_p3):,} players with actual contract data)  |  "
                        f"League-average PPEL3: **{_lg_pvp3:.3f}** WAR per $M"
                    )

            else:  # PPEL5 — last 5 years, actual contract data only
                _all_yrs_sorted5 = sorted(raw["Year"].dropna().unique())
                _last5 = _all_yrs_sorted5[-5:] if len(_all_yrs_sorted5) >= 5 else _all_yrs_sorted5

                _raw5 = raw[raw["Year"].isin(_last5) & (raw["Salary_M"] > 0)].copy()
                if _league_sel != "Both Leagues":
                    _raw5 = _raw5[_raw5["League"] == _league_sel]
                if "All Teams" not in _team_sel and _team_sel:
                    _raw5 = _raw5[_raw5["Team"].isin(_team_sel)]
                if _stage_sel:
                    _raw5 = _raw5[_raw5["Stage_Clean"].isin(_stage_sel)]
                if _pos_sel:
                    _raw5 = _raw5[_raw5["Position"].isin(_pos_sel)]
                if _ptype_sel != "All":
                    _raw5 = _raw5[_raw5["Player_Type"] == _ptype_sel]

                _p5 = (
                    _raw5.groupby("Player")
                    .agg(
                        WAR5    =("WAR_Total",   "sum"),
                        Sal5    =("Salary_M",    "sum"),
                        Seasons =("Year",        "nunique"),
                        Team    =("Team",        "last"),
                        Stage   =("Stage_Clean", "last"),
                    )
                    .reset_index()
                )
                _p5 = _p5[(_p5["Sal5"] > 0) & (_p5["WAR5"] >= _pv_min_war)].copy()
                _p5["PVP5"] = _p5["WAR5"] / _p5["Sal5"].clip(lower=0.01)

                if len(_p5) < 5:
                    st.warning("Not enough players with 5-year data under current filters.")
                else:
                    _x5 = _p5["WAR5"].values.astype(float)
                    _y5 = _p5["Sal5"].values.astype(float)
                    if _reg_method == "OLS":
                        import statsmodels.api as _sm5
                        _m5  = _sm5.OLS(_y5, _sm5.add_constant(_x5)).fit()
                        _p5["_pred5"] = _m5.predict(_sm5.add_constant(_x5))
                        _x5seq = np.linspace(_x5.min(), _x5.max(), 200)
                        _y5seq = _m5.params[0] + _m5.params[1] * _x5seq
                        _cel5_lbl = f"PPEL5 (OLS · R²={_m5.rsquared:.3f})"
                    elif _reg_method == "LOESS":
                        _lo5 = _sm_lowess(_y5, _x5, frac=0.75, return_sorted=True)
                        _p5["_pred5"] = np.interp(_x5, _lo5[:, 0], _lo5[:, 1])
                        _x5seq, _y5seq = _lo5[:, 0], _lo5[:, 1]
                        _cel5_lbl = "PPEL5 (LOESS)"
                    else:
                        _cf5 = np.polyfit(_x5, _y5, 2)
                        _pf5 = np.poly1d(_cf5)
                        _p5["_pred5"] = _pf5(_x5)
                        _x5seq = np.linspace(_x5.min(), _x5.max(), 200)
                        _y5seq = _pf5(_x5seq)
                        _cel5_lbl = "PPEL5 (Poly)"

                    _p5["_resid5"] = _p5["Sal5"] - _p5["_pred5"]
                    _p5["_above5"] = _p5["_resid5"] <= 0

                    _p5_hover = _p5.apply(lambda r: (
                        f"<b>{r['Player']}</b><br>"
                        + f"{r['Team']}  ({int(r['Seasons'])} seasons)<br>"
                        + f"5yr WAR: {r['WAR5']:.1f}  |  5yr Salary: ${r['Sal5']:.1f}M<br>"
                        + f"PPEL5: {r['PVP5']:.3f}  |  {r['Stage']}"
                    ), axis=1)

                    _clrs5 = ["#22c55e" if v else "#ef4444" for v in _p5["_above5"]]
                    fig_p5 = go.Figure()
                    fig_p5.add_trace(go.Scatter(
                        x=_p5["WAR5"], y=_p5["Sal5"],
                        mode="markers",
                        marker=dict(color=_clrs5, size=8, opacity=0.8,
                                    line=dict(color="rgba(0,0,0,0)", width=0)),
                        text=_p5_hover, hovertemplate="%{text}<extra></extra>",
                        name="Players", showlegend=False,
                    ))
                    fig_p5.add_trace(go.Scatter(
                        x=_x5seq, y=_y5seq, mode="lines",
                        line=dict(color="#f4a261", width=2.5),
                        name=_cel5_lbl,
                    ))
                    fig_p5.update_layout(**_pt(
                        title=f"PPEL5 — 5-Year Cumulative WAR vs Salary  ({_last5[0]}–{_last5[-1]})",
                        xaxis=dict(title="5-Year Total WAR"),
                        yaxis=dict(title="5-Year Total Salary ($M)"),
                        height=640, showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hoverlabel=dict(bgcolor="#18243a", bordercolor="#253d58",
                                        font=dict(color="#d6e8f8", size=12), namelength=-1),
                    ))
                    st.plotly_chart(fig_p5, width="stretch")

                    _p5_ranked = (
                        _p5.nlargest(_pv_n, "PVP5") if _pv_sort.startswith("Best")
                        else _p5.nsmallest(_pv_n, "PVP5")
                    ).sort_values("PVP5", ascending=_pv_sort.startswith("Worst"))
                    _p5_ranked["_lbl"] = (
                        _p5_ranked["Player"] + "  (" + _p5_ranked["Team"] + ")"
                    )
                    fig_bar5 = go.Figure(go.Bar(
                        y=_p5_ranked["_lbl"], x=_p5_ranked["PVP5"], orientation="h",
                        marker=dict(color=["#22c55e" if v else "#ef4444"
                                           for v in (_p5_ranked["_above5"])]),
                        text=_p5_ranked["PVP5"].round(3).astype(str),
                        textposition="outside", textfont=dict(color="#d6e8f8", size=9),
                        hovertemplate=(
                            "%{y}<br>PPEL5: %{x:.3f}<br>"
                            "5yr WAR: " + _p5_ranked["WAR5"].round(1).astype(str)
                            + "  5yr Salary: $" + _p5_ranked["Sal5"].round(1).astype(str) + "M"
                            + "<extra></extra>"
                        ).tolist(),
                    ))
                    fig_bar5.update_layout(**_pt(
                        title="PPEL5 Ranking (5-Year WAR / 5-Year Salary $M)",
                        xaxis=dict(title="PPEL5"), yaxis=dict(autorange="reversed"),
                        height=max(350, _pv_n * 22), margin=dict(l=230, r=60, t=40, b=40),
                    ))
                    st.plotly_chart(fig_bar5, width="stretch")
                    _lg_pvp5 = _p5["WAR5"].sum() / _p5["Sal5"].sum()
                    st.caption(
                        f"Seasons used: **{', '.join(str(y) for y in _last5)}** "
                        f"({len(_p5):,} players with 5-year contract data)  |  "
                        f"League-average PPEL5: **{_lg_pvp5:.3f}** WAR per $M"
                    )

        # ── Tab 7 — Pre-Arb Explorer ──────────────────────────────────────
        with t7:
            _render_glossary([
                ("Pre-Arb", "Pre-Arbitration",
                 "Players in their first 1–3 years of service time earning near the league minimum "
                 "($740K–$780K). They often deliver elite WAR per dollar."),
                ("WAR Trajectory", "Year-over-Year WAR Change",
                 "How a player's WAR evolves from season to season. Rising = development; "
                 "falling = regression or injury impact."),
                ("SD Threshold", "Classification Threshold",
                 "Players whose average annual WAR change exceeds ±(threshold × global SD) "
                 "are labelled Improving or Declining; the rest are Neutral."),
            ], title="📖 Pre-Arb Explorer", cols=3)

            _pa_c1, _pa_c2, _pa_c3 = st.columns([1, 1, 1])
            with _pa_c1:
                _pa_min_seasons = st.slider("Min seasons", 2, 5, 2, key="ef_pa_min_seasons")
            with _pa_c2:
                _pa_sd_thresh = st.slider("Trend SD threshold", 0.25, 1.5, 0.5, step=0.05,
                                          key="ef_pa_sd")
            with _pa_c3:
                _pa_ptype = st.selectbox("Player type", ["All", "Position Player", "Pitcher"],
                                         key="ef_pa_ptype")

            # Build pre-arb subset from the full raw dataset (all years, ignoring year filter)
            _pa_raw = raw[raw["Stage_Clean"] == "Pre-Arbitration"].copy()
            if _pa_ptype != "All":
                _pa_raw = _pa_raw[_pa_raw["Player_Type"] == _pa_ptype]
            if _league_sel != "Both Leagues":
                _pa_raw = _pa_raw[_pa_raw["League"] == _league_sel]
            if "All Teams" not in _team_sel and _team_sel:
                _pa_raw = _pa_raw[_pa_raw["Team"].isin(_team_sel)]

            _pa_counts  = _pa_raw.groupby("Player")["Year"].nunique()
            _pa_players = _pa_counts[_pa_counts >= _pa_min_seasons].index
            _pa_df      = (_pa_raw[_pa_raw["Player"].isin(_pa_players)]
                           .sort_values(["Player", "Year"]).copy())

            if len(_pa_df) < 3:
                st.info("No Pre-Arbitration players with enough seasons under current filters.")
            else:
                _pa_df["WAR_delta"] = _pa_df.groupby("Player")["WAR_Total"].diff()

                _pa_summary = (
                    _pa_df.groupby("Player")
                    .agg(
                        avg_delta=("WAR_delta",  "mean"),
                        seasons  =("Year",       "nunique"),
                        last_team=("Team",       "last"),
                        last_war =("WAR_Total",  "last"),
                        last_sal =("Salary_M",   "last"),
                        last_age =("Age",        "last"),
                        total_war=("WAR_Total",  "sum"),
                    )
                    .reset_index()
                    .dropna(subset=["avg_delta"])
                )

                _pa_sd_global = float(_pa_summary["avg_delta"].std()) or 1.0
                _pa_summary["Trend"] = _pa_summary["avg_delta"].apply(
                    lambda d: "Improving" if d > _pa_sd_thresh * _pa_sd_global
                    else ("Declining" if d < -_pa_sd_thresh * _pa_sd_global else "Neutral")
                )
                _TREND_COLORS = {
                    "Improving": "#22c55e",
                    "Neutral":   "#fbbf24",
                    "Declining": "#ef4444",
                }

                _pa_n_imp = int((_pa_summary["Trend"] == "Improving").sum())
                _pa_n_neu = int((_pa_summary["Trend"] == "Neutral").sum())
                _pa_n_dec = int((_pa_summary["Trend"] == "Declining").sum())
                _pa_n_tot = len(_pa_summary)

                st.markdown(f"""<div style="background:#090f1a;border:1px solid #1e3a5c;
border-radius:10px;padding:12px 16px;margin-bottom:12px;
display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
  <div style="text-align:center;">
    <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;">Pre-Arb Players</div>
    <div style="font-size:20px;font-weight:700;color:#e8f4ff;">{_pa_n_tot}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;">Improving</div>
    <div style="font-size:20px;font-weight:700;color:#22c55e;">{_pa_n_imp}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;">Neutral</div>
    <div style="font-size:20px;font-weight:700;color:#fbbf24;">{_pa_n_neu}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;">Declining</div>
    <div style="font-size:20px;font-weight:700;color:#ef4444;">{_pa_n_dec}</div>
  </div>
</div>""", unsafe_allow_html=True)

                # ── Player search filter ──────────────────────────────────
                _pa_search_col, _pa_reset_col = st.columns([3, 1])
                with _pa_search_col:
                    _pa_all_players = sorted(_pa_summary["Player"].unique())
                    _pa_player_sel = st.multiselect(
                        "Filter by player(s)", _pa_all_players,
                        placeholder="Type or select player names…",
                        key="ef_pa_player_sel",
                    )
                with _pa_reset_col:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    if st.button("↺  Reset", use_container_width=True, key="ef_pa_player_reset"):
                        if "ef_pa_player_sel" in st.session_state:
                            del st.session_state["ef_pa_player_sel"]
                        st.rerun()

                if _pa_player_sel:
                    _pa_df      = _pa_df[_pa_df["Player"].isin(_pa_player_sel)]
                    _pa_summary = _pa_summary[_pa_summary["Player"].isin(_pa_player_sel)]

                # WAR trajectory lines, one per player, coloured by trend
                fig_pa = go.Figure()
                _trend_map = dict(zip(_pa_summary["Player"], _pa_summary["Trend"]))
                _legend_shown = set()
                for _pname, _pgrp in _pa_df.groupby("Player"):
                    _trend = _trend_map.get(_pname, "Neutral")
                    _t_color = _TREND_COLORS.get(_trend, "#fbbf24")
                    _pg = _pgrp.sort_values("Year")
                    _show_leg = _trend not in _legend_shown
                    if _show_leg:
                        _legend_shown.add(_trend)
                    fig_pa.add_trace(go.Scatter(
                        x=_pg["Year"].astype(int), y=_pg["WAR_Total"],
                        mode="lines+markers",
                        name=_trend,
                        legendgroup=_trend,
                        showlegend=_show_leg,
                        line=dict(color=_t_color, width=1.5),
                        marker=dict(size=6, color=_t_color),
                        hovertemplate=(
                            f"<b>{_pname}</b><br>"
                            "Year: %{x}<br>WAR: %{y:.1f}"
                            f"<br>Team: {_pg['Team'].iloc[-1]}"
                            "<extra></extra>"
                        ),
                        opacity=0.65,
                    ))
                fig_pa.update_layout(**_pt(
                    title="Pre-Arbitration Player WAR Trajectories",
                    xaxis=dict(title="Season", tickformat="d"),
                    yaxis=dict(title="WAR"),
                    height=580, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                ))
                st.plotly_chart(fig_pa, width="stretch")
                if st.button("↺  Reset Legend Filter", key="ef_pa_reset_legend"):
                    st.rerun()

                # Ranked summary table
                _pa_disp = (
                    _pa_summary
                    .sort_values("avg_delta", ascending=False)
                    .rename(columns={
                        "avg_delta": "Avg WAR Δ/yr", "seasons":   "Seasons",
                        "last_team": "Last Team",    "last_war":  "Last WAR",
                        "last_sal":  "Last Sal ($M)", "last_age": "Age",
                        "total_war": "Total WAR",
                    })
                )

                def _clr_trend(val):
                    if val == "Improving": return "background-color:#14532d;color:white"
                    if val == "Declining": return "background-color:#7f1d1d;color:white"
                    return "background-color:#713f12;color:white"

                st.dataframe(
                    _pa_disp[["Player", "Last Team", "Trend", "Age", "Seasons",
                               "Avg WAR Δ/yr", "Last WAR", "Total WAR", "Last Sal ($M)"]]
                    .style
                    .format({"Avg WAR Δ/yr": "{:+.2f}", "Last WAR": "{:.1f}",
                             "Total WAR": "{:.1f}", "Last Sal ($M)": "${:.2f}M",
                             "Age": "{:.0f}"}, na_rep="—")
                    .map(_clr_trend, subset=["Trend"]),
                    use_container_width=True, hide_index=True, height=410,
                )
                st.caption(
                    f"SD threshold = {_pa_sd_thresh:.2f} × global SD ({_pa_sd_global:.2f} WAR/yr).  "
                    f"Improving: avg Δ > +{_pa_sd_thresh * _pa_sd_global:.2f}  |  "
                    f"Declining: avg Δ < −{_pa_sd_thresh * _pa_sd_global:.2f}"
                )

        # ── Tab 8 — WAR Stability (WSR) ─────────────────────────────────
        with t8:
            st.markdown(
                "<div style='background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;"
                "padding:14px 18px;margin-bottom:12px;'>"
                "<div style='font-size:1rem;font-weight:700;color:#d6e8f8;margin-bottom:6px;'>fWAR Stability Rating (WSR)</div>"
                "<div style='font-size:0.85rem;color:#93b8d8;line-height:1.7;'>"
                "Not all high-fWAR players are equally reliable. This chart plots each player's "
                "<b>average fWAR</b> (X-axis) against their <b>standard deviation</b> (Y-axis) — "
                "a measure of how much their production swings from year to year.<br><br>"
                "<b>How to read it:</b> Bottom-right = <span style='color:#22c55e;font-weight:600;'>Cornerstone</span> "
                "(high production, low variance — the most reliable stars). "
                "Top-right = <span style='color:#ef4444;font-weight:600;'>Star but Risky</span> "
                "(high ceiling but unpredictable). Bottom-left = consistent depth pieces. "
                "Top-left = volatile fringe players.<br><br>"
                "<b>Standard Deviation (SD)</b> measures how spread out a player's year-to-year fWAR values are. "
                "An SD of 0.5 means very consistent; SD of 2.0+ means huge swings between great and poor seasons. "
                "<b>WSR</b> = average fWAR ÷ (1 + SD) — higher WSR means more reliable, bankable production."
                "</div></div>",
                unsafe_allow_html=True,
            )

            _wsr_min_pa = st.slider("Min PA (hitters) / IP (pitchers)", 50, 300, 150,
                                     key="v2_war_stability_min_pa")

            # Recompute WSR with user threshold
            _wsr_raw = raw.copy()
            for _nc in ["WAR_Total", "PA", "IP"]:
                if _nc in _wsr_raw.columns:
                    _wsr_raw[_nc] = pd.to_numeric(_wsr_raw[_nc], errors="coerce")

            _is_pit = _wsr_raw["Position"].isin(["SP", "RP", "P", "TWP"])
            _wsr_qual = _wsr_raw[
                (_is_pit & (_wsr_raw["IP"].fillna(0) >= _wsr_min_pa)) |
                (~_is_pit & (_wsr_raw["PA"].fillna(0) >= _wsr_min_pa))
            ]
            _wsr_grp = _wsr_qual.groupby("Player").agg(
                WAR_Mean=("WAR_Total", "mean"),
                WAR_Std=("WAR_Total", "std"),
                Seasons=("Year", "nunique"),
                Team=("Team", "last"),
            ).reset_index()
            _wsr_grp = _wsr_grp[_wsr_grp["Seasons"] >= 2].copy()
            _wsr_grp["WAR_Std"] = _wsr_grp["WAR_Std"].fillna(0)
            _wsr_grp["WSR"] = (_wsr_grp["WAR_Mean"] / (1 + _wsr_grp["WAR_Std"])).round(3)

            def _wsr_tier(v):
                if v >= 3.5: return "Elite"
                if v >= 2.0: return "Reliable"
                if v >= 1.0: return "Volatile"
                return "Unstable"
            _wsr_grp["Tier"] = _wsr_grp["WSR"].apply(_wsr_tier)

            _WSR_COLORS = {"Elite": "#22c55e", "Reliable": "#14b8a6",
                           "Volatile": "#f59e0b", "Unstable": "#ef4444"}

            if not _wsr_grp.empty:
                _wsr_hover = _wsr_grp.apply(lambda r: (
                    f"<b>{r['Player']}</b><br>"
                    + f"{r['Team']} · {int(r['Seasons'])} seasons<br>"
                    + f"Mean WAR: {r['WAR_Mean']:.2f} · Std: {r['WAR_Std']:.2f}<br>"
                    + f"WSR: {r['WSR']:.3f} · {r['Tier']}"
                ), axis=1)

                fig_wsr = go.Figure()
                for tier, color in _WSR_COLORS.items():
                    mask = _wsr_grp["Tier"] == tier
                    if mask.any():
                        sub = _wsr_grp[mask]
                        fig_wsr.add_trace(go.Scatter(
                            x=sub["WAR_Mean"], y=sub["WAR_Std"],
                            mode="markers", name=tier,
                            marker=dict(color=color, size=8, opacity=0.8),
                            text=_wsr_hover[mask],
                            hovertemplate="%{text}<extra></extra>",
                        ))

                _xm = float(_wsr_grp["WAR_Mean"].median())
                _ym = float(_wsr_grp["WAR_Std"].median())
                for _ax, _ay, _atxt in [
                    (_xm * 1.8, _ym * 1.8, "Star but Risky"),
                    (_xm * 0.3, _ym * 1.8, "Fringe Volatile"),
                    (_xm * 1.8, _ym * 0.3, "Cornerstone"),
                    (_xm * 0.3, _ym * 0.3, "Consistent Depth"),
                ]:
                    fig_wsr.add_annotation(x=_ax, y=_ay, text=_atxt,
                                           showarrow=False, font=dict(color="#4a687e", size=10), opacity=0.6)

                fig_wsr.update_layout(**_pt(
                    title="WAR Stability — Mean WAR vs Standard Deviation",
                    xaxis=dict(title="Mean WAR (2021-2025)"),
                    yaxis=dict(title="WAR Std Dev"),
                    height=580, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                    font=dict(color="#dbeafe", size=12)),
                ))
                st.plotly_chart(fig_wsr, width="stretch")
            else:
                st.info("No players meet the minimum qualifying threshold.")


# ---------------------------------------------------------------------------
# League Analysis page
# ---------------------------------------------------------------------------

def _render_league_analysis():
    from pages.player_analysis import render
    render(_data_url, _read_csv, _R2_MODE, _cached_simulator_data,
           _cached_mlbam_lookup, _RAZZBALL_PATH, _file_hash,
           _load_base_config, _resolve_data_path, _DEFAULT_CONFIG,
           _cached_2026_payroll, _dir_hash)


# (Team-level league sections moved to /rankings page)
if False:  # noqa: dead code preserved for reference
    if _R2_MODE:
        # In production, all analysis outputs are pre-built and served from R2
        analysis_dir  = None
        script_path   = None
        scatter_path  = _data_url("efficiency_scatter.png")
        ranking_path  = _data_url("efficiency_ranking.png")
        position_path = _data_url("position_breakdown.png")
        table_csv     = _data_url("al_nl_ranking_table.csv")
        detail_csv    = _data_url("efficiency_detail.csv")
    else:
        analysis_dir  = _ROOT_DIR
        script_path   = os.path.join(analysis_dir, "mlb_efficiency_analysis.py")
        scatter_path  = os.path.join(analysis_dir, "efficiency_scatter.png")
        ranking_path  = os.path.join(analysis_dir, "efficiency_ranking.png")
        position_path = os.path.join(analysis_dir, "position_breakdown.png")
        table_csv     = os.path.join(analysis_dir, "al_nl_ranking_table.csv")
        detail_csv    = os.path.join(analysis_dir, "efficiency_detail.csv")

    st.markdown(
        "### 📉 League-Wide Team Spending Efficiency\n"
        "How much is each team spending above or below the **cost effective line** "
        "(the regression line of payroll vs wins)? Which positions drive the gap? "
        "Does staying close to the frontier predict playoff success?"
    )

    _data_ready = _R2_MODE or (os.path.exists(scatter_path) and os.path.exists(table_csv))

    if not _R2_MODE:
        col_btn, col_status = st.columns([2, 5])
        with col_btn:
            regen = st.button(
                "Regenerate Analysis",
                type="primary" if not _data_ready else "secondary",
                help="Re-runs mlb_efficiency_analysis.py to refresh all charts and tables.",
                key="league_regen_btn",
            )
        with col_status:
            if _data_ready:
                import datetime as _dt
                _mtime = os.path.getmtime(table_csv)
                _age   = _dt.datetime.now() - _dt.datetime.fromtimestamp(_mtime)
                _days  = _age.days
                st.caption(
                    f"Analysis last generated {_days} day{'s' if _days != 1 else ''} ago. "
                    "Click **Regenerate Analysis** to refresh with the latest data."
                )
            else:
                st.info(
                    "**No analysis data found yet.** Click **Regenerate Analysis** to build all charts "
                    "and tables. This runs a one-time calculation (~15 seconds) that produces:\n\n"
                    "- Payroll vs Wins frontier chart for each season\n"
                    "- Team efficiency ranking (who overspends vs who extracts value)\n"
                    "- Playoff Success vs Efficiency 4-quadrant view\n"
                    "- Full AL/NL ranking table with playoff and WS records\n"
                    "- PPR correlation analysis\n\n"
                    "You only need to regenerate when you want updated numbers."
                )
        if regen:
            if not os.path.exists(script_path):
                st.error(f"Analysis script not found:\n`{script_path}`")
            else:
                with st.spinner("Running efficiency analysis (may take 10-20 seconds) ..."):
                    result = subprocess.run(
                        ["python", script_path],
                        capture_output=True, text=True, cwd=analysis_dir,
                    )
                if result.returncode == 0:
                    st.success("Analysis complete. Charts and tables refreshed.")
                else:
                    st.error(f"Script failed (exit {result.returncode}):\n```\n{result.stderr[-1500:]}\n```")
    else:
        regen = False

    st.markdown("---")

    if _R2_MODE or os.path.exists(scatter_path):
        st.subheader("Payroll vs Wins — Cost Effective Line by Season")
        st.caption(
            "Yellow line = OLS regression frontier. "
            "Teams above the line get more wins per dollar than expected; "
            "teams below are underperforming relative to their spend. "
            "Star = WS champion, diamond = WS runner-up. Faded = missed playoffs."
        )
        st.image(_r2_image(scatter_path), width="stretch")
    else:
        st.info("efficiency_scatter.png not found -- click Regenerate Analysis.")

    st.markdown("---")

    st.subheader("Team Efficiency Ranking -- $M Above/Below Frontier")
    st.caption(
        "Bars show average dollars a team overspent (+, red) or underspent (−, green) "
        "relative to what their wins should have cost, averaged 2021-2025. "
        "Gold outline = WS champion, silver = WS runner-up."
    )
    if _R2_MODE or os.path.exists(table_csv):
        try:
            _rdf = _read_csv(table_csv)
            if "Avg_Gap_M" in _rdf.columns:
                _team_col = "Abbr" if "Abbr" in _rdf.columns else "Team"
                _rdf = _rdf.sort_values("Avg_Gap_M").reset_index(drop=True)
                _gaps  = _rdf["Avg_Gap_M"].tolist()
                _teams = _rdf[_team_col].tolist()
                _bclrs = ["#22c55e" if v <= 0 else "#ef4444" for v in _gaps]
                # Gold border for WS champs, silver for WS runners-up
                _lclrs = ["#f59e0b" if _rdf.at[i, "WS_Wins"] > 0
                           else "#94a3b8" if _rdf.at[i, "WS_Apps"] > 0
                           else "rgba(0,0,0,0)"
                           for i in range(len(_rdf))] if "WS_Wins" in _rdf.columns else "rgba(0,0,0,0)"
                _fig_rank = go.Figure(data=[go.Bar(
                    x=_gaps, y=_teams, orientation="h",
                    marker=dict(color=_bclrs, line=dict(color=_lclrs, width=2)),
                    text=[f"${v:+.0f}M" for v in _gaps],
                    textposition="outside",
                    textfont=dict(color="#dbeafe", size=9),
                    hovertemplate="%{y}: %{x:+.1f}M vs frontier<extra></extra>",
                )])
                _fig_rank.add_vline(x=0, line_color="#7aa2c0", line_width=1, opacity=0.4)
                _fig_rank.update_layout(**_pt(
                    title="Avg $M Above / Below Cost Effective Line  (2021–2025)",
                    xaxis=dict(title="$M vs Frontier  (positive = overspent)"),
                    height=max(420, len(_rdf) * 17),
                    margin=dict(l=60, r=80, t=45, b=50),
                ))
                st.plotly_chart(_fig_rank, width="stretch",
                                config={"displayModeBar": False})
            else:
                st.image(_r2_image(ranking_path), width="stretch")
        except Exception:
            if _R2_MODE or os.path.exists(ranking_path):
                st.image(_r2_image(ranking_path), width="stretch")
            else:
                st.info("efficiency_ranking.png not found — click Regenerate Analysis.")
    elif _R2_MODE or os.path.exists(ranking_path):
        st.image(_r2_image(ranking_path), width="stretch")
    else:
        st.info("efficiency_ranking.png not found — click Regenerate Analysis.")

    st.markdown("---")

    if _R2_MODE or os.path.exists(position_path):
        st.subheader("WAR by Position -- Efficient vs Inefficient Teams")
        st.caption(
            "Top row = 5 most efficient teams (spending least above frontier). "
            "Bottom row = 5 most inefficient. Each bar = average WAR from that position group."
        )
        st.image(_r2_image(position_path), width="stretch")
    else:
        st.info("position_breakdown.png not found -- click Regenerate Analysis.")

    # ── Q3: Playoff Success vs Pay Efficiency — 4-Quadrant ──────────────────
    if _R2_MODE or os.path.exists(table_csv):
        try:
            _q3 = _read_csv(table_csv)
            if {"Avg_Gap_M", "Avg_Wins", "Playoff_Apps"}.issubset(_q3.columns):
                st.markdown("---")
                st.subheader("Playoff Success vs Pay Efficiency — 4-Quadrant View")
                st.caption(
                    "X-axis: average $ above/below the cost effective line (negative = efficient). "
                    "Y-axis: 5-year average wins. "
                    "Color = playoff appearances 2021–2025. "
                    "Star marker = World Series champion. "
                    "Dotted lines split at break-even efficiency (x=0) and .500 record (y=81)."
                )

                _tc = "Abbr" if "Abbr" in _q3.columns else "Team"
                _q3["_tier"] = _q3["Playoff_Apps"].apply(
                    lambda x: "4–5 apps" if x >= 4 else ("2–3 apps" if x >= 2 else ("1 app" if x == 1 else "No playoffs"))
                )
                _tier_cfg = {
                    "4–5 apps":  ("#f59e0b", 12),
                    "2–3 apps":  ("#3b82f6", 10),
                    "1 app":     ("#94a3b8", 9),
                    "No playoffs": ("#4b5563", 9),
                }

                _fig_q3 = go.Figure()
                for tier, (clr, sz) in _tier_cfg.items():
                    _t = _q3[_q3["_tier"] == tier]
                    if _t.empty:
                        continue
                    # WS winners get star, rest get circle
                    for sym, mask in [("star", _t["WS_Wins"] > 0), ("circle", _t["WS_Wins"] == 0)]:
                        _sub = _t[mask]
                        if _sub.empty:
                            continue
                        _fig_q3.add_trace(go.Scatter(
                            x=_sub["Avg_Gap_M"],
                            y=_sub["Avg_Wins"],
                            mode="markers+text",
                            text=_sub[_tc],
                            textposition="top center",
                            textfont=dict(size=9, color="#dbeafe"),
                            marker=dict(
                                color=clr,
                                size=sz + (6 if sym == "star" else 0),
                                symbol=sym,
                                opacity=0.9,
                                line=dict(color="#0e1117", width=1),
                            ),
                            name=tier + (" ★" if sym == "star" else ""),
                            showlegend=(sym == "circle"),   # show one legend entry per tier
                            hovertemplate=(
                                "%{text}<br>"
                                "Avg Wins: %{y:.1f}<br>"
                                "Avg Gap $M: %{x:+.0f}<br>"
                                f"Playoff Apps: {_sub['Playoff_Apps'].iloc[0] if len(_sub) == 1 else 'varies'}<extra></extra>"
                            ),
                        ))

                # Quadrant dividers
                _fig_q3.add_vline(x=0,  line=dict(color="#7aa2c0", dash="dot", width=1.2), opacity=0.5)
                _fig_q3.add_hline(y=81, line=dict(color="#7aa2c0", dash="dot", width=1.2), opacity=0.5)

                # Quadrant labels
                _xmin = _q3["Avg_Gap_M"].min(); _xmax = _q3["Avg_Gap_M"].max()
                _ymin = _q3["Avg_Wins"].min();  _ymax = _q3["Avg_Wins"].max()
                for x_a, y_a, txt, xanch in [
                    (_xmin, _ymax, "Smart Champions",  "left"),
                    (_xmax, _ymax, "Bought Wins",      "right"),
                    (_xmin, _ymin, "Efficient Losers", "left"),
                    (_xmax, _ymin, "Expensive Losers", "right"),
                ]:
                    _fig_q3.add_annotation(
                        x=x_a, y=y_a, text=f"<i>{txt}</i>",
                        showarrow=False, xanchor=xanch,
                        font=dict(size=11, color="#4a6a8a"),
                    )

                _fig_q3.update_layout(**_pt(
                    title="Wins vs Efficiency — Playoff Performance (2021–2025)",
                    xaxis=dict(title="Avg $M vs Frontier  (negative = efficient)"),
                    yaxis=dict(title="Avg Wins (2021–2025)"),
                    height=540,
                    showlegend=True,
                ))
                st.plotly_chart(_fig_q3, width="stretch", config={"displayModeBar": False})
        except Exception:
            pass  # silently skip if ranking CSV unavailable

    st.markdown("---")

    if _R2_MODE or os.path.exists(table_csv):
        st.subheader("Full AL / NL Ranking Table")
        rank_df = _read_csv(table_csv)

        league_filter = st.radio("League", ["Both", "AL", "NL"], horizontal=True, key="league_filter")
        sort_col = st.selectbox(
            "Sort by",
            ["Leag_Rank", "Avg_Gap_M", "Avg_Wins", "WS_Wins", "Playoff_Apps",
             "Avg_Pay_M", "Avg_$/WAR_M", "Playoff_Rnds"],
            key="league_sort_col",
        )
        sort_asc = st.checkbox("Ascending", value=True, key="league_sort_asc")

        tbl = rank_df.copy()
        if league_filter != "Both":
            tbl = tbl[tbl["League"] == league_filter]
        tbl = tbl.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

        show_cols = [
            "League", "Division", "Team", "W2025", "W2024", "W2023", "W2022", "W2021",
            "Avg_Wins", "Avg_Pay_M", "Avg_$/WAR_M", "Avg_Gap_M",
            "Playoff_Apps", "WS_Apps", "WS_Wins", "Playoff_Rnds",
        ]
        show_cols = [c for c in show_cols if c in tbl.columns]

        def _highlight_gap(val):
            try:
                v = float(val)
                if v <= -50: return "background-color: #1a4731; color: white"
                if v <= 0:   return "background-color: #1a3a20; color: white"
                if v <= 50:  return "background-color: #3a1a1a; color: white"
                return "background-color: #5a1a1a; color: white"
            except (TypeError, ValueError):
                return ""

        def _highlight_ws(val):
            try:
                if int(val) > 0:
                    return "background-color: #7a5a00; color: #FFD700; font-weight: bold"
            except (TypeError, ValueError):
                pass
            return ""

        styled = (
            tbl[show_cols]
            .rename(columns={
                "Avg_Gap_M":   "$vsLine ($M)",
                "Avg_Pay_M":   "Avg Payroll ($M)",
                "Avg_$/WAR_M": "$/fWAR ($M)",
                "Avg_Wins":    "Avg W",
                "Playoff_Apps":"Playoffs",
                "WS_Apps":     "WS Apps",
                "WS_Wins":     "WS Wins",
                "Playoff_Rnds":"PO Rounds",
            })
            .style
            .applymap(_highlight_gap, subset=["$vsLine ($M)"])
            .applymap(_highlight_ws,  subset=["WS Wins"])
            .format({
                "Avg W":             "{:.1f}",
                "Avg Payroll ($M)":  "${:.0f}M",
                "$/WAR ($M)":        "${:.1f}M",
                "$vsLine ($M)":      "${:+.0f}M",
            }, na_rep="-")
        )
        st.dataframe(styled, width="stretch", height=600)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download Ranking Table (CSV)",
                data=rank_df.to_csv(index=False),
                file_name="al_nl_ranking_table.csv",
                mime="text/csv",
                key="dl_ranking",
            )
        if _R2_MODE or os.path.exists(detail_csv):
            detail_df = _read_csv(detail_csv)
            with dl2:
                st.download_button(
                    "Download Year-by-Year Detail (CSV)",
                    data=detail_df.to_csv(index=False),
                    file_name="efficiency_detail.csv",
                    mime="text/csv",
                    key="dl_detail",
                )

        st.markdown("---")
        st.subheader("Key Finding: Efficiency vs Winning")
        c1, c2, c3 = st.columns(3)
        if _R2_MODE or os.path.exists(detail_csv):
            det = _read_csv(detail_csv)
            from scipy.stats import pearsonr, pointbiserialr
            r_w,  _ = pearsonr(det["dollar_gap_M"], det["Wins"])
            r_po, _ = pointbiserialr(det["dollar_gap_M"], det["in_playoffs"].astype(int))
            c1.metric("$ Gap vs Wins (r)",    f"{r_w:+.3f}",
                      help="Pearson r. Negative = efficient teams win more.")
            c2.metric("$ Gap vs Playoffs (r)", f"{r_po:+.3f}",
                      help="Point-biserial r. Negative = efficient teams make playoffs more.")
            c3.metric("Interpretation", "Efficient teams win more",
                      delta="Significant at p<0.001", delta_color="normal")

        # ── PPR vs Wins correlation ──────────────────────────────────────
        st.markdown("---")
        st.subheader("Pay vs Play Ratio (PPR) vs Team Performance")
        st.caption(
            "PPR = projected WAR across full contract ÷ total contract $M. "
            "Higher PPR = more production per dollar committed over the life of the contract. "
            "Avg PPR = mean across all rostered players; Cum PPR = total roster PPR sum. "
            "Does building a high-PPR roster translate to winning?"
        )

        _COMBINED = _data_url("data/mlb_combined_2021_2025.csv")
        _IND_2025  = _data_url("data/2025mlbshared.csv")
        if _R2_MODE or os.path.exists(_COMBINED):
            import hashlib as _hl
            _fhash = "r2-remote" if _R2_MODE else _hl.md5(open(_COMBINED, "rb").read(65536)).hexdigest()
            sim_df = _cached_simulator_data(_COMBINED, _IND_2025, _fhash)

            ppr_raw = sim_df.dropna(subset=["PPR", "Team"])
            team_agg = (
                ppr_raw.groupby("Team")["PPR"]
                .agg(avg_ppr="mean", cum_ppr="sum")
                .reset_index()
            )

            if "Abbr" in rank_df.columns:
                ppr_wins = team_agg.merge(
                    rank_df[["Abbr", "Team", "Avg_Wins", "W2025", "League"]],
                    left_on="Team", right_on="Abbr", how="inner",
                )
                # After merge: Team_x = abbreviation, Team_y = full name
                ppr_wins = ppr_wins.rename(columns={"Team_x": "Abbr_code", "Team_y": "TeamName"})

                if len(ppr_wins) >= 5:
                    from scipy.stats import pearsonr as _pr
                    r_avg, _ = _pr(ppr_wins["avg_ppr"], ppr_wins["Avg_Wins"])
                    r_cum, _ = _pr(ppr_wins["cum_ppr"], ppr_wins["Avg_Wins"])
                    r_avg25, _ = _pr(ppr_wins["avg_ppr"], ppr_wins["W2025"])

                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Avg PPR vs Avg Wins (r)", f"{r_avg:+.3f}",
                               help="Pearson r: average team PPR vs 5-year avg wins.")
                    pm2.metric("Cum PPR vs Avg Wins (r)", f"{r_cum:+.3f}",
                               help="Pearson r: total roster PPR vs 5-year avg wins.")
                    pm3.metric("Avg PPR vs 2025 Wins (r)", f"{r_avg25:+.3f}",
                               help="Pearson r: average team PPR vs 2025 season wins.")
                    pm4.metric("Teams matched", str(len(ppr_wins)))

                    try:
                        al_mask = ppr_wins["League"] == "AL"
                        nl_mask = ppr_wins["League"] == "NL"
                        pc1, pc2 = st.columns(2)

                        for col, x_col, x_label, r_val in [
                            (pc1, "avg_ppr",  "Avg PPR",        r_avg),
                            (pc2, "cum_ppr",  "Cumulative PPR", r_cum),
                        ]:
                            z = np.polyfit(ppr_wins[x_col], ppr_wins["Avg_Wins"], 1)
                            x_line = np.linspace(ppr_wins[x_col].min(), ppr_wins[x_col].max(), 100)
                            y_line = np.poly1d(z)(x_line)

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=ppr_wins.loc[al_mask, x_col],
                                y=ppr_wins.loc[al_mask, "Avg_Wins"],
                                mode="markers+text",
                                text=ppr_wins.loc[al_mask, "Abbr_code"],
                                textposition="top center",
                                textfont=dict(color="#7aa2c0", size=9),
                                marker=dict(color="#3b82f6", size=9, opacity=0.9,
                                            line=dict(color="#1e3a5f", width=1)),
                                name="AL",
                                hovertemplate="%{text}<br>" + x_label + ": %{x:.2f}<br>Avg Wins: %{y:.1f}<extra></extra>",
                            ))
                            fig.add_trace(go.Scatter(
                                x=ppr_wins.loc[nl_mask, x_col],
                                y=ppr_wins.loc[nl_mask, "Avg_Wins"],
                                mode="markers+text",
                                text=ppr_wins.loc[nl_mask, "Abbr_code"],
                                textposition="top center",
                                textfont=dict(color="#f59e0b", size=9),
                                marker=dict(color="#f59e0b", size=9, opacity=0.9,
                                            line=dict(color="#1e3a5f", width=1)),
                                name="NL",
                                hovertemplate="%{text}<br>" + x_label + ": %{x:.2f}<br>Avg Wins: %{y:.1f}<extra></extra>",
                            ))
                            fig.add_trace(go.Scatter(
                                x=x_line, y=y_line,
                                mode="lines",
                                line=dict(color="#60a5fa", dash="dash", width=1.5),
                                opacity=0.7,
                                showlegend=False,
                                hoverinfo="skip",
                            ))
                            fig.update_layout(**_pt(
                                title=f"{x_label} vs Avg Wins  (r = {r_val:+.3f})",
                                xaxis=dict(title=x_label),
                                yaxis=dict(title="Avg Wins (2021–2025)"),
                                height=380,
                                showlegend=True,
                                margin=dict(l=50, r=20, t=45, b=50),
                            ))
                            with col:
                                st.plotly_chart(fig, width="stretch",
                                                config={"displayModeBar": False})
                    except Exception as _e:
                        st.info(f"PPR correlation chart unavailable: {_e}")
                else:
                    st.info("Not enough team PPR data to compute correlation.")
        else:
            st.info("Player database not found — cannot compute PPR correlation.")

    else:
        st.info("al_nl_ranking_table.csv not found -- click Regenerate Analysis.")

    # ── Pre-Arbitration Value Analysis ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Pre-Arbitration Talent: Peak Performance vs Cost")
    st.caption(
        "Pre-arb players earn near league minimum (~$0.7–1.0M), making high-WAR pre-arb talent "
        "the greatest value in baseball. This section evaluates each pre-arb player's peak "
        "single-season WAR (2021–2025) against their 2026 salary."
    )

    _pa_combined = os.path.join(_ROOT_DIR, "data", "mlb_combined_2021_2025.csv")
    _pa_payroll  = os.path.join(_ROOT_DIR, "2026 Payroll")

    if os.path.exists(_pa_combined) and os.path.exists(_pa_payroll):
        try:
            import hashlib as _hl26
            _pa_hash = _hl26.md5(open(_pa_combined, "rb").read(65536)).hexdigest()
            _pa_hist = _cached_player_history(_pa_combined, _pa_hash)
            _pa_df26 = _cached_2026_payroll(_pa_payroll, _pa_combined, _dir_hash(_pa_payroll))

            # Pre-arb players from 2026 payroll
            _pre = _pa_df26[_pa_df26["Stage_Clean"] == "Pre-Arb"].copy()

            if not _pre.empty:
                # Compute peak WAR (best single season 2021-2025) for each player
                _pa_hist["WAR_Total"] = pd.to_numeric(_pa_hist["WAR_Total"], errors="coerce")
                _pa_hist["Year"]      = pd.to_numeric(_pa_hist["Year"],      errors="coerce")
                _peak = (
                    _pa_hist[_pa_hist["WAR_Total"].notna()]
                    .loc[lambda d: d.groupby("Player")["WAR_Total"].idxmax()]
                    [["Player", "WAR_Total", "Year"]]
                    .rename(columns={"WAR_Total": "Peak_WAR", "Year": "Peak_Year"})
                )
                _pre = _pre.merge(_peak, on="Player", how="left")
                _pre = _pre.dropna(subset=["Peak_WAR"])
                _pre["Peak_WAR"]  = pd.to_numeric(_pre["Peak_WAR"],  errors="coerce")
                _pre["Salary_M"]  = pd.to_numeric(_pre["Salary_M"],  errors="coerce").fillna(0.72)
                _pre["Peak_Year"] = pd.to_numeric(_pre["Peak_Year"], errors="coerce").astype("Int64")

                # Summary metrics
                _solid_pre = _pre[_pre["Peak_WAR"] >= 2.0]
                _pa_m1, _pa_m2, _pa_m3, _pa_m4 = st.columns(4)
                _pa_m1.metric("Total Pre-Arb Players", str(len(_pre)))
                _pa_m2.metric("Peak WAR ≥ 2.0", str(len(_solid_pre)),
                              help="Pre-arb players with at least one 2+ WAR season.")
                _pa_m3.metric("Avg Peak WAR (≥1 WAR)", (
                    f"{_pre[_pre['Peak_WAR'] >= 1.0]['Peak_WAR'].mean():.1f}"
                    if len(_pre[_pre["Peak_WAR"] >= 1.0]) > 0 else "—"
                ))
                _pa_m4.metric("Max Peak WAR", (
                    f"{_pre['Peak_WAR'].max():.1f} — {_pre.loc[_pre['Peak_WAR'].idxmax(), 'Player']}"
                    if not _pre.empty else "—"
                ))

                # Colour by WAR tier
                def _peak_clr(w):
                    if w >= 4.0: return "#10b981"   # emerald — elite
                    if w >= 2.0: return "#3b82f6"   # blue — solid
                    if w >= 0.5: return "#f59e0b"   # amber — average
                    return "#6b7280"                # gray — depth

                _chart_pre = (
                    _pre[_pre["Peak_WAR"] >= 1.0]
                    .sort_values("Peak_WAR", ascending=True)
                    .tail(25)
                    .copy()
                )
                _chart_pre["color"] = _chart_pre["Peak_WAR"].apply(_peak_clr)
                _chart_pre["label"] = _chart_pre.apply(
                    lambda r: f"Peak: {r['Peak_WAR']:.1f} WAR ({int(r['Peak_Year']) if pd.notna(r['Peak_Year']) else '?'})",
                    axis=1,
                )

                _pa_col1, _pa_col2 = st.columns(2)

                with _pa_col1:
                    _fig_pa = go.Figure(go.Bar(
                        x=_chart_pre["Peak_WAR"],
                        y=_chart_pre["Player"],
                        orientation="h",
                        marker_color=_chart_pre["color"],
                        text=_chart_pre["label"],
                        textposition="outside",
                        textfont=dict(size=10, color="#7aa2c0"),
                        hovertemplate="%{y}<br>Peak WAR: %{x:.1f}<extra></extra>",
                    ))
                    _fig_pa.update_layout(**_pt(
                        title="Top Pre-Arb Players by Peak WAR (2021–2025)",
                        xaxis=dict(title="Peak Single-Season WAR"),
                        yaxis=dict(title=""),
                        height=500,
                        margin=dict(l=160, r=120, t=45, b=40),
                    ))
                    st.plotly_chart(_fig_pa, width="stretch",
                                    config={"displayModeBar": False})

                with _pa_col2:
                    # Scatter: Peak WAR vs Age — salary is meaningless (all ~league min)
                    _sc_pre = _pre[_pre["Peak_WAR"] >= 0.5].copy()
                    _sc_pre["Age"] = pd.to_numeric(_sc_pre["Age"], errors="coerce")
                    _sc_pre = _sc_pre.dropna(subset=["Age"])
                    _sc_pre["color"] = _sc_pre["Peak_WAR"].apply(_peak_clr)
                    # Label top 15 by peak WAR
                    _top15 = set(_sc_pre.nlargest(15, "Peak_WAR")["Player"])
                    _sc_pre["text_label"] = _sc_pre.apply(
                        lambda r: r["Player"].split()[-1] if r["Player"] in _top15 else "",
                        axis=1,
                    )

                    _fig_sc = go.Figure(go.Scatter(
                        x=_sc_pre["Age"],
                        y=_sc_pre["Peak_WAR"],
                        mode="markers+text",
                        text=_sc_pre["text_label"],
                        textposition="top center",
                        textfont=dict(size=9, color="#7aa2c0"),
                        marker=dict(
                            color=_sc_pre["color"],
                            size=9,
                            opacity=0.85,
                            line=dict(color="#1e3a5f", width=1),
                        ),
                        hovertemplate=(
                            "%{customdata}<br>"
                            "Age: %{x}<br>"
                            "Peak WAR: %{y:.1f}<extra></extra>"
                        ),
                        customdata=_sc_pre["Player"],
                    ))
                    # 2 WAR threshold — below this you're unlikely to crack a roster
                    _fig_sc.add_hline(
                        y=2.0,
                        line=dict(color="#f59e0b", dash="dash", width=1.2),
                        annotation_text="2 WAR — meaningful starter",
                        annotation_position="bottom right",
                        annotation_font=dict(color="#f59e0b", size=10),
                    )
                    # Age 25 reference — typical arb clock starts ~3 yrs service time
                    _fig_sc.add_vline(
                        x=25,
                        line=dict(color="#6b7280", dash="dot", width=1),
                        annotation_text="Age 25",
                        annotation_position="top right",
                        annotation_font=dict(color="#6b7280", size=9),
                    )
                    # Quadrant labels
                    _age_min = int(_sc_pre["Age"].min())
                    _age_max = int(_sc_pre["Age"].max())
                    _war_max = _sc_pre["Peak_WAR"].max()
                    _fig_sc.add_annotation(
                        x=_age_min + 0.3, y=_war_max * 0.97,
                        text="<b>Franchise Core</b>",
                        showarrow=False, xanchor="left",
                        font=dict(color="#22c55e", size=10),
                    )
                    _fig_sc.add_annotation(
                        x=_age_max - 0.3, y=_war_max * 0.97,
                        text="<b>Extension Decision</b>",
                        showarrow=False, xanchor="right",
                        font=dict(color="#f59e0b", size=10),
                    )
                    _fig_sc.update_layout(**_pt(
                        title="Peak WAR vs Age — Pre-Arb Players",
                        xaxis=dict(title="Age  (← More pre-arb years remaining)"),
                        yaxis=dict(title="Peak Single-Season WAR"),
                        height=500,
                    ))
                    st.plotly_chart(_fig_sc, width="stretch",
                                    config={"displayModeBar": False})

                # Top pre-arb value table
                st.markdown("##### Top Pre-Arb Value Players")
                _tbl_pre = (
                    _pre[_pre["Peak_WAR"] >= 1.0]
                    .sort_values("Peak_WAR", ascending=False)
                    .head(20)[["Player", "Team", "Position", "Age",
                                "Peak_WAR", "Peak_Year", "Salary_M", "W_per_M"]]
                    .reset_index(drop=True)
                )
                _tbl_pre.columns = ["Player", "Team", "Pos", "Age",
                                    "Peak WAR", "Peak Year", "Sal '26 $M", "WAR/$M"]
                st.dataframe(
                    _tbl_pre.style.format({
                        "Peak WAR": "{:.1f}", "Sal '26 $M": "${:.2f}M", "WAR/$M": "{:.2f}",
                        "Age": "{:.0f}", "Peak Year": "{:.0f}",
                    }, na_rep="—"),
                    hide_index=True,
                    use_container_width=True,
                    height=min(70 + len(_tbl_pre) * 35, 500),
                )
            else:
                st.info("No pre-arbitration players found in 2026 payroll data.")
        except Exception as _pa_err:
            st.info(f"Pre-arb analysis unavailable: {_pa_err}")
    else:
        st.info("Pre-arb analysis requires 2026 payroll data and mlb_combined_2021_2025.csv.")

    _render_feedback_widget("league")


# ---------------------------------------------------------------------------
# League Rankings page
# ---------------------------------------------------------------------------

def _render_rankings_page():
    """League Rankings -- WAR, Salary, and Efficiency ranked by team."""
    from pages.rankings import render as _rankings_render
    _rankings_render(
        data_url=_data_url,
        read_csv=_read_csv,
        team_logo_url=_team_logo_url,
        cached_mlbam_lookup=_cached_mlbam_lookup,
        razzball_path=_RAZZBALL_PATH,
        r2_mode=_R2_MODE,
    )

# ---------------------------------------------------------------------------
# Team Analysis page
# ---------------------------------------------------------------------------

def _render_team_analysis_page():
    """Full team deep-dive page — delegated to pages.team_analysis."""
    from app.pages.team_analysis import render
    render(_data_url, _read_csv, _load_enriched_roster, _team_logo_url,
           _fetch_2026_standings, _fetch_2026_standings_full, _fetch_2026_team_stats, _R2_MODE)


# ---------------------------------------------------------------------------
# Methodology & Data Sources page
# ---------------------------------------------------------------------------

def _render_glossary_page():
    """Methodology & data sources reference page — delegated to pages.methodology."""
    from app.pages.methodology import render as _meth_render
    _meth_render()


# ---------------------------------------------------------------------------
# Feedback page
# ---------------------------------------------------------------------------

def _render_feedback_page():
    """Feedback & suggestions page — delegated to pages.feedback."""
    from app.pages.feedback import render as _fb_render
    _fb_render()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _valid_pages = {"home", "league", "simulator", "roster_optimizer", "rankings", "glossary", "team", "feedback"}
    if "page" not in st.session_state:
        # First load or browser refresh — restore from URL query param
        qp = st.query_params.get("page", "home")
        st.session_state["page"] = qp if qp in _valid_pages else "home"

    # Keep URL bar in sync with current page (no extra rerun)
    try:
        st.query_params["page"] = st.session_state["page"]
    except Exception:
        pass

    page = st.session_state.get("page", "league")

    # Skip nav bar on the home page — it has its own navigation cards
    if page != "home":
        _render_nav_bar()

    if page == "home":
        _render_home_page()
    elif page == "rankings":
        _render_rankings_page()
    elif page == "league":
        _render_league_analysis()
    elif page == "simulator":
        _render_simulator_page()
    elif page == "roster_optimizer":
        base_cfg = _load_base_config(_DEFAULT_CONFIG) if os.path.exists(_DEFAULT_CONFIG) else {}
        _render_roster_optimizer_page(base_cfg)
    elif page == "team":
        _render_team_analysis_page()
    elif page == "glossary":
        _render_glossary_page()
    elif page == "feedback":
        _render_feedback_page()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
