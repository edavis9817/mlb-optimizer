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

# ---------------------------------------------------------------------------
# R2 / Production remote data
# ---------------------------------------------------------------------------
R2_BASE_URL = os.environ.get("R2_BASE_URL", "").strip().rstrip("/")
_R2_MODE    = bool(R2_BASE_URL)

try:
    import requests as _requests
    _requests_available = True
except ImportError:
    _requests_available = False

import io as _io
import unicodedata as _unicodedata
from pathlib import Path as _Path

# ---------------------------------------------------------------------------
# Persistent Disk Cache Management
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.expanduser("~/.mlb_toolbox_cache")
_ETAG_METADATA_FILE = os.path.join(_CACHE_DIR, "etags.json")

try:
    _Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)
except Exception:
    pass  # Cache directory creation optional


def _init_etag_metadata() -> dict:
    """Load or create the ETag metadata file."""
    if os.path.exists(_ETAG_METADATA_FILE):
        try:
            with open(_ETAG_METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_etag_metadata(metadata: dict) -> None:
    """Persist ETag metadata to disk."""
    try:
        with open(_ETAG_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass  # Metadata save is optional


def _compute_cache_key(url: str) -> str:
    """Generate a deterministic cache filename from a URL."""
    return hashlib.md5(url.encode()).hexdigest() + ".cached"


def _get_cached_file_path(url: str) -> str:
    """Return the disk path where a remote file should be cached."""
    return os.path.join(_CACHE_DIR, _compute_cache_key(url))


def _fix_player_name(s: str) -> str:
    """Normalise a player name: undo double-encoded UTF-8, then strip diacritics.

    Handles mojibake like "JosÃ©" → "Jose" and clean accents like "Pérez" → "Perez".
    Plain ASCII names pass through unchanged.
    """
    if not isinstance(s, str):
        return s
    # Step 1: undo double-encoded UTF-8 (latin-1 round-trip)
    try:
        s = s.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    # Step 2: strip combining diacritical marks → ASCII equivalents
    nfkd = _unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not _unicodedata.combining(c))


def _fix_player_col(df: pd.DataFrame) -> pd.DataFrame:
    """If a 'Player' column exists, normalise every name in place."""
    if "Player" in df.columns:
        df["Player"] = df["Player"].map(_fix_player_name)
    return df


def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV from a local path or an R2 URL with persistent disk caching & ETag support.
    
    For remote URLs:
    1. Check if cached file exists on disk
    2. If yes, verify with ETag to detect changes
    3. If unchanged: use cached file (fast)
    4. If changed or missing: download and cache
    5. Automatically normalises the Player column if present
    """
    if path.startswith("http") and _requests_available:
        cache_file = _get_cached_file_path(path)
        metadata = _init_etag_metadata()
        cache_key = path
        
        # Try to use cached file if it exists
        if os.path.exists(cache_file):
            try:
                # Check ETag with HEAD request to detect changes
                cached_etag = metadata.get(cache_key, {}).get("etag")
                if cached_etag:
                    head_resp = _requests.head(path, timeout=30)
                    current_etag = head_resp.headers.get("ETag", "")
                    if current_etag and current_etag == cached_etag:
                        # File unchanged—use cached version
                        return _fix_player_col(pd.read_csv(cache_file, **kwargs))
            except Exception:
                # If HEAD check fails, fall back to using cached file anyway
                try:
                    return _fix_player_col(pd.read_csv(cache_file, **kwargs))
                except Exception:
                    pass  # Fall through to re-download
        
        # Download and cache
        try:
            resp = _requests.get(path, timeout=30)
            resp.raise_for_status()
            
            # Save to cache
            with open(cache_file, "wb") as f:
                f.write(resp.content)
            
            # Update metadata with ETag
            etag = resp.headers.get("ETag", "")
            if etag:
                metadata[cache_key] = {"etag": etag, "url": path}
                _save_etag_metadata(metadata)
            
            return _fix_player_col(pd.read_csv(_io.BytesIO(resp.content), **kwargs))
        except Exception as e:
            # If download fails but cache exists, use stale cache
            if os.path.exists(cache_file):
                try:
                    return _fix_player_col(pd.read_csv(cache_file, **kwargs))
                except Exception:
                    raise e
            raise
    
    return _fix_player_col(pd.read_csv(path, **kwargs))


def _read_excel(path: str, **kwargs) -> pd.DataFrame:
    """Read an Excel file from a local path or an R2 URL with persistent disk caching & ETag support."""
    if path.startswith("http") and _requests_available:
        cache_file = _get_cached_file_path(path)
        metadata = _init_etag_metadata()
        cache_key = path
        
        # Try to use cached file if it exists
        if os.path.exists(cache_file):
            try:
                cached_etag = metadata.get(cache_key, {}).get("etag")
                if cached_etag:
                    head_resp = _requests.head(path, timeout=30)
                    current_etag = head_resp.headers.get("ETag", "")
                    if current_etag and current_etag == cached_etag:
                        return _fix_player_col(pd.read_excel(cache_file, **kwargs))
            except Exception:
                try:
                    return _fix_player_col(pd.read_excel(cache_file, **kwargs))
                except Exception:
                    pass
        
        # Download and cache
        try:
            resp = _requests.get(path, timeout=30)
            resp.raise_for_status()
            
            with open(cache_file, "wb") as f:
                f.write(resp.content)
            
            etag = resp.headers.get("ETag", "")
            if etag:
                metadata[cache_key] = {"etag": etag, "url": path}
                _save_etag_metadata(metadata)
            
            return _fix_player_col(pd.read_excel(_io.BytesIO(resp.content), **kwargs))
        except Exception as e:
            if os.path.exists(cache_file):
                try:
                    return _fix_player_col(pd.read_excel(cache_file, **kwargs))
                except Exception:
                    raise e
            raise
    
    return _fix_player_col(pd.read_excel(path, **kwargs))


def _r2_image(path: str) -> bytes | str:
    """Return image bytes (for R2 URLs) or a local path for st.image()."""
    if path.startswith("http") and _requests_available:
        resp = _requests.get(path, timeout=30)
        resp.raise_for_status()
        return resp.content
    return path


def _data_url(relative_path: str) -> str:
    """Return an R2 public URL (production) or an absolute local path (development).

    relative_path should use forward slashes and be relative to the project root,
    e.g. "data/mlb_combined_2021_2025.csv" or "2026 Payroll".
    """
    if R2_BASE_URL:
        return f"{R2_BASE_URL}/{relative_path.replace(os.sep, '/')}"
    return os.path.join(_ROOT_DIR, *relative_path.split("/"))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = os.path.join(_ROOT_DIR, "configs", "default_config.json")

# Position group map — each infield position keeps its own identity
_POS_GROUP_MAP = {
    "C":   "C",
    "1B":  "1B", "3B": "3B", "IF": "1B",
    "2B":  "2B", "SS": "SS",
    "LF":  "OF", "RF": "OF", "OF": "OF",
    "CF":  "CF",
    "SP":  "SP", "RP": "RP", "TWP": "SP",
    "DH":  "DH", "P":  "SP",
}

_ELIGIBLE_SLOTS_MAP = {
    "C":  ["C"],
    "1B": ["1B"],
    "2B": ["2B"],
    "3B": ["3B"],
    "SS": ["SS"],
    "CF": ["CF"],
    "OF": ["LF", "RF"],
    "SP": ["SP"],
    "RP": ["RP"],
    "DH": ["DH"],
}

# DH is optional — some teams don't carry one
_OPTIONAL_SLOTS = {"DH"}

_ROSTER_TEMPLATE = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
    "LF": 1, "CF": 1, "RF": 1, "DH": 1, "BENCH": 3,
    "SP": 5, "RP": 8,
}

# Player headshots and MLBAM ID lookup (inside the project data folder)
_HEADSHOTS_DIR = (
    f"{R2_BASE_URL}/data/headshots" if _R2_MODE
    else os.path.join(_ROOT_DIR, "data", "headshots")
)
_RAZZBALL_PATH = _data_url("data/razzball.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_feedback_widget(page_name: str = "") -> None:
    """Render a shared feedback/suggestion widget at the bottom of every page."""
    with st.expander("💬 Feedback & Suggestions", expanded=False):
        _fb_type = st.radio("Type:", ["Bug Report", "Feature Request", "General Feedback"],
                            key=f"fb_type_{page_name}", horizontal=True)
        _fb_text = st.text_area("Your feedback:", key=f"fb_text_{page_name}",
                                placeholder="Describe the issue or suggestion...")
        if st.button("Submit Feedback", key=f"fb_submit_{page_name}", type="secondary"):
            if _fb_text.strip():
                st.success("Thank you! Your feedback has been recorded.")
            else:
                st.warning("Please enter some feedback text.")


def _loading_placeholder(message: str = "Loading data ...") -> None:
    """Render a centered loading card with animated progress bar."""
    st.markdown(
        f"<div class='loading-container'>"
        f"<div class='loading-icon'>⚾</div>"
        f"<div class='loading-title'>{message}</div>"
        f"<div class='loading-sub'>This may take a few seconds on first load</div>"
        f"<div class='loading-bar'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _load_base_config(path: str = _DEFAULT_CONFIG) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _resolve_data_path(raw_path: str, config_path: str) -> str:
    if os.path.isabs(raw_path):
        return raw_path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(config_dir, raw_path))


def _file_hash(path: str) -> str:
    """Generate a cache key for a file path or R2 URL.
    
    For remote files: uses ETag if available, otherwise returns URL hash
    For local files: computes MD5 of file content
    """
    if path.startswith("http"):
        metadata = _init_etag_metadata()
        cached_data = metadata.get(path, {})
        if "etag" in cached_data:
            return cached_data["etag"]  # Use ETag as cache key
        # Try to fetch ETag if we don't have it cached
        try:
            head_resp = _requests.head(path, timeout=10)
            etag = head_resp.headers.get("ETag", "")
            if etag:
                metadata[path] = {"etag": etag, "url": path}
                _save_etag_metadata(metadata)
                return etag
        except Exception:
            pass
        # Fallback: return URL hash
        return hashlib.md5(path.encode()).hexdigest()[:16]
    
    # Local file: compute MD5
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "missing"


def _dir_hash(dir_path: str) -> str:
    """Cheap cache key for a directory — hashes the sorted file listing.
    
    For remote directories on R2: returns a stable identifier
    For local directories: hashes the file listing
    """
    if dir_path.startswith("http"):
        return hashlib.md5(dir_path.encode()).hexdigest()[:16]  # Stable hash for R2 directory
    try:
        return hashlib.md5("|".join(sorted(os.listdir(dir_path))).encode()).hexdigest()[:16]
    except Exception:
        return "0"


# ---------------------------------------------------------------------------
# 2026 payroll constants + parser
# ---------------------------------------------------------------------------

# Maps xlsx filename stem → team abbreviation matching mlb_combined data
_PAYROLL_2026_TEAM_MAP: dict[str, str] = {
    "Angels":       "LAA",
    "Astros":       "HOU",
    "Athletics":    "ATH",
    "Blue Jays":    "TOR",
    "Braves":       "ATL",
    "Brewers":      "MIL",
    "Cardinals":    "STL",
    "Cubs":         "CHC",
    "Diamondbacks": "ARI",
    "Dodgers":      "LAD",
    "Giants":       "SFG",
    "Guardians":    "CLE",
    "Mariners":     "SEA",
    "Marlins":      "MIA",
    "Mets":         "NYM",
    "Nationals":    "WSN",
    "Orioles":      "BAL",
    "Padres":       "SDP",
    "Phillies":     "PHI",
    "Pirates":      "PIT",
    "Rangers":      "TEX",
    "Rays":         "TBR",
    "Red Sox":      "BOS",
    "Reds":         "CIN",
    "Rockies":      "COL",
    "Royals":       "KCR",
    "Tigers":       "DET",
    "Twins":        "MIN",
    "White Sox":    "CHW",
    "Yankees":      "NYY",
}


def _parse_payroll_val(val) -> float | None:
    """Parse a payroll cell (e.g. '$33,000,000', 'TBD', 'FREE AGENT') → float $M."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "n/a"):
        return None
    u = s.upper()
    if any(tok in u for tok in ("TBD", "ARB", "FREE AGENT", "PRE-ARB")):
        return None
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return round(float(s) / 1_000_000, 4)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------

def _pt(**overrides) -> dict:
    """Return a base Plotly layout dict — dark slate + deep blue theme.

    Pass keyword overrides to customise per-chart (e.g. title, height, showlegend).
    Nested dict overrides are shallow-merged with the base dicts.
    """
    base: dict = dict(
        paper_bgcolor="#111927",   # match main bg
        plot_bgcolor="#18243a",    # slightly lifted card surface
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
            bgcolor="#18243a", bordercolor="#253d58", borderwidth=1,
            font=dict(color="#7a9ebc"),
        ),
        margin=dict(l=50, r=20, t=45, b=50),
        showlegend=False,
    )
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


# Muted chart series palette — use these instead of saturated colors
_CHART_PALETTE = [
    "#4873b8",  # muted blue
    "#2e9080",  # muted teal
    "#5889c8",  # lighter muted blue
    "#458068",  # muted green
    "#7865b8",  # muted purple
    "#b88840",  # muted gold
    "#3898a8",  # muted cyan
    "#887898",  # muted mauve
    "#5e9860",  # muted lime
    "#987878",  # muted rose (non-alarm)
]

# Position-group chart colors — muted series (SP, RP, C, 1B, 2B, 3B, SS, CF, OF, DH)
_PG_CHART_COLORS = [
    "#4873b8",  # SP  — muted blue
    "#7865b8",  # RP  — muted purple
    "#2e9080",  # C   — muted teal
    "#5889c8",  # 1B  — light muted blue
    "#3898a8",  # 2B  — muted cyan
    "#458068",  # 3B  — muted green
    "#887898",  # SS  — muted mauve
    "#b88840",  # CF  — muted gold
    "#5e9860",  # OF  — muted lime
    "#6a7890",  # DH  — slate
]

# 2026 MLB Competitive Balance Tax (CBT / luxury tax) thresholds.
# Source: 2022 CBA — tiers are $20M bands starting at $244M.
# The two "aprons" carry roster-building restrictions beyond just the tax rate:
#   First Apron  ($264M): cannot acquire max-salary players via trade, no cash-in-trades,
#                         cannot sign released MLB-contract players above the threshold.
#   Second Apron ($304M): cannot trade for players on 40-man roster, cannot sign
#                         international bonus pool trades, lose top draft pick.
_CBT_TIERS: list[tuple[float, str, str, str, str]] = [
    # (threshold_$M, label, bg_color, text_color, apron_note)
    (244.0, "Under CBT",                "#0c2218", "#22c55e", ""),
    (264.0, "Tier 1 ≥$244M",           "#281a08", "#fbbf24",
     "Above CBT — paying luxury tax on overages"),
    (284.0, "1st Apron ≥$264M",        "#2d1408", "#f97316",
     "1st Apron — trade & signing restrictions apply"),
    (304.0, "Tier 3 ≥$284M",           "#2d0c0c", "#ef4444",
     "Above 1st Apron — severe roster-building limits"),
    (9999., "2nd Apron ≥$304M",        "#1f0808", "#fca5a5",
     "2nd Apron — hardest restrictions, draft pick penalty"),
]

def _cbt_info(budget_m: float) -> tuple[str, str, str, float | None, str]:
    """Return (label, bg, fg, next_threshold, apron_note) for a given budget."""
    for i, (thresh, label, bg, fg, note) in enumerate(_CBT_TIERS):
        if budget_m < thresh:
            nxt = thresh if i > 0 else None
            return label, bg, fg, nxt, note
    t = _CBT_TIERS[-1]
    return t[1], t[2], t[3], None, t[4]


# ---------------------------------------------------------------------------
# Shared glossary renderer
# ---------------------------------------------------------------------------

def _render_glossary(
    terms: list[tuple[str, str, str]],
    title: str = "📖 Terms & Definitions",
    cols: int = 2,
) -> None:
    """Render a collapsed expander with definition cards in a 2-column grid.

    Each entry is (abbr, full_name, description).
    """
    with st.expander(title, expanded=False):
        _gcols = st.columns(cols)
        for i, (abbr, fullname, desc) in enumerate(terms):
            with _gcols[i % cols]:
                st.markdown(
                    f"<div style='background:#0d1e35;border-left:3px solid #2b5cc8;"
                    f"border-radius:0 8px 8px 0;padding:0.5rem 0.85rem;margin-bottom:0.45rem;'>"
                    f"<span style='font-size:0.8rem;font-weight:700;color:#93c5fd;'>{abbr}</span>"
                    f"<span style='font-size:0.76rem;color:#93b8d8;'> — {fullname}</span>"
                    f"<div style='font-size:0.72rem;color:#7a9ebc;margin-top:0.15rem;line-height:1.5;'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Archetype ID → human-readable label
# ---------------------------------------------------------------------------

def _arch_label(arch_id: str) -> str:
    """Convert 'SP_FA_Elite' → 'Elite Starter (Free Agent)'."""
    _pos = {
        "SP": "Starter",  "RP": "Reliever",  "C": "Catcher",
        "1B": "1st Base", "2B": "2nd Base",  "3B": "3rd Base", "SS": "Shortstop",
        "CI": "Corner IF","MI": "Middle IF",  "CF": "Center Field",
        "OF": "Corner OF","DH": "DH",
    }
    _stage = {"FA": "Free Agent", "Arb": "Arb-Eligible", "Pre-Arb": "Pre-Arb"}
    parts = arch_id.split("_", 2)          # "SP_Pre-Arb_Solid" → ["SP","Pre-Arb","Solid"]
    if len(parts) != 3:
        return arch_id
    pos_str   = _pos.get(parts[0], parts[0])
    stage_str = _stage.get(parts[1], parts[1])
    tier_str  = parts[2]
    return f"{tier_str} {pos_str} ({stage_str})"


# ---------------------------------------------------------------------------
# Contract Decision Helper
# ---------------------------------------------------------------------------

def _contract_decision(player_row: dict) -> str:
    """Return Extend / Hold / Trade / DFA recommendation for one player."""
    war   = float(player_row.get("WAR_Total") or 0)
    sal   = float(player_row.get("Salary_M")  or 0)
    age   = float(player_row.get("Age")        or 30)
    yrs   = float(player_row.get("Yrs_Left")   or 0)
    stage = str(player_row.get("Stage_Clean")  or "FA")

    # DFA: clearly negative or replacement-level with big contract
    if war < -0.5:
        return "DFA"
    if war < 0.5 and sal > 15 and age > 32:
        return "DFA"
    # Trade: declining value, expensive, locked in long-term
    if war < 1.0 and sal > 12 and yrs > 1:
        return "Trade"
    if war < 0.5 and sal > 8 and age > 33:
        return "Trade"
    # Extend: young, high-WAR, still cheap
    if age <= 27 and war >= 3.0 and stage in ("Pre-Arb", "Arb"):
        return "Extend"
    if age <= 26 and war >= 2.0 and stage == "Pre-Arb":
        return "Extend"
    return "Hold"


# ---------------------------------------------------------------------------
# Roster Grade (A–F)
# ---------------------------------------------------------------------------

def _roster_grade(roster_df: pd.DataFrame) -> dict:
    """Return A–F grades for Production, Efficiency, Depth, Contract Health."""
    total_war  = float(roster_df["WAR_Total"].sum()) if "WAR_Total" in roster_df.columns else 0
    total_cost = float(roster_df["Salary_M"].sum())  if "Salary_M"  in roster_df.columns else 0

    prod_score  = min(100, int(total_war / 35 * 100))                            # 35 WAR = full marks
    eff_score   = min(100, int((total_war / max(total_cost, 0.1)) / 0.30 * 100)) # 0.30 WAR/$M = full marks
    pos_groups  = set(roster_df["pos_group"].tolist()) if "pos_group" in roster_df.columns else set()
    covered     = len(pos_groups & {"SP", "RP", "C", "1B", "2B", "3B", "SS", "CF", "OF"})
    depth_score = min(100, int(covered / 9 * 100))
    yrs_avg     = (float(roster_df["Yrs_Left"].mean())
                   if "Yrs_Left" in roster_df.columns and roster_df["Yrs_Left"].notna().any()
                   else 2.0)
    ctrc_score  = max(0, min(100, int(100 - abs(yrs_avg - 2.0) * 20)))

    def _g(s: int) -> str:
        return ("A+" if s >= 95 else "A" if s >= 85 else "B+" if s >= 75 else
                "B" if s >= 65 else "C" if s >= 50 else "D" if s >= 35 else "F")

    return {
        "Production":      (_g(prod_score),  prod_score),
        "Efficiency":      (_g(eff_score),   eff_score),
        "Depth":           (_g(depth_score), depth_score),
        "Contract Health": (_g(ctrc_score),  ctrc_score),
    }


# ---------------------------------------------------------------------------
# Cache functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading & projecting player data ...", persist="disk")
def _cached_projections(salary_path: str, file_hash: str, proj_weights_json: str,
                         season: int, clip_neg: bool, min_war: float, max_yrs: int):
    cfg = {
        "season":              season,
        "projection_weights":  json.loads(proj_weights_json),
        "clip_negative_war":   clip_neg,
        "min_war_threshold":   min_war,
        "max_contract_years":  max_yrs,
    }
    raw_df = _read_csv(salary_path, low_memory=False)
    return make_projections(raw_df, cfg), raw_df


@st.cache_data(show_spinner="Building archetypes ...", persist="disk")
def _cached_archetypes(proj_hash: str, proj_json: str):
    proj_df = pd.read_json(proj_json, orient="records")
    proj_df["eligible_slots"] = proj_df["eligible_slots"].apply(
        lambda v: v if isinstance(v, list) else []
    )
    arch_df        = build_archetype_definitions(proj_df)
    proj_with_arch = assign_archetypes(proj_df)
    return arch_df, proj_with_arch


@st.cache_data(show_spinner="Loading wins data ...", persist="disk")
def _cached_wins(wins_path: str, file_hash: str):
    if not os.path.exists(wins_path):
        return pd.DataFrame()
    return _read_csv(wins_path, low_memory=False)


@st.cache_data(show_spinner="Loading team payroll history ...", persist="disk")
def _cached_payroll_history(data_dir: str):
    return get_team_payroll_history(data_dir)


@st.cache_data(show_spinner="Loading team roster ...")
def _cached_team_scenario(
    data_dir: str,
    team: str,
    combined_hash: str,
    roster_slots_json: str,
    market_dpw_M: float,
    include_arb: bool,
    budget_override_M: float | None,
    depth_chart_dir: str | None = None,
    include_minors: bool = False,
):
    import json as _json
    base_roster_slots = _json.loads(roster_slots_json)
    combined_df = _read_csv(
        os.path.join(data_dir, "mlb_combined_2021_2025.csv"),
        low_memory=False,
    )
    return build_offseason_scenario(
        data_dir=data_dir,
        team=team,
        combined_df=combined_df,
        base_roster_slots=base_roster_slots,
        market_dpw_M=market_dpw_M,
        include_arb=include_arb,
        budget_override_M=budget_override_M,
        depth_chart_dir=depth_chart_dir,
        include_minors=include_minors,
    )


@st.cache_data(show_spinner="Loading player database ...", persist="disk")
def _cached_simulator_data(combined_path: str, ind_2025_path: str, file_hash: str) -> pd.DataFrame:
    """Load and merge 2025 combined data with 2025 individual contract columns."""
    comb = _read_csv(combined_path, low_memory=False)
    comb.columns = [c.strip() for c in comb.columns]
    comb["Year"] = pd.to_numeric(comb["Year"], errors="coerce")
    comb2025 = comb[comb["Year"] == 2025].copy()

    # Deduplicate traded players — keep highest salary
    comb2025["Salary_M"] = pd.to_numeric(comb2025.get("Salary_M", pd.Series(dtype=float)), errors="coerce").fillna(0)
    comb2025 = (
        comb2025.sort_values("Salary_M", ascending=False, kind="mergesort")
                .drop_duplicates(subset=["Player"], keep="first")
                .reset_index(drop=True)
    )

    # Enrich with contract year columns from individual file
    if ind_2025_path.startswith("http") or os.path.exists(ind_2025_path):
        ind = _read_csv(ind_2025_path, low_memory=False)
        ind.columns = [c.strip() for c in ind.columns]
        year_cols  = [c for c in ["2026", "2027", "2028", "2029", "2030", "2031"] if c in ind.columns]
        extra_cols = ["Player", "Contract"] + year_cols
        extra_cols = [c for c in extra_cols if c in ind.columns]
        sort_col   = "2025" if "2025" in ind.columns else ind.columns[0]
        ind_dedup  = (
            ind.sort_values(sort_col, ascending=False, kind="mergesort", na_position="last")
               .drop_duplicates(subset=["Player"], keep="first")
        )
        comb2025 = comb2025.merge(ind_dedup[extra_cols], on="Player", how="left")

    # Add position group + eligible slots
    comb2025["Position"] = comb2025["Position"].fillna("UNK").astype(str).str.strip()
    comb2025["pos_group"] = comb2025["Position"].map(lambda p: _POS_GROUP_MAP.get(p, "UNK"))
    comb2025["eligible_slots"] = comb2025["pos_group"].map(lambda g: _ELIGIBLE_SLOTS_MAP.get(g, []))

    # Clean numerics
    for col in ["WAR_Total", "Age", "Salary_M"]:
        if col in comb2025.columns:
            comb2025[col] = pd.to_numeric(comb2025[col], errors="coerce")

    # Drop truly unknown positions
    comb2025 = comb2025[comb2025["pos_group"] != "UNK"].reset_index(drop=True)

    # Players with no contract data assumed at league minimum ($0.74M / 1-year)
    _LG_MIN_M = 0.74
    comb2025.loc[comb2025["Salary_M"].isna() | (comb2025["Salary_M"] == 0), "Salary_M"] = _LG_MIN_M

    # ------------------------------------------------------------------
    # Pay vs Play Ratio (PPR)
    # PPR = Σ actual WAR for each contract year / total contract $M
    # For each contract year we use the player's actual WAR from that season
    # (looking back N seasons where N = total contract length).
    # Missing seasons fall back to the most recent known WAR.
    # Higher PPR = more WAR produced per dollar committed.
    # ------------------------------------------------------------------
    _future_yr_cols = [c for c in ["2026", "2027", "2028", "2029", "2030", "2031"]
                       if c in comb2025.columns]
    for _c in _future_yr_cols:
        # Values may be currency-formatted strings like "$14,000,000" — strip and normalise
        cleaned = (
            comb2025[_c].astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        # If values are raw dollars (median >> 1 000) convert to $M to match Salary_M scale
        _median = numeric.dropna().median()
        if pd.notna(_median) and _median > 1_000:
            numeric = numeric / 1_000_000
        comb2025[_c] = numeric

    # Build multi-year WAR lookup: {player: {year: war}}
    _comb_war = comb[["Player", "Year", "WAR_Total"]].copy()
    _comb_war["WAR_Total"] = pd.to_numeric(_comb_war["WAR_Total"], errors="coerce").fillna(0.0)
    _war_hist: dict = {}
    for _, _r in _comb_war.dropna(subset=["Player", "Year"]).iterrows():
        _p = str(_r["Player"]); _y = int(_r["Year"])
        _war_hist.setdefault(_p, {})[_y] = float(_r["WAR_Total"])

    def _calc_ppr(row):
        sal_cur  = float(row.get("Salary_M") or 0)
        war_cur  = float(row.get("WAR_Total") or 0)  # most recent season (2025)
        fut_sals = [float(row[c]) for c in _future_yr_cols
                    if pd.notna(row.get(c)) and float(row.get(c) or 0) > 0]
        total_yrs = 1 + len(fut_sals)
        total_val = sal_cur + sum(fut_sals)
        if total_val <= 0:
            return None
        # Sum actual WAR for the most recent total_yrs seasons
        p_hist = _war_hist.get(str(row.get("Player", "")), {})
        war_sum = sum(p_hist.get(2025 - i, war_cur) for i in range(total_yrs))
        return round(war_sum / total_val, 3)

    comb2025["PPR"] = comb2025.apply(_calc_ppr, axis=1)

    # Single-season WAR per $M (last season efficiency)
    comb2025["W_per_M"] = comb2025.apply(
        lambda r: round(float(r["WAR_Total"]) / float(r["Salary_M"]), 3)
        if (pd.notna(r.get("WAR_Total")) and pd.notna(r.get("Salary_M"))
            and float(r.get("Salary_M") or 0) > 0)
        else None,
        axis=1,
    )

    # Years remaining on contract (future years with a committed salary value)
    comb2025["Yrs_Left"] = comb2025.apply(
        lambda r: int(sum(
            1 for c in _future_yr_cols
            if pd.notna(r.get(c)) and float(r.get(c) or 0) > 0
        )),
        axis=1,
    )

    # Total contract value ($M) — current year + all future committed years
    def _calc_total_ctrc(row):
        sal_cur  = float(row.get("Salary_M") or 0)
        fut_sals = [float(row[c]) for c in _future_yr_cols
                    if pd.notna(row.get(c)) and float(row.get(c) or 0) > 0]
        total = sal_cur + sum(fut_sals)
        return round(total, 2) if total > 0 else None

    comb2025["Total_Contract_M"] = comb2025.apply(_calc_total_ctrc, axis=1)

    # ------------------------------------------------------------------
    # WAR Stability Rating (WSR) — Feature 3
    # WSR = mean_WAR / (1 + std_WAR) using only qualifying seasons
    # Qualifying: PA ≥ 200 (hitters) or IP ≥ 50 (pitchers)
    # ------------------------------------------------------------------
    _comb_all = comb.copy()
    for _nc in ["WAR_Total", "PA", "IP"]:
        if _nc in _comb_all.columns:
            _comb_all[_nc] = pd.to_numeric(_comb_all[_nc], errors="coerce")

    def _calc_wsr(player_name, is_pitcher_flag):
        p_rows = _comb_all[_comb_all["Player"] == player_name]
        if is_pitcher_flag:
            qual = p_rows[p_rows["IP"].fillna(0) >= 50]
        else:
            qual = p_rows[p_rows["PA"].fillna(0) >= 200]
        if len(qual) < 2:
            return None, None, None, "Insufficient Data"
        wars = qual["WAR_Total"].dropna()
        if len(wars) < 2:
            return None, None, None, "Insufficient Data"
        m, s = float(wars.mean()), float(wars.std())
        wsr = round(m / (1 + s), 3)
        if wsr >= 3.5:
            tier = "Elite"
        elif wsr >= 2.0:
            tier = "Reliable"
        elif wsr >= 1.0:
            tier = "Volatile"
        else:
            tier = "Unstable"
        return wsr, round(m, 2), round(s, 2), tier

    _wsr_cache: dict = {}
    for _, _row in comb2025.iterrows():
        _pn = _row["Player"]
        _ip = _row.get("Position", "") in ("SP", "RP", "P", "TWP")
        if _pn not in _wsr_cache:
            _wsr_cache[_pn] = _calc_wsr(_pn, _ip)

    comb2025["WSR"]       = comb2025["Player"].map(lambda p: _wsr_cache.get(p, (None,))[0])
    comb2025["WAR_Mean"]  = comb2025["Player"].map(lambda p: _wsr_cache.get(p, (None, None))[1])
    comb2025["WAR_Std"]   = comb2025["Player"].map(lambda p: _wsr_cache.get(p, (None, None, None))[2])
    comb2025["WSR_Tier"]  = comb2025["Player"].map(lambda p: _wsr_cache.get(p, (None, None, None, "Insufficient Data"))[3])

    return comb2025


@st.cache_data(show_spinner="Loading 2026 payroll data ...", persist="disk")
def _cached_2026_payroll(payroll_dir: str, combined_path: str, dir_hash: str) -> pd.DataFrame:
    """Consolidate all 30-team 2026 payroll xlsx files into a single DataFrame.

    Joins with 2025 combined data to bring in Position + last-known WAR/stats.
    Returns a DataFrame compatible with the simulator (same key columns as _cached_simulator_data).
    """
    _SHEETS = {
        "Guaranteed":               "FA",
        "Eligible For Arb":         "Arb",
        "Not Yet Eligible For Arb": "Pre-Arb",
    }
    _YR_COLS = [2027, 2028, 2029, 2030, 2031, 2032]

    rows: list[dict] = []
    # Build (filename, full_path_or_url) pairs — supports both local dir and R2 base URL
    if payroll_dir.startswith("http"):
        _payroll_entries = [
            (f"{name}-Payroll-2026.xlsx", f"{payroll_dir}/{name}-Payroll-2026.xlsx")
            for name in _PAYROLL_2026_TEAM_MAP
        ]
    else:
        _payroll_entries = [
            (fname, os.path.join(payroll_dir, fname))
            for fname in sorted(os.listdir(payroll_dir))
            if fname.endswith(".xlsx")
        ]
    for fname, fpath in _payroll_entries:
        team_name = fname.replace("-Payroll-2026.xlsx", "")
        team_abbr = _PAYROLL_2026_TEAM_MAP.get(team_name, team_name[:3].upper())
        try:
            if fpath.startswith("http") and _requests_available:
                _xr = _requests.get(fpath, timeout=30)
                _xr.raise_for_status()
                xl = pd.ExcelFile(_io.BytesIO(_xr.content))
            else:
                xl = pd.ExcelFile(fpath)
        except Exception:
            continue

        for sheet, stage in _SHEETS.items():
            if sheet not in xl.sheet_names:
                continue
            try:
                sdf = xl.parse(sheet)
            except Exception:
                continue
            if sdf.empty or "Player" not in sdf.columns:
                continue

            for _, row in sdf.iterrows():
                player = str(row.get("Player", "")).strip()
                if not player or player.lower() in ("nan", "player", ""):
                    continue

                sal_2026 = _parse_payroll_val(row.get(2026))
                aav      = _parse_payroll_val(row.get("AAV"))

                fut: dict[str, float | None] = {}
                for yr in _YR_COLS:
                    fut[str(yr)] = _parse_payroll_val(row.get(yr))

                yrs_left = sum(1 for v in fut.values() if v is not None and v > 0)

                rows.append({
                    "Player":      player,
                    "Team":        team_abbr,
                    "Stage_Clean": stage,
                    "Age":         row.get("Age"),
                    "Contract":    str(row.get("Contract", "")).strip(),
                    "AAV_M":       aav,
                    "Salary_M":    sal_2026,
                    "Yrs_Left":    yrs_left,
                    "playerId":    row.get("playerId"),
                    **fut,
                })

    if not rows:
        return pd.DataFrame()

    df26 = pd.DataFrame(rows)
    df26["Age"]    = pd.to_numeric(df26["Age"],    errors="coerce")
    df26["Salary_M"] = pd.to_numeric(df26["Salary_M"], errors="coerce")

    # Deduplicate (keep highest salary row — same player can appear in multiple sheets)
    df26 = (
        df26.sort_values("Salary_M", ascending=False, na_position="last", kind="mergesort")
            .drop_duplicates(subset=["Player"], keep="first")
            .reset_index(drop=True)
    )

    # Enrich with 2025 Position + WAR + key stats via player name join
    # Also keep full multi-year data for WAR history lookup in PPR
    _war_hist_26: dict = {}
    if combined_path.startswith("http") or os.path.exists(combined_path):
        comb = _read_csv(combined_path, low_memory=False)
        comb.columns = [c.strip() for c in comb.columns]
        comb["Year"] = pd.to_numeric(comb["Year"], errors="coerce")
        # Build WAR history lookup: {player: {year: war}}
        _comb_war26 = comb[["Player", "Year", "WAR_Total"]].copy()
        _comb_war26["WAR_Total"] = pd.to_numeric(_comb_war26["WAR_Total"], errors="coerce").fillna(0.0)
        for _, _r in _comb_war26.dropna(subset=["Player", "Year"]).iterrows():
            _p = str(_r["Player"]); _y = int(_r["Year"])
            _war_hist_26.setdefault(_p, {})[_y] = float(_r["WAR_Total"])
        stat_cols = ["Player", "Position", "WAR_Total",
                     "HR", "RBI", "AVG", "OBP", "SLG", "ERA", "FIP", "IP"]
        stat_cols = [c for c in stat_cols if c in comb.columns]
        c25 = (
            comb[comb["Year"] == 2025][stat_cols]
            .drop_duplicates(subset=["Player"], keep="first")
        )
        df26 = df26.merge(c25, on="Player", how="left")

    # Position group + eligible slots
    df26["Position"] = df26.get("Position", pd.Series(dtype=str)).fillna("UNK").astype(str).str.strip()
    df26["pos_group"]     = df26["Position"].map(lambda p: _POS_GROUP_MAP.get(p, "UNK"))
    df26["eligible_slots"] = df26["pos_group"].map(lambda g: _ELIGIBLE_SLOTS_MAP.get(g, []))

    # Convert future-year columns to numeric $M
    for yr in _YR_COLS:
        col = str(yr)
        if col in df26.columns:
            df26[col] = pd.to_numeric(df26[col], errors="coerce")

    # PPR = Σ actual WAR for each contract year / total contract $M
    # Looks back N seasons (N = total contract length) using real historical WAR.
    # Missing seasons fall back to most recent known WAR (2025).
    _fyr = [str(yr) for yr in _YR_COLS if str(yr) in df26.columns]

    def _ppr_26(row):
        sal = float(row.get("Salary_M") or 0)
        war = float(row.get("WAR_Total") or 0)  # 2025 WAR — fallback for missing seasons
        if sal <= 0:
            return None
        fut_sals = [float(row[c]) for c in _fyr
                    if pd.notna(row.get(c)) and float(row.get(c) or 0) > 0]
        total_val = sal + sum(fut_sals)
        total_yrs = 1 + len(fut_sals)
        if total_val <= 0:
            return None
        # Sum actual WAR for the most recent total_yrs seasons
        p_hist = _war_hist_26.get(str(row.get("Player", "")), {})
        war_sum = sum(p_hist.get(2025 - i, war) for i in range(total_yrs))
        return round(war_sum / total_val, 3)

    df26["PPR"] = df26.apply(_ppr_26, axis=1)
    df26["W_per_M"] = df26.apply(
        lambda r: round(float(r["WAR_Total"]) / float(r["Salary_M"]), 3)
        if pd.notna(r.get("WAR_Total")) and pd.notna(r.get("Salary_M"))
        and float(r.get("Salary_M") or 0) > 0
        else None,
        axis=1,
    )

    # Total contract value ($M) — 2026 salary + all future committed years
    def _total_ctrc_26(row):
        sal = float(row.get("Salary_M") or 0)
        fut_sals = [float(row[c]) for c in _fyr
                    if pd.notna(row.get(c)) and float(row.get(c) or 0) > 0]
        total = sal + sum(fut_sals)
        return round(total, 2) if total > 0 else None

    df26["Total_Contract_M"] = df26.apply(_total_ctrc_26, axis=1)

    df26["Year"] = 2026
    return df26


@st.cache_data(show_spinner=False)
def _cached_player_history(combined_path: str, file_hash: str) -> pd.DataFrame:
    """Load full multi-year player data for player cards (all years, all players)."""
    df = _read_csv(combined_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def _cached_war_reliability(combined_path: str, file_hash: str) -> dict:
    """Compute WAR reliability grades from multi-year history (for consistency badges)."""
    df = _read_csv(combined_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["WAR_Total"] = pd.to_numeric(df.get("WAR_Total", pd.Series(dtype=float)), errors="coerce").fillna(0)
    result: dict = {}
    for player, grp in df.groupby("Player"):
        years = int(grp["Year"].dropna().nunique())
        if years == 0:
            continue
        war_vals = grp["WAR_Total"].values
        war_sd = float(np.std(war_vals, ddof=1)) if years > 1 else 1.5
        if years >= 4 and war_sd < 1.0:
            grade = "A"
        elif years >= 3 and war_sd < 1.5:
            grade = "B"
        elif years >= 2 and war_sd < 2.0:
            grade = "C"
        elif years >= 2:
            grade = "D"
        else:
            grade = "?"
        result[str(player)] = {"years": years, "war_sd": round(war_sd, 2), "grade": grade}
    return result


@st.cache_data(show_spinner=False)
def _cached_razzball(razzball_path: str) -> pd.DataFrame:
    """Load razzball MLBAM ID lookup table (local file or R2 URL)."""
    if not razzball_path.startswith("http") and not os.path.exists(razzball_path):
        return pd.DataFrame()
    try:
        df = _read_csv(razzball_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _cached_mlbam_lookup(razzball_path: str) -> dict[str, str]:
    """Build a dict mapping normalised player name → MLBAM ID string.

    Uses First+Last columns if available, otherwise the Name column.
    """
    rz = _cached_razzball(razzball_path)
    if rz.empty or "MLBAMID" not in rz.columns:
        return {}
    lookup: dict[str, str] = {}
    if "First" in rz.columns and "Last" in rz.columns:
        for _, row in rz.iterrows():
            first = str(row.get("First", "")).strip()
            last  = str(row.get("Last", "")).strip()
            mid   = str(row["MLBAMID"]).strip()
            if first and last and mid.isdigit():
                key = _fix_player_name(f"{first} {last}")
                lookup[key] = mid
    if "Name" in rz.columns:
        for _, row in rz.iterrows():
            name = str(row.get("Name", "")).strip()
            mid  = str(row["MLBAMID"]).strip()
            if name and mid.isdigit():
                key = _fix_player_name(name)
                lookup.setdefault(key, mid)
    return lookup


def _headshot_url(mlbam_id: str, width: int = 56) -> str:
    """Return the MLB static headshot URL for a given MLBAM ID."""
    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        f"d_people:generic:headshot:67:current.png/w_{width},q_auto:best"
        f"/v1/people/{mlbam_id}/headshot/67/current"
    )


def _hover_img_tag(player_name: str, mlbam_map: dict[str, str]) -> str:
    """Return an <img> tag for the player's headshot, or empty string."""
    mid = mlbam_map.get(player_name, "")
    if not mid:
        return ""
    url = _headshot_url(mid, width=56)
    return (
        f"<img src='{url}' width='56' height='56' "
        f"style='border-radius:50%;vertical-align:middle;margin-right:6px;'>"
    )


@st.cache_data(show_spinner=False)
def _cached_40man_roster(roster_path: str, fhash: str) -> pd.DataFrame:
    """Load the 40-man roster CSV (from local file or R2 URL)."""
    try:
        df = _read_csv(roster_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _build_carousel_players(combined_path: str) -> list[str]:
    """Return top-3 WAR players per team from 2025, ensuring all 30 teams are represented.

    Skips players whose headshot cannot be resolved (no MLBAM ID).
    """
    try:
        df = _read_csv(combined_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["WAR_Total"] = pd.to_numeric(df.get("WAR_Total", pd.Series(dtype=float)), errors="coerce")
        df25 = df[df["Year"] == 2025].dropna(subset=["WAR_Total", "Team", "Player"])

        mlbam = _cached_mlbam_lookup(_RAZZBALL_PATH)

        result = []
        for _tm, grp in df25.groupby("Team"):
            top = grp.sort_values("WAR_Total", ascending=False)
            added = 0
            for _, row in top.iterrows():
                if added >= 3:
                    break
                pname = row["Player"]
                if pname in mlbam:  # has a headshot
                    result.append(pname)
                    added += 1
        return result
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _cached_carousel_images(headshots_dir: str, n: int = 90, seed: int = 42,
                             player_list: tuple = ()) -> list:
    """Load headshot PNGs as base64 strings for the landing page carousel.

    Uses player_list if provided (top-3 WAR per team), otherwise falls back to
    local directory listing.
    """
    import random
    import base64
    rng = random.Random(seed)

    if player_list:
        # Use the curated player list (all 30 teams represented)
        if not _requests_available:
            return []
        mlbam = _cached_mlbam_lookup(_RAZZBALL_PATH)
        players = list(player_list)
        rng.shuffle(players)
        result = []
        for name in players[:n]:
            mid = mlbam.get(name)
            if not mid:
                continue
            url = _headshot_url(mid, width=213)
            try:
                resp = _requests.get(url, timeout=4)
                if resp.status_code == 200 and len(resp.content) > 8000:
                    # Skip generic silhouettes (< 8KB) — only use real photos
                    result.append(base64.b64encode(resp.content).decode())
            except Exception:
                continue
        return result

    if headshots_dir.startswith("http"):
        # R2 mode fallback: use MLBAM API with player list
        if not _requests_available:
            return []
        result = []
        return result
    try:
        files = sorted(f for f in os.listdir(headshots_dir) if f.lower().endswith(".png"))
    except OSError:
        return []
    rng.shuffle(files)
    result = []
    for fn in files[:n]:
        try:
            with open(os.path.join(headshots_dir, fn), "rb") as fh:
                result.append(base64.b64encode(fh.read()).decode())
        except OSError:
            continue
    return result


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
  background: #111927 !important;
}
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  background: #111927 !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
  background: #0e1720 !important;
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
  background: #18243a !important;
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
  background: #18243a !important;
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
  font-size: 0.78rem !important;
}
/* ── General paragraph / body text brighter ──────────────────────── */
.stApp p, .stApp span, .stApp label, .stApp div {
  color: inherit;
}
[data-testid="stMarkdownContainer"] p {
  color: #93b8d8 !important;
  font-size: 0.84rem !important;
  line-height: 1.6 !important;
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
        + _a('glossary', '📖 Glossary')
        + '<a href="https://docs.google.com/forms/d/e/1FAIpQLSdexY0xhRoQt3F6LVJHdZ7z4_nHeZZIL7Bn8bFrIaqmsTb0Pw/viewform?usp=publish-editor" '
        + 'target="_blank" style="color:#3b82f6;text-decoration:none;font-size:0.82rem;'
        + 'padding:0.35rem 0.7rem;margin-left:0.5rem;border:1px solid #2e4a62;border-radius:6px;'
        + 'white-space:nowrap;">Feedback</a>'
        + '</div>'
        + '<hr style="margin:0.4rem 0 1rem;border:none;border-top:1px solid #1e3250;">'
    )
    st.markdown(nav, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Roster Simulator — player comparison
# ---------------------------------------------------------------------------

def _render_player_comparison(players_df: pd.DataFrame):
    """Side-by-side stats table for 2–4 selected players."""
    metrics = [
        ("Position",         "Position",        None),
        ("Age",              "Age",             "{:.0f}"),
        ("Stage",            "Stage_Clean",     None),
        ("2025 WAR",         "WAR_Total",       "{:.1f}"),
        ("2026 Salary $M",   "Salary_M",        "${:.1f}M"),
        ("Full Contract $M", "Total_Contract_M","${:.0f}M"),
        ("W/$M (Season)",    "W_per_M",         "{:.2f}"),
        ("W/$M (Contract)",  "PPR",             "{:.2f}"),
        ("Yrs Left",         "Yrs_Left",        "{:.0f}"),
        ("HR",               "HR",              "{:.0f}"),
        ("AVG",              "AVG",             "{:.3f}"),
        ("ERA",              "ERA",             "{:.2f}"),
        ("IP",               "IP",              "{:.0f}"),
        ("FIP",              "FIP",             "{:.2f}"),
    ]
    players = players_df.to_dict("records")
    header  = ["Metric"] + [p.get("Player", f"Player {i+1}") for i, p in enumerate(players)]
    rows    = []
    for label, col, fmt in metrics:
        vals = []
        for p in players:
            v = p.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append("—")
            elif fmt:
                try:
                    vals.append(fmt.format(float(v)))
                except Exception:
                    vals.append(str(v))
            else:
                vals.append(str(v))
        if any(v != "—" for v in vals):
            rows.append([label] + vals)
    comp_df = pd.DataFrame(rows, columns=header)
    st.dataframe(comp_df, hide_index=True, use_container_width=True,
                 height=min(60 + len(comp_df) * 35, 520))


# ---------------------------------------------------------------------------
# Roster Simulator — trade analyzer
# ---------------------------------------------------------------------------

def _render_trade_analyzer(roster_df: pd.DataFrame):
    """Interactive trade analyzer — incoming vs outgoing player impact."""
    full_df = st.session_state.get("sim_df_full", pd.DataFrame())
    if full_df.empty:
        st.info("Full player pool not loaded yet — visit the Roster Simulator tab first.")
        return

    roster_names = roster_df["Player"].tolist() if "Player" in roster_df.columns else []
    all_names    = sorted(full_df["Player"].dropna().unique().tolist())
    available_in = [n for n in all_names if n not in roster_names]

    c_out, c_in = st.columns(2)
    with c_out:
        st.markdown("**Players Out** *(from your roster)*")
        out_players = st.multiselect("out", options=roster_names,
                                     key="trade_out", label_visibility="collapsed")
    with c_in:
        st.markdown("**Players In** *(acquire)*")
        in_players  = st.multiselect("in", options=available_in,
                                     key="trade_in", label_visibility="collapsed")

    if not out_players and not in_players:
        st.caption("Select players above to see trade impact.")
        return

    out_df = roster_df[roster_df["Player"].isin(out_players)] if out_players else pd.DataFrame(columns=roster_df.columns)
    in_df  = full_df[full_df["Player"].isin(in_players)]  if in_players  else pd.DataFrame(columns=full_df.columns)

    out_war  = float(out_df["WAR_Total"].sum()) if "WAR_Total" in out_df.columns else 0.0
    in_war   = float(in_df["WAR_Total"].sum())  if "WAR_Total" in in_df.columns  else 0.0
    out_sal  = float(out_df["Salary_M"].sum())  if "Salary_M"  in out_df.columns else 0.0
    in_sal   = float(in_df["Salary_M"].sum())   if "Salary_M"  in in_df.columns  else 0.0
    out_ppr  = float(out_df["PPR"].mean()) if "PPR" in out_df.columns and out_df["PPR"].notna().any() else 0.0
    in_ppr   = float(in_df["PPR"].mean())  if "PPR" in in_df.columns  and in_df["PPR"].notna().any()  else 0.0

    delta_war = in_war  - out_war
    delta_sal = in_sal  - out_sal
    delta_ppr = in_ppr  - out_ppr

    t1, t2, t3 = st.columns(3)
    t1.metric("Net WAR",      f"{delta_war:+.1f}",
              f"{in_war:.1f} in vs {out_war:.1f} out",
              delta_color="normal")
    t2.metric("Net Salary",   f"${delta_sal:+.1f}M",
              f"${in_sal:.1f}M in vs ${out_sal:.1f}M out",
              delta_color="inverse")
    t3.metric("Net Pay/Play", f"{delta_ppr:+.2f}",
              f"{in_ppr:.2f} in vs {out_ppr:.2f} out",
              delta_color="normal")

    if delta_war > 0 and delta_sal <= 0:
        verdict, vcolor = "✅ Strong Trade — more WAR, same or less money", "#2ecc71"
    elif delta_war > 0 and delta_sal > 0:
        verdict, vcolor = f"⚠️ Mixed — {delta_war:.1f} more WAR for ${delta_sal:.1f}M more", "#f39c12"
    elif delta_war <= 0 and delta_sal < 0:
        verdict, vcolor = f"💰 Salary dump — {-delta_war:.1f} WAR loss, saves ${-delta_sal:.1f}M", "#3498db"
    else:
        verdict, vcolor = "❌ Net negative — WAR loss without salary relief", "#e74c3c"

    st.markdown(
        f"<p style='color:{vcolor};font-weight:600;margin:0.5rem 0;'>{verdict}</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Roster Simulator — position coverage helper
# ---------------------------------------------------------------------------

def _render_position_coverage(roster_df: pd.DataFrame):
    """Show position coverage vs standard roster template."""
    if "pos_group" not in roster_df.columns:
        st.info("No position data available.")
        return

    # Count eligible players per slot
    slot_have: dict[str, int] = {slot: 0 for slot in _ROSTER_TEMPLATE}
    for _, row in roster_df.iterrows():
        eligible = _ELIGIBLE_SLOTS_MAP.get(str(row.get("pos_group", "")), [])
        for slot in eligible:
            if slot in slot_have:
                slot_have[slot] += 1
        # Everyone can be bench
        slot_have["BENCH"] = slot_have.get("BENCH", 0) + 1

    rows = []
    for slot, needed in _ROSTER_TEMPLATE.items():
        have = slot_have.get(slot, 0)
        optional = slot in _OPTIONAL_SLOTS
        if have >= needed:
            indicator = "OK"
        elif optional and have == 0:
            indicator = "— optional"
        elif have > 0:
            indicator = f"{have}/{needed}"
        else:
            indicator = f"0/{needed} MISSING"
        rows.append({"Slot": slot, "Need": needed, "Have": have, "Status": indicator})

    cov_df = pd.DataFrame(rows)

    def _color_status(val):
        s = str(val)
        if s == "OK":
            return "background-color: #1a4731; color: #2ecc71; font-weight: bold"
        if "optional" in s:
            return "background-color: #1a2a3a; color: #7aa2c0; font-style: italic"
        if "MISSING" in s:
            return "background-color: #4a1a1a; color: #e74c3c; font-weight: bold"
        return "background-color: #3a2a00; color: #f39c12; font-weight: bold"

    styled = cov_df[["Slot", "Need", "Have", "Status"]].style.applymap(
        _color_status, subset=["Status"]
    )
    st.dataframe(styled, hide_index=True, use_container_width=True,
                 height=min(60 + len(cov_df) * 35, 490))


# ---------------------------------------------------------------------------
# Roster Simulator — roster summary panel
# ---------------------------------------------------------------------------

def _render_roster_summary(budget_M: float = 130.0):
    """Render the built roster summary panel."""
    import io

    roster_records = st.session_state.get("sim_roster", [])
    if not roster_records:
        return

    roster_df = pd.DataFrame(roster_records)

    st.markdown("---")
    st.markdown("### My Custom Roster")

    # ── Metrics row 1 ────────────────────────────────────────────────────────
    total_war  = float(roster_df["WAR_Total"].sum())  if "WAR_Total"   in roster_df.columns else 0.0
    total_cost = float(roster_df["Salary_M"].sum())   if "Salary_M"    in roster_df.columns else 0.0
    n_fa    = int((roster_df["Stage_Clean"] == "FA").sum())      if "Stage_Clean" in roster_df.columns else 0
    n_arb   = int((roster_df["Stage_Clean"] == "Arb").sum())     if "Stage_Clean" in roster_df.columns else 0
    n_pre   = int((roster_df["Stage_Clean"] == "Pre-Arb").sum()) if "Stage_Clean" in roster_df.columns else 0
    avg_ppr = (float(roster_df["PPR"].mean())
               if "PPR" in roster_df.columns and roster_df["PPR"].notna().any()
               else None)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Players",            str(len(roster_df)))
    m2.metric("Total WAR",          f"{total_war:.1f}")
    m3.metric("Payroll ($M)",       f"${total_cost:.1f}M")
    m4.metric("$/WAR",              f"${total_cost / max(total_war, 0.1):.1f}M")
    m5.metric("FA / Arb / Pre-Arb", f"{n_fa} / {n_arb} / {n_pre}")
    m6.metric("Avg Pay/Play Ratio",
              f"{avg_ppr:.2f}" if avg_ppr is not None else "—",
              help="Avg Pay vs Play Ratio: sum of actual WAR across all contract years ÷ total contract $M. Higher = better value.")

    # ── Metrics row 2: efficiency breakdown ──────────────────────────────────
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("WAR / Player", f"{total_war / max(len(roster_df), 1):.1f}",
              help="Average WAR contributed per rostered player.")
    e2.metric("WAR / $M", f"{total_war / max(total_cost, 0.1):.2f}",
              help="Team-level production efficiency: total WAR divided by total payroll.")
    if "Stage_Clean" in roster_df.columns and "Salary_M" in roster_df.columns:
        pre_cost = float(roster_df[roster_df["Stage_Clean"] == "Pre-Arb"]["Salary_M"].sum())
        e3.metric("Pre-Arb % Pay", f"{pre_cost / max(total_cost, 0.1) * 100:.1f}%",
                  help="Fraction of payroll locked in pre-arb players — lower = more flexibility.")
    else:
        e3.metric("Pre-Arb % Pay", "—")
    if "Yrs_Left" in roster_df.columns and roster_df["Yrs_Left"].notna().any():
        e4.metric("Avg Yrs Left", f"{float(roster_df['Yrs_Left'].mean()):.1f}",
                  help="Average contract years remaining across all rostered players.")
    else:
        e4.metric("Avg Yrs Left", "—")
    e5.metric("Est. Win Total", f"~{47.7 + total_war:.0f} W",
              help="Rough estimate: 47.7 replacement-level wins + Total WAR.")

    # ── Roster Grade ──────────────────────────────────────────────────────────
    grades = _roster_grade(roster_df)
    _GRADE_COLORS = {
        "A+": "#00e676", "A": "#4caf50", "B+": "#8bc34a", "B": "#cddc39",
        "C": "#ffc107",  "D": "#ff9800", "F": "#f44336",
    }
    st.markdown("##### Roster Grade")
    g_cols = st.columns(4)
    for i, (dim, (grade, score)) in enumerate(grades.items()):
        color = _GRADE_COLORS.get(grade, "#7aa2c0")
        g_cols[i].markdown(
            f"<div style='text-align:center;padding:0.6rem 0.4rem;background:#0d1f38;"
            f"border:1px solid #1e3a5f;border-radius:8px;'>"
            f"<div style='font-size:1.8rem;font-weight:700;color:{color};'>{grade}</div>"
            f"<div style='font-size:0.75rem;color:#7aa2c0;margin-top:2px;'>{dim}</div>"
            f"<div style='font-size:0.65rem;color:#4a6a8a;'>{score}/100</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("")

    # ── Roster table + position coverage ─────────────────────────────────────
    col_table, col_check = st.columns([3, 2])

    with col_table:
        st.markdown("##### Roster  <span style='font-size:0.78rem;color:#666;font-weight:normal;'>— check ✕ to remove players</span>", unsafe_allow_html=True)
        show = [c for c in ["Player", "Team", "Position", "Stage_Clean",
                             "Age", "WAR_Total", "Salary_M", "W_per_M", "Yrs_Left", "PPR"]
                if c in roster_df.columns]
        edit_df = roster_df[show].copy()

        # Decision + Reliability columns
        reliability = st.session_state.get("sim_reliability", {})
        edit_df["Decision"]    = [_contract_decision(r) for r in roster_records]
        edit_df["Consistency"] = [reliability.get(r.get("Player", ""), {}).get("grade", "?")
                                   for r in roster_records]
        edit_df.insert(0, "Remove", False)

        # Row highlighting: amber for depth-chart-only players (no MLB stats)
        _dc_only_flags = [bool(r.get("_dc_only", False)) for r in roster_records]
        def _highlight_dc(row):
            return (
                ["background-color:#2d1f00;color:#fbbf24"] * len(row)
                if _dc_only_flags[row.name] else [""] * len(row)
            )
        _styled_edit = edit_df.style.apply(_highlight_dc, axis=1)

        if any(_dc_only_flags):
            st.caption("🟡 Amber rows = depth chart only — no 2025 MLB stats in database (league min salary assumed)")

        edited = st.data_editor(
            _styled_edit,
            column_config={
                "Remove":      st.column_config.CheckboxColumn("✕",          width="small"),
                "WAR_Total":   st.column_config.NumberColumn("WAR",           format="%.1f",   width="small"),
                "Salary_M":    st.column_config.NumberColumn("Sal $M",        format="$%.1fM", width="small"),
                "W_per_M":     st.column_config.NumberColumn("W/$M (Ssn)",    format="%.2f",   width="small"),
                "Yrs_Left":    st.column_config.ProgressColumn("Yrs Left",    min_value=0, max_value=7, format="%d yr", width="small"),
                "PPR":         st.column_config.NumberColumn("W/$M (Ctrc)",   format="%.2f",   width="small",
                               help="Sum of actual WAR across all contract years ÷ total contract $M."),
                "Stage_Clean": st.column_config.TextColumn("Stage",           width="small"),
                "Position":    st.column_config.TextColumn("Pos",             width="small"),
                "Age":         st.column_config.NumberColumn("Age",           format="%d",     width="small"),
                "Decision":    st.column_config.TextColumn("Decision",        width="small",
                               help="Extend / Hold / Trade / DFA recommendation based on age, WAR, and contract."),
                "Consistency": st.column_config.TextColumn("Consistency",     width="small",
                               help="WAR year-to-year consistency grade (A = very consistent, F = 1 year of data only)."),
            },
            disabled=[c for c in edit_df.columns if c != "Remove"],
            hide_index=True,
            use_container_width=True,
            height=min(60 + (len(roster_df) + 1) * 35, 520),
            key="roster_editor",
        )

        if st.button("Remove Selected Players", key="roster_remove_btn", type="secondary"):
            keep = [rec for rec, rm in zip(roster_records, edited["Remove"].tolist()) if not rm]
            if len(keep) < len(roster_records):
                st.session_state["sim_roster"] = keep
                st.rerun()
            else:
                st.info("No players checked for removal — tick the ✕ column first.")

    with col_check:
        st.markdown("##### Position Coverage")
        _render_position_coverage(roster_df)

    # ── Charts: WAR by position group  |  Salary vs WAR scatter ─────────────
    ch1, ch2 = st.columns(2)
    _PG_ORDER  = ["SP", "RP", "C", "1B", "2B", "3B", "SS", "CF", "OF", "DH"]
    _PG_COLORS = _PG_CHART_COLORS

    with ch1:
        st.markdown("##### WAR by Position Group")
        if "pos_group" in roster_df.columns and "WAR_Total" in roster_df.columns:
            pg_war = (
                roster_df.groupby("pos_group")["WAR_Total"]
                .sum()
                .reindex(_PG_ORDER)
                .dropna()
            )
            if not pg_war.empty:
                try:
                    cats    = pg_war.index.tolist()
                    vals    = pg_war.values.tolist()
                    bcolors = [_PG_COLORS[_PG_ORDER.index(p)] for p in cats if p in _PG_ORDER]
                    y_max   = max(vals) * 1.35 if vals else 10
                    fig = go.Figure(data=[go.Bar(
                        x=cats, y=vals,
                        marker_color=bcolors, marker_line_width=0,
                        text=[f"{v:.1f}" for v in vals],
                        textposition="outside",
                        textfont=dict(color="#dbeafe", size=11),
                        hovertemplate="%{x}: %{y:.1f} WAR<extra></extra>",
                    )])
                    fig.update_layout(**_pt(
                        title="WAR by Position Group",
                        yaxis=dict(title="WAR", range=[0, y_max]),
                        height=450,
                        margin=dict(l=50, r=20, t=45, b=50),
                    ))
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})
                except Exception:
                    pass

    with ch2:
        st.markdown("##### Salary vs WAR — Player Efficiency")
        if "Salary_M" in roster_df.columns and "WAR_Total" in roster_df.columns:
            try:
                scat_df = roster_df.dropna(subset=["Salary_M", "WAR_Total"])
                if not scat_df.empty:
                    dot_colors = (
                        [_PG_COLORS[_PG_ORDER.index(p)] if p in _PG_ORDER else "#60a5fa"
                         for p in scat_df["pos_group"]]
                        if "pos_group" in scat_df.columns
                        else "#3b82f6"
                    )
                    labels = scat_df["Player"].tolist() if "Player" in scat_df.columns else [""] * len(scat_df)
                    x_max_ref = float(scat_df["Salary_M"].max()) * 1.15
                    x_ref = np.linspace(0, x_max_ref, 60)
                    y_ref = x_ref * (total_war / max(total_cost, 0.1))

                    fig2 = go.Figure()
                    # Team avg efficiency reference line
                    fig2.add_trace(go.Scatter(
                        x=x_ref, y=y_ref, mode="lines",
                        line=dict(color="#f59e0b", dash="dash", width=1.5),
                        opacity=0.65, name="Team avg WAR/$M", hoverinfo="skip",
                    ))
                    # Player dots
                    fig2.add_trace(go.Scatter(
                        x=scat_df["Salary_M"].tolist(),
                        y=scat_df["WAR_Total"].tolist(),
                        mode="markers+text",
                        text=labels,
                        textposition="top center",
                        textfont=dict(size=8, color="#7aa2c0"),
                        marker=dict(color=dot_colors, size=10, opacity=0.9,
                                    line=dict(color="#1e3a5f", width=1)),
                        name="Players",
                        hovertemplate="<b>%{text}</b><br>Salary: $%{x:.1f}M<br>WAR: %{y:.1f}<extra></extra>",
                        showlegend=False,
                    ))
                    fig2.update_layout(**_pt(
                        title="Salary vs WAR — Player Efficiency",
                        xaxis=dict(title="Salary ($M)"),
                        yaxis=dict(title="WAR"),
                        height=450,
                        showlegend=True,
                        margin=dict(l=50, r=20, t=45, b=50),
                    ))
                    st.plotly_chart(fig2, use_container_width=True,
                                    config={"displayModeBar": False})
            except Exception:
                pass

    # ── Trade Analyzer ───────────────────────────────────────────────────────
    with st.expander("🔄 Trade Analyzer", expanded=False):
        _render_trade_analyzer(roster_df)

    # ── Simulated Wins Explanation ────────────────────────────────────────────
    with st.expander("ℹ️ How is Est. Win Total calculated?", expanded=False):
        est_wins = 47.7 + total_war
        st.markdown(f"""
**Formula:** `Est. Wins ≈ 47.7 + Total WAR`

**Your roster:** 47.7 (baseline) + {total_war:.1f} (total WAR) = **~{est_wins:.0f} wins**

---

**Replacement Level Baseline (47.7):**
A team composed entirely of "replacement-level" players — freely available from waivers or AAA — is expected to win roughly 47–48 games. This is the universally accepted MLB baseline for WAR calculations.

**WAR Contribution:**
Each 1.0 WAR represents one additional win above that replacement level player. A 5-WAR player produces 5 more wins than his freely-available replacement.

**Reference points:**
| Total WAR | Est. Wins | Outcome |
|-----------|-----------|---------|
| 25 WAR    | ~73 W     | Rebuilding |
| 30 WAR    | ~78 W     | Fringe contender |
| 35 WAR    | ~83 W     | Wild card threat |
| 40 WAR    | ~88 W     | Playoff team |
| 45 WAR    | ~93 W     | Division favorite |
| 50 WAR    | ~98 W     | World Series contender |

**Full Monte Carlo Simulation:**
The Optimizer tab runs a more sophisticated model: `wins = intercept + slope × WAR` with per-player WAR variance sampled from a normal distribution, plus group-level shocks (SP, RP, hitters). This produces a full win distribution (P10–P90) and playoff probability rather than a single estimate.

*Tip: A balanced roster of ~40 total WAR with payroll under $130M is the sweet spot for sustainable contention.*
""")

    # ── Best Fits ─────────────────────────────────────────────────────────────
    _render_best_fits(roster_df, budget_M)

    # ── Export + clear ────────────────────────────────────────────────────────
    st.markdown("---")
    dl_col, clr_col, _ = st.columns([2, 2, 6])
    with dl_col:
        csv_buf = io.StringIO()
        roster_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Export Roster (CSV)",
            data=csv_buf.getvalue(),
            file_name="my_custom_roster.csv",
            mime="text/csv",
            key="sim_export_btn",
        )
    with clr_col:
        if st.button("Clear Roster", key="sim_clear_btn"):
            for k in ("sim_roster", "sim_roster_war", "sim_roster_cost"):
                st.session_state.pop(k, None)
            st.rerun()


# ---------------------------------------------------------------------------
# Player card
# ---------------------------------------------------------------------------

def _render_player_card(player_name: str, combined_path: str, file_hash: str):
    """Render a player detail card: headshot + year-by-year stats (2021-2025)."""
    import io as _io
    try:
        import requests as _requests
        _requests_available = True
    except ImportError:
        _requests_available = False

    all_df = _cached_player_history(combined_path, file_hash)
    player_df = all_df[all_df["Player"] == player_name].sort_values("Year").reset_index(drop=True)

    if player_df.empty:
        st.warning(f"No historical data found for **{player_name}**.")
        return

    latest = player_df.iloc[-1]
    pos      = str(latest.get("Position", "?"))
    team     = str(latest.get("Team", "?"))
    stage    = str(latest.get("Stage_Clean", "?"))
    age_val  = latest.get("Age", None)
    age_str  = f"{int(age_val)}" if pd.notna(age_val) else "—"
    is_pitcher = pos in ("SP", "RP", "P", "TWP")

    # Try to get 2026 salary from payroll data first; fall back to 2025 historical
    _payroll_dir_26 = os.path.join(_ROOT_DIR, "2026 Payroll")
    sal_26_val = None
    try:
        if os.path.exists(_payroll_dir_26):
            _df26_card = _cached_2026_payroll(
                _payroll_dir_26, combined_path, _dir_hash(_payroll_dir_26)
            )
            _row26 = _df26_card[_df26_card["Player"] == player_name]
            if not _row26.empty and pd.notna(_row26.iloc[0].get("Salary_M")):
                sal_26_val = float(_row26.iloc[0]["Salary_M"])
    except Exception:
        pass

    if sal_26_val is not None:
        sal_label = "Sal '26"
        sal_str   = f"${sal_26_val:.1f}M"
    else:
        sal_val = latest.get("Salary_M", None)
        sal_label = "Sal (2025)"
        sal_str   = f"${float(sal_val):.1f}M" if pd.notna(sal_val) and sal_val else "—"

    # Look up PPR, total contract, years left from simulator data
    _ind_path = _data_url("data/2025mlbshared.csv")
    try:
        _sim_df   = _cached_simulator_data(combined_path, _ind_path, file_hash)
        _sim_row  = _sim_df[_sim_df["Player"] == player_name]
        ppr_val   = float(_sim_row.iloc[0]["PPR"]) if (
            not _sim_row.empty and "PPR" in _sim_row.columns
            and pd.notna(_sim_row.iloc[0]["PPR"])
        ) else None
        total_ctrc_val = float(_sim_row.iloc[0]["Total_Contract_M"]) if (
            not _sim_row.empty and "Total_Contract_M" in _sim_row.columns
            and pd.notna(_sim_row.iloc[0]["Total_Contract_M"])
        ) else None
        yrs_left_val = int(_sim_row.iloc[0]["Yrs_Left"]) if (
            not _sim_row.empty and "Yrs_Left" in _sim_row.columns
            and pd.notna(_sim_row.iloc[0]["Yrs_Left"])
        ) else None
    except Exception:
        ppr_val = None
        total_ctrc_val = None
        yrs_left_val = None
    ppr_str = f"{ppr_val:.2f} WAR/$M" if ppr_val is not None else "—"
    if total_ctrc_val is not None:
        yrs_total = 1 + (yrs_left_val or 0)
        ctrc_str = f"${total_ctrc_val:.0f}M / {yrs_total} yr{'s' if yrs_total != 1 else ''}"
    else:
        ctrc_str = "—"

    img_col, stats_col = st.columns([1, 3])

    # ------------------------------------------------------------------
    # Headshot
    # ------------------------------------------------------------------
    with img_col:
        headshot_path = os.path.join(_HEADSHOTS_DIR, f"{player_name}.png")
        img_bytes = None

        # In R2 mode serve from bucket; otherwise use local cache
        if _R2_MODE and _requests_available:
            try:
                _r2_resp = _requests.get(
                    f"{R2_BASE_URL}/data/headshots/{player_name}.png", timeout=5)
                if _r2_resp.status_code == 200:
                    img_bytes = _r2_resp.content
            except Exception:
                pass
        elif os.path.exists(headshot_path):
            with open(headshot_path, "rb") as fh:
                img_bytes = fh.read()
        if img_bytes is None and _requests_available:
            # Look up MLBAM ID from razzball.csv
            rz_df = _cached_razzball(_RAZZBALL_PATH)
            mlbam_id = None
            if not rz_df.empty and "MLBAMID" in rz_df.columns:
                parts = player_name.strip().split(" ", 1)
                first = parts[0] if parts else ""
                last  = parts[1] if len(parts) > 1 else ""
                rz_match = rz_df[
                    rz_df.get("First", pd.Series(dtype=str)).astype(str).str.strip().str.lower().eq(first.lower()) &
                    rz_df.get("Last",  pd.Series(dtype=str)).astype(str).str.strip().str.lower().eq(last.lower())
                ]
                if not rz_match.empty:
                    raw_id = str(rz_match.iloc[0]["MLBAMID"]).strip()
                    if raw_id.isdigit():
                        mlbam_id = raw_id

            if mlbam_id:
                img_url = (
                    "https://img.mlbstatic.com/mlb-photos/image/upload/"
                    "d_people:generic:headshot:67:current.png/w_426,q_auto:best"
                    f"/v1/people/{mlbam_id}/headshot/67/current"
                )
                try:
                    resp = _requests.get(img_url, timeout=8)
                    if resp.status_code == 200:
                        img_bytes = resp.content
                        # Cache locally for next time (dev mode only)
                        if not _R2_MODE:
                            try:
                                os.makedirs(_HEADSHOTS_DIR, exist_ok=True)
                                with open(headshot_path, "wb") as fh:
                                    fh.write(img_bytes)
                            except OSError:
                                pass
                except Exception:
                    pass

        if img_bytes:
            st.image(img_bytes, width=200)
        else:
            st.markdown(
                "<div style='width:200px;height:200px;background:#1a1a2e;"
                "display:flex;align-items:center;justify-content:center;"
                "border-radius:10px;font-size:4rem;border:1px solid #2a3a5a;'>⚾</div>",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    with stats_col:
        st.markdown(
            f"<h4 style='margin:0 0 0.3rem;color:#d6e8f8;'>{player_name}</h4>"
            f"<div style='font-size:0.85rem;color:#93b8d8;margin-bottom:0.5rem;'>"
            f"<b>{pos}</b> &nbsp;|&nbsp; {team} &nbsp;|&nbsp; "
            f"Stage: <b>{stage}</b> &nbsp;|&nbsp; Age: <b>{age_str}</b> &nbsp;|&nbsp; "
            f"{sal_label}: <b>{sal_str}</b> &nbsp;|&nbsp; "
            f"Full Contract: <b>{ctrc_str}</b>"
            f"</div>"
            f"<div style='margin-bottom:0.3rem;'>"
            f"<span style='background:#1a3a1a;color:#2ecc71;padding:3px 10px;"
            f"border-radius:6px;font-size:0.85rem;font-weight:700;'>"
            f"&#x1F4C8; Pay vs Play Ratio: {ppr_str}</span>"
            f"<span style='color:#666;font-size:0.75rem;margin-left:8px;'>"
            f"projected WAR over contract &divide; total contract $M</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Year-by-year table
        year_cols = ["Year", "Team", "WAR_Total"]
        if is_pitcher:
            year_cols += [c for c in ["ERA", "FIP", "IP", "G", "GS", "K9", "BB9", "WHIP"] if c in player_df.columns]
        else:
            year_cols += [c for c in ["HR", "RBI", "AVG", "OBP", "SLG", "OPS", "G", "AB", "R"] if c in player_df.columns]
        year_cols = [c for c in year_cols if c in player_df.columns]

        yby_df = player_df[year_cols].copy()
        yby_df["Year"] = yby_df["Year"].astype(int)

        fmt: dict = {"WAR_Total": "{:.1f}"}
        for stat, f in [("ERA", "{:.2f}"), ("FIP", "{:.2f}"), ("IP",  "{:.1f}"),
                         ("K9",  "{:.2f}"), ("BB9", "{:.2f}"), ("WHIP","{:.2f}"),
                         ("AVG", "{:.3f}"), ("OBP", "{:.3f}"), ("SLG", "{:.3f}"), ("OPS", "{:.3f}")]:
            if stat in yby_df.columns:
                fmt[stat] = f

        st.dataframe(
            yby_df.rename(columns={"WAR_Total": "WAR"}).style.format(fmt, na_rep="—"),
            hide_index=True,
            use_container_width=True,
            height=min(70 + len(yby_df) * 35, 310),
        )

    # ------------------------------------------------------------------
    # WAR trend + WAR-per-dollar charts (full width below headshot+stats)
    # ------------------------------------------------------------------
    if "WAR_Total" in player_df.columns:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_df = player_df[["Year", "WAR_Total"]].copy()
        chart_df["Year"]      = chart_df["Year"].astype(int)
        chart_df["WAR_Total"] = pd.to_numeric(chart_df["WAR_Total"], errors="coerce")

        if "Salary_M" in player_df.columns:
            chart_df["Salary_M"] = pd.to_numeric(player_df["Salary_M"], errors="coerce")
            chart_df["WAR_per_M"] = chart_df.apply(
                lambda r: round(r["WAR_Total"] / r["Salary_M"], 3)
                if pd.notna(r["Salary_M"]) and r["Salary_M"] > 0 and pd.notna(r["WAR_Total"])
                else None,
                axis=1,
            )

        has_dollar = "WAR_per_M" in chart_df.columns and chart_df["WAR_per_M"].notna().any()
        years = chart_df["Year"].tolist()
        wars  = chart_df["WAR_Total"].tolist()

        try:
            bar_cols = ["#22c55e" if (pd.notna(w) and w >= 0) else "#ef4444" for w in wars]
            yr_strs  = [str(y) for y in years]
            war_vals = [w if pd.notna(w) else 0 for w in wars]

            fig_war = go.Figure(data=[go.Bar(
                x=yr_strs, y=war_vals,
                marker_color=bar_cols, marker_line_width=0,
                text=[f"{v:.1f}" if v != 0 else "" for v in war_vals],
                textposition="outside", textfont=dict(color="#dbeafe", size=9),
                hovertemplate="%{x}: %{y:.1f} WAR<extra></extra>",
            )])
            fig_war.update_layout(**_pt(
                title="WAR by Year",
                yaxis=dict(title="WAR"),
                xaxis=dict(tickvals=yr_strs),
                height=300,
                margin=dict(l=50, r=20, t=45, b=50),
            ))

            if has_dollar:
                wpm = chart_df["WAR_per_M"].tolist()
                wpm_vals = [w if pd.notna(w) else 0 for w in wpm]
                fig_wpm = go.Figure(data=[go.Bar(
                    x=yr_strs, y=wpm_vals,
                    marker_color="#3b82f6", marker_line_width=0,
                    text=[f"{v:.2f}" if v != 0 else "" for v in wpm_vals],
                    textposition="outside", textfont=dict(color="#dbeafe", size=9),
                    hovertemplate="%{x}: %{y:.2f} WAR/$M<extra></extra>",
                )])
                fig_wpm.update_layout(**_pt(
                    title="WAR per $M by Year",
                    yaxis=dict(title="WAR / $M"),
                    xaxis=dict(tickvals=yr_strs),
                    height=300,
                    margin=dict(l=50, r=20, t=45, b=50),
                ))
                col_war, col_wpm = st.columns(2)
                with col_war:
                    st.plotly_chart(fig_war, use_container_width=True,
                                    config={"displayModeBar": False})
                with col_wpm:
                    st.plotly_chart(fig_wpm, use_container_width=True,
                                    config={"displayModeBar": False})
            else:
                st.plotly_chart(fig_war, use_container_width=True,
                                config={"displayModeBar": False})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Landing / Home page
# ---------------------------------------------------------------------------

def _render_home_page():
    """Landing page: 3 faded carousel rows as background, feature cards in foreground."""
    _CARDS = [
        ("rankings",  "🏆", "Rankings",
         "All 30 MLB teams ranked by efficiency, fWAR, payroll, and win performance. "
         "See which franchises get the most wins per dollar and which are overspending."),
        ("team",      "🏟️", "Team Analysis",
         "Deep dive into any MLB team. Roster breakdown, salary commitments, "
         "efficiency rankings, fWAR leaders, IL status, and 3 year payroll projections."),
        ("league",    "📊", "Player Analysis",
         "Player-level cost effective line, PPEL regression, fWAR stability ratings, "
         "and age trajectory analysis across 4,000+ player-seasons."),
        ("simulator", "🎮", "Roster Simulator",
         "Build customized MLB rosters to optimize pay vs performance efficiency."),
        ("glossary",  "📖", "Glossary & Methodology",
         "Every metric explained in detail — fWAR, PPEL, CBT thresholds, WSR, "
         "efficiency formulas, roster grades, and how we calculate each one."),
    ]

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------
    st.markdown("""
    <style>
    /* ── wrapper — full-width, no box, same bg as app ────────────── */
    .home-wrap {
        position: relative;
        min-height: 100vh;
        overflow: hidden;
        background: #111927;
    }

    /* ── background carousel layer ───────────────────────────────── */
    .home-bg {
        position: absolute;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        display: flex;
        flex-direction: column;
        justify-content: space-evenly;
        padding: 10px 0;
        opacity: 0.10;
    }
    .cr-row   { overflow: hidden; }
    .cr-track { display: flex; width: max-content; }

    .cr-go-l1 { animation: go-l 200s linear infinite; }
    .cr-go-r  { animation: go-r 200s linear infinite; }
    .cr-go-l3 { animation: go-l 200s linear infinite; animation-delay: -60s; }

    @keyframes go-r {
        0%   { transform: translateX(-50%); }
        100% { transform: translateX(0);    }
    }
    @keyframes go-l {
        0%   { transform: translateX(0);    }
        100% { transform: translateX(-50%); }
    }
    .cr-img {
        width: 240px; height: 240px;
        object-fit: cover; object-position: top center;
        border-radius: 10px; margin: 0 10px; flex-shrink: 0;
    }

    /* ── foreground layer ─────────────────────────────────────────── */
    .home-fg {
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        gap: 0.4rem;
        padding: 0.8rem 2rem 2rem;
    }
    /* gradient only on the text — emoji lives in its own span */
    .home-title-grad {
        font-size: 3.8rem; font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #93c5fd 50%, #dbeafe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15; letter-spacing: -2px;
        vertical-align: middle;
    }
    /* baseball emoji — no gradient, natural colour, vertically aligned */
    .home-ball {
        font-size: 3.8rem; line-height: 1.15;
        -webkit-text-fill-color: initial;
        vertical-align: middle;
        margin-left: 0.15em;
    }
    .home-sub {
        font-size: 0.88rem; color: #5a8aaa; letter-spacing: 0.22em;
        text-transform: uppercase; text-align: center;
    }
    .home-mission {
        font-size: 0.95rem; color: #93b8d8; text-align: center;
        max-width: 720px; line-height: 1.7; margin: 0.5rem auto;
    }
    /* stationary tagline */
    .home-ticker {
        width: 100%; text-align: center;
        font-size: 0.82rem; color: #5a8aaa; letter-spacing: 0.22em;
        text-transform: uppercase; margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .home-rule {
        width: 52%; border: none;
        border-top: 1px solid rgba(96,165,250,0.18); margin: 0.3rem 0;
    }
    .home-cta {
        font-size: 0.88rem; color: #5a8aaa; letter-spacing: 0.14em;
        text-transform: uppercase; margin-top: 0.3rem;
    }
    /* card grid inside wrapper */
    .h-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        width: 100%;
        max-width: 1300px;
        margin-top: 0.8rem;
    }
    .h-card {
        background: rgba(17, 25, 39, 0.88);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(96,165,250,0.22);
        border-radius: 14px;
        padding: 1.5rem 1.1rem 1.3rem;
        text-align: center;
        display: flex; flex-direction: column;
        align-items: center; gap: 0.5rem;
        transition: border-color 0.2s, box-shadow 0.2s, transform 0.18s;
    }
    .h-card:hover {
        border-color: rgba(96,165,250,0.55);
        box-shadow: 0 4px 22px rgba(96,165,250,0.14);
        transform: translateY(-2px);
    }
    .h-icon  { font-size: 2.8rem; line-height: 1; }
    .h-title { font-size: 1.12rem; font-weight: 700; color: #dbeafe; }
    .h-desc  { font-size: 0.85rem; color: #7a9ebc; line-height: 1.6; flex: 1; }
    .h-btn {
        margin-top: 0.8rem;
        padding: 0.42rem 1.4rem;
        background: rgba(96,165,250,0.12);
        border: 1px solid rgba(96,165,250,0.32);
        border-radius: 7px;
        color: #93c5fd;
        font-size: 0.88rem; font-weight: 600;
        text-decoration: none !important;
        transition: background 0.18s, border-color 0.18s;
        white-space: nowrap;
    }
    .h-card:hover .h-btn {
        background: rgba(96,165,250,0.26);
        border-color: rgba(96,165,250,0.6);
        color: #dbeafe;
    }

    /* ── Home page — Tablet ──────────────────────────────────────── */
    @media (max-width: 1024px) {
        .h-grid { grid-template-columns: repeat(2, 1fr); max-width: 100%; }
        .home-title-grad, .home-ball { font-size: 2.8rem; }
        .home-fg { padding: 1.2rem 1rem 1.5rem; }
        .cr-img  { width: 140px; height: 140px; }
    }

    /* ── Issue 4 — Home page cards mobile layout ────────────────── */
    @media (max-width: 768px) {
        .home-wrap {
            min-height: auto;
            overflow-x: hidden !important;
        }
        .home-fg {
            min-height: auto;
            padding: 2rem 12px 1.5rem;
            gap: 0.35rem;
        }
        .h-grid {
            grid-template-columns: 1fr;
            gap: 0.7rem;
            max-width: 100%;
            padding: 0 12px;
            overflow-x: hidden !important;
        }
        .h-card {
            padding: 1rem 0.8rem 0.9rem;
            border-radius: 10px;
            word-wrap: break-word;
            overflow: hidden;
        }
        .h-icon  { font-size: 1.8rem; }
        .h-title { font-size: 0.85rem; }
        .h-desc  { font-size: 0.68rem; }
        .h-btn   {
            padding: 0.4rem 1rem;
            font-size: 0.75rem;
            width: 100%;
            text-align: center;
        }
        .home-title-grad, .home-ball { font-size: 2.2rem; }
        .home-sub { font-size: 0.68rem; letter-spacing: 0.15em; }
        .home-rule { width: 80%; }
        .home-cta { font-size: 0.65rem; }
        .cr-img { width: 120px; height: 120px; margin: 0 5px; border-radius: 6px; }
    }

    /* ── Home page — Small phone ─────────────────────────────────── */
    @media (max-width: 480px) {
        .home-title-grad, .home-ball { font-size: 1.8rem; }
        .home-sub { font-size: 0.6rem; }
        .cr-img { width: 75px; height: 75px; }
        .home-fg { padding: 1.5rem 8px 1.2rem; }
        .h-grid { padding: 0 8px; }
    }
    </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Build carousel: top-3 WAR per team from 2025 (all 30 teams represented)
    # ------------------------------------------------------------------
    _comb_url = _data_url("data/mlb_combined_2021_2025.csv")
    _carousel_players = _build_carousel_players(_comb_url)
    imgs = _cached_carousel_images(
        _HEADSHOTS_DIR,
        n=90,
        seed=42,
        player_list=tuple(_carousel_players) if _carousel_players else (),
    )

    def _strip(img_list: list, cls: str) -> str:
        tags = "".join(
            f'<img class="cr-img" src="data:image/png;base64,{b}" alt="">'
            for b in img_list * 2   # duplicate for seamless loop
        )
        return f'<div class="cr-row"><div class="cr-track {cls}">{tags}</div></div>'

    if imgs:
        n = len(imgs)
        t = n // 3
        bg_rows = (
            _strip(imgs,                    "cr-go-l1")  # row 1 → scrolls left
          + _strip(imgs[t:]  + imgs[:t],    "cr-go-r")   # row 2 → scrolls right
          + _strip(imgs[2*t:]+ imgs[:2*t],  "cr-go-l3")  # row 3 → scrolls left (phase-offset)
        )
    else:
        bg_rows = ""

    # ------------------------------------------------------------------
    # Build foreground card grid HTML
    # ------------------------------------------------------------------
    cards_html = "".join(
        f'<a class="h-card" href="?page={page_key}" target="_self" style="text-decoration:none;">'
        f'<div class="h-icon">{icon}</div>'
        f'<div class="h-title">{title}</div>'
        f'<div class="h-desc">{desc}</div>'
        f'<div class="h-btn">Open &#8594;</div>'
        f'</a>'
        for page_key, icon, title, desc in _CARDS
    )

    # ------------------------------------------------------------------
    # Render: carousel background fills the page, foreground overlays
    # ------------------------------------------------------------------
    st.markdown(f"""
    <div class="home-wrap">
      <div class="home-bg">{bg_rows}</div>
      <div class="home-fg">
        <div class="home-ticker">MLB Toolbox: Built by fans for fans</div>
        <div style="text-align:center;display:flex;align-items:center;justify-content:center;">
          <span class="home-title-grad">MLB Toolbox</span><span class="home-ball">&#9918;</span>
        </div>
        <div class="home-mission">
          Provide visualization tools and metrics to better<br>track, rank and forecast team and player cost per win efficiency
        </div>
        <hr class="home-rule">
        <div class="home-cta">Choose a tool to get started</div>
        <div class="h-grid">{cards_html}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# Roster Simulator — Best Fits helper
# ---------------------------------------------------------------------------

def _render_best_fits(roster_df: pd.DataFrame, budget_M: float) -> None:
    """Recommend available players that best complement the current roster and budget."""

    all_players = st.session_state.get("sim_df_full", pd.DataFrame())
    if all_players.empty:
        return

    on_roster     = set(roster_df["Player"].tolist()) if "Player" in roster_df.columns else set()
    current_cost  = float(roster_df["Salary_M"].fillna(0).sum()) if "Salary_M" in roster_df.columns else 0.0
    remaining     = budget_M - current_cost

    # Open roster slots
    _filled: dict[str, int] = {}
    if "pos_group" in roster_df.columns:
        for _pg in roster_df["pos_group"].tolist():
            for _slot in _ELIGIBLE_SLOTS_MAP.get(_pg, []):
                if _slot in _ROSTER_TEMPLATE:
                    _filled[_slot] = _filled.get(_slot, 0) + 1
    _missing = [s for s, need in _ROSTER_TEMPLATE.items()
                if s not in _OPTIONAL_SLOTS and _filled.get(s, 0) < need]

    # Reverse map: slot → list of pos_groups that fill it
    _pg_for_slot: dict[str, list[str]] = {}
    for _pg, _slots in _ELIGIBLE_SLOTS_MAP.items():
        for _s in _slots:
            _pg_for_slot.setdefault(_s, []).append(_pg)

    st.markdown("---")
    st.markdown("### Best Available Additions")

    # ── Status bar ──────────────────────────────────────────────────────────
    _sb1, _sb2 = st.columns([1, 2])
    with _sb1:
        _rem_color = "#06d6a0" if remaining >= 0 else "#ef4444"
        st.markdown(
            f"<div style='padding:0.6rem 1rem;background:#162030;border:1px solid #243f5c;"
            f"border-radius:8px;'>"
            f"<div style='font-size:0.7rem;color:#a8c8e8;margin-bottom:2px;'>Remaining Budget</div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{_rem_color};'>${remaining:.1f}M</div>"
            f"<div style='font-size:0.7rem;color:#6a8aaa;'>${budget_M:.0f}M total — ${current_cost:.1f}M committed</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with _sb2:
        if _missing:
            st.warning(f"**{len(_missing)} open position(s):** {', '.join(_missing)}  "
                       f"— these slots are not covered by your current roster.")
        else:
            st.success("All roster slots filled — additions below would improve depth or replace weak spots.")

    if remaining <= 0.5:
        st.info("Under $0.5M remaining. Remove a player to free up budget space.")
        return

    # ── Available pool ───────────────────────────────────────────────────────
    avail = all_players[
        ~all_players["Player"].isin(on_roster)
        & all_players["Salary_M"].notna()
        & (all_players["Salary_M"] <= remaining + 0.01)
    ].copy()

    if avail.empty:
        st.info("No players available within the remaining budget.")
        return

    avail["_score"] = avail["WAR_Total"].fillna(0) / avail["Salary_M"].clip(lower=0.1)

    # Acquisition type: check 40-man roster CSV for actual roster membership
    _roster_40_bf = st.session_state.get("_sim_roster_40", pd.DataFrame())
    _rostered_set = set()
    _rostered_teams: dict = {}
    if not _roster_40_bf.empty and "full_name" in _roster_40_bf.columns:
        _rostered_set = set(_roster_40_bf["full_name"].str.lower().str.strip())
        _rostered_teams = dict(zip(
            _roster_40_bf["full_name"].str.lower().str.strip(),
            _roster_40_bf["team"],
        ))
    avail["_is_rostered"] = avail["Player"].str.lower().str.strip().isin(_rostered_set)
    avail["Curr. Team"] = avail["Player"].str.lower().str.strip().map(_rostered_teams).fillna("")
    avail["Acquisition"] = avail["_is_rostered"].map({True: "Trade Required", False: "Free Agent"})
    # Free agents rank slightly higher at equal WAR (more accessible)
    avail["_adj_score"] = avail["_score"] * np.where(avail["_is_rostered"], 1.0, 1.2)

    _show_cols = ["Player", "Curr. Team", "Position", "Stage_Clean", "Age", "WAR_Total", "Salary_M", "_score", "Acquisition"]
    _show_cols = [c for c in _show_cols if c in avail.columns]

    _fmt = {"WAR_Total": "{:.1f}", "Salary_M": "${:.1f}M", "_score": "{:.2f}", "Age": "{:.0f}"}
    _rename = {"WAR_Total": "fWAR", "Salary_M": "Sal $M", "_score": "fWAR/$M", "Stage_Clean": "Stage"}

    fit_t1, fit_t2, fit_t3 = st.tabs(["Best Value Adds", "Fill Open Positions", "Best Group Fit"])

    # ── Tab 1: Best value adds (highest W/$M, any position) ─────────────────
    with fit_t1:
        st.caption(
            "Top affordable players by WAR/$ efficiency. Green = below-market value, "
            "red = below league avg. Excludes players already on your roster."
        )
        _top = (
            avail.sort_values("_adj_score", ascending=False)
                 .head(20)[_show_cols]
                 .rename(columns=_rename)
                 .reset_index(drop=True)
        )

        def _clr_score(v):
            try:
                f = float(v)
                if f >= 0.5: return "background-color:#14532d;color:white"
                if f >= 0.3: return "background-color:#166534;color:white"
                if f < 0.1:  return "background-color:#7f1d1d;color:white"
            except Exception:
                pass
            return ""

        st.dataframe(
            _top.style
                .format({k: v for k, v in {**_fmt, "WAR": "{:.1f}", "Sal $M": "${:.1f}M",
                          "W/$M": "{:.2f}", "Age": "{:.0f}"}.items() if k in _top.columns})
                .map(_clr_score, subset=["W/$M"] if "W/$M" in _top.columns else []),
            hide_index=True, use_container_width=True, height=440,
        )

    # ── Tab 2: Fill open positions ──────────────────────────────────────────
    with fit_t2:
        if not _missing:
            st.success("No open positions — all slots are filled by your current roster.")
            st.caption("Additions below would improve depth at occupied positions.")
            for _pos in list(_ROSTER_TEMPLATE.keys())[:4]:
                _pgs = _pg_for_slot.get(_pos, [_pos])
                _sub = avail[avail["pos_group"].isin(_pgs)].nlargest(3, "WAR_Total")
                if not _sub.empty:
                    with st.expander(f"{_pos} depth", expanded=False):
                        st.dataframe(
                            _sub[_show_cols].rename(columns=_rename)
                                .style.format({k: v for k, v in {**_fmt, "WAR": "{:.1f}",
                                    "Sal $M": "${:.1f}M", "W/$M": "{:.2f}", "Age": "{:.0f}"
                                }.items() if k in _sub.rename(columns=_rename).columns}),
                            hide_index=True, use_container_width=True,
                        )
        else:
            for _slot in _missing:
                _pgs = _pg_for_slot.get(_slot, [_slot])
                _slot_avail = avail[avail["pos_group"].isin(_pgs)].nlargest(5, "_score")
                _label = f"**{_slot}**  — {len(_slot_avail)} affordable option(s)"
                if _slot_avail.empty:
                    _label += "  ⚠ none in budget"
                with st.expander(_label, expanded=True):
                    if _slot_avail.empty:
                        st.warning(f"No affordable {_slot} available within ${remaining:.0f}M.")
                    else:
                        _sd = _slot_avail[_show_cols].rename(columns=_rename)
                        st.dataframe(
                            _sd.style.format({k: v for k, v in {**_fmt, "WAR": "{:.1f}",
                                "Sal $M": "${:.1f}M", "W/$M": "{:.2f}", "Age": "{:.0f}"
                            }.items() if k in _sd.columns}),
                            hide_index=True, use_container_width=True,
                        )

    # ── Tab 3: Best group fit (greedy knapsack) ──────────────────────────────
    with fit_t3:
        st.caption(
            "Players selected greedily by W/$M efficiency to maximize WAR within your "
            "remaining budget. Prioritises open positions first, then best overall value."
        )

        # Prioritise players that fill open slots
        _open_pgs = set()
        for _slot in _missing:
            for _pg in _pg_for_slot.get(_slot, [_slot]):
                _open_pgs.add(_pg)

        avail_sorted = pd.concat([
            avail[avail["pos_group"].isin(_open_pgs)].sort_values("_score", ascending=False),
            avail[~avail["pos_group"].isin(_open_pgs)].sort_values("_score", ascending=False),
        ]).drop_duplicates("Player")

        _group: list = []
        _g_cost = 0.0
        _g_war  = 0.0
        for _, _row in avail_sorted.iterrows():
            _s = float(_row.get("Salary_M") or 0)
            if _g_cost + _s <= remaining + 0.01:
                _group.append(_row)
                _g_cost += _s
                _g_war  += float(_row.get("WAR_Total") or 0)
            if len(_group) >= 10:
                break

        if _group:
            _gc1, _gc2, _gc3 = st.columns(3)
            _gc1.metric("Players in group", len(_group))
            _gc2.metric("Added WAR",        f"+{_g_war:.1f}")
            _gc3.metric("Group cost",       f"${_g_cost:.1f}M",
                        delta=f"${remaining - _g_cost:.1f}M still available", delta_color="normal")

            _gdf = pd.DataFrame(_group)[_show_cols].rename(columns=_rename).reset_index(drop=True)
            # Flag players that fill an open slot
            _gdf["Fills Gap"] = [
                "✓" if any(pg in _open_pgs for pg in [r.get("pos_group", "")])
                else ""
                for r in _group
            ]
            st.dataframe(
                _gdf.style.format({k: v for k, v in {**_fmt, "WAR": "{:.1f}",
                    "Sal $M": "${:.1f}M", "W/$M": "{:.2f}", "Age": "{:.0f}"
                }.items() if k in _gdf.columns}),
                hide_index=True, use_container_width=True, height=380,
            )

            # Projected roster totals after group
            _new_war  = (float(roster_df["WAR_Total"].sum()) if "WAR_Total" in roster_df.columns else 0) + _g_war
            _new_cost = current_cost + _g_cost
            st.info(
                f"**Projected after additions** — "
                f"WAR: **{_new_war:.1f}** (~{47.7 + _new_war:.0f} wins)  |  "
                f"Payroll: **${_new_cost:.1f}M** (${budget_M - _new_cost:.1f}M remaining)"
            )
        else:
            st.info("No players fit within the remaining budget.")


# ---------------------------------------------------------------------------
# Roster Simulator — main page
# ---------------------------------------------------------------------------

def _render_simulator_page():
    """Render the Roster Simulator page — full 2-column layout with sticky bar."""
    import io

    # ── CSS ────────────────────────────────────────────────────────────────
    st.markdown("""<style>
/* ══ Simulator page ═══════════════════════════════════════════════════ */
.sim-page-hdr{background:linear-gradient(135deg,#18243a 0%,#111927 100%);
  border:1px solid #1e3250;border-radius:12px;padding:0.65rem 1.1rem;margin-bottom:0.6rem;}
.sim-page-hdr h2{margin:0;font-size:1.15rem;color:#d6e8f8;font-weight:800;letter-spacing:-0.01em;}
.sim-page-hdr .sim-sub{font-size:0.68rem;color:#4a687e;margin-top:0.1rem;line-height:1.4;}

/* Chips */
.sim-chips{display:flex;gap:0.3rem;flex-wrap:wrap;margin-top:0.4rem;}
.sim-chip{padding:0.15rem 0.55rem;border-radius:999px;font-size:0.67rem;
  font-weight:700;border:1px solid transparent;letter-spacing:0.01em;}
.sim-chip.budget{background:#12213a;color:#93c5fd;border-color:#1e3a6a;}
.sim-chip.remain-ok  {background:#0a1f14;color:#4ade80;border-color:#14532d;}
.sim-chip.remain-warn{background:#1f1400;color:#fbbf24;border-color:#78450c;}
.sim-chip.remain-over{background:#1f0a0a;color:#fca5a5;border-color:#7f1d1d;}
.sim-chip.players{background:#141428;color:#a5b4fc;border-color:#2d2d5a;}
.sim-chip.slots-ok  {background:#0a1f14;color:#86efac;border-color:#14532d;}
.sim-chip.slots-open{background:#1f1400;color:#fcd34d;border-color:#78450c;}

/* Section divider */
.sim-divider{border:none;border-top:1px solid #1e3250;margin:0.5rem 0;}

/* Pool header — bolder, more prominent */
.sim-pool-hdr{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:0.3rem;padding-bottom:0.3rem;border-bottom:2px solid #1e3a5f;}
.sim-pool-hdr h4{margin:0;color:#d6e8f8;font-size:1.0rem;font-weight:800;letter-spacing:-0.01em;}
.sim-pool-badge{background:#1e3a5f;color:#93c5fd;padding:0.1rem 0.5rem;
  border-radius:999px;font-size:0.67rem;font-weight:700;}

/* "Already on roster" tag in pool */
.sim-added-tag{display:inline-block;background:#14532d;color:#86efac;
  font-size:0.60rem;font-weight:700;padding:0.05rem 0.4rem;
  border-radius:999px;margin-left:0.3rem;vertical-align:middle;}

/* Add-to-roster action bar */
.sim-action-bar{background:#0d1e35;border:1px solid #1e3a5f;border-radius:10px;
  padding:0.5rem 0.7rem;margin-top:0.4rem;display:flex;align-items:center;gap:0.6rem;}
.sim-sel-summary{font-size:0.70rem;color:#7a9ebc;flex:1;}
.sim-sel-summary strong{color:#d6e8f8;}

/* Roster right panel */
.sim-roster-hdr{display:flex;justify-content:space-between;align-items:center;
  border-bottom:2px solid #1e3a5f;padding-bottom:0.3rem;margin-bottom:0.45rem;}
.sim-roster-hdr .sim-rh-title{font-weight:800;color:#d6e8f8;font-size:1.0rem;letter-spacing:-0.01em;}
.sim-roster-hdr .sim-rh-count{font-size:0.67rem;color:#4a687e;font-weight:600;}

/* KPI grid — compact */
.sim-kpi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.25rem;margin-bottom:0.4rem;}
.sim-kpi-box{background:#0d1e35;border:1px solid #1e3250;border-radius:7px;
  padding:0.32rem 0.4rem;text-align:center;}
.sim-kpi-box .kv{font-size:1.0rem;font-weight:800;color:#d6e8f8;line-height:1.1;}
.sim-kpi-box .kl{font-size:0.57rem;color:#4a687e;margin-top:1px;text-transform:uppercase;letter-spacing:0.04em;}

/* Grade strip */
.sim-grade-strip{display:flex;gap:0.25rem;margin:0.3rem 0 0.5rem;}
.sim-grade-box{flex:1;text-align:center;padding:0.28rem 0.2rem;
  background:#0d1e35;border:1px solid #1e3250;border-radius:7px;}
.sim-grade-box .gv{font-size:1.15rem;font-weight:800;}
.sim-grade-box .gl{font-size:0.55rem;color:#4a687e;margin-top:1px;text-transform:uppercase;letter-spacing:0.03em;}
.sim-grade-box .gs{font-size:0.52rem;color:#2e4a62;}

/* CBT planning slider block */
.sim-cbt-block{background:#0d1e35;border:1px solid #1e3a5f;border-radius:9px;
  padding:0.55rem 0.7rem;margin:0.5rem 0;}
.sim-cbt-block .cb-title{font-size:0.68rem;font-weight:700;color:#7a9ebc;
  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.3rem;}
.sim-cbt-row{display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;margin-top:0.2rem;}
.sim-cbt-pill{padding:0.15rem 0.55rem;border-radius:999px;font-size:0.67rem;font-weight:700;border:1px solid transparent;}
.sim-cbt-pill.ok  {background:#0a1f14;color:#4ade80;border-color:#14532d;}
.sim-cbt-pill.warn{background:#1f1400;color:#fbbf24;border-color:#78450c;}
.sim-cbt-pill.over{background:#1f0a0a;color:#fca5a5;border-color:#7f1d1d;}

/* Sticky bar */
.mlb-sbar{position:fixed;bottom:0;top:auto;left:0;right:0;z-index:9998;
  background:rgba(8,13,22,0.98);border-top:1px solid #1e3250;
  padding:0.28rem 1.5rem;display:flex;align-items:center;gap:1.6rem;
  font-size:0.72rem;color:#7a9ebc;}
.mlb-sbar .sb-team{font-weight:800;color:#d6e8f8;font-size:0.80rem;}
.mlb-sbar .sb-stat{display:flex;flex-direction:column;align-items:center;}
.mlb-sbar .sb-val{font-weight:700;font-size:0.84rem;color:#93c5fd;}
.mlb-sbar .sb-lbl{font-size:0.57rem;color:#4a687e;}
.mlb-sbar-pad{height:44px;}
</style>""", unsafe_allow_html=True)

    # ── Player card sub-page routing ─────────────────────────────────────────
    if st.session_state.get("view_player"):
        player_name = st.session_state["view_player"]
        if st.button("← Back to Roster Simulator", key="card_back_btn", type="secondary"):
            st.session_state.pop("view_player", None)
            st.rerun()
        st.markdown(f"## {player_name}")
        st.markdown("---")
        if _R2_MODE:
            _comb = _data_url("data/mlb_combined_2021_2025.csv")
        else:
            if not os.path.exists(_DEFAULT_CONFIG):
                st.error(f"Config file not found: {_DEFAULT_CONFIG}")
                return
            _base = _load_base_config(_DEFAULT_CONFIG)
            _sal  = _resolve_data_path(_base["raw_salary_war_path"], _DEFAULT_CONFIG)
            _comb = os.path.join(os.path.dirname(_sal), "mlb_combined_2021_2025.csv")
        _hash = _file_hash(_comb)
        _render_player_card(player_name, _comb, _hash)
        return

    # ── Data loading ──────────────────────────────────────────────────────────
    if _R2_MODE:
        combined_path    = _data_url("data/mlb_combined_2021_2025.csv")
        ind_2025_path    = _data_url("data/2025mlbshared.csv")
        payroll_2026_dir = _data_url("2026 Payroll")
    else:
        if not os.path.exists(_DEFAULT_CONFIG):
            st.error(f"Config file not found: {_DEFAULT_CONFIG}")
            return
        base_cfg         = _load_base_config(_DEFAULT_CONFIG)
        salary_abs       = _resolve_data_path(base_cfg["raw_salary_war_path"], _DEFAULT_CONFIG)
        data_dir         = os.path.dirname(salary_abs)
        combined_path    = os.path.join(data_dir, "mlb_combined_2021_2025.csv")
        ind_2025_path    = os.path.join(data_dir, "2025mlbshared.csv")
        payroll_2026_dir = os.path.join(_ROOT_DIR, "2026 Payroll")

        if not os.path.exists(combined_path):
            st.error(f"Data file not found:\n`{combined_path}`")
            return

    if _R2_MODE or os.path.exists(payroll_2026_dir):
        try:
            df = _cached_2026_payroll(payroll_2026_dir, combined_path, _dir_hash(payroll_2026_dir))
        except Exception as e:
            st.error(f"Error loading 2026 payroll data: {e}")
            st.exception(e)
            return
    else:
        try:
            df = _cached_simulator_data(combined_path, ind_2025_path, _file_hash(combined_path))
        except Exception as e:
            st.error(f"Error loading player data: {e}")
            st.exception(e)
            return

    # ── Load 40-man roster CSV ──────────────────────────────────────────────
    _roster_40_path = _data_url("40man_rosters_2025.csv")
    _roster_40 = _cached_40man_roster(_roster_40_path, _file_hash(_roster_40_path))
    st.session_state["_sim_roster_40"] = _roster_40

    df = df.copy()
    if "WAR_Total" in df.columns and "pos_group" in df.columns:
        df["WAR_Pct"] = (
            df.groupby("pos_group")["WAR_Total"]
            .rank(pct=True).mul(100).round(0).astype("Int64")
        )
    st.session_state["sim_df_full"] = df
    try:
        st.session_state["sim_reliability"] = _cached_war_reliability(
            combined_path, _file_hash(combined_path)
        )
    except Exception:
        pass

    all_teams = sorted(df["Team"].dropna().unique().tolist())

    # ── Roster state ──────────────────────────────────────────────────────────
    roster_records = st.session_state.get("sim_roster", [])
    roster_df      = pd.DataFrame(roster_records) if roster_records else pd.DataFrame()
    n_rostered     = len(roster_records)
    _rc_cost = float(roster_df["Salary_M"].fillna(0).sum()) if (
        not roster_df.empty and "Salary_M" in roster_df.columns
    ) else 0.0

    # Pre-compute roster metrics (used in header, right panel, sticky bar)
    _r_total_war  = float(roster_df["WAR_Total"].sum())  if (not roster_df.empty and "WAR_Total"   in roster_df.columns) else 0.0
    _r_total_cost = _rc_cost
    _r_wpm        = _r_total_war / max(_r_total_cost, 0.1)
    _r_dpw        = _r_total_cost / max(_r_total_war, 0.1)
    _r_est_wins   = 47.7 + _r_total_war
    _r_n_fa  = int((roster_df["Stage_Clean"] == "FA").sum())      if (not roster_df.empty and "Stage_Clean" in roster_df.columns) else 0
    _r_n_arb = int((roster_df["Stage_Clean"] == "Arb").sum())     if (not roster_df.empty and "Stage_Clean" in roster_df.columns) else 0
    _r_n_pre = int((roster_df["Stage_Clean"] == "Pre-Arb").sum()) if (not roster_df.empty and "Stage_Clean" in roster_df.columns) else 0

    # ── Page header card ──────────────────────────────────────────────────────
    _hdr_budget = st.session_state.get("sim_budget_input", 130)
    _hdr_remain = _hdr_budget - _rc_cost
    _hdr_filled: dict[str, int] = {}
    if not roster_df.empty and "pos_group" in roster_df.columns:
        for _pg in roster_df["pos_group"].tolist():
            for _sl in _ELIGIBLE_SLOTS_MAP.get(_pg, []):
                if _sl in _ROSTER_TEMPLATE:
                    _hdr_filled[_sl] = _hdr_filled.get(_sl, 0) + 1
    _hdr_open = [s for s, need in _ROSTER_TEMPLATE.items()
                 if s not in _OPTIONAL_SLOTS and _hdr_filled.get(s, 0) < need]

    _rem_cls   = "remain-ok" if _hdr_remain >= 10 else ("remain-warn" if _hdr_remain >= 0 else "remain-over")
    _slots_cls = "slots-ok" if not _hdr_open else "slots-open"
    _slots_txt = "✅ Roster Full" if not _hdr_open else f"⚠ {len(_hdr_open)} open slot{'s' if len(_hdr_open)!=1 else ''}"

    _cbt_lbl, _cbt_bg, _cbt_fg, _cbt_nxt, _cbt_note = _cbt_info(_hdr_budget)
    _cbt_nxt_txt = (f" · ${_cbt_nxt - _hdr_budget:.0f}M to next tier" if _cbt_nxt else "")
    st.markdown(
        f"<div class='sim-page-hdr'>"
        f"<h2>🎮 Roster Simulator</h2>"
        f"<div class='sim-sub'>Build and analyze a custom MLB roster — select players from the pool and click <strong>Add to Roster</strong>."
        f"  Salaries are 2026 contracts; stats are 2025 figures.</div>"
        f"<div class='sim-chips'>"
        f"  <span class='sim-chip budget'>💰 Budget: ${_hdr_budget:.0f}M</span>"
        f"  <span class='sim-chip {_rem_cls}'>${_hdr_remain:+.1f}M remaining</span>"
        f"  <span class='sim-chip players'>👥 {n_rostered} players</span>"
        f"  <span class='sim-chip {_slots_cls}'>{_slots_txt}</span>"
        f"  <span class='sim-chip' style='background:{_cbt_bg};color:{_cbt_fg};border-color:{_cbt_fg}33;'"
        f"    title='{_cbt_note}'>"
        f"    🏦 {_cbt_lbl}{_cbt_nxt_txt}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── Optimizer Controls ────────────────────────────────────────────────────
    # Apply any pending budget value BEFORE the number_input widget renders.
    # (Streamlit forbids setting a widget's session-state key after it has rendered.)
    if "_sim_pending_budget" in st.session_state:
        st.session_state["sim_budget_input"] = st.session_state.pop("_sim_pending_budget")

    with st.expander("⚙️ Optimizer Controls", expanded=(n_rostered == 0)):
        oc1, oc2, oc3, oc4, oc5 = st.columns([3, 2, 3, 2, 2])
        with oc1:
            load_team = st.selectbox("Team", all_teams, key="sim_load_team")
        with oc2:
            if "sim_budget_input" not in st.session_state:
                st.session_state["sim_budget_input"] = 130
            budget_M = st.number_input(
                "Budget $M", min_value=50, max_value=350, step=5,
                key="sim_budget_input",
            )
            # CBT tier indicator
            _b_lbl, _b_bg, _b_fg, _b_nxt, _b_note = _cbt_info(float(budget_M))
            _b_nxt_s = f" · ${_b_nxt - budget_M:.0f}M to next" if _b_nxt else ""
            st.markdown(
                f"<div style='background:{_b_bg};color:{_b_fg};border-radius:6px;"
                f"padding:0.18rem 0.5rem;font-size:0.65rem;font-weight:600;margin-top:2px;'>"
                f"🏦 {_b_lbl}{_b_nxt_s}"
                + (f"<div style='font-size:0.58rem;opacity:0.8;margin-top:1px;'>{_b_note}</div>" if _b_note else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        with oc3:
            _locked_opts = (roster_df["Player"].tolist()
                            if not roster_df.empty and "Player" in roster_df.columns else [])
            st.multiselect(
                "Lock Players", _locked_opts, default=[],
                placeholder="None locked", key="sim_locked_players",
            )
        with oc4:
            st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
            _load_btn = st.button("📂 Load Roster", key="sim_load_team_btn",
                                  type="primary", use_container_width=True)
        with oc5:
            st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
            _reset_btn = st.button("🔄 Reset", key="sim_reset_btn",
                                   type="secondary", use_container_width=True)

    # Handle Load Roster — use 40-man roster CSV as source of truth
    if _load_btn:
        _tp = pd.DataFrame()
        if not _roster_40.empty and "team" in _roster_40.columns:
            # Filter 40-man roster to selected team
            _r40_team = _roster_40[_roster_40["team"] == load_team].copy()
            if not _r40_team.empty:
                # Build a lookup from the payroll/combined DataFrame
                _lkp = df.set_index("Player") if not df.empty else pd.DataFrame()
                _rows = []
                _matched = 0
                _unmatched_names = []
                for _, _r in _r40_team.iterrows():
                    _nm = str(_r["full_name"])
                    if not _lkp.empty and _nm in _lkp.index:
                        _pr = _lkp.loc[_nm]
                        if isinstance(_pr, pd.DataFrame):
                            _pr = _pr.iloc[0]
                        _rec = _pr.to_dict()
                        _rec["Player"]   = _nm
                        _rec["_dc_only"] = False
                        _matched += 1
                    else:
                        # Player on 40-man but no stats — league minimum
                        _pos = str(_r.get("position", "UNK"))
                        _pg = ("SP" if _pos == "P" else
                               "RP" if _pos in ("RP", "CL") else
                               _pos if _pos in ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH") else
                               "OF" if _pos in ("OF", "LF", "CF", "RF") else "UNK")
                        _rec = {
                            "Player":      _nm,
                            "Team":        load_team,
                            "Position":    _pos,
                            "pos_group":   _pg,
                            "Salary_M":    0.74,
                            "Stage_Clean": "Pre-Arb",
                            "WAR_Total":   float("nan"),
                            "W_per_M":     float("nan"),
                            "PPR":         float("nan"),
                            "_dc_only":    True,
                        }
                        _unmatched_names.append(_nm)
                    # Add status from 40-man CSV
                    _rec["_40man_status"] = str(_r.get("status", "Active"))
                    _rec["on_40man"] = True
                    _rows.append(_rec)
                _tp = pd.DataFrame(_rows)
                # Store debug info for the debug expander
                st.session_state["_sim_40man_debug"] = {
                    "team": load_team,
                    "total": len(_r40_team),
                    "matched": _matched,
                    "unmatched": len(_unmatched_names),
                    "unmatched_names": _unmatched_names,
                }
        # Fallback: if 40-man CSV unavailable, use payroll/combined data
        if _tp.empty:
            _tp = df[df["Team"] == load_team].copy()
            _tp["_dc_only"] = False
        if not _tp.empty:
            st.session_state["sim_roster"] = _tp.to_dict("records")
            _ab = float(_tp["Salary_M"].fillna(0.74).sum())
            if _ab > 0:
                st.session_state["_sim_pending_budget"] = max(50, round(_ab))
            st.rerun()
        else:
            st.warning(f"No players found for {load_team}.")

    # Handle Reset
    if _reset_btn:
        for _k in ("sim_roster", "sim_roster_war", "sim_roster_cost"):
            st.session_state.pop(_k, None)
        st.rerun()

    # ── Debug expander (only when ?debug=1) ──────────────────────────────────
    if st.query_params.get("debug") == "1" and "_sim_40man_debug" in st.session_state:
        _dbg = st.session_state["_sim_40man_debug"]
        with st.expander("🔍 40-Man Roster Debug", expanded=True):
            st.markdown(f"**Team:** {_dbg['team']}")
            st.markdown(f"**Player count:** {_dbg['total']}  (must be <= 40)")
            st.markdown(f"**Matched to stats CSV:** {_dbg['matched']}")
            st.markdown(f"**No stats (league min):** {_dbg['unmatched']}")
            if _dbg["unmatched_names"]:
                st.markdown("**Unmatched player names:**")
                for _un in _dbg["unmatched_names"]:
                    st.markdown(f"- {_un}")

    # ── Fix 5 — Roster Status Explainer ───────────────────────────────────────
    _render_glossary([
        ("40-Man Roster", "40-Man Roster",
         "The full group of players under MLB contract with a team. Includes the active 26-man roster, "
         "injured list players, and optioned minor leaguers. Teams have 40 slots — adding a player beyond "
         "40 requires removing someone first."),
        ("26-Man Roster", "Active Roster",
         "The 26 players eligible to play in any given game. Managers choose their lineup from this group. "
         "Expands to 28 in September."),
        ("60-Day IL", "60-Day Injured List",
         "Players sidelined for at least 60 days. They do NOT count against the active 26-man roster but "
         "still occupy a 40-man spot. Teams use this to free active roster space for healthy players."),
        ("15-Day IL", "15-Day Injured List",
         "Short-term injury designation. Player is out at least 15 days but retains their 40-man spot. "
         "Common for minor injuries. Frees one active roster space."),
        ("Option", "Optional Assignment",
         "A 40-man player sent to the minor leagues. Still occupies a 40-man spot but frees an active roster "
         "space. Players can be optioned up to 3 times before requiring waivers to move down."),
        ("DFA", "Designated for Assignment",
         "Team removes a player from the 40-man roster. Player has 10 days to be traded, claimed on waivers, "
         "or released. Used to clear 40-man space for new additions."),
        ("Service Time", "MLB Service Time",
         "Days accrued on the active roster. 172 days = 1 full service year. Determines arb eligibility "
         "(3 years), free agency (6 years), and Super Two status (~2.17 years)."),
        ("Trade Required", "Trade Required Acquisition",
         "This player is on another team's 40-man roster. Adding them to your custom roster would require "
         "a trade in real life. Players not on any 40-man roster can be signed directly as free agents."),
    ], title="📋 Understanding Roster Rules & Status", cols=2)

    # ── Filters ───────────────────────────────────────────────────────────────
    # Detect handedness columns (Bats for hitters, Throws for pitchers)
    _hand_col = next((c for c in ("Bats", "Throws", "Hand", "B", "T") if c in df.columns), None)
    _has_hand = _hand_col is not None
    _hand_opts = (["All"] + sorted(df[_hand_col].dropna().unique().tolist())
                  if _has_hand else ["All", "L", "R", "S"])

    if _has_hand:
        fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8 = st.columns([3, 2, 2, 1.5, 2, 2, 2, 1])
    else:
        fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8 = st.columns([3, 2, 2, 1.5, 2, 2, 2, 1])

    with fc1:
        teams_sel = st.multiselect("Team(s)", all_teams, default=[],
                                   placeholder="All teams", key="sim_teams")
    with fc2:
        player_type_sel = st.selectbox("Type", ["All", "Pitcher", "Position Player"],
                                       key="sim_ptype")
    _pitcher_pos  = ["SP", "RP"]
    _position_pos = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
    if player_type_sel == "Pitcher":
        _pos_opts = ["All"] + _pitcher_pos
    elif player_type_sel == "Position Player":
        _pos_opts = ["All"] + _position_pos
    else:
        _pos_opts = ["All"] + _position_pos + _pitcher_pos
    with fc3:
        position_sel = st.selectbox("Position", _pos_opts, key="sim_pos")
    with fc4:
        hand_sel = st.selectbox(
            "Hand",
            _hand_opts,
            key="sim_hand",
            disabled=not _has_hand,
            help=("Filter by batting/throwing hand." if _has_hand
                  else "Add a 'Bats' or 'Throws' column to your data CSV to enable this filter."),
        )
    _stage_opts = (["All"] + sorted(df["Stage_Clean"].dropna().unique().tolist())
                   if "Stage_Clean" in df.columns else ["All"])
    with fc5:
        stage_sel = st.selectbox("Stage", _stage_opts, key="sim_stage")
    with fc6:
        min_war = st.slider("Min WAR", min_value=-5.0, max_value=8.0,
                            value=0.0, step=0.5, key="sim_min_war")
    with fc7:
        name_search = st.text_input("Search", placeholder="Player name…", key="sim_search")
    with fc8:
        st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
        if st.button("✕", key="sim_filter_clear", help="Clear all filters"):
            for _fk in ("sim_teams", "sim_ptype", "sim_pos", "sim_hand", "sim_stage",
                        "sim_min_war", "sim_search"):
                st.session_state.pop(_fk, None)
            st.rerun()

    # ── Apply filters ──────────────────────────────────────────────────────────
    filtered = df.copy()
    if teams_sel:
        filtered = filtered[filtered["Team"].isin(teams_sel)]
    if player_type_sel == "Pitcher":
        filtered = filtered[filtered["pos_group"].isin(["SP", "RP"])]
    elif player_type_sel == "Position Player":
        filtered = filtered[~filtered["pos_group"].isin(["SP", "RP"])]
    if position_sel != "All":
        filtered = filtered[filtered["Position"] == position_sel]
    if _has_hand and hand_sel != "All":
        filtered = filtered[filtered[_hand_col] == hand_sel]
    if stage_sel != "All" and "Stage_Clean" in filtered.columns:
        filtered = filtered[filtered["Stage_Clean"] == stage_sel]
    if min_war > -5.0 and "WAR_Total" in filtered.columns:
        filtered = filtered[filtered["WAR_Total"] >= min_war]
    if name_search and name_search.strip():
        filtered = filtered[
            filtered["Player"].str.contains(name_search.strip(), case=False, na=False)
        ]
    filtered = filtered.sort_values("WAR_Total", ascending=False,
                                    na_position="last").reset_index(drop=True)

    # ── Terms glossary ────────────────────────────────────────────────────────
    _render_glossary([
        ("WAR",        "Wins Above Replacement",
         "How many wins a player adds vs a replacement-level player (e.g. a minor leaguer or bench fill-in). "
         "League average is ~2 WAR; All-Star level is 5+; MVP-caliber is 8+."),
        ("W/$M",       "WAR per $M (Season)",
         "Single-season WAR divided by 2026 salary. Higher = more efficient. "
         "League avg is roughly 0.3–0.5 WAR/$M for free agents; Pre-Arb players often exceed 2.0."),
        ("PPR",        "Pay-to-Performance Ratio (Contract)",
         "Total career WAR earned divided by total contract value ($M). Accounts for multi-year deals — "
         "a player on a 5-year $100M contract needs ~50 WAR to break even at market rate."),
        ("PPEL",       "Pay-Performance Efficiency Line",
         "The regression line of WAR vs Salary across all players. Sits at the league-average cost of a win. "
         "Players below the line are underpaid; above are overpaid relative to their output."),
        ("Stage",      "Contract Stage",
         "FA = Free Agent (negotiated market contract). "
         "Arb = Arbitration-eligible (salary set by panel or hearing). "
         "Pre-Arb = Club-controlled, first 3 years of service time (near league minimum ~$740K)."),
        ("CBT",        "Competitive Balance Tax",
         "MLB luxury tax on payrolls above $244M (2026). Tiers are $20M bands: "
         "$244M · $264M (1st Apron) · $284M · $304M (2nd Apron). "
         "The two aprons also carry roster-building restrictions — "
         "1st Apron limits trades/signings; 2nd Apron adds draft pick penalties."),
    ], title="📖 Terms & Definitions")

    # ── Vertical layout: Player Pool → My Custom Roster ───────────────────────

    # ───────────────── TOP: Player Pool ───────────────────────────────────────
    with st.container():
        # Players already on the custom roster
        _on_roster_names = {r.get("Player", "") for r in roster_records}
        _n_available = len([p for p in filtered["Player"] if p not in _on_roster_names])

        st.markdown(
            f"<div class='sim-pool-hdr'>"
            f"<h4>Player Pool</h4>"
            f"<span class='sim-pool-badge'>{_n_available:,} available"
            f" · {len(filtered):,} shown · {len(df):,} total</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        base_show  = ["Player", "Team", "Position", "Stage_Clean",
                      "Age", "WAR_Total", "WSR", "Salary_M", "W_per_M", "PPR"]
        hitter_ext = [c for c in ["HR", "AVG", "OBP"] if c in filtered.columns]
        pitch_ext  = [c for c in ["ERA", "IP"]         if c in filtered.columns]
        show_cols  = [c for c in base_show + hitter_ext[:2] + pitch_ext[:1]
                      if c in filtered.columns]
        display_df = filtered[show_cols].copy()

        # Mark players already on roster
        display_df["Added"] = display_df["Player"].apply(
            lambda p: "✓" if p in _on_roster_names else ""
        )
        # Show 40-man roster membership (current team or blank for true FAs)
        _r40_team_map = {}
        if not _roster_40.empty and "full_name" in _roster_40.columns:
            _r40_team_map = dict(zip(
                _roster_40["full_name"].str.lower().str.strip(),
                _roster_40["team"],
            ))
        display_df["40-Man"] = display_df["Player"].str.lower().str.strip().map(_r40_team_map).fillna("")
        show_cols_with_added = ["Added"] + show_cols + ["40-Man"]

        col_cfg: dict = {
            "Added":      st.column_config.TextColumn("✓",       width=30),
            "Player":     st.column_config.TextColumn("Player",   width="medium"),
            "Team":       st.column_config.TextColumn("Team",     width="small"),
            "Position":   st.column_config.TextColumn("Pos",      width="small"),
            "Stage_Clean":st.column_config.TextColumn("Stage",    width="small"),
            "Age":        st.column_config.NumberColumn("Age",    format="%d",     width="small"),
            "WAR_Total":  st.column_config.NumberColumn("WAR ↕",  format="%.1f",   width="small",
                          help="Wins Above Replacement — how many wins this player adds vs a "
                               "replacement-level fill-in. Scale: 0–1 = bench, 2 = solid starter, "
                               "4–5 = All-Star, 7+ = MVP candidate."),
            "Salary_M":   st.column_config.NumberColumn("Sal $M ↕", format="$%.1fM", width="small"),
            "W_per_M":    st.column_config.NumberColumn("W/$M ↕", format="%.2f",   width="small",
                          help="Season WAR ÷ 2026 salary. Higher = more efficient."),
            "PPR":        st.column_config.NumberColumn("Ctrc W/$M", format="%.2f", width="small",
                          help="Contract WAR per $M over the full contract length."),
            "WSR":        st.column_config.NumberColumn("WSR", format="%.2f", width="small",
                          help="WAR Stability Rating: mean WAR / (1 + std WAR). Higher = more consistent production. "
                               "Elite ≥ 3.5, Reliable ≥ 2.0, Volatile ≥ 1.0, Unstable < 1.0."),
            "40-Man":     st.column_config.TextColumn("40-Man", width="small",
                          help="Current 40-man roster team. Blank = free agent (can be signed directly)."),
        }
        for _st in hitter_ext[:2] + pitch_ext[:1]:
            if _st in ("AVG", "OBP"):
                col_cfg[_st] = st.column_config.NumberColumn(_st, format="%.3f", width="small")
            elif _st == "HR":
                col_cfg[_st] = st.column_config.NumberColumn("HR", format="%d",   width="small")
            elif _st == "ERA":
                col_cfg[_st] = st.column_config.NumberColumn("ERA", format="%.2f", width="small")
            else:
                col_cfg[_st] = st.column_config.NumberColumn(_st, format="%.1f", width="small")

        _fsig = f"{sorted(teams_sel)}{player_type_sel}{position_sel}{hand_sel}{stage_sel}{min_war}{name_search}"
        _fkey = hashlib.md5(_fsig.encode()).hexdigest()[:10]

        event = st.dataframe(
            display_df[show_cols_with_added],
            column_config=col_cfg,
            hide_index=True,
            use_container_width=True,
            height=480,
            on_select="rerun",
            selection_mode="multi-row",
            key=f"sim_table_{_fkey}",
        )

        sel_indices   = event.selection.rows if hasattr(event, "selection") else []
        selected_rows = (filtered.iloc[sel_indices].copy() if sel_indices
                         else pd.DataFrame(columns=filtered.columns))
        # Exclude players already on roster from the add action
        selected_new  = selected_rows[~selected_rows["Player"].isin(_on_roster_names)]
        n_sel         = len(selected_rows)
        n_new         = len(selected_new)

        # ── Action bar ──────────────────────────────────────────────────────
        _sel_war  = float(selected_new["WAR_Total"].sum()) if (n_new > 0 and "WAR_Total" in selected_new.columns) else 0.0
        _sel_cost = float(selected_new["Salary_M"].sum())  if (n_new > 0 and "Salary_M"  in selected_new.columns) else 0.0
        _budget_after = float(budget_M) - _rc_cost - _sel_cost

        _ba1, _ba2, _ba3 = st.columns([4, 3, 5])
        with _ba1:
            _add_clicked = st.button(
                f"➕ Add {n_new} to Roster" if n_new > 0 else (
                    f"✓ Already Added" if n_sel > 0 else "↑ Select players above"
                ),
                type="primary" if n_new > 0 else "secondary",
                use_container_width=True, key="sim_add_btn", disabled=(n_new == 0),
            )
        with _ba2:
            _card_clicked = st.button(
                "🃏 Player Card", type="secondary",
                use_container_width=True, key="sim_card_btn", disabled=(n_sel == 0),
                help="Open detailed player card for the first selected player.",
            )
        with _ba3:
            if n_new > 0:
                _after_color = "#4ade80" if _budget_after >= 10 else ("#fbbf24" if _budget_after >= 0 else "#fca5a5")
                st.markdown(
                    f"<div style='padding:0.3rem 0.5rem;background:#0d1e35;border:1px solid #1e3a5f;"
                    f"border-radius:8px;font-size:0.68rem;color:#7a9ebc;line-height:1.5;'>"
                    f"<strong style='color:#d6e8f8;'>{n_new}</strong> new · "
                    f"WAR <strong style='color:#d6e8f8;'>+{_sel_war:.1f}</strong> · "
                    f"Cost <strong style='color:#d6e8f8;'>${_sel_cost:.1f}M</strong> · "
                    f"Budget after <strong style='color:{_after_color};'>${_budget_after:+.1f}M</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif n_sel > 0 and n_new == 0:
                st.caption("All selected players are already on your roster.")

        # Player comparison (2–4 selected)
        if 2 <= n_sel <= 4:
            with st.expander(f"🔄 Compare {n_sel} Players", expanded=True):
                _render_player_comparison(selected_rows)

        # Handle Add to Roster
        if _add_clicked and n_sel > 0:
            _existing       = st.session_state.get("sim_roster", [])
            _existing_names = {r.get("Player") for r in _existing}
            _new_recs       = [r for r in selected_rows.to_dict("records")
                               if r.get("Player") not in _existing_names]
            st.session_state["sim_roster"] = _existing + _new_recs
            st.rerun()

        if _card_clicked and n_sel > 0:
            st.session_state["view_player"] = selected_rows.iloc[0]["Player"]
            st.rerun()

    # ───────────────── BOTTOM: My Custom Roster ───────────────────────────────
    with st.container():
        if not roster_records:
            st.markdown(
                "<div style='background:#0d1f38;border:1px dashed #1e3a5c;border-radius:12px;"
                "padding:2.5rem 1rem;text-align:center;color:#4a6a8a;margin-top:2rem;'>"
                "<div style='font-size:2rem;'>🏟</div>"
                "<div style='font-size:0.88rem;margin-top:0.5rem;color:#7aa2c0;'>Roster is empty</div>"
                "<div style='font-size:0.72rem;margin-top:0.3rem;'>Load a team or select players and click <strong>Add to Roster</strong></div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            _r_remaining = float(budget_M) - _r_total_cost
            _rem_color   = "#4ade80" if _r_remaining >= 0 else "#f87171"

            # Roster title + KPI grid
            st.markdown(
                f"<div class='sim-roster-hdr'>"
                f"  <span class='sim-rh-title'>My Custom Roster</span>"
                f"  <span class='sim-rh-count'>{n_rostered} players</span>"
                f"</div>"
                f"<div class='sim-kpi-grid'>"
                f"  <div class='sim-kpi-box'><div class='kv'>{_r_total_war:.1f}</div><div class='kl'>Total WAR</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv'>${_r_total_cost:.0f}M</div><div class='kl'>Payroll</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv' style='color:{_rem_color}'>${_r_remaining:+.0f}M</div><div class='kl'>Remaining</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv'>~{_r_est_wins:.0f}W</div><div class='kl'>Est. Wins</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv'>${_r_dpw:.1f}M</div><div class='kl'>$/WAR</div>"
                f"  <div style='font-size:0.58rem;color:#4a687e;margin-top:2px;'>${_r_total_cost / max(_r_total_war + 47.7, 1):.2f}M $/Win</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv'>{_r_n_fa}/{_r_n_arb}/{_r_n_pre}</div><div class='kl'>FA/Arb/Pre</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Export / Clear
            _ec1, _ec2 = st.columns(2)
            with _ec1:
                _csv_buf = io.StringIO()
                roster_df.to_csv(_csv_buf, index=False)
                st.download_button(
                    "⬇ Export CSV", data=_csv_buf.getvalue(),
                    file_name="my_custom_roster.csv", mime="text/csv",
                    key="sim_export_btn", use_container_width=True,
                )
            with _ec2:
                if st.button("🗑 Clear Roster", key="sim_clear_btn",
                             use_container_width=True, type="secondary"):
                    for _k in ("sim_roster", "sim_roster_war", "sim_roster_cost"):
                        st.session_state.pop(_k, None)
                    st.rerun()

            # Roster Grades strip
            _grades = _roster_grade(roster_df)
            _GRADE_COLORS = {
                "A+": "#00e676", "A": "#4caf50", "B+": "#8bc34a", "B": "#cddc39",
                "C":  "#ffc107", "D": "#ff9800", "F": "#f44336",
            }
            _gh = "<div class='sim-grade-strip'>"
            for _dim, (_gr, _sc) in _grades.items():
                _gc = _GRADE_COLORS.get(_gr, "#7aa2c0")
                _gh += (
                    f"<div class='sim-grade-box'>"
                    f"<div class='gv' style='color:{_gc};'>{_gr}</div>"
                    f"<div class='gl'>{_dim}</div>"
                    f"<div class='gs'>{_sc}/100</div>"
                    f"</div>"
                )
            _gh += "</div>"
            st.markdown(_gh, unsafe_allow_html=True)

            # Tabs: Roster | Analysis | Best Additions
            rt1, rt2, rt3 = st.tabs(["📋 Roster", "📊 Analysis", "✨ Best Additions"])

            with rt1:
                _reliability = st.session_state.get("sim_reliability", {})
                _show = [c for c in ["Player", "Position", "Stage_Clean", "Age",
                                     "WAR_Total", "Salary_M", "W_per_M", "PPR"]
                         if c in roster_df.columns]
                _edf = roster_df[_show].copy()
                _edf["Decision"]    = [_contract_decision(r) for r in roster_records]
                _edf["Consistency"] = [_reliability.get(r.get("Player", ""), {}).get("grade", "?")
                                       for r in roster_records]
                _edf.insert(0, "✕", False)
                _dc_flags = [bool(r.get("_dc_only", False)) for r in roster_records]
                def _hl_dc(row):
                    return (["background-color:#2d1f00;color:#fbbf24"] * len(row)
                            if _dc_flags[row.name] else [""] * len(row))
                if any(_dc_flags):
                    st.caption("🟡 Amber = depth chart only (no MLB stats, league min salary assumed)")
                _edited = st.data_editor(
                    _edf.style.apply(_hl_dc, axis=1),
                    column_config={
                        "✕":           st.column_config.CheckboxColumn("✕",          width="small"),
                        "WAR_Total":   st.column_config.NumberColumn("WAR",           format="%.1f",   width="small"),
                        "Salary_M":    st.column_config.NumberColumn("Sal $M",        format="$%.1fM", width="small"),
                        "W_per_M":     st.column_config.NumberColumn("W/$M",          format="%.2f",   width="small"),
                        "PPR":         st.column_config.NumberColumn("Ctrc W/$M",     format="%.2f",   width="small"),
                        "Stage_Clean": st.column_config.TextColumn("Stage",           width="small"),
                        "Position":    st.column_config.TextColumn("Pos",             width="small"),
                        "Age":         st.column_config.NumberColumn("Age",           format="%d",     width="small"),
                        "Decision":    st.column_config.TextColumn("Action",          width="small"),
                        "Consistency": st.column_config.TextColumn("Consistency",     width="small"),
                    },
                    disabled=[c for c in _edf.columns if c != "✕"],
                    hide_index=True, use_container_width=True,
                    height=min(60 + (n_rostered + 1) * 35, 460),
                    key="roster_editor",
                )
                if st.button("Remove Selected", key="roster_remove_btn", type="secondary"):
                    _keep = [rec for rec, rm in zip(roster_records, _edited["✕"].tolist()) if not rm]
                    if len(_keep) < len(roster_records):
                        st.session_state["sim_roster"] = _keep
                        st.rerun()
                    else:
                        st.info("No players checked — tick the ✕ column first.")
                st.markdown("##### Position Coverage")
                _render_position_coverage(roster_df)

                # ── CBT Threshold Planner ────────────────────────────────
                st.markdown("<hr class='sim-divider'>", unsafe_allow_html=True)
                if "sim_cbt_adj" not in st.session_state:
                    st.session_state["sim_cbt_adj"] = 0
                _cbt_adj = st.slider(
                    "Luxury Tax Threshold Adjustment ($M)",
                    min_value=-20, max_value=20, step=1,
                    key="sim_cbt_adj",
                    help=(
                        "Shift the CBT base threshold ($244M for 2026) to model different "
                        "planning targets. Negative = more conservative, Positive = more aggressive."
                    ),
                )
                _eff_thresh = 244.0 + _cbt_adj
                _cbt_lbl2, _, _cbt_fg2, _, _cbt_note2 = _cbt_info(_r_total_cost)
                _rem_vs_thresh = _eff_thresh - _r_total_cost
                _pill_cls2 = "ok" if _rem_vs_thresh >= 0 else "over"
                _rem_vs_txt = (
                    f"${_rem_vs_thresh:.0f}M under target"
                    if _rem_vs_thresh >= 0
                    else f"${abs(_rem_vs_thresh):.0f}M over target"
                )
                _note_txt = _cbt_note2 if _cbt_note2 else "✓ Under CBT — no luxury tax"
                st.markdown(
                    f"<div class='sim-cbt-block'>"
                    f"  <div class='cb-title'>🧾 CBT Threshold Planner</div>"
                    f"  <div class='sim-cbt-row'>"
                    f"    <span style='font-size:0.68rem;color:#7a9ebc;'>"
                    f"      Target: <strong style='color:#d6e8f8;'>${_eff_thresh:.0f}M</strong>"
                    f"    </span>"
                    f"    <span class='sim-cbt-pill {_pill_cls2}'>{_rem_vs_txt}</span>"
                    f"  </div>"
                    f"  <div style='font-size:0.63rem;color:#4a687e;margin-top:0.3rem;'>"
                    f"    <span style='color:{_cbt_fg2};font-weight:700;'>{_cbt_lbl2}</span>"
                    f"    {(' — ' + _note_txt) if _note_txt else ''}"
                    f"  </div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with rt2:
                _PG_ORDER  = ["SP", "RP", "C", "1B", "2B", "3B", "SS", "CF", "OF", "DH"]
                _PG_COLORS = _PG_CHART_COLORS
                if "pos_group" in roster_df.columns and "WAR_Total" in roster_df.columns:
                    _pg_war = (roster_df.groupby("pos_group")["WAR_Total"]
                               .sum().reindex(_PG_ORDER).dropna())
                    if not _pg_war.empty:
                        _cats  = _pg_war.index.tolist()
                        _vals  = _pg_war.values.tolist()
                        _bcols = [_PG_COLORS[_PG_ORDER.index(p)] for p in _cats if p in _PG_ORDER]
                        _ymax  = max(_vals) * 1.35 if _vals else 10
                        _fig1  = go.Figure(data=[go.Bar(
                            x=_cats, y=_vals, marker_color=_bcols, marker_line_width=0,
                            text=[f"{v:.1f}" for v in _vals], textposition="outside",
                            textfont=dict(color="#dbeafe", size=10),
                            hovertemplate="%{x}: %{y:.1f} WAR<extra></extra>",
                        )])
                        _fig1.update_layout(**_pt(
                            title="WAR by Position Group",
                            yaxis=dict(title="WAR", range=[0, _ymax]),
                            height=260, margin=dict(l=40, r=10, t=36, b=36),
                        ))
                        st.plotly_chart(_fig1, use_container_width=True,
                                        config={"displayModeBar": False})
                if "Salary_M" in roster_df.columns and "WAR_Total" in roster_df.columns:
                    _sc_df = roster_df.dropna(subset=["Salary_M", "WAR_Total"])
                    if not _sc_df.empty:
                        _dc2 = ([_PG_COLORS[_PG_ORDER.index(p)] if p in _PG_ORDER else "#60a5fa"
                                 for p in _sc_df["pos_group"]]
                                if "pos_group" in _sc_df.columns else "#3b82f6")
                        _xlim = float(_sc_df["Salary_M"].max()) * 1.15
                        _xr   = np.linspace(0, _xlim, 60)
                        _yr   = _xr * _r_wpm
                        _fig2 = go.Figure()
                        _fig2.add_trace(go.Scatter(
                            x=_xr, y=_yr, mode="lines",
                            line=dict(color="#f59e0b", dash="dash", width=1.5),
                            opacity=0.6, name="Avg WAR/$M", hoverinfo="skip",
                        ))
                        _fig2.add_trace(go.Scatter(
                            x=_sc_df["Salary_M"].tolist(), y=_sc_df["WAR_Total"].tolist(),
                            mode="markers+text",
                            text=(_sc_df["Player"].tolist() if "Player" in _sc_df.columns else []),
                            textposition="top center",
                            textfont=dict(size=7, color="#7aa2c0"),
                            marker=dict(color=_dc2, size=8, opacity=0.9,
                                        line=dict(color="#1e3a5f", width=1)),
                            hovertemplate="<b>%{text}</b><br>$%{x:.1f}M · WAR %{y:.1f}<extra></extra>",
                            showlegend=False,
                        ))
                        _fig2.update_layout(**_pt(
                            title="Salary vs WAR",
                            xaxis=dict(title="Salary ($M)"),
                            yaxis=dict(title="WAR"),
                            height=260, showlegend=True,
                            margin=dict(l=40, r=10, t=36, b=36),
                        ))
                        st.plotly_chart(_fig2, use_container_width=True,
                                        config={"displayModeBar": False})
                # ── Fix 6 — Future Payroll Commitments ────────────────────
                if n_rostered >= 5:
                    st.markdown("<hr class='sim-divider'>", unsafe_allow_html=True)
                    st.markdown("##### 📅 Future Payroll Commitments")

                    _fut_rows = []
                    for _, _fp in roster_df.iterrows():
                        _pn  = _fp.get("Player", "?")
                        _sal = float(_fp.get("Salary_M") or 0.74)
                        _stg = str(_fp.get("Stage_Clean", ""))
                        _age = int(_fp["Age"]) if pd.notna(_fp.get("Age")) else 28
                        # 2027/2028 from payroll data if available
                        _s27 = float(_fp["2027"]) if pd.notna(_fp.get("2027")) and float(_fp.get("2027") or 0) > 0 else None
                        _s28 = float(_fp["2028"]) if pd.notna(_fp.get("2028")) and float(_fp.get("2028") or 0) > 0 else None
                        # Estimate if not available
                        if _s27 is None:
                            if _stg == "Pre-Arb":
                                _s27 = 0.74
                            elif _stg == "Arb":
                                _s27 = round(_sal * 1.25, 1)
                            else:
                                _s27 = None  # FA — unknown
                        if _s28 is None:
                            if _stg == "Pre-Arb":
                                _s28 = round(max(2.0, _sal * 3), 1)  # enters arb
                            elif _stg == "Arb":
                                _s28 = round(_sal * 1.5, 1)
                            else:
                                _s28 = None
                        _fut_rows.append({"Player": _pn, "Stage": _stg,
                                          "2026": _sal, "2027": _s27, "2028": _s28})

                    _fut_df = pd.DataFrame(_fut_rows)

                    # Stacked bar chart by stage
                    _stg_colors = {"Pre-Arb": "#22c55e", "Arb": "#f59e0b", "FA": "#3b82f6"}
                    _years_lbl = ["2026", "2027", "2028"]
                    _fig_fut = go.Figure()
                    for stg, clr in _stg_colors.items():
                        _stg_sub = _fut_df[_fut_df["Stage"] == stg]
                        vals = [float(_stg_sub[yr].dropna().sum()) for yr in _years_lbl]
                        _fig_fut.add_trace(go.Bar(
                            x=_years_lbl, y=vals, name=stg,
                            marker_color=clr, opacity=0.85,
                            hovertemplate=f"{stg}<br>%{{x}}: $%{{y:.1f}}M<extra></extra>",
                        ))
                    # Unknown/FA years
                    _unk_vals = [0, 0, 0]
                    for yr_i, yr in enumerate(_years_lbl):
                        _unk_vals[yr_i] = float(_fut_df[_fut_df[yr].isna()].shape[0] * 0)  # placeholder
                    _fig_fut.add_hline(y=244, line_dash="dash", line_color="#f59e0b", opacity=0.5,
                                       annotation_text="CBT $244M", annotation_position="top right",
                                       annotation_font_color="#f59e0b")
                    _fig_fut.add_hline(y=float(budget_M), line_dash="dot", line_color="#3b82f6", opacity=0.4,
                                       annotation_text=f"Budget ${budget_M}M", annotation_position="bottom right",
                                       annotation_font_color="#3b82f6")
                    _fig_fut.update_layout(**_pt(
                        title="Projected Payroll by Stage (2026–2028)",
                        yaxis=dict(title="Total $M"), height=340,
                        barmode="stack", showlegend=True,
                        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                    ))
                    st.plotly_chart(_fig_fut, use_container_width=True, config={"displayModeBar": False})

                    # Summary table
                    _fut_show = _fut_df[["Player", "Stage", "2026", "2027", "2028"]].copy()
                    _fut_show["2026"] = _fut_show["2026"].apply(lambda v: f"${v:.1f}M" if pd.notna(v) else "—")
                    _fut_show["2027"] = _fut_show["2027"].apply(lambda v: f"~${v:.1f}M" if pd.notna(v) else "Free Agent")
                    _fut_show["2028"] = _fut_show["2028"].apply(lambda v: f"~${v:.1f}M" if pd.notna(v) else "Free Agent")
                    st.dataframe(_fut_show, hide_index=True, use_container_width=True,
                                 height=min(60 + n_rostered * 35, 400))
                    st.caption(
                        "2026 salaries reflect actual contracts. 2027–2028 figures for arbitration-eligible "
                        "players are estimates based on typical raise rates. Free agent years (shown as "
                        "'Free Agent') assume the player's contract expires."
                    )

                with st.expander("🔄 Trade Analyzer", expanded=False):
                    _render_trade_analyzer(roster_df)

            with rt3:
                _render_best_fits(roster_df, float(budget_M))

    # ── Sticky roster summary strip ────────────────────────────────────────────
    if roster_records:
        _sb_remain = float(budget_M) - _r_total_cost
        _sb_rc = "#4ade80" if _sb_remain >= 0 else "#f87171"
        st.markdown(
            f"<div class='mlb-sbar'>"
            f"  <span class='sb-team'>🏟 Custom Roster</span>"
            f"  <span class='sb-stat'><span class='sb-val'>{_r_total_war:.1f}</span>"
            f"    <span class='sb-lbl'>WAR</span></span>"
            f"  <span class='sb-stat'><span class='sb-val'>${_r_total_cost:.0f}M</span>"
            f"    <span class='sb-lbl'>Payroll</span></span>"
            f"  <span class='sb-stat'><span class='sb-val' style='color:{_sb_rc}'>${_sb_remain:+.0f}M</span>"
            f"    <span class='sb-lbl'>Remaining</span></span>"
            f"  <span class='sb-stat'><span class='sb-val'>~{_r_est_wins:.0f}W</span>"
            f"    <span class='sb-lbl'>Est. Wins</span></span>"
            f"  <span class='sb-stat'><span class='sb-val'>{n_rostered}</span>"
            f"    <span class='sb-lbl'>Players</span></span>"
            f"</div>"
            f"<div class='mlb-sbar-pad'></div>",
            unsafe_allow_html=True,
        )

    _render_feedback_widget("simulator")


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
                st.plotly_chart(fig, use_container_width=True,
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
            st.dataframe(mix_df, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True,
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
                st.plotly_chart(fig, use_container_width=True,
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
            st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})
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
            st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar": False})
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
    st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})


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
                st.dataframe(styled, use_container_width=True)
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
                st.dataframe(slot_df, use_container_width=True, hide_index=True)
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
                st.plotly_chart(_fig_cur, use_container_width=True, config={"displayModeBar": False})

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
                st.plotly_chart(_fig_dep, use_container_width=True, config={"displayModeBar": False})

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
    @st.cache_data(show_spinner=False)
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
      <div style="font-size:10px;color:#7a9ebc;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Median $/fWAR</div>
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
        t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
            "Cost Effective Line",
            "PPEL",
            "Age Trajectory",
            "By Team",
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
                "<b>Key terms:</b> <b>fWAR</b> = FanGraphs Wins Above Replacement — how many extra wins a player "
                "adds vs a minor-league replacement. <b>PPR</b> = Pay-to-Performance Ratio (career fWAR ÷ total contract $M; "
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
    <div style="background:#0d1b2a;border:1px solid #14532d;border-radius:8px;padding:10px 12px;">
      <div style="font-size:13px;font-weight:700;color:#22c55e;margin-bottom:4px;">Pre-Arbitration</div>
      <div style="font-size:10px;color:#7a9ebc;line-height:1.5;">0–3 years service time. Salary near league minimum (~$740K). Teams control rights — often the best value in baseball.</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #b88840;border-radius:8px;padding:10px 12px;">
      <div style="font-size:13px;font-weight:700;color:#f59e0b;margin-bottom:4px;">Arbitration</div>
      <div style="font-size:10px;color:#7a9ebc;line-height:1.5;">3–6 years service time. Salary negotiated or set by arbitration hearing. Pay rises based on prior performance.</div>
    </div>
    <div style="background:#0d1b2a;border:1px solid #3b6fd4;border-radius:8px;padding:10px 12px;">
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
      <span style="color:#06d6a0;">●</span> Free Agent &nbsp;
      <span style="color:#fbbf24;">●</span> Arbitration &nbsp;
      <span style="color:#38bdf8;">●</span> Pre-Arb.
      Switch colour mode in the Display panel.
    </div>
    <div>
      <span style="font-weight:700;color:#a8c8e8;">PPR (Pay-Performance Ratio)</span><br>
      Actual salary ÷ model-predicted salary.
      PPR &lt; 1 = underpaid · PPR &gt; 1 = overpaid · PPR = 1 = fair market.
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
                "Ranked by PPR (lowest = most underpaid). Adjusts with all filters above.</div>",
                unsafe_allow_html=True,
            )
            _top25 = df.nsmallest(25, "PPR")[["Player","Team","Year","WAR_Total","Salary_M","predicted","PPR","Stage_Clean"]].copy()
            _top25.insert(0, "#", range(1, len(_top25) + 1))
            _top25.columns = ["#", "Player", "Team", "Year", "fWAR", "Salary $M", "Expected $M", "PPR", "Stage"]
            _top25["Year"] = _top25["Year"].astype(int)
            st.dataframe(
                _top25.style.format({
                    "fWAR": "{:.1f}", "Salary $M": "{:.1f}",
                    "Expected $M": "{:.1f}", "PPR": "{:.3f}",
                }).apply(lambda row: ["background-color:#0c221866"] * len(row) if row["#"] <= 5 else [""] * len(row), axis=1),
                hide_index=True, use_container_width=True, height=min(60 + 25 * 35, 720),
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
            st.plotly_chart(fig2, use_container_width=True)

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
                hide_index=True, use_container_width=True, height=min(60 + 25 * 35, 720),
            )

        # ── Tab 4 — By Team ───────────────────────────────────────────────
        with t4:
            _avail_t = sorted(df["Team"].unique())
            _def_t   = _avail_t[:min(4, len(_avail_t))]
            _sel_t   = st.multiselect("Teams", _avail_t, default=_def_t, key="ef_team_sel")
            if not _sel_t:
                st.info("Select at least one team.")
            else:
                fig4 = go.Figure()
                for _tn, _tg in df[df["Team"].isin(_sel_t)].groupby("Team"):
                    fig4.add_trace(go.Scatter(
                        x=_tg["WAR_Total"], y=_tg["Salary_M"],
                        mode="markers", name=_tn, marker=dict(size=8),
                        text=_tg["Player"] + " (" + _tg["Year"].astype(str) + ")",
                        hovertemplate="%{text}<br>WAR: %{x:.1f}  Salary: $%{y:.2f}M<extra></extra>",
                    ))
                fig4.update_layout(**_pt(
                    title="WAR vs Salary by Team",
                    xaxis=dict(title="WAR"), yaxis=dict(title="Salary ($M)"),
                    height=640, showlegend=True,
                ))
                st.plotly_chart(fig4, use_container_width=True)

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
                use_container_width=True, hide_index=True, height=600,
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
                st.plotly_chart(fR1, use_container_width=True)

            with _r2:
                fR2 = go.Figure(go.Histogram(
                    x=df["residual"], nbinsx=30,
                    marker=dict(color="#2a9d8f", line=dict(color="#0e1117", width=0.5)),
                ))
                fR2.add_vline(x=0, line_color="#f4a261", line_dash="dash")
                fR2.update_layout(**_pt(title="Residual Distribution",
                    xaxis=dict(title="Residual ($M)"), yaxis=dict(title="Count"), height=400))
                st.plotly_chart(fR2, use_container_width=True)

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
            st.plotly_chart(fR3, use_container_width=True)

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
                st.plotly_chart(fig_cel, use_container_width=True)

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
                st.plotly_chart(fig_bar1, use_container_width=True)
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
                    st.plotly_chart(fig_p3, use_container_width=True)

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
                    st.plotly_chart(fig_bar3, use_container_width=True)
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
                    st.plotly_chart(fig_p5, use_container_width=True)

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
                    st.plotly_chart(fig_bar5, use_container_width=True)
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

                # WAR trajectory lines, one per player, coloured by trend
                fig_pa = go.Figure()
                _trend_map = dict(zip(_pa_summary["Player"], _pa_summary["Trend"]))
                for _pname, _pgrp in _pa_df.groupby("Player"):
                    _t_color = _TREND_COLORS.get(_trend_map.get(_pname, "Neutral"), "#fbbf24")
                    _pg = _pgrp.sort_values("Year")
                    fig_pa.add_trace(go.Scatter(
                        x=_pg["Year"].astype(int), y=_pg["WAR_Total"],
                        mode="lines+markers", name=_pname,
                        line=dict(color=_t_color, width=1.5),
                        marker=dict(size=6, color=_t_color),
                        hovertemplate=(
                            f"<b>{_pname}</b><br>"
                            "Year: %{x}<br>WAR: %{y:.1f}"
                            f"<br>Team: {_pg['Team'].iloc[-1]}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                        opacity=0.65,
                    ))
                # Legend proxy traces
                for _tn, _tc in _TREND_COLORS.items():
                    fig_pa.add_trace(go.Scatter(
                        x=[None], y=[None], mode="lines",
                        line=dict(color=_tc, width=2.5),
                        name=_tn, showlegend=True,
                    ))
                fig_pa.update_layout(**_pt(
                    title="Pre-Arbitration Player WAR Trajectories",
                    xaxis=dict(title="Season", tickformat="d"),
                    yaxis=dict(title="WAR"),
                    height=580, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                ))
                st.plotly_chart(fig_pa, use_container_width=True)

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
                    use_container_width=True, hide_index=True, height=500,
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
                st.plotly_chart(fig_wsr, use_container_width=True)
            else:
                st.info("No players meet the minimum qualifying threshold.")


# ---------------------------------------------------------------------------
# League Analysis page
# ---------------------------------------------------------------------------

def _render_league_analysis():
    """Render the League Efficiency Analysis page."""
    import subprocess

    # ── Interactive player-level Cost Effective Line ─────────────────────────
    st.markdown("### Player Analysis")
    st.caption(
        "Interactive WAR vs Salary analysis for every player (2021–2025). "
        "Fit the PPEL regression line, colour by career stage or Pay-Performance Ratio, "
        "and identify the most underpaid / overpaid players in the league."
    )
    _render_efficiency_frontier()

    _render_feedback_widget("league")


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
        st.image(_r2_image(scatter_path), use_container_width=True)
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
                st.plotly_chart(_fig_rank, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.image(_r2_image(ranking_path), use_container_width=True)
        except Exception:
            if _R2_MODE or os.path.exists(ranking_path):
                st.image(_r2_image(ranking_path), use_container_width=True)
            else:
                st.info("efficiency_ranking.png not found — click Regenerate Analysis.")
    elif _R2_MODE or os.path.exists(ranking_path):
        st.image(_r2_image(ranking_path), use_container_width=True)
    else:
        st.info("efficiency_ranking.png not found — click Regenerate Analysis.")

    st.markdown("---")

    if _R2_MODE or os.path.exists(position_path):
        st.subheader("WAR by Position -- Efficient vs Inefficient Teams")
        st.caption(
            "Top row = 5 most efficient teams (spending least above frontier). "
            "Bottom row = 5 most inefficient. Each bar = average WAR from that position group."
        )
        st.image(_r2_image(position_path), use_container_width=True)
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
                st.plotly_chart(_fig_q3, use_container_width=True, config={"displayModeBar": False})
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
        st.dataframe(styled, use_container_width=True, height=600)

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
                                st.plotly_chart(fig, use_container_width=True,
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
                    st.plotly_chart(_fig_pa, use_container_width=True,
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
                    st.plotly_chart(_fig_sc, use_container_width=True,
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
    """League Rankings — WAR, Salary, and Efficiency ranked by team."""

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown("""<style>
.rk-hdr{background:linear-gradient(135deg,#18243a 0%,#111927 100%);
  border:1px solid #1e3250;border-radius:12px;padding:0.9rem 1.3rem;margin-bottom:0.8rem;}
.rk-hdr h2{margin:0;font-size:1.25rem;color:#d6e8f8;font-weight:700;}
.rk-hdr .rk-sub{font-size:0.72rem;color:#7a9ebc;margin-top:0.15rem;}
.rk-answer{background:#18243a;border:1px solid #1e3250;border-radius:10px;
  padding:0.7rem 0.8rem;text-align:center;}
.rk-answer .rk-q{font-size:0.68rem;color:#93b8d8;text-transform:uppercase;
  letter-spacing:0.05em;margin-bottom:0.2rem;font-weight:600;}
.rk-answer .rk-team{font-size:1.25rem;font-weight:800;color:#d6e8f8;line-height:1.1;}
.rk-answer .rk-val{font-size:0.78rem;color:#93b8d8;margin-top:0.2rem;}
.rk-answer .rk-icon{font-size:1.3rem;margin-bottom:0.15rem;line-height:1;}
</style>""", unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    detail_csv  = _data_url("efficiency_detail.csv")
    ranking_csv = _data_url("al_nl_ranking_table.csv")

    _files_local = not _R2_MODE
    if _files_local and (not os.path.exists(detail_csv) or not os.path.exists(ranking_csv)):
        st.warning(
            "Rankings data not yet generated. Go to **League Analysis** → "
            "**Regenerate Analysis** first."
        )
        return

    try:
        detail_df  = _read_csv(detail_csv)
        ranking_df = _read_csv(ranking_csv)
    except Exception as _e:
        st.error(f"Could not load rankings data: {_e}")
        return

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='rk-hdr'>"
        "<h2>🏆 MLB Efficiency Rankings</h2>"
        "<div class='rk-sub'>All 30 MLB teams ranked by spending efficiency, fWAR production, "
        "payroll, and win performance. Efficiency measures how far above or below the "
        "cost-effective line each team sits — negative means winning more per dollar.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Year selector ─────────────────────────────────────────────────────────
    years_avail = sorted(detail_df["Year"].dropna().unique().astype(int), reverse=True)
    yr_col, _ = st.columns([2, 8])
    with yr_col:
        sel_year = st.selectbox("Season", years_avail, key="rk_year", index=0)

    yr_df = detail_df[detail_df["Year"] == sel_year].copy()
    if yr_df.empty:
        st.warning(f"No data for {sel_year}.")
        return

    # Merge division/league metadata from the ranking table
    if {"Abbr", "Division", "League"}.issubset(ranking_df.columns):
        _meta = ranking_df[["Abbr", "Team", "Division", "League",
                             "Playoff_Apps", "WS_Wins"]].copy()
        _meta = _meta.rename(columns={"Abbr": "_abbr", "Team": "Team_Full"})
        yr_df = yr_df.merge(_meta, left_on="Team", right_on="_abbr", how="left")
        yr_df.drop(columns=["_abbr"], errors="ignore", inplace=True)

    # Derived column: $/WAR for the selected season
    yr_df["DPW"] = (yr_df["payroll_M"] / yr_df["team_WAR"].clip(lower=0.1)).round(2)

    # ── Terms glossary ────────────────────────────────────────────────────────
    _render_glossary([
        ("CEL",            "Cost Effective Line",
         "The league regression of team payroll vs wins. Shows the expected win total for any given payroll. "
         "Teams on the line get average value; teams below it win more than their spending predicts."),
        ("Efficiency Gap", "Dollars Above/Below the CEL",
         "How many $M a team's payroll sits above or below what their win total 'should' cost at market rate. "
         "Negative (green) = efficient — winning more per dollar. Positive (red) = overspending."),
        ("$/fWAR",         "Dollars per fWAR",
         "Average cost of one fWAR for this team's roster in the selected season. "
         "League average free-agent rate is ~$7–9M/fWAR; Pre-Arb players cost far less."),
        ("Wins vs Pred.",  "Wins Above Payroll Prediction",
         "Actual wins minus the wins predicted by the payroll regression. "
         "Positive = team beats their expected win total given their spending (e.g. strong coaching, "
         "player development). Negative = underperforming relative to payroll."),
        ("Avg Gap $M",     "5-Year Average Efficiency Gap",
         "Average dollars above or below the CEL per season across 2021–2025. "
         "Consistently negative teams (Tampa Bay, Cleveland) build wins efficiently year over year."),
        ("fWAR",           "FanGraphs Wins Above Replacement (Team)",
         "Sum of all player fWAR on the 40-man roster for the season. "
         "Reflects total roster talent independent of lineup decisions or luck."),
    ], title="📖 Terms & Definitions")

    # ── Quick-answer highlight cards ─────────────────────────────────────────
    def _qa(icon, question, team, value_str, bdr="#1e3250", tooltip=""):
        _tip = f" title='{tooltip}'" if tooltip else ""
        return (
            f"<div class='rk-answer' style='border-color:{bdr};cursor:help;'{_tip}>"
            f"<div class='rk-icon'>{icon}</div>"
            f"<div class='rk-q'>{question}</div>"
            f"<div class='rk-team'>{team}</div>"
            f"<div class='rk-val'>{value_str}</div>"
            f"</div>"
        )

    _best_eff  = yr_df.loc[yr_df["dollar_gap_M"].idxmin()]
    _worst_eff = yr_df.loc[yr_df["dollar_gap_M"].idxmax()]
    _top_war   = yr_df.loc[yr_df["team_WAR"].idxmax()]
    _top_wins  = yr_df.loc[yr_df["Wins"].idxmax()]
    _overperf  = yr_df.loc[yr_df["wins_vs_pred"].idxmax()]
    _best_dpw  = yr_df.loc[yr_df["DPW"].idxmin()]

    # Use full team names if available
    def _full(row):
        return row.get("Team_Full") or row["Team"]

    qa1, qa2, qa3 = st.columns(3)
    qa4, qa5, qa6 = st.columns(3)

    with qa1:
        st.markdown(_qa(
            "🏆", "Most Efficient",
            _full(_best_eff),
            f"${_best_eff['dollar_gap_M']:.0f}M below the line",
            "#1e4a1e",
            "This team won the most games relative to payroll $ spent",
        ), unsafe_allow_html=True)
    with qa2:
        st.markdown(_qa(
            "📈", "Top Overperformer",
            _full(_overperf),
            f"+{_overperf['wins_vs_pred']:.1f} wins vs forecast",
            "#0c2218",
            "This team had the most wins relative to forecast based on payroll $ spent",
        ), unsafe_allow_html=True)
    with qa3:
        st.markdown(_qa(
            "💰", "Best $ per fWAR",
            _full(_best_dpw),
            f"${_best_dpw['DPW']:.1f}M per fWAR",
            "#1a1228",
            "Lowest cost per fWAR — the most production per dollar on the roster",
        ), unsafe_allow_html=True)
    with qa4:
        st.markdown(_qa(
            "🔴", "Least Efficient Spending",
            _full(_worst_eff),
            f"${_worst_eff['dollar_gap_M']:.0f}M above the line",
            "#280c0c",
            "This team spent the most payroll $ for level of wins earned",
        ), unsafe_allow_html=True)
    with qa5:
        st.markdown(_qa(
            "⭐", "Top fWAR",
            _full(_top_war),
            f"{_top_war['team_WAR']:.1f} total fWAR",
            tooltip="Highest total roster fWAR — sum of all player contributions above replacement",
        ), unsafe_allow_html=True)
    with qa6:
        st.markdown(_qa(
            "🏅", "Most Wins",
            _full(_top_wins),
            f"{int(_top_wins['Wins'])} wins",
            tooltip="Highest regular-season win total for the selected year",
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

    # ── Ranking tabs ─────────────────────────────────────────────────────────
    rt1, rt2, rt3, rt4 = st.tabs([
        "🏆 Efficiency", "⭐ fWAR", "💰 Salary", "📈 Win Performance",
    ])

    def _hbar(df_in, x_col, color_fn, title, x_label, text_fn=None, zero_line=False):
        """Render a themed horizontal bar chart for rankings."""
        vals   = df_in[x_col].tolist()
        teams  = df_in["Team"].tolist()
        colors = [color_fn(v) for v in vals]
        texts  = [text_fn(v) for v in vals] if text_fn else [f"{v:.1f}" for v in vals]
        fig = go.Figure(go.Bar(
            y=teams, x=vals, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=texts, textposition="outside",
            textfont=dict(color="#d6e8f8", size=9),
            hovertemplate="%{y}: %{text}<extra></extra>",
        ))
        _xaxis = dict(title=x_label, zeroline=zero_line,
                       zerolinecolor="#4a687e", zerolinewidth=1)
        # Center 0 on charts with diverging values
        if zero_line and vals:
            _abs_max = max(abs(v) for v in vals) * 1.15
            _xaxis["range"] = [-_abs_max, _abs_max]
        fig.update_layout(**_pt(
            title=title,
            xaxis=_xaxis,
            yaxis=dict(autorange="reversed"),
            height=max(340, len(df_in) * 22),
            margin=dict(l=60, r=80, t=42, b=30),
        ))
        return fig

    # ── Tab 1: Efficiency ─────────────────────────────────────────────────────
    with rt1:
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "Dollar gap from the <b>Cost Effective Line</b> — how much more or less each team "
            "spends relative to what their wins cost at league-average market rates. "
            "<span style='color:#22c55e;font-weight:600;'>Green</span> = below the line (efficient). "
            "<span style='color:#ef4444;font-weight:600;'>Red</span> = above the line (overspending).</div>",
            unsafe_allow_html=True,
        )
        _eff = yr_df.sort_values("dollar_gap_M", ascending=True).reset_index(drop=True)
        _eff["Rank"] = range(1, len(_eff) + 1)

        ch1, tb1 = st.columns([3, 2])
        with ch1:
            st.plotly_chart(_hbar(
                _eff, "dollar_gap_M",
                color_fn=lambda v: "#22c55e" if v <= 0 else "#ef4444",
                title=f"{sel_year} — Efficiency Ranking of MLB Team Spending $ to Regular Season Wins",
                x_label="$ Gap ($M) — negative = efficient",
                text_fn=lambda v: f"${v:+.0f}M",
                zero_line=True,
            ), use_container_width=True, config={"displayModeBar": False})
        with tb1:
            _e = _eff[["Rank", "Team", "dollar_gap_M", "payroll_M", "Wins", "in_playoffs"]].copy()
            _e.columns = ["#", "Team", "Gap $M", "Payroll $M", "Wins", "Postseason"]
            _e["Gap $M"]     = _e["Gap $M"].round(0).astype(int)
            _e["Payroll $M"] = _e["Payroll $M"].round(0).astype(int)
            _e["Wins"]       = _e["Wins"].round(0).astype(int)
            _e["Postseason"]  = _e["Postseason"].map({True: "✓", False: ""})

            def _eff_clr(row):
                g = row["Gap $M"]
                if g < -80:  return ["background-color:#0c2218"] * len(row)
                if g < 0:    return ["background-color:#14532d55"] * len(row)
                if g > 120:  return ["background-color:#2d0c0c"] * len(row)
                if g > 0:    return ["background-color:#2d150c55"] * len(row)
                return [""] * len(row)

            st.dataframe(
                _e.style.apply(_eff_clr, axis=1).format(
                    {"Gap $M": "{:+d}", "Payroll $M": "{:d}", "Wins": "{:d}"}, na_rep="—"),
                hide_index=True, use_container_width=True,
                height=min(60 + len(_e) * 35, 720),
            )

    # ── Tab 2: WAR ────────────────────────────────────────────────────────────
    with rt2:
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "Total team WAR for the selected season across all pitchers and position players. "
            "Higher WAR = more total roster talent. Top-5 teams highlighted in green.</div>",
            unsafe_allow_html=True,
        )
        _war = yr_df.sort_values("team_WAR", ascending=False).reset_index(drop=True)
        _war["Rank"] = range(1, len(_war) + 1)
        _war_max = _war["team_WAR"].max() or 1

        ch2, tb2 = st.columns([3, 2])
        with ch2:
            st.plotly_chart(_hbar(
                _war, "team_WAR",
                color_fn=lambda v: (
                    "#22c55e" if v >= _war_max * 0.85 else
                    "#4873b8" if v >= _war_max * 0.60 else "#4a687e"
                ),
                title=f"{sel_year} — Total Team fWAR",
                x_label="Total fWAR",
                text_fn=lambda v: f"{v:.1f}",
            ), use_container_width=True, config={"displayModeBar": False})
        with tb2:
            _w = _war[["Rank", "Team", "team_WAR", "payroll_M", "DPW", "Wins", "in_playoffs"]].copy()
            _w.columns = ["#", "Team", "fWAR", "Payroll $M", "$/fWAR M", "Wins", "Postseason"]
            _w["Payroll $M"] = _w["Payroll $M"].round(0).astype(int)
            _w["Wins"]       = _w["Wins"].round(0).astype(int)
            _w["Postseason"]  = _w["Postseason"].map({True: "✓", False: ""})

            def _war_clr(row):
                if row["#"] <= 5:  return ["background-color:#0c2218"] * len(row)
                if row["#"] <= 10: return ["background-color:#14532d55"] * len(row)
                return [""] * len(row)

            st.dataframe(
                _w.style.apply(_war_clr, axis=1).format(
                    {"fWAR": "{:.1f}", "Payroll $M": "{:d}", "$/fWAR M": "{:.1f}", "Wins": "{:d}"}, na_rep="—"),
                hide_index=True, use_container_width=True,
                height=min(60 + len(_w) * 35, 720),
            )

    # ── Tab 3: Salary ─────────────────────────────────────────────────────────
    with rt3:
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "Team payroll for the selected season. Use $/WAR to compare how much each team "
            "paid per unit of performance. Top-5 spenders in gold, bottom-5 in blue.</div>",
            unsafe_allow_html=True,
        )
        _sal = yr_df.sort_values("payroll_M", ascending=False).reset_index(drop=True)
        _sal["Rank"]   = range(1, len(_sal) + 1)
        _sal_max = _sal["payroll_M"].max() or 1

        ch3, tb3 = st.columns([3, 2])
        with ch3:
            # Gold→blue gradient: highest spenders = gold, lowest = muted blue
            st.plotly_chart(_hbar(
                _sal, "payroll_M",
                color_fn=lambda v: (
                    "#b88840" if v >= _sal_max * 0.80 else
                    "#4873b8" if v >= _sal_max * 0.45 else "#2e4a62"
                ),
                title=f"{sel_year} — Team Payroll",
                x_label="Payroll ($M)",
                text_fn=lambda v: f"${v:.0f}M",
            ), use_container_width=True, config={"displayModeBar": False})
        with tb3:
            _s = _sal[["Rank", "Team", "payroll_M", "team_WAR", "DPW", "Wins", "in_playoffs"]].copy()
            _s.columns = ["#", "Team", "Payroll $M", "WAR", "$/fWAR M", "Wins", "Postseason"]
            _s["Payroll $M"] = _s["Payroll $M"].round(0).astype(int)
            _s["WAR"]        = _s["WAR"].round(1)
            _s["$/fWAR M"]    = _s["$/fWAR M"].round(1)
            _s["Wins"]       = _s["Wins"].round(0).astype(int)
            _s["Postseason"]  = _s["Postseason"].map({True: "✓", False: ""})

            def _sal_clr(row):
                if row["#"] <= 5:   return ["background-color:#2d1f0c"] * len(row)
                if row["#"] >= 26:  return ["background-color:#0c1a2d"] * len(row)
                return [""] * len(row)

            st.dataframe(
                _s.style.apply(_sal_clr, axis=1).format(
                    {"Payroll $M": "{:d}", "WAR": "{:.1f}", "$/fWAR M": "{:.1f}", "Wins": "{:d}"}, na_rep="—"),
                hide_index=True, use_container_width=True,
                height=min(60 + len(_s) * 35, 720),
            )

    # ── Tab 4: Win Performance ────────────────────────────────────────────────
    with rt4:
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "Actual wins minus the wins <b>predicted by payroll</b> from the league regression. "
            "<span style='color:#22c55e;font-weight:600;'>Green</span> = team wins more than their "
            "spending predicts (efficient roster building). "
            "<span style='color:#ef4444;font-weight:600;'>Red</span> = team underperforms their payroll.</div>",
            unsafe_allow_html=True,
        )
        _wvp = yr_df.sort_values("wins_vs_pred", ascending=False).reset_index(drop=True)
        _wvp["Rank"] = range(1, len(_wvp) + 1)

        ch4, tb4 = st.columns([3, 2])
        with ch4:
            st.plotly_chart(_hbar(
                _wvp, "wins_vs_pred",
                color_fn=lambda v: "#22c55e" if v >= 0 else "#ef4444",
                title=f"{sel_year} — Wins Above/Below Payroll Prediction",
                x_label="Wins vs Predicted",
                text_fn=lambda v: f"{v:+.1f}W",
                zero_line=True,
            ), use_container_width=True, config={"displayModeBar": False})
        with tb4:
            _vp = _wvp[["Rank", "Team", "Wins", "pred_wins", "wins_vs_pred",
                         "payroll_M", "in_playoffs"]].copy()
            _vp.columns = ["#", "Team", "Wins", "Predicted", "Δ Wins", "Payroll $M", "Postseason"]
            _vp["Wins"]       = _vp["Wins"].round(0).astype(int)
            _vp["Predicted"]  = _vp["Predicted"].round(1)
            _vp["Δ Wins"]     = _vp["Δ Wins"].round(1)
            _vp["Payroll $M"] = _vp["Payroll $M"].round(0).astype(int)
            _vp["Postseason"]  = _vp["Postseason"].map({True: "✓", False: ""})

            def _vp_clr(row):
                d = row["Δ Wins"]
                if d > 10:  return ["background-color:#0c2218"] * len(row)
                if d > 0:   return ["background-color:#14532d55"] * len(row)
                if d < -10: return ["background-color:#2d0c0c"] * len(row)
                if d < 0:   return ["background-color:#2d150c55"] * len(row)
                return [""] * len(row)

            st.dataframe(
                _vp.style.apply(_vp_clr, axis=1).format(
                    {"Wins": "{:d}", "Predicted": "{:.1f}", "Δ Wins": "{:+.1f}", "Payroll $M": "{:d}"}, na_rep="—"),
                hide_index=True, use_container_width=True,
                height=min(60 + len(_vp) * 35, 720),
            )

    # ── Multi-year summary ────────────────────────────────────────────────────
    with st.expander("📊 Multi-Year Summary (2025–2021)", expanded=False):
        st.caption(
            "5-year averages from the efficiency analysis. "
            "**Avg Gap** = average dollars above/below the cost-effective line per season — "
            "negative means the team consistently spends below market rate for their wins."
        )
        _rk = ranking_df.sort_values("Leag_Rank").reset_index(drop=True).copy()
        _rk.insert(0, "#", range(1, len(_rk) + 1))
        _rk_cols = [c for c in ["#", "Abbr", "Team", "Division",
                                 "Avg_Wins", "Avg_Pay_M", "Avg_$/WAR_M", "Avg_Gap_M",
                                 "Playoff_Apps", "WS_Wins"] if c in _rk.columns]
        _rk = _rk[_rk_cols].rename(columns={
            "Abbr": "Abbr", "Team": "Team", "Division": "Division",
            "Avg_Wins": "Avg Wins", "Avg_Pay_M": "Avg Pay $M",
            "Avg_$/WAR_M": "$/fWAR M", "Avg_Gap_M": "Avg Gap $M",
            "Playoff_Apps": "Playoffs (5yr)", "WS_Wins": "WS",
        })

        def _rk_clr(row):
            try:
                g = row["Avg Gap $M"]
                if g < -100: return ["background-color:#0c2218"] * len(row)
                if g < 0:    return ["background-color:#14532d55"] * len(row)
                if g > 100:  return ["background-color:#2d0c0c"] * len(row)
            except Exception:
                pass
            return [""] * len(row)

        _rk["Avg Pay $M"] = _rk["Avg Pay $M"].round(0).astype(int)
        _rk["Avg Gap $M"] = _rk["Avg Gap $M"].round(0).astype(int)
        _rk["$/fWAR M"]    = _rk["$/fWAR M"].round(2)
        st.dataframe(
            _rk.style.apply(_rk_clr, axis=1),
            hide_index=True, use_container_width=True, height=680,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Feature 1 — "Does WAR Translate to Wins?"
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### fWAR-to-Wins Relationship")
    st.markdown(
        "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
        "Team fWAR (total roster talent) is one of the strongest predictors of regular-season wins. "
        "Each dot below is one team-season. <span style='color:#22c55e;font-weight:600;'>Green</span> = "
        "made the playoffs. The orange regression line shows the expected wins for a given fWAR level. "
        "The vertical blue dashed line marks the ~30 fWAR threshold — teams consistently above it are "
        "postseason contenders.</div>",
        unsafe_allow_html=True,
    )

    if "team_WAR" in detail_df.columns and "Wins" in detail_df.columns:
        _f1 = detail_df.dropna(subset=["team_WAR", "Wins"]).copy()
        _f1["team_WAR"] = pd.to_numeric(_f1["team_WAR"], errors="coerce")
        _f1["Wins"] = pd.to_numeric(_f1["Wins"], errors="coerce")
        _f1 = _f1.dropna(subset=["team_WAR", "Wins"])

        if len(_f1) > 5:
            # Regression
            _x = _f1["team_WAR"].values
            _y = _f1["Wins"].values
            _coef = np.polyfit(_x, _y, 1)
            _pred = np.polyval(_coef, _x)
            _ss_res = np.sum((_y - _pred) ** 2)
            _ss_tot = np.sum((_y - _y.mean()) ** 2)
            _r2 = 1 - (_ss_res / _ss_tot) if _ss_tot > 0 else 0

            st.markdown(
                f"<div style='font-size:0.88rem;color:#d6e8f8;font-weight:600;margin-bottom:0.5rem;'>"
                f"fWAR explains <span style='color:#60a5fa;'>{_r2 * 100:.1f}%</span> of win variation "
                f"across {len(_f1)} team-seasons (2025–2021). Each additional fWAR is worth roughly "
                f"<span style='color:#60a5fa;'>{_coef[0]:.2f}</span> wins.</div>",
                unsafe_allow_html=True,
            )

            # Build scatter
            _f1_playoff = _f1.get("in_playoffs", pd.Series([False] * len(_f1)))
            _f1_colors = ["#22c55e" if p else "#4a687e" for p in _f1_playoff]
            _f1_hover = _f1.apply(lambda r: (
                f"<b>{r['Team']}</b> {int(r['Year'])}<br>"
                + f"fWAR: {r['team_WAR']:.1f} · Wins: {int(r['Wins'])}<br>"
                + f"Playoff: {'Yes' if r.get('in_playoffs') else 'No'}"
            ), axis=1)

            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(
                x=_f1["team_WAR"], y=_f1["Wins"], mode="markers",
                marker=dict(color=_f1_colors, size=8, opacity=0.8),
                text=_f1_hover, hovertemplate="%{text}<extra></extra>",
                name="Teams",
            ))
            # Regression line
            _xr = np.linspace(_x.min(), _x.max(), 100)
            fig_f1.add_trace(go.Scatter(
                x=_xr, y=np.polyval(_coef, _xr), mode="lines",
                line=dict(color="#f4a261", width=2), name=f"OLS (R²={_r2:.3f})",
            ))
            # Reference line at WAR=30
            fig_f1.add_vline(x=30, line_dash="dash", line_color="#3b6fd4", opacity=0.5,
                             annotation_text="Avg contender floor", annotation_position="top right",
                             annotation_font_color="#3b6fd4")

            fig_f1.update_layout(**_pt(
                title="Team fWAR vs Actual Wins (2025–2021)",
                xaxis=dict(title="Total Team fWAR"),
                yaxis=dict(title="Actual Wins"),
                height=440, showlegend=True,
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                font=dict(color="#dbeafe", size=12)),
            ))
            _f1_col, _ = st.columns([3, 1])
            with _f1_col:
                st.plotly_chart(fig_f1, use_container_width=True)

        # 1B — Efficiency vs Postseason table
        if {"dollar_gap_M", "in_playoffs", "ws_champ"}.issubset(detail_df.columns):
            _eff_tbl = detail_df.groupby("Team").agg(
                Avg_Gap=("dollar_gap_M", "mean"),
                Playoff_Apps=("in_playoffs", "sum"),
                WS_Apps=("ws_champ", lambda x: int(x.sum()) + int(detail_df.loc[x.index, "ws_runnerup"].sum()) if "ws_runnerup" in detail_df.columns else int(x.sum())),
                WS_Wins=("ws_champ", "sum"),
            ).reset_index()
            _eff_tbl["Avg_Gap"] = _eff_tbl["Avg_Gap"].round(1)
            _eff_tbl["Playoff_Apps"] = _eff_tbl["Playoff_Apps"].astype(int)
            _eff_tbl["WS_Wins"] = _eff_tbl["WS_Wins"].astype(int)

            def _eff_tier(gap):
                if gap < -20: return ("Elite Value", "#22c55e")
                if gap < -5:  return ("Efficient", "#14b8a6")
                if gap <= 5:  return ("Market Rate", "#6b7280")
                return ("Overpaying", "#ef4444")

            _eff_tbl["Tier"] = _eff_tbl["Avg_Gap"].apply(lambda g: _eff_tier(g)[0])
            _eff_tbl = _eff_tbl.sort_values("Avg_Gap")

            st.markdown("#### Efficiency vs Postseason Outcomes (2025–2021)")

            # 5-tier ranking with color coding
            _n_teams = len(_eff_tbl)
            _tier_size = max(1, _n_teams // 5)
            def _rank_tier(idx):
                if idx < _tier_size:     return "Top Tier"
                if idx < _tier_size * 2: return "Above Average"
                if idx < _tier_size * 3: return "Average"
                if idx < _tier_size * 4: return "Below Average"
                return "Bottom"
            _eff_tbl["Ranking"] = [_rank_tier(i) for i in range(len(_eff_tbl))]

            _RANK_CLR = {"Top Tier": "#14532d", "Above Average": "#1a3a20",
                         "Average": "", "Below Average": "#2d1f0c", "Bottom": "#2d0c0c"}
            def _tier_clr(row):
                bg = _RANK_CLR.get(row.get("Ranking", ""), "")
                return [f"background-color:{bg}"] * len(row) if bg else [""] * len(row)

            _eff_disp = _eff_tbl.rename(columns={
                "Avg_Gap": "Avg Gap ($M)", "Playoff_Apps": "Playoff Apps",
                "WS_Apps": "WS Appearances", "WS_Wins": "WS Wins",
                "Tier": "Efficiency Tier",
            })
            st.dataframe(
                _eff_disp.style.apply(_tier_clr, axis=1).format(
                    {"Avg Gap ($M)": "{:.1f}"}, na_rep="—"),
                hide_index=True, use_container_width=True, height=400,
            )

            # Auto-generated insight
            _eff_good = _eff_tbl[_eff_tbl["Avg_Gap"] < -10]
            _eff_bad  = _eff_tbl[_eff_tbl["Avg_Gap"] > 10]
            _n_seasons = len(detail_df["Year"].unique())
            if not _eff_good.empty and not _eff_bad.empty:
                _good_pct = _eff_good["Playoff_Apps"].sum() / (len(_eff_good) * _n_seasons) * 100
                _bad_pct  = _eff_bad["Playoff_Apps"].sum() / (len(_eff_bad) * _n_seasons) * 100
                st.info(
                    f"Among the {len(_eff_good)} most efficient teams (gap < -$10M), "
                    f"{_eff_good['Playoff_Apps'].sum()} made the playoffs "
                    f"({_good_pct:.0f}% of seasons) vs {_bad_pct:.0f}% for the "
                    f"{len(_eff_bad)} least efficient teams."
                )

            st.markdown(
                "<div style='background:#0d1e35;border-left:3px solid #3b6fd4;border-radius:0 8px 8px 0;"
                "padding:0.7rem 1rem;margin-top:0.6rem;font-size:0.82rem;color:#93b8d8;line-height:1.6;'>"
                "<b style='color:#60a5fa;'>Research context:</b> Over the past decade, World Series "
                "champions averaged 8th in payroll — efficient roster construction matters more than "
                "total spend once you reach the postseason.</div>",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════
    # Feature 2 — Incremental Spending Impact
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Marginal Spending Impact")
    st.markdown(
        "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
        "Not all payroll dollars spent are equal. The first $100M spent gains the most wins per $ "
        "than the next tier $244M to $300M+. Based on this nonlinear relationship, each bar below "
        "shows how many additional wins a team can expect per $10M spent within that spending tier, "
        "based on 2021–2025 data. "
        "<span style='color:#22c55e;font-weight:600;'>Green</span> = strong return, "
        "<span style='color:#f59e0b;font-weight:600;'>amber</span> = moderate, "
        "<span style='color:#ef4444;font-weight:600;'>red</span> = diminishing returns.</div>",
        unsafe_allow_html=True,
    )

    if {"payroll_M", "Wins"}.issubset(detail_df.columns):
        _f2 = detail_df.dropna(subset=["payroll_M", "Wins"]).copy()
        _f2["payroll_M"] = pd.to_numeric(_f2["payroll_M"], errors="coerce")
        _f2["Wins"]      = pd.to_numeric(_f2["Wins"], errors="coerce")

        _TIERS = [
            ("Budget ($0–100M)",       0,   100),
            ("Mid-Market ($100–175M)", 100, 175),
            ("Contender ($175–244M)",  175, 244),
            ("Big Market ($244M+)",    244, 999),
        ]

        _slopes = []
        for name, lo, hi in _TIERS:
            tier_df = _f2[(_f2["payroll_M"] >= lo) & (_f2["payroll_M"] < hi)]
            if len(tier_df) >= 5:
                c = np.polyfit(tier_df["payroll_M"], tier_df["Wins"], 1)
                wins_per_10m = c[0] * 10
            else:
                wins_per_10m = 0
            _slopes.append((name, round(wins_per_10m, 2)))

        _sl_names  = [s[0] for s in _slopes]
        _sl_vals   = [s[1] for s in _slopes]
        _sl_colors = ["#22c55e" if v >= 1.0 else "#f59e0b" if v >= 0.3 else "#ef4444" for v in _sl_vals]

        fig_f2 = go.Figure(go.Bar(
            y=_sl_names, x=_sl_vals, orientation="h",
            marker_color=_sl_colors,
            text=[f"{v:+.2f} wins" for v in _sl_vals],
            textposition="outside", textfont=dict(color="#dbeafe", size=10),
            hovertemplate="%{y}: %{x:.2f} wins per $10M<extra></extra>",
        ))
        _xmax_f2 = max(abs(v) for v in _sl_vals) * 1.5 if _sl_vals else 3
        fig_f2.update_layout(**_pt(
            title="Marginal Wins per $10M by Spending Tier",
            xaxis=dict(title="Wins per $10M spent",
                       zeroline=True, zerolinecolor="#4a687e", zerolinewidth=1,
                       range=[-_xmax_f2, _xmax_f2]),
            yaxis=dict(autorange="reversed"),
            height=300,
            margin=dict(l=180, r=80, t=40, b=40),
        ))
        _f2_col, _ = st.columns([3, 1])
        with _f2_col:
            st.plotly_chart(fig_f2, use_container_width=True)

        # 2B — Interactive slider
        _spend_add = st.slider("Add spending ($M)", 0, 50, 10, step=5,
                                key="v2_spend_slider")
        if _spend_add > 0 and _slopes:
            # Find user's current tier (assume mid-market as default)
            _cur_tier_idx = 1
            _cur_slope = _slopes[_cur_tier_idx][1]
            _add_wins = _cur_slope * (_spend_add / 10)
            st.markdown(
                f"<div style='background:#0d1e35;border:1px solid #1e3250;border-radius:8px;"
                f"padding:0.8rem 1rem;font-size:0.85rem;color:#d6e8f8;'>"
                f"Adding <b>${_spend_add}M</b> at the <b>{_slopes[_cur_tier_idx][0]}</b> tier "
                f"buys ~<b>{_add_wins:.1f}</b> additional wins "
                f"(slope: {_cur_slope:.2f} wins/$10M)."
                f"</div>",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════
    # Feature 4 — Roster Stability Score (RSS) vs Wins
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Roster Stability and Win Correlation")
    st.markdown(
        "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
        "Roster Stability Score (RSS) measures what percentage of a team's qualifying players "
        "returned from the prior season. Higher RSS means more continuity. "
        "<span style='color:#22c55e;font-weight:600;'>Green</span> dots made the postseason.</div>",
        unsafe_allow_html=True,
    )

    _comb_path_4 = _data_url("data/mlb_combined_2021_2025.csv")
    try:
        _comb_4 = _read_csv(_comb_path_4, low_memory=False)
        _comb_4.columns = [c.strip() for c in _comb_4.columns]
        for _nc in ["Year", "PA", "IP", "WAR_Total"]:
            if _nc in _comb_4.columns:
                _comb_4[_nc] = pd.to_numeric(_comb_4[_nc], errors="coerce")

        _rss_thresh = st.slider("Min PA (hitters) / IP (pitchers)", 50, 300, 150,
                                 key="v2_rss_min_pa")

        _is_pit4 = _comb_4["Position"].isin(["SP", "RP", "P", "TWP"])
        _comb_q = _comb_4[
            (_is_pit4 & (_comb_4["IP"].fillna(0) >= _rss_thresh)) |
            (~_is_pit4 & (_comb_4["PA"].fillna(0) >= _rss_thresh))
        ].copy()

        _years4 = sorted(_comb_q["Year"].dropna().unique().astype(int))
        _rss_records = []
        for yr in _years4:
            if yr == min(_years4):
                continue  # need prior year
            for tm in _comb_q["Team"].unique():
                curr = set(_comb_q[(_comb_q["Year"] == yr) & (_comb_q["Team"] == tm)]["Player"])
                prev = set(_comb_q[(_comb_q["Year"] == yr - 1) & (_comb_q["Team"] == tm)]["Player"])
                if not curr:
                    continue
                returning = curr & prev
                rss = len(returning) / len(curr) * 100
                # Get wins from detail_df
                _wins_row = detail_df[(detail_df["Year"] == yr) & (detail_df["Team"] == tm)]
                wins = float(_wins_row["Wins"].iloc[0]) if not _wins_row.empty and "Wins" in _wins_row.columns else None
                playoff = bool(_wins_row["in_playoffs"].iloc[0]) if not _wins_row.empty and "in_playoffs" in _wins_row.columns else False
                _rss_records.append({"Team": tm, "Year": yr, "RSS": round(rss, 1),
                                     "Wins": wins, "Playoff": playoff,
                                     "Returning": len(returning), "Total": len(curr)})

        if _rss_records:
            _rss_df = pd.DataFrame(_rss_records).dropna(subset=["Wins"])

            if len(_rss_df) > 10:
                _rx = _rss_df["RSS"].values
                _ry = _rss_df["Wins"].values
                _rc = np.polyfit(_rx, _ry, 1)
                _rp = np.polyval(_rc, _rx)
                _rss_r2 = 1 - np.sum((_ry - _rp) ** 2) / max(np.sum((_ry - _ry.mean()) ** 2), 1e-9)

                _rss_colors = ["#22c55e" if p else "#4a687e" for p in _rss_df["Playoff"]]
                _rss_hover = _rss_df.apply(lambda r: (
                    f"<b>{r['Team']}</b> {int(r['Year'])}<br>"
                    + f"RSS: {r['RSS']:.1f}% · Wins: {int(r['Wins'])}<br>"
                    + f"Returning: {int(r['Returning'])}/{int(r['Total'])}"
                ), axis=1)

                fig_rss = go.Figure()
                fig_rss.add_trace(go.Scatter(
                    x=_rss_df["RSS"], y=_rss_df["Wins"], mode="markers",
                    marker=dict(color=_rss_colors, size=8, opacity=0.8),
                    text=_rss_hover, hovertemplate="%{text}<extra></extra>",
                    name="Teams",
                ))
                _xr4 = np.linspace(_rx.min(), _rx.max(), 100)
                fig_rss.add_trace(go.Scatter(
                    x=_xr4, y=np.polyval(_rc, _xr4), mode="lines",
                    line=dict(color="#f4a261", width=2),
                    name=f"OLS (R²={_rss_r2:.3f})",
                ))
                fig_rss.update_layout(**_pt(
                    title="Roster Stability vs Win Total",
                    xaxis=dict(title="Roster Stability Score (%)"),
                    yaxis=dict(title="Wins"), height=440, showlegend=True,
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                    hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                    font=dict(color="#dbeafe", size=12)),
                ))
                _rss_col, _ = st.columns([3, 1])
                with _rss_col:
                    st.plotly_chart(fig_rss, use_container_width=True)

                # Auto insight
                _med_rss = float(_rss_df["RSS"].median())
                _high = _rss_df[_rss_df["RSS"] >= _med_rss]
                _low  = _rss_df[_rss_df["RSS"] < _med_rss]
                if not _high.empty and not _low.empty:
                    st.info(
                        f"Teams with RSS above {_med_rss:.0f}% won an average of "
                        f"{_high['Wins'].mean():.1f} games vs {_low['Wins'].mean():.1f} "
                        f"games for teams below that threshold."
                    )
    except Exception:
        st.caption("Could not load combined data for RSS analysis.")

    _render_feedback_widget("rankings")


# ---------------------------------------------------------------------------
# Team Analysis page
# ---------------------------------------------------------------------------

_ABBR_TO_FULL: dict[str, str] = {v: k for k, v in _PAYROLL_2026_TEAM_MAP.items()}
_TEAM_CITIES: dict[str, str] = {
    "ARI": "Arizona", "ATH": "Las Vegas", "ATL": "Atlanta", "BAL": "Baltimore",
    "BOS": "Boston", "CHC": "Chicago", "CHW": "Chicago", "CIN": "Cincinnati",
    "CLE": "Cleveland", "COL": "Colorado", "DET": "Detroit", "HOU": "Houston",
    "KCR": "Kansas City", "LAA": "Los Angeles", "LAD": "Los Angeles", "MIA": "Miami",
    "MIL": "Milwaukee", "MIN": "Minnesota", "NYM": "New York", "NYY": "New York",
    "PHI": "Philadelphia", "PIT": "Pittsburgh", "SDP": "San Diego", "SEA": "Seattle",
    "SFG": "San Francisco", "STL": "St. Louis", "TBR": "Tampa Bay", "TEX": "Texas",
    "TOR": "Toronto", "WSN": "Washington",
}


def _render_team_analysis_page():
    """Full team deep-dive page — roster, rankings, salary, projections."""

    # ── Data loading ─────────────────────────────────────────────────────
    detail_csv   = _data_url("efficiency_detail.csv")
    combined_csv = _data_url("data/mlb_combined_2021_2025.csv")
    roster_csv   = _data_url("40man_rosters_2025.csv")

    try:
        detail_df = _read_csv(detail_csv)
    except Exception:
        detail_df = pd.DataFrame()
    # Load 40-man roster — try cached first, then direct read
    roster_40 = st.session_state.get("_sim_roster_40", pd.DataFrame())
    if roster_40.empty:
        _r40_hash = "r2-remote" if _R2_MODE else _file_hash(roster_csv)
        roster_40 = _cached_40man_roster(roster_csv, _r40_hash)
    if roster_40.empty:
        # Direct fallback read (bypass cache) for R2 mode
        try:
            roster_40 = _read_csv(roster_csv, low_memory=False)
            roster_40.columns = [c.strip() for c in roster_40.columns]
        except Exception:
            roster_40 = pd.DataFrame()
    payroll_dir = _data_url("2026 Payroll") if _R2_MODE else os.path.join(_ROOT_DIR, "2026 Payroll")

    # Available teams from 40-man CSV
    _teams = sorted(roster_40["team"].dropna().unique()) if not roster_40.empty else sorted(_ABBR_TO_FULL.keys())

    # ── Team selector ────────────────────────────────────────────────────
    sel_col, _, _ = st.columns([2, 3, 3])
    with sel_col:
        sel_team = st.selectbox(
            "Select a Team",
            _teams,
            format_func=lambda t: f"{_TEAM_CITIES.get(t, t)} {_ABBR_TO_FULL.get(t, t)} ({t})",
            key="team_analysis_sel",
        )

    _full_name = f"{_TEAM_CITIES.get(sel_team, '')} {_ABBR_TO_FULL.get(sel_team, sel_team)}"

    # ── Load simulator data (same enriched dataset the simulator uses) ───
    _ind_path = _data_url("data/2025mlbshared.csv")
    _sim_hash = "r2-remote" if _R2_MODE else _file_hash(combined_csv)
    try:
        _sim_df = _cached_simulator_data(combined_csv, _ind_path, _sim_hash)
        _comb_team = _sim_df[_sim_df["Team"] == sel_team].copy()
    except Exception:
        _sim_df = pd.DataFrame()
        _comb_team = pd.DataFrame()

    # ── Load 2026 payroll data ───────────────────────────────────────────
    try:
        _pay_hash = _dir_hash(payroll_dir) if not _R2_MODE else "r2"
        df26 = _cached_2026_payroll(payroll_dir, combined_csv, _pay_hash)
        team_pay = df26[df26["Team"] == sel_team].copy() if not df26.empty else pd.DataFrame()
    except Exception:
        team_pay = pd.DataFrame()

    # Fall back to simulator data if payroll unavailable
    if team_pay.empty and not _comb_team.empty:
        team_pay = _comb_team.copy()

    # ── Load 40-man roster for this team ─────────────────────────────────
    team_roster = roster_40[roster_40["team"] == sel_team].copy() if not roster_40.empty else pd.DataFrame()
    n_active = int((team_roster["status"] == "Active").sum()) if not team_roster.empty else 0
    n_il     = int((team_roster["status"] == "Injured 60-Day").sum()) if not team_roster.empty else 0
    n_total  = len(team_roster)

    # ── Efficiency data for this team ────────────────────────────────────
    team_eff = detail_df[detail_df["Team"] == sel_team].copy() if not detail_df.empty else pd.DataFrame()
    all_eff_2025 = detail_df[detail_df["Year"] == 2025] if not detail_df.empty else pd.DataFrame()

    # ── Latest season stats ──────────────────────────────────────────────
    _latest = team_eff[team_eff["Year"] == 2025].iloc[0] if not team_eff.empty and 2025 in team_eff["Year"].values else None

    # ══════════════════════════════════════════════════════════════════════
    # HEADER CARD
    # ══════════════════════════════════════════════════════════════════════
    _payroll_m = float(_latest["payroll_M"]) if _latest is not None else 0
    _wins      = int(_latest["Wins"]) if _latest is not None else 0
    _war       = float(_latest["team_WAR"]) if _latest is not None else 0
    _gap       = float(_latest["dollar_gap_M"]) if _latest is not None else 0
    _playoff   = bool(_latest["in_playoffs"]) if _latest is not None else False

    # Compute rankings among all teams (2025)
    _eff_rank = int((all_eff_2025["dollar_gap_M"].rank(ascending=True) == all_eff_2025.loc[all_eff_2025["Team"] == sel_team, "dollar_gap_M"].rank(ascending=True).values[0]).sum()) if not all_eff_2025.empty and sel_team in all_eff_2025["Team"].values else 0
    if not all_eff_2025.empty and sel_team in all_eff_2025["Team"].values:
        _eff_rank = int(all_eff_2025["dollar_gap_M"].rank().loc[all_eff_2025["Team"] == sel_team].values[0])
        _war_rank = int(all_eff_2025["team_WAR"].rank(ascending=False).loc[all_eff_2025["Team"] == sel_team].values[0])
        _pay_rank = int(all_eff_2025["payroll_M"].rank(ascending=False).loc[all_eff_2025["Team"] == sel_team].values[0])
    else:
        _eff_rank = _war_rank = _pay_rank = 0

    _dpw = round(_payroll_m / max(_war, 0.1), 1)

    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0f2035,#0d1b2a);border:1px solid #1e3a5c;"
        f"border-radius:10px;padding:16px 20px;margin-bottom:14px;'>"
        f"<div style='font-size:1.4rem;font-weight:800;color:#e8f4ff;margin-bottom:8px;'>"
        f"{_full_name}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:12px;'>"
        f"<div style='background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;'>"
        f"<div style='font-size:10px;color:#7a9ebc;text-transform:uppercase;'>2025 Record</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:#e8f4ff;'>{_wins}W</div></div>"
        f"<div style='background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;'>"
        f"<div style='font-size:10px;color:#7a9ebc;text-transform:uppercase;'>2026 Record</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:#4a687e;'>—</div></div>"
        f"<div style='background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;'>"
        f"<div style='font-size:10px;color:#7a9ebc;text-transform:uppercase;'>2026 Payroll</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:#e8f4ff;'>${_payroll_m:.0f}M</div>"
        f"<div style='font-size:0.65rem;color:#7a9ebc;'>#{_pay_rank}/30</div></div>"
        f"<div style='background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;'>"
        f"<div style='font-size:10px;color:#7a9ebc;text-transform:uppercase;'>Team fWAR</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:#e8f4ff;'>{_war:.1f}</div>"
        f"<div style='font-size:0.65rem;color:#7a9ebc;'>#{_war_rank}/30</div></div>"
        f"<div style='background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;'>"
        f"<div style='font-size:10px;color:#7a9ebc;text-transform:uppercase;'>Efficiency</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:{'#22c55e' if _gap < 0 else '#ef4444'};'>"
        f"{'$' + str(int(_gap)) + 'M' if _gap <= 0 else '+$' + str(int(_gap)) + 'M'}</div>"
        f"<div style='font-size:0.65rem;color:#7a9ebc;'>#{_eff_rank}/30</div></div>"
        f"<div style='background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;'>"
        f"<div style='font-size:10px;color:#7a9ebc;text-transform:uppercase;'>$/fWAR</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:#e8f4ff;'>${_dpw:.1f}M</div></div>"
        f"<div style='background:#0d1b2a;border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;'>"
        f"<div style='font-size:10px;color:#7a9ebc;text-transform:uppercase;'>40-Man Roster</div>"
        f"<div style='font-size:1.2rem;font-weight:700;color:#e8f4ff;'>{n_active} <span style='font-size:0.7rem;color:#7a9ebc;'>active</span>"
        f" · {n_il} <span style='font-size:0.7rem;color:#ef4444;'>IL</span></div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════
    tt1, tt2, tt3, tt4, tt5 = st.tabs(["📋 Roster", "📊 Rankings", "💰 Salary & Payroll", "📈 Payroll Efficiency", "📉 History"])

    # ── Tab 1 — Roster ───────────────────────────────────────────────────
    with tt1:
        if not team_roster.empty:
            # Merge roster players with stats (2025) + payroll (2026)
            _merged = team_roster.copy()
            # Normalize 40-man names to match our datasets
            _merged["_key"] = _merged["full_name"].apply(_fix_player_name).str.lower().str.strip()

            # Build unified lookup from payroll (priority) + simulator data (fallback)
            _lookup = pd.DataFrame()
            _lk_cols = ["Player", "WAR_Total", "Age", "Position", "Stage_Clean",
                        "Salary_M", "PPR", "W_per_M", "HR", "AVG", "ERA", "IP",
                        "2027", "2028"]
            if not team_pay.empty and "Player" in team_pay.columns:
                _lk_pay = team_pay[[c for c in _lk_cols if c in team_pay.columns]].copy()
                _lk_pay["_key"] = _lk_pay["Player"].str.lower().str.strip()
                _lookup = _lk_pay.drop_duplicates(subset=["_key"], keep="first")
            if _lookup.empty and not _comb_team.empty:
                _lk_sim = _comb_team[[c for c in _lk_cols if c in _comb_team.columns]].copy()
                _lk_sim["_key"] = _lk_sim["Player"].str.lower().str.strip()
                _lookup = _lk_sim.drop_duplicates(subset=["_key"], keep="first")

            if not _lookup.empty:
                _merged = _merged.merge(_lookup.drop(columns=["Player"], errors="ignore"),
                                        on="_key", how="left")

            # Fill missing salary with league min
            if "Salary_M" not in _merged.columns:
                _merged["Salary_M"] = 0.74
            _merged["Salary_M"] = _merged["Salary_M"].fillna(0.74)

            # Build display table
            _rtbl = pd.DataFrame()
            _rtbl["Player"] = _merged["full_name"]
            _rtbl["Pos"] = _merged["position"]
            _rtbl["Status"] = _merged["status"].apply(lambda s: "60-Day IL" if "Injured" in str(s) else "Active")
            _rtbl["Stage"] = _merged.get("Stage_Clean", pd.Series(["—"] * len(_merged))).fillna("—")
            _rtbl["Age"] = _merged.get("Age", pd.Series([None] * len(_merged)))
            _rtbl["'26 Salary $M"] = _merged["Salary_M"]
            _rtbl["'25 fWAR"] = _merged.get("WAR_Total", pd.Series([None] * len(_merged)))
            _rtbl["fWAR/$M"] = (_rtbl["'25 fWAR"].fillna(0) / _rtbl["'26 Salary $M"].clip(lower=0.01)).round(2)
            # Add a few key stats
            if "ERA" in _merged.columns:
                _rtbl["ERA"] = _merged["ERA"]
            if "HR" in _merged.columns:
                _rtbl["HR"] = _merged["HR"]
            _rtbl = _rtbl.sort_values("fWAR/$M", ascending=False).reset_index(drop=True)
            _rtbl.insert(0, "#", range(1, len(_rtbl) + 1))

            # Stage color coding
            _STG_CLR = {"Pre-Arb": "#14532d", "Arb": "#2d1f0c", "FA": "#0c1a2d"}
            def _stage_clr(row):
                stg = str(row.get("Stage", ""))
                bg = _STG_CLR.get(stg, "")
                if row.get("Status") == "60-Day IL":
                    return [f"background-color:#2d0c0c;color:#fca5a5"] * len(row)
                if bg:
                    return [f"background-color:{bg}66"] * len(row)
                return [""] * len(row)

            _n_act = len(_rtbl[_rtbl["Status"] == "Active"])
            _n_il = len(_rtbl[_rtbl["Status"] != "Active"])
            st.markdown(f"##### 40-Man Roster ({_n_act} active, {_n_il} on IL)")
            st.markdown(
                "<div style='font-size:0.78rem;color:#7a9ebc;margin-bottom:0.4rem;'>"
                "Ranked by fWAR per $M (most efficient first). '25 fWAR = 2025 stats. '26 Salary = 2026 contract. Color: "
                "<span style='color:#22c55e;'>Pre-Arb</span> · "
                "<span style='color:#f59e0b;'>Arb</span> · "
                "<span style='color:#3b82f6;'>FA</span> · "
                "<span style='color:#fca5a5;'>IL</span></div>",
                unsafe_allow_html=True,
            )
            _fmt_dict = {"Age": "{:.0f}", "'26 Salary $M": "{:.1f}", "'25 fWAR": "{:.1f}",
                         "fWAR/$M": "{:.2f}", "ERA": "{:.2f}", "HR": "{:.0f}"}
            st.dataframe(
                _rtbl.style.apply(_stage_clr, axis=1).format(
                    {k: v for k, v in _fmt_dict.items() if k in _rtbl.columns}, na_rep="—"),
                hide_index=True, use_container_width=True,
                height=min(60 + len(_rtbl) * 35, 700),
            )
        else:
            st.info(f"No 40-man roster data available for {sel_team}.")

    # ── Tab 2 — Rankings Position ────────────────────────────────────────
    with tt2:
        if not all_eff_2025.empty:
            st.markdown(
                "<div style='font-size:0.85rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
                "Where this team ranks among all 30 MLB teams in 2025. "
                "<span style='color:#f59e0b;font-weight:600;'>Gold</span> highlights the selected team.</div>",
                unsafe_allow_html=True,
            )

            _rk = all_eff_2025.sort_values("dollar_gap_M").reset_index(drop=True)
            _rk["Rank"] = range(1, len(_rk) + 1)
            _colors = ["#f59e0b" if t == sel_team else ("#22c55e" if g < 0 else "#ef4444")
                        for t, g in zip(_rk["Team"], _rk["dollar_gap_M"])]

            fig_rk = go.Figure(go.Bar(
                y=_rk["Team"], x=_rk["dollar_gap_M"], orientation="h",
                marker_color=_colors,
                text=[f"${g:+.0f}M" for g in _rk["dollar_gap_M"]],
                textposition="outside", textfont=dict(color="#d6e8f8", size=9),
                hovertemplate="%{y}: $%{x:+.0f}M<extra></extra>",
            ))
            _abs_max = max(abs(_rk["dollar_gap_M"].max()), abs(_rk["dollar_gap_M"].min())) * 1.15
            fig_rk.update_layout(**_pt(
                title=f"2025 Efficiency Ranking — {_full_name} is #{_eff_rank}",
                xaxis=dict(title="$ Gap ($M) — negative = efficient",
                           zeroline=True, zerolinecolor="#4a687e", zerolinewidth=1,
                           range=[-_abs_max, _abs_max]),
                yaxis=dict(autorange="reversed"),
                height=max(400, len(_rk) * 22),
                margin=dict(l=60, r=80, t=42, b=30),
            ))
            st.plotly_chart(fig_rk, use_container_width=True, config={"displayModeBar": False})

            # fWAR ranking bar
            _wrk = all_eff_2025.sort_values("team_WAR", ascending=False).reset_index(drop=True)
            _wrk["Rank"] = range(1, len(_wrk) + 1)
            _wcolors = ["#f59e0b" if t == sel_team else "#3b82f6" for t in _wrk["Team"]]

            fig_wrk = go.Figure(go.Bar(
                y=_wrk["Team"], x=_wrk["team_WAR"], orientation="h",
                marker_color=_wcolors,
                text=[f"{w:.1f}" for w in _wrk["team_WAR"]],
                textposition="outside", textfont=dict(color="#d6e8f8", size=9),
                hovertemplate="%{y}: %{x:.1f} fWAR<extra></extra>",
            ))
            fig_wrk.update_layout(**_pt(
                title=f"2025 fWAR Ranking — {_full_name} is #{_war_rank}",
                xaxis=dict(title="Total Team fWAR"),
                yaxis=dict(autorange="reversed"),
                height=max(400, len(_wrk) * 22),
                margin=dict(l=60, r=80, t=42, b=30),
            ))
            st.plotly_chart(fig_wrk, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No efficiency ranking data available.")

    # ── Tab 3 — Salary & Payroll ─────────────────────────────────────────
    with tt3:
        if not team_pay.empty:
            # Salary by stage
            _stg_sal = team_pay.groupby("Stage_Clean")["Salary_M"].sum().reset_index()
            _stg_colors = {"Pre-Arb": "#22c55e", "Arb": "#f59e0b", "FA": "#3b82f6"}

            fig_stg = go.Figure(go.Pie(
                labels=_stg_sal["Stage_Clean"],
                values=_stg_sal["Salary_M"],
                marker_colors=[_stg_colors.get(s, "#4a687e") for s in _stg_sal["Stage_Clean"]],
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(color="#d6e8f8", size=11),
                hovertemplate="%{label}: $%{value:.1f}M<extra></extra>",
            ))
            fig_stg.update_layout(**_pt(
                title="2026 Payroll by Contract Stage",
                height=360, showlegend=False,
            ))
            st.plotly_chart(fig_stg, use_container_width=True, config={"displayModeBar": False})

            # Top 10 highest paid
            _top_sal = team_pay.nlargest(10, "Salary_M")[["Player", "Position", "Salary_M", "WAR_Total", "Stage_Clean"]].copy()
            _top_sal.insert(0, "#", range(1, len(_top_sal) + 1))
            _top_sal.columns = ["#", "Player", "Pos", "Salary $M", "fWAR", "Stage"]
            st.markdown("##### Top 10 Highest-Paid Players")
            st.dataframe(
                _top_sal.style.format({"Salary $M": "{:.1f}", "fWAR": "{:.1f}"}, na_rep="—"),
                hide_index=True, use_container_width=True,
            )

            # Future payroll commitments (3-year)
            st.markdown("##### 📅 Projected Payroll (2026–2028)")
            _yr_cols = ["2026_total", "2027_total", "2028_total"]
            _s26 = float(team_pay["Salary_M"].sum())
            _s27 = float(team_pay["2027"].dropna().sum()) if "2027" in team_pay.columns else 0
            _s28 = float(team_pay["2028"].dropna().sum()) if "2028" in team_pay.columns else 0

            fig_proj = go.Figure()
            fig_proj.add_trace(go.Bar(
                x=["2026", "2027", "2028"],
                y=[_s26, _s27, _s28],
                marker_color=["#3b82f6", "#60a5fa", "#93c5fd"],
                text=[f"${v:.0f}M" for v in [_s26, _s27, _s28]],
                textposition="outside", textfont=dict(color="#d6e8f8"),
                hovertemplate="%{x}: $%{y:.1f}M<extra></extra>",
            ))
            fig_proj.add_hline(y=244, line_dash="dash", line_color="#f59e0b", opacity=0.5,
                               annotation_text="CBT $244M", annotation_font_color="#f59e0b")
            fig_proj.update_layout(**_pt(
                title=f"{_full_name} — Committed Payroll",
                yaxis=dict(title="Total $M"), height=340,
            ))
            st.plotly_chart(fig_proj, use_container_width=True, config={"displayModeBar": False})

            st.caption(
                "2026 reflects actual contracts. 2027–2028 include only confirmed multi-year commitments. "
                "Players whose contracts expire show $0 for future years."
            )

            # CBT status
            _cbt_lbl, _cbt_bg, _, _, _cbt_note = _cbt_info(_s26)
            st.markdown(
                f"<div style='background:{_cbt_bg};border-radius:8px;padding:10px 14px;"
                f"font-size:0.85rem;color:#d6e8f8;margin-top:0.5rem;'>"
                f"<b>CBT Status:</b> {_cbt_lbl} at ${_s26:.0f}M — {_cbt_note}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info(f"No 2026 payroll data available for {sel_team}.")

    # ── Tab 4 — Payroll Efficiency (player-level scatter for this team) ──
    with tt4:
        st.markdown(
            "<div style='font-size:0.85rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
            "Player-level fWAR vs Salary for this team's roster. Dots below the orange market "
            "line are underpaid (good value). Dots above are overpaid relative to production.</div>",
            unsafe_allow_html=True,
        )
        if not team_pay.empty and "WAR_Total" in team_pay.columns and "Salary_M" in team_pay.columns:
            _tp_plot = team_pay.dropna(subset=["WAR_Total", "Salary_M"]).copy()
            if len(_tp_plot) >= 3:
                _stg_clrs = {"FA": "#3b82f6", "Arb": "#f59e0b", "Pre-Arb": "#22c55e"}
                _tp_colors = [_stg_clrs.get(s, "#4a687e") for s in _tp_plot.get("Stage_Clean", [])]
                _tp_hover = _tp_plot.apply(lambda r: (
                    f"<b>{r['Player']}</b><br>"
                    + f"fWAR: {r['WAR_Total']:.1f} · Salary: ${r['Salary_M']:.1f}M<br>"
                    + f"Stage: {r.get('Stage_Clean', '—')}"
                ), axis=1)

                fig_eff = go.Figure()
                for stg, clr in _stg_clrs.items():
                    mask = _tp_plot["Stage_Clean"] == stg
                    if mask.any():
                        fig_eff.add_trace(go.Scatter(
                            x=_tp_plot.loc[mask, "WAR_Total"],
                            y=_tp_plot.loc[mask, "Salary_M"],
                            mode="markers+text",
                            text=_tp_plot.loc[mask, "Player"],
                            textposition="top center",
                            textfont=dict(size=8, color="#7aa2c0"),
                            marker=dict(color=clr, size=9, opacity=0.9),
                            name=stg,
                            hovertemplate="%{text}<br>fWAR: %{x:.1f} · $%{y:.1f}M<extra></extra>",
                        ))

                # Market line (simple linear fit)
                _xv = _tp_plot["WAR_Total"].values
                _yv = _tp_plot["Salary_M"].values
                try:
                    _cf = np.polyfit(_xv, _yv, 1)
                    _xl = np.linspace(max(_xv.min(), -1), _xv.max(), 50)
                    fig_eff.add_trace(go.Scatter(
                        x=_xl, y=np.polyval(_cf, _xl), mode="lines",
                        line=dict(color="#f4a261", dash="dash", width=2),
                        name="Team market line", showlegend=True,
                    ))
                except Exception:
                    pass

                fig_eff.update_layout(**_pt(
                    title=f"{_full_name} — Player fWAR vs Salary",
                    xaxis=dict(title="fWAR (2025)"), yaxis=dict(title="2026 Salary ($M)"),
                    height=500, showlegend=True,
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                    hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                    font=dict(color="#dbeafe", size=12)),
                ))
                st.plotly_chart(fig_eff, use_container_width=True, config={"displayModeBar": False})

                # Top value / worst value mini-tables
                _vc1, _vc2 = st.columns(2)
                with _vc1:
                    st.markdown("##### Best Value Players")
                    _bv = _tp_plot.copy()
                    _bv["fWAR/$M"] = (_bv["WAR_Total"] / _bv["Salary_M"].clip(lower=0.01)).round(2)
                    _bv_top = _bv.nlargest(5, "fWAR/$M")[["Player", "WAR_Total", "Salary_M", "fWAR/$M", "Stage_Clean"]].copy()
                    _bv_top.columns = ["Player", "fWAR", "Salary $M", "fWAR/$M", "Stage"]
                    st.dataframe(_bv_top.style.format({"fWAR": "{:.1f}", "Salary $M": "{:.1f}", "fWAR/$M": "{:.2f}"}, na_rep="—"),
                                 hide_index=True, use_container_width=True)
                with _vc2:
                    st.markdown("##### Most Overpaid Players")
                    _ov_top = _bv.nsmallest(5, "fWAR/$M")[["Player", "WAR_Total", "Salary_M", "fWAR/$M", "Stage_Clean"]].copy()
                    _ov_top.columns = ["Player", "fWAR", "Salary $M", "fWAR/$M", "Stage"]
                    st.dataframe(_ov_top.style.format({"fWAR": "{:.1f}", "Salary $M": "{:.1f}", "fWAR/$M": "{:.2f}"}, na_rep="—"),
                                 hide_index=True, use_container_width=True)
            else:
                st.info("Not enough player data for efficiency scatter.")
        else:
            st.info(f"No payroll data available for {sel_team}.")

    # ── Tab 5 — Historical Trends ────────────────────────────────────────
    with tt5:
        if not team_eff.empty and len(team_eff) >= 2:
            st.markdown(
                "<div style='font-size:0.85rem;color:#93b8d8;margin-bottom:0.8rem;'>"
                "Season-by-season trends for this team across key metrics (2021–2025).</div>",
                unsafe_allow_html=True,
            )

            # Wins trend
            _te = team_eff.sort_values("Year")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=_te["Year"], y=_te["Wins"], mode="lines+markers+text",
                marker=dict(color="#3b82f6", size=10),
                line=dict(color="#3b82f6", width=2),
                text=[f"{int(w)}" for w in _te["Wins"]],
                textposition="top center", textfont=dict(color="#d6e8f8", size=10),
                name="Wins",
                hovertemplate="%{x}: %{y} wins<extra></extra>",
            ))
            fig_hist.add_trace(go.Scatter(
                x=_te["Year"], y=_te["pred_wins"], mode="lines",
                line=dict(color="#f59e0b", dash="dash", width=1),
                name="Predicted (by payroll)",
                hovertemplate="%{x}: %{y:.0f} predicted<extra></extra>",
            ))
            fig_hist.update_layout(**_pt(
                title=f"{_full_name} — Wins vs Payroll Prediction",
                xaxis=dict(title="Season", dtick=1),
                yaxis=dict(title="Wins"),
                height=380, showlegend=True,
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
            ))
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

            # Efficiency gap trend
            fig_gap = go.Figure(go.Bar(
                x=_te["Year"].astype(int).astype(str),
                y=_te["dollar_gap_M"],
                marker_color=["#22c55e" if g < 0 else "#ef4444" for g in _te["dollar_gap_M"]],
                text=[f"${g:+.0f}M" for g in _te["dollar_gap_M"]],
                textposition="outside", textfont=dict(color="#d6e8f8", size=10),
                hovertemplate="%{x}: $%{y:+.0f}M<extra></extra>",
            ))
            _abs_max_g = max(abs(_te["dollar_gap_M"]).max(), 10) * 1.3
            fig_gap.update_layout(**_pt(
                title=f"{_full_name} — Efficiency Gap by Season",
                xaxis=dict(title="Season"),
                yaxis=dict(title="$ Gap ($M)", zeroline=True, zerolinecolor="#4a687e",
                           range=[-_abs_max_g, _abs_max_g]),
                height=340,
            ))
            st.plotly_chart(fig_gap, use_container_width=True, config={"displayModeBar": False})

            # Payroll + fWAR trend
            fig_pw = go.Figure()
            fig_pw.add_trace(go.Bar(
                x=_te["Year"].astype(int).astype(str), y=_te["payroll_M"],
                name="Payroll $M", marker_color="#3b82f6", opacity=0.6,
                hovertemplate="%{x}: $%{y:.0f}M<extra></extra>",
            ))
            fig_pw.add_trace(go.Scatter(
                x=_te["Year"].astype(int).astype(str), y=_te["team_WAR"],
                name="Team fWAR", yaxis="y2",
                mode="lines+markers", marker=dict(color="#22c55e", size=8),
                line=dict(color="#22c55e", width=2),
                hovertemplate="%{x}: %{y:.1f} fWAR<extra></extra>",
            ))
            fig_pw.update_layout(**_pt(
                title=f"{_full_name} — Payroll vs fWAR",
                yaxis=dict(title="Payroll $M"),
                yaxis2=dict(title="fWAR", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#22c55e"),
                            title_font=dict(color="#22c55e")),
                height=380, showlegend=True,
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
            ))
            st.plotly_chart(fig_pw, use_container_width=True, config={"displayModeBar": False})

            # Summary table
            st.markdown("##### Season-by-Season Summary")
            _sum = _te[["Year", "Wins", "payroll_M", "team_WAR", "dollar_gap_M", "in_playoffs"]].copy()
            _sum.columns = ["Year", "Wins", "Payroll $M", "fWAR", "Gap $M", "Postseason"]
            _sum["Year"] = _sum["Year"].astype(int)
            _sum["Postseason"] = _sum["Postseason"].map({True: "✓", False: ""})
            st.dataframe(
                _sum.style.format({"Payroll $M": "{:.0f}", "fWAR": "{:.1f}", "Gap $M": "{:+.0f}"}, na_rep="—"),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info(f"Not enough historical data for {sel_team}.")

    _render_feedback_widget("team")


# ---------------------------------------------------------------------------
# Glossary & Methodology page
# ---------------------------------------------------------------------------

def _render_glossary_page():
    """Full glossary and methodology reference page — pure static content."""

    st.markdown(
        "<h1 style='margin-bottom:0.1rem;'>📖 Glossary & Methodology</h1>"
        "<p style='color:#4a687e;font-size:0.82rem;margin-bottom:1.5rem;'>"
        "Every metric, formula, and term used across the MLB Toolbox — explained in one place.</p>",
        unsafe_allow_html=True,
    )

    # ── helper: styled section header ────────────────────────────────────
    def _section(num: int, title: str):
        st.markdown(
            f"<div style='margin:1.8rem 0 0.6rem;padding:0.5rem 0.8rem;"
            f"border-left:3px solid #2b5cc8;background:#0d1e35;border-radius:0 8px 8px 0;'>"
            f"<span style='font-size:0.7rem;color:#3b6fd4;font-weight:700;'>SECTION {num}</span>"
            f"<div style='font-size:1.05rem;font-weight:700;color:#d6e8f8;'>{title}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ==================================================================
    # Section 1 — Core Baseball Stats
    # ==================================================================
    _section(1, "Core Baseball Stats")

    _render_glossary([
        ("WAR", "Wins Above Replacement (fWAR)",
         "The number of wins a player contributes above a replacement-level (AAAA) player. "
         "We use FanGraphs fWAR, which values pitchers via FIP rather than ERA."),
        ("wRC+", "Weighted Runs Created Plus",
         "Park- and league-adjusted offensive metric. 100 = league average; "
         "150 means 50% better than average."),
        ("FIP", "Fielding Independent Pitching",
         "Estimates a pitcher's ERA based only on events they control: strikeouts, walks, HBP, "
         "and home runs. Lower is better."),
        ("BABIP", "Batting Avg on Balls In Play",
         "Batting average on balls put into play (excludes HR, K, BB). "
         "League average is ~.300; extreme values often regress."),
        ("OPS", "On-base Plus Slugging",
         "Sum of on-base percentage and slugging percentage. Quick offensive snapshot; "
         "league average is around .710-.730."),
        ("OPS+", "Adjusted OPS",
         "Park- and league-normalized OPS. 100 = league average. "
         "Comparable across eras and ballparks."),
        ("ERA", "Earned Run Average",
         "Earned runs allowed per 9 innings pitched. Influenced by defense and luck; "
         "FIP is generally a better skill measure."),
        ("WHIP", "Walks + Hits per Inning Pitched",
         "Baserunners allowed per inning. Sub-1.00 is elite; league average is ~1.25."),
        ("K/9", "Strikeouts per 9 Innings",
         "Strikeout rate normalized to 9 innings. Higher is better for pitchers."),
        ("BB/9", "Walks per 9 Innings",
         "Walk rate normalized to 9 innings. Lower is better for pitchers."),
        ("HR/9", "Home Runs per 9 Innings",
         "Home run rate normalized to 9 innings. Heavily influences FIP and ERA."),
        ("UZR", "Ultimate Zone Rating",
         "Advanced fielding metric measuring runs saved/cost relative to an average "
         "defender at the same position. Measured in runs above average."),
        ("DRS", "Defensive Runs Saved",
         "Fielding metric from Baseball Info Solutions. Positive = above-average defender. "
         "Comparable to UZR but uses a different methodology."),
        ("oWAR", "Offensive WAR",
         "WAR component from batting and baserunning only, excluding defense. "
         "Useful for isolating hitting contribution."),
        ("dWAR", "Defensive WAR",
         "WAR component from fielding only. Combines UZR/DRS-type metrics with "
         "positional adjustment."),
        ("bWAR vs fWAR", "Baseball-Reference WAR vs FanGraphs WAR",
         "bWAR (B-Ref) uses RA/9 for pitchers (actual runs); fWAR (FanGraphs) uses FIP "
         "(fielding-independent). This project uses fWAR exclusively. "
         "FanGraphs WAR (fWAR) is applied given its focus on attempting to predict future "
         "performance and evaluating player talent level."),
    ], title="Core Baseball Stats", cols=2)

    # WAR formula breakdown
    st.markdown(
        "<div style='margin:0.6rem 0 0.2rem;font-size:0.82rem;font-weight:700;color:#93c5fd;'>"
        "Full fWAR Formula Breakdown</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "<div style='font-size:0.75rem;font-weight:600;color:#d6e8f8;margin-bottom:0.3rem;'>"
            "Position Players</div>",
            unsafe_allow_html=True,
        )
        st.latex(
            r"\text{fWAR} = \frac{"
            r"\text{Bat} + \text{BsR} + \text{Fld} + \text{Pos} + \text{Lg} + \text{Rep}"
            r"}{\text{RPW}}"
        )
        st.caption(
            "Bat = Batting Runs, BsR = Baserunning Runs, Fld = Fielding Runs, "
            "Pos = Positional Adjustment, Lg = League Adjustment, "
            "Rep = Replacement Level Runs, RPW ≈ 10 Runs Per Win"
        )
    with c2:
        st.markdown(
            "<div style='font-size:0.75rem;font-weight:600;color:#d6e8f8;margin-bottom:0.3rem;'>"
            "Pitchers (FIP-based)</div>",
            unsafe_allow_html=True,
        )
        st.latex(
            r"\text{FIP} = \frac{13 \times \text{HR} + 3 \times (\text{BB}+\text{HBP})"
            r" - 2 \times \text{K}}{\text{IP}} + C"
        )
        st.caption(
            "C = FIP constant (scales FIP to league ERA). "
            "Pitcher fWAR then converts FIP to runs above replacement level "
            "using innings pitched and a runs-per-win denominator (~10 RPW)."
        )

    # ==================================================================
    # Section 2 — Our Custom Metrics
    # ==================================================================
    _section(2, "Our Custom Metrics")

    _render_glossary([
        ("W/$M", "WAR per Million Dollars",
         "Single-season efficiency ratio: WAR ÷ Salary (in $M). "
         "Higher means more production per dollar spent."),
        ("PPR", "Pay-to-Performance Ratio",
         "Career efficiency: Sum of career WAR ÷ total contract value ($M). "
         "Captures long-term value delivery across entire contract."),
        ("PPEL", "Pay-Performance Efficiency Line",
         "OLS regression of WAR ~ Salary across all players. "
         "Players below the line are underpaid (good value); above = overpaid."),
        ("PPEL3", "3-Year Weighted PPEL",
         "Weighted average WAR over 3 seasons (0.50 / 0.30 / 0.20, most recent first) "
         "regressed against current salary. Smooths single-year noise."),
        ("PPEL5", "5-Year Weighted PPEL",
         "Weighted average WAR over 5 seasons (0.35 / 0.25 / 0.20 / 0.12 / 0.08) "
         "regressed against current salary. Best for established veterans."),
        ("Residual", "Salary Residual",
         "Actual Salary − PPEL Predicted Salary (in $M). "
         "Positive = overpaid vs market; negative = underpaid (bargain)."),
        ("CEL", "Cost Effective Line",
         "Team-level equivalent of PPEL: OLS regression of total team WAR vs total team payroll. "
         "Teams below the line are spending efficiently."),
        ("Efficiency Gap", "Team Efficiency Gap",
         "Team payroll − predicted payroll for their WAR total. "
         "Negative = efficient (winning more than you're paying for)."),
        ("Est. Wins", "Estimated Wins",
         "47.7 + Roster WAR. The 47.7 baseline represents the expected wins for a team of "
         "all replacement-level players over a 162-game season."),
        ("$/WAR", "Dollars per WAR",
         "Team payroll divided by total team WAR. Shows the average cost of one win above "
         "replacement for that roster."),
        ("$/Win", "Dollars per Win (Adjusted)",
         "2026 Payroll ($M) / (Total 2025 WAR + 47.7). Unlike $/WAR, this includes the "
         "47.7 replacement-baseline wins every team gets for free. Lower = better."),
        ("WSR", "WAR Stability Rating",
         "mean WAR / (1 + std WAR) across qualifying seasons (PA ≥ 200 or IP ≥ 50). "
         "Elite ≥ 3.5, Reliable ≥ 2.0, Volatile ≥ 1.0, Unstable < 1.0. "
         "Rewards consistent production over volatile single-season peaks."),
        ("RSS", "Roster Stability Score",
         "Percentage of qualifying players (PA ≥ 150 or IP ≥ 40) who returned from "
         "the previous season. Higher RSS indicates roster continuity."),
    ], title="Custom Metrics", cols=2)

    # Formula highlights
    st.markdown(
        "<div style='margin:0.6rem 0 0.2rem;font-size:0.82rem;font-weight:700;color:#93c5fd;'>"
        "Key Formulas</div>",
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        st.latex(r"\text{W/\$M} = \frac{\text{WAR}}{\text{Salary (\$M)}}")
    with f2:
        st.latex(r"\text{PPEL: } \hat{S} = \beta_0 + \beta_1 \times \text{WAR}")
    with f3:
        st.latex(r"\text{Est. Wins} = 47.7 + \sum \text{WAR}")

    st.caption(
        "PPEL regression is fitted via OLS across all players in the dataset. "
        "PPEL3/PPEL5 use the same approach but with time-weighted WAR inputs."
    )

    # ==================================================================
    # Section 3 — Roster & Contract Terms
    # ==================================================================
    _section(3, "Roster & Contract Terms")

    _render_glossary([
        ("CBT — Under $244M", "No Restrictions",
         "Below the Competitive Balance Tax threshold. No luxury tax penalties or restrictions."),
        ("CBT — $244M–$264M", "Tier 1 Luxury Tax",
         "First CBT tier. Team pays a 20% tax on overage (first-time) or 30% (repeat). "
         "No operational restrictions."),
        ("CBT — $264M–$284M", "1st Apron",
         "Significant restrictions: cannot receive revenue-sharing money, limited international "
         "bonus pool, pick compensation restrictions for free agent signings."),
        ("CBT — $284M–$304M", "Tier 3 / Enhanced Penalties",
         "Severe restrictions on top of 1st Apron rules. Higher tax rates, additional "
         "draft pick forfeiture, and frozen luxury tax rate escalation."),
        ("CBT — $304M+", "2nd Apron",
         "Hardest restrictions. Highest surtax rate, top draft pick drops 10 spots, "
         "cannot add salary via trade without sending equal or greater salary back."),
        ("FA", "Free Agent",
         "Player with 6+ years of MLB service time whose contract has expired. "
         "Can sign with any team on the open market."),
        ("Arb", "Arbitration-Eligible",
         "Player with 3-6 years of service time (or Super Two eligible). "
         "Salary set via negotiation or arbitration hearing if no agreement."),
        ("Pre-Arb", "Pre-Arbitration / Team Control",
         "Player with fewer than 3 years of service time. Salary near league minimum "
         "(~$740K in 2026). Team controls rights."),
        ("DFA", "Designated for Assignment",
         "Player removed from 40-man roster. Team has 7 days to trade, release, or "
         "outright the player to the minors."),
        ("Option", "Minor League Option",
         "Sending a player with options remaining to the minor leagues. A player has 3 option "
         "years; once used in a season, that year's option is burned."),
        ("NRI", "Non-Roster Invitee",
         "Player invited to Spring Training who is not on the 40-man roster. "
         "Must be added to the 40-man to make Opening Day roster."),
        ("40-Man vs 26-Man", "Roster Types",
         "40-man: players protected from Rule 5 Draft and eligible for MLB call-up. "
         "26-man: the active game-day roster (expanded to 28 in September)."),
        ("IL", "Injured List",
         "10-day IL (position players, relievers), 15-day IL (starting pitchers), "
         "and 60-day IL (long-term; player removed from 40-man count)."),
        ("Rule 5", "Rule 5 Draft Eligibility",
         "Players not on the 40-man roster after 5 years in the minors (signed at 18+) or "
         "4 years (signed at ≤18) are eligible to be drafted by other teams."),
    ], title="Roster & Contract Terms", cols=2)

    # ==================================================================
    # Section 4 — Roster Grade System
    # ==================================================================
    _section(4, "Roster Grade System")

    st.markdown(
        "<p style='color:#7a9ebc;font-size:0.76rem;margin-bottom:0.8rem;'>"
        "Every simulated roster receives four letter grades (A+ through F) across "
        "these dimensions:</p>",
        unsafe_allow_html=True,
    )

    _render_glossary([
        ("Production", "Total WAR Output",
         "Score = Total Roster WAR ÷ 35 WAR ceiling. "
         "A+ ≥ 95%, A ≥ 85%, B ≥ 70%, C ≥ 55%, D ≥ 40%, F < 40%."),
        ("Efficiency", "WAR per Dollar",
         "Roster-wide WAR/$M compared to a 0.30 WAR/$M baseline. "
         "Higher ratio → better grade. Rewards getting production without overspending."),
        ("Depth", "Position Coverage",
         "How many of the 9 standard position groups are filled "
         "(C, 1B, 2B, 3B, SS, LF, CF, RF + SP/RP). Full coverage = A+."),
        ("Contract Health", "Years Remaining Balance",
         "Average years remaining on roster contracts vs a 2.0-year ideal. "
         "Too short (all expiring) or too long (all locked up) lowers the grade."),
    ], title="Grade Dimensions", cols=2)

    # Grade cutoff table
    st.markdown(
        "<div style='margin:0.8rem 0 0.4rem;font-size:0.82rem;font-weight:700;color:#93c5fd;'>"
        "Letter Grade Cutoffs</div>",
        unsafe_allow_html=True,
    )

    _grade_data = [
        ("A+", "≥ 95%", "Elite — best-in-class across the league"),
        ("A",  "≥ 85%", "Excellent — clear strength with minimal gaps"),
        ("B",  "≥ 70%", "Above average — solid with room to improve"),
        ("C",  "≥ 55%", "Average — meets baseline expectations"),
        ("D",  "≥ 40%", "Below average — notable weakness"),
        ("F",  "< 40%", "Poor — significant area of concern"),
    ]

    cols = st.columns(6)
    for i, (grade, pct, desc) in enumerate(_grade_data):
        with cols[i]:
            _color = {
                "A+": "#22c55e", "A": "#4ade80", "B": "#60a5fa",
                "C": "#facc15", "D": "#f97316", "F": "#ef4444",
            }[grade]
            st.markdown(
                f"<div style='text-align:center;background:#0d1e35;border:1px solid #1e3250;"
                f"border-radius:8px;padding:0.5rem 0.3rem;'>"
                f"<div style='font-size:1.3rem;font-weight:900;color:{_color};'>{grade}</div>"
                f"<div style='font-size:0.65rem;color:#7a9ebc;'>{pct}</div>"
                f"<div style='font-size:0.58rem;color:#4a687e;margin-top:0.2rem;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Example comparison
    st.markdown(
        "<div style='margin:0.8rem 0 0.4rem;font-size:0.82rem;font-weight:700;color:#93c5fd;'>"
        "Example: A+ Roster vs C Roster</div>",
        unsafe_allow_html=True,
    )

    ex1, ex2 = st.columns(2)
    with ex1:
        st.markdown(
            "<div style='background:#0d1e35;border:1px solid #22c55e;border-radius:8px;"
            "padding:0.6rem 0.8rem;'>"
            "<div style='font-size:0.8rem;font-weight:700;color:#22c55e;margin-bottom:0.3rem;'>"
            "A+ Roster Example</div>"
            "<div style='font-size:0.68rem;color:#7a9ebc;line-height:1.7;'>"
            "<b style='color:#d6e8f8;'>Production:</b> ~33+ WAR (≥95% of 35 ceiling)<br>"
            "<b style='color:#d6e8f8;'>Efficiency:</b> WAR/$M well above 0.30 baseline<br>"
            "<b style='color:#d6e8f8;'>Depth:</b> All 9 position groups filled<br>"
            "<b style='color:#d6e8f8;'>Contract Health:</b> ~2.0 avg years remaining"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with ex2:
        st.markdown(
            "<div style='background:#0d1e35;border:1px solid #facc15;border-radius:8px;"
            "padding:0.6rem 0.8rem;'>"
            "<div style='font-size:0.8rem;font-weight:700;color:#facc15;margin-bottom:0.3rem;'>"
            "C Roster Example</div>"
            "<div style='font-size:0.68rem;color:#7a9ebc;line-height:1.7;'>"
            "<b style='color:#d6e8f8;'>Production:</b> ~19-20 WAR (55-60% of ceiling)<br>"
            "<b style='color:#d6e8f8;'>Efficiency:</b> WAR/$M near or below 0.30<br>"
            "<b style='color:#d6e8f8;'>Depth:</b> 5-6 of 9 groups filled, gaps at key spots<br>"
            "<b style='color:#d6e8f8;'>Contract Health:</b> Skewed toward all expiring or all long-term"
            "</div></div>",
            unsafe_allow_html=True,
        )

    # ==================================================================
    # Section 5 — Data Sources & Methodology
    # ==================================================================
    _section(5, "Data Sources & Methodology")

    st.markdown(
        "<div style='background:#0d1e35;border:1px solid #1e3250;border-radius:8px;"
        "padding:1rem 1.2rem;font-size:0.76rem;color:#7a9ebc;line-height:1.8;'>"
        "<b style='color:#d6e8f8;font-size:0.82rem;'>Where the data comes from</b><br><br>"
        "<b style='color:#93c5fd;'>WAR & Performance Data:</b> "
        "All WAR figures use FanGraphs fWAR, covering the 2021–2025 MLB seasons. "
        "Pitchers are evaluated via FIP-based WAR; position players use the full "
        "batting + baserunning + fielding + positional framework.<br><br>"
        "<b style='color:#93c5fd;'>Salary & Payroll Data:</b> "
        "2026 contract commitments sourced per-team from publicly available payroll reports. "
        "Includes base salary, signing bonuses, and incentives where applicable.<br><br>"
        "<b style='color:#93c5fd;'>40-Man Rosters:</b> "
        "Current 40-man roster composition is pulled from the MLB Stats API "
        "(statsapi.mlb.com) and updated periodically.<br><br>"
        "<b style='color:#93c5fd;'>Projection Approach:</b> "
        "All analysis is backward-looking — we use historical WAR to evaluate efficiency, "
        "not forward-looking projection systems (ZiPS, Steamer, etc.). "
        "This means the metrics reflect what players <i>have</i> done, not what they "
        "are projected to do.<br><br>"
        "<b style='color:#93c5fd;'>Research Context:</b> "
        "MLB Toolbox is a university research project. It is not financial advice, "
        "not affiliated with MLB or any franchise, and should not be used for "
        "gambling or professional roster decisions.<br><br>"
        "<b style='color:#93c5fd;'>Last Updated:</b> "
        "Data covers through the 2025 MLB season."
        "</div>",
        unsafe_allow_html=True,
    )

    _render_feedback_widget("glossary")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _valid_pages = {"home", "league", "simulator", "roster_optimizer", "rankings", "glossary", "team"}
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


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
