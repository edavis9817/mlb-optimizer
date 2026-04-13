"""MLB Toolbox — data loading utilities.

All data-loading, caching, and remote-fetch functions extracted from
streamlit_app.py so they can be imported directly by any page module.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests as _requests
    _requests_available = True
except ImportError:
    _requests_available = False

from utils.player_utils import fix_player_name, fix_player_col
from utils.constants import (
    PAYROLL_TEAM_MAP,
    MLB_TEAM_ID_MAP,
    STAGE_DISPLAY_MAP,
    POS_GROUP_MAP,
    ELIGIBLE_SLOTS_MAP,
    LOGO_FILE_NAMES,
)
from utils.player_utils import headshot_url

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_APP_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT_DIR = os.path.dirname(_APP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.team_mode import (
    get_team_payroll_history,
    build_offseason_scenario,
)

# ---------------------------------------------------------------------------
# R2 / Production remote data
# ---------------------------------------------------------------------------
R2_BASE_URL = os.environ.get("R2_BASE_URL", "").strip().rstrip("/")
R2_MODE     = bool(R2_BASE_URL)

# ---------------------------------------------------------------------------
# Persistent Disk Cache Management
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.expanduser("~/.mlb_toolbox_cache")
_ETAG_METADATA_FILE = os.path.join(_CACHE_DIR, "etags.json")

try:
    Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)
except Exception:
    pass  # Cache directory creation optional

# ---------------------------------------------------------------------------
# Module-level convenience paths (mirror streamlit_app.py originals)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = os.path.join(_ROOT_DIR, "configs", "default_config.json")
ROOT_DIR = _ROOT_DIR


# ---------------------------------------------------------------------------
# ETag helpers
# ---------------------------------------------------------------------------

def init_etag_metadata() -> dict:
    """Load or create the ETag metadata file."""
    if os.path.exists(_ETAG_METADATA_FILE):
        try:
            with open(_ETAG_METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_etag_metadata(metadata: dict) -> None:
    """Persist ETag metadata to disk."""
    try:
        with open(_ETAG_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass  # Metadata save is optional


def compute_cache_key(url: str) -> str:
    """Generate a deterministic cache filename from a URL."""
    return hashlib.md5(url.encode()).hexdigest() + ".cached"


def get_cached_file_path(url: str) -> str:
    """Return the disk path where a remote file should be cached."""
    return os.path.join(_CACHE_DIR, compute_cache_key(url))


# ---------------------------------------------------------------------------
# CSV / Excel / image readers
# ---------------------------------------------------------------------------

def read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV from a local path or an R2 URL with persistent disk caching & ETag support.

    For remote URLs:
    1. Check if cached file exists on disk
    2. If yes, verify with ETag to detect changes
    3. If unchanged: use cached file (fast)
    4. If changed or missing: download and cache
    5. Automatically normalises the Player column if present
    """
    if path.startswith("http") and _requests_available:
        cache_file = get_cached_file_path(path)
        metadata = init_etag_metadata()
        cache_key = path

        # Try to use cached file if it exists
        if os.path.exists(cache_file):
            try:
                cached_meta  = metadata.get(cache_key, {})
                cached_etag  = cached_meta.get("etag")
                checked_at   = cached_meta.get("checked_at", 0)
                if cached_etag:
                    # Skip HEAD request if we verified within the last hour
                    if time.time() - checked_at < 3600:
                        return fix_player_col(pd.read_csv(cache_file, **kwargs))
                    head_resp = _requests.head(path, timeout=30)
                    current_etag = head_resp.headers.get("ETag", "")
                    if current_etag and current_etag == cached_etag:
                        # File unchanged—update timestamp and use cached version
                        metadata[cache_key]["checked_at"] = time.time()
                        save_etag_metadata(metadata)
                        return fix_player_col(pd.read_csv(cache_file, **kwargs))
            except Exception:
                # If HEAD check fails, fall back to using cached file anyway
                try:
                    return fix_player_col(pd.read_csv(cache_file, **kwargs))
                except Exception:
                    pass  # Fall through to re-download

        # Download and cache
        try:
            resp = _requests.get(path, timeout=30)
            resp.raise_for_status()

            # Save to cache
            with open(cache_file, "wb") as f:
                f.write(resp.content)

            # Update metadata with ETag and check timestamp
            etag = resp.headers.get("ETag", "")
            if etag:
                metadata[cache_key] = {"etag": etag, "url": path, "checked_at": time.time()}
                save_etag_metadata(metadata)

            return fix_player_col(pd.read_csv(io.BytesIO(resp.content), **kwargs))
        except Exception as e:
            # If download fails but cache exists, use stale cache
            if os.path.exists(cache_file):
                try:
                    return fix_player_col(pd.read_csv(cache_file, **kwargs))
                except Exception:
                    raise e
            raise

    return fix_player_col(pd.read_csv(path, **kwargs))


def read_excel(path: str, **kwargs) -> pd.DataFrame:
    """Read an Excel file from a local path or an R2 URL with persistent disk caching & ETag support."""
    if path.startswith("http") and _requests_available:
        cache_file = get_cached_file_path(path)
        metadata = init_etag_metadata()
        cache_key = path

        # Try to use cached file if it exists
        if os.path.exists(cache_file):
            try:
                cached_meta  = metadata.get(cache_key, {})
                cached_etag  = cached_meta.get("etag")
                checked_at   = cached_meta.get("checked_at", 0)
                if cached_etag:
                    # Skip HEAD request if we verified within the last hour
                    if time.time() - checked_at < 3600:
                        return fix_player_col(pd.read_excel(cache_file, **kwargs))
                    head_resp = _requests.head(path, timeout=30)
                    current_etag = head_resp.headers.get("ETag", "")
                    if current_etag and current_etag == cached_etag:
                        metadata[cache_key]["checked_at"] = time.time()
                        save_etag_metadata(metadata)
                        return fix_player_col(pd.read_excel(cache_file, **kwargs))
            except Exception:
                try:
                    return fix_player_col(pd.read_excel(cache_file, **kwargs))
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
                metadata[cache_key] = {"etag": etag, "url": path, "checked_at": time.time()}
                save_etag_metadata(metadata)

            return fix_player_col(pd.read_excel(io.BytesIO(resp.content), **kwargs))
        except Exception as e:
            if os.path.exists(cache_file):
                try:
                    return fix_player_col(pd.read_excel(cache_file, **kwargs))
                except Exception:
                    raise e
            raise

    return fix_player_col(pd.read_excel(path, **kwargs))


def r2_image(path: str) -> bytes | str:
    """Return image bytes (for R2 URLs) or a local path for st.image()."""
    if path.startswith("http") and _requests_available:
        resp = _requests.get(path, timeout=30)
        resp.raise_for_status()
        return resp.content
    return path


def data_url(relative_path: str) -> str:
    """Return an R2 public URL (production) or an absolute local path (development).

    relative_path should use forward slashes and be relative to the project root,
    e.g. "data/mlb_combined_2021_2025.csv" or "2026 Payroll".
    """
    if R2_BASE_URL:
        return f"{R2_BASE_URL}/{relative_path.replace(os.sep, '/')}"
    return os.path.join(_ROOT_DIR, *relative_path.split("/"))


# Now that data_url is defined, set RAZZBALL_PATH properly
RAZZBALL_PATH = data_url("data/razzball.csv")
HEADSHOTS_DIR = (
    f"{R2_BASE_URL}/data/headshots" if R2_MODE
    else os.path.join(_ROOT_DIR, "data", "headshots")
)


# ---------------------------------------------------------------------------
# Config & path helpers
# ---------------------------------------------------------------------------

def load_base_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as fh:
        return json.load(fh)


def resolve_data_path(raw_path: str, config_path: str) -> str:
    if os.path.isabs(raw_path):
        return raw_path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(config_dir, raw_path))


def file_hash(path: str) -> str:
    """Generate a cache key for a file path or R2 URL.

    For remote files: uses ETag if available, otherwise returns URL hash
    For local files: computes MD5 of file content
    """
    if path.startswith("http"):
        metadata = init_etag_metadata()
        cached_data = metadata.get(path, {})
        if "etag" in cached_data:
            return cached_data["etag"]  # Use ETag as cache key
        # Try to fetch ETag if we don't have it cached
        try:
            head_resp = _requests.head(path, timeout=10)
            etag = head_resp.headers.get("ETag", "")
            if etag:
                metadata[path] = {"etag": etag, "url": path}
                save_etag_metadata(metadata)
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


def dir_hash(dir_path: str) -> str:
    """Cheap cache key for a directory -- hashes the sorted file listing.

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
# Payroll parser
# ---------------------------------------------------------------------------

def parse_payroll_val(val) -> float | None:
    """Parse a payroll cell (e.g. '$33,000,000', 'TBD', 'FREE AGENT') -> float $M."""
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
# Team logo URL helper
# ---------------------------------------------------------------------------

def team_logo_url(abbr: str) -> str:
    """Return the R2 logo URL for a team abbreviation."""
    return data_url(f"logos/{LOGO_FILE_NAMES.get(abbr, abbr)}.png")


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading & projecting player data ...", persist="disk", ttl=86400)
def cached_projections(salary_path: str, file_hash_val: str, proj_weights_json: str,
                       season: int, clip_neg: bool, min_war: float, max_yrs: int):
    from src.projections import make_projections
    cfg = {
        "season":              season,
        "projection_weights":  json.loads(proj_weights_json),
        "clip_negative_war":   clip_neg,
        "min_war_threshold":   min_war,
        "max_contract_years":  max_yrs,
    }
    raw_df = read_csv(salary_path, low_memory=False)
    return make_projections(raw_df, cfg), raw_df


@st.cache_data(show_spinner="Building archetypes ...", persist="disk", ttl=86400)
def cached_archetypes(proj_hash: str, proj_json: str):
    from src.archetypes import build_archetype_definitions, assign_archetypes
    proj_df = pd.read_json(proj_json, orient="records")
    proj_df["eligible_slots"] = proj_df["eligible_slots"].apply(
        lambda v: v if isinstance(v, list) else []
    )
    arch_df        = build_archetype_definitions(proj_df)
    proj_with_arch = assign_archetypes(proj_df)
    return arch_df, proj_with_arch


@st.cache_data(show_spinner="Loading wins data ...", persist="disk", ttl=86400)
def cached_wins(wins_path: str, file_hash_val: str):
    if not os.path.exists(wins_path):
        return pd.DataFrame()
    return read_csv(wins_path, low_memory=False)


@st.cache_data(show_spinner="Loading team payroll history ...", persist="disk", ttl=86400)
def cached_payroll_history(data_dir: str):
    return get_team_payroll_history(data_dir)


@st.cache_data(show_spinner="Loading team roster ...", ttl=86400)
def cached_team_scenario(
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
    combined_df = read_csv(
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


@st.cache_data(show_spinner="Loading player database ...", persist="disk", ttl=86400)
def cached_simulator_data(combined_path: str, ind_2025_path: str, file_hash_val: str) -> pd.DataFrame:
    """Load and merge 2025 combined data with 2025 individual contract columns."""
    comb = read_csv(combined_path, low_memory=False)
    comb.columns = [c.strip() for c in comb.columns]
    comb["Year"] = pd.to_numeric(comb["Year"], errors="coerce")
    comb2025 = comb[comb["Year"] == 2025].copy()

    # Deduplicate traded players -- keep highest salary
    comb2025["Salary_M"] = pd.to_numeric(comb2025.get("Salary_M", pd.Series(dtype=float)), errors="coerce").fillna(0)
    comb2025 = (
        comb2025.sort_values("Salary_M", ascending=False, kind="mergesort")
                .drop_duplicates(subset=["Player"], keep="first")
                .reset_index(drop=True)
    )

    # Enrich with contract year columns from individual file
    if ind_2025_path.startswith("http") or os.path.exists(ind_2025_path):
        ind = read_csv(ind_2025_path, low_memory=False)
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
    comb2025["pos_group"] = comb2025["Position"].map(lambda p: POS_GROUP_MAP.get(p, "UNK"))
    comb2025["eligible_slots"] = comb2025["pos_group"].map(lambda g: ELIGIBLE_SLOTS_MAP.get(g, []))

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
    # PPR = Sigma actual WAR for each contract year / total contract $M
    # For each contract year we use the player's actual WAR from that season
    # (looking back N seasons where N = total contract length).
    # Missing seasons fall back to the most recent known WAR.
    # Higher PPR = more WAR produced per dollar committed.
    # ------------------------------------------------------------------
    _future_yr_cols = [c for c in ["2026", "2027", "2028", "2029", "2030", "2031"]
                       if c in comb2025.columns]
    for _c in _future_yr_cols:
        # Values may be currency-formatted strings like "$14,000,000" -- strip and normalise
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

    # Total contract value ($M) -- current year + all future committed years
    def _calc_total_ctrc(row):
        sal_cur  = float(row.get("Salary_M") or 0)
        fut_sals = [float(row[c]) for c in _future_yr_cols
                    if pd.notna(row.get(c)) and float(row.get(c) or 0) > 0]
        total = sal_cur + sum(fut_sals)
        return round(total, 2) if total > 0 else None

    comb2025["Total_Contract_M"] = comb2025.apply(_calc_total_ctrc, axis=1)

    # ------------------------------------------------------------------
    # WAR Stability Rating (WSR) -- Feature 3
    # WSR = mean_WAR / (1 + std_WAR) using only qualifying seasons
    # Qualifying: PA >= 200 (hitters) or IP >= 50 (pitchers)
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


@st.cache_data(show_spinner="Loading 2026 payroll data ...", persist="disk", ttl=86400)
def cached_2026_payroll(payroll_dir: str, combined_path: str, dir_hash_val: str) -> pd.DataFrame:
    """Consolidate all 30-team 2026 payroll xlsx files into a single DataFrame.

    Joins with 2025 combined data to bring in Position + last-known WAR/stats.
    Returns a DataFrame compatible with the simulator (same key columns as cached_simulator_data).
    """
    _SHEETS = {
        "Guaranteed":               "FA",
        "Eligible For Arb":         "Arb",
        "Not Yet Eligible For Arb": "Pre-Arb",
    }
    _YR_COLS = [2027, 2028, 2029, 2030, 2031, 2032]

    rows: list[dict] = []
    # Build (filename, full_path_or_url) pairs -- supports both local dir and R2 base URL
    if payroll_dir.startswith("http"):
        _payroll_entries = [
            (f"{name}-Payroll-2026.xlsx", f"{payroll_dir}/{name}-Payroll-2026.xlsx")
            for name in PAYROLL_TEAM_MAP
        ]
    else:
        _payroll_entries = [
            (fname, os.path.join(payroll_dir, fname))
            for fname in sorted(os.listdir(payroll_dir))
            if fname.endswith(".xlsx")
        ]
    for fname, fpath in _payroll_entries:
        team_name = fname.replace("-Payroll-2026.xlsx", "")
        team_abbr = PAYROLL_TEAM_MAP.get(team_name, team_name[:3].upper())
        try:
            if fpath.startswith("http") and _requests_available:
                _xr = _requests.get(fpath, timeout=30)
                _xr.raise_for_status()
                xl = pd.ExcelFile(io.BytesIO(_xr.content))
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

                sal_2026 = parse_payroll_val(row.get(2026))
                aav      = parse_payroll_val(row.get("AAV"))

                fut: dict[str, float | None] = {}
                for yr in _YR_COLS:
                    fut[str(yr)] = parse_payroll_val(row.get(yr))

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

    # Deduplicate (keep highest salary row -- same player can appear in multiple sheets)
    df26 = (
        df26.sort_values("Salary_M", ascending=False, na_position="last", kind="mergesort")
            .drop_duplicates(subset=["Player"], keep="first")
            .reset_index(drop=True)
    )

    # Enrich with 2025 Position + WAR + key stats via player name join
    # Also keep full multi-year data for WAR history lookup in PPR
    _war_hist_26: dict = {}
    if combined_path.startswith("http") or os.path.exists(combined_path):
        comb = read_csv(combined_path, low_memory=False)
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
    df26["pos_group"]     = df26["Position"].map(lambda p: POS_GROUP_MAP.get(p, "UNK"))
    df26["eligible_slots"] = df26["pos_group"].map(lambda g: ELIGIBLE_SLOTS_MAP.get(g, []))

    # Convert future-year columns to numeric $M
    for yr in _YR_COLS:
        col = str(yr)
        if col in df26.columns:
            df26[col] = pd.to_numeric(df26[col], errors="coerce")

    # PPR = Sigma actual WAR for each contract year / total contract $M
    # Looks back N seasons (N = total contract length) using real historical WAR.
    # Missing seasons fall back to most recent known WAR (2025).
    _fyr = [str(yr) for yr in _YR_COLS if str(yr) in df26.columns]

    def _ppr_26(row):
        sal = float(row.get("Salary_M") or 0)
        war = float(row.get("WAR_Total") or 0)  # 2025 WAR -- fallback for missing seasons
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

    # Total contract value ($M) -- 2026 salary + all future committed years
    def _total_ctrc_26(row):
        sal = float(row.get("Salary_M") or 0)
        fut_sals = [float(row[c]) for c in _fyr
                    if pd.notna(row.get(c)) and float(row.get(c) or 0) > 0]
        total = sal + sum(fut_sals)
        return round(total, 2) if total > 0 else None

    df26["Total_Contract_M"] = df26.apply(_total_ctrc_26, axis=1)

    df26["Year"] = 2026
    return df26


@st.cache_data(show_spinner=False, ttl=86400)
def cached_player_history(combined_path: str, file_hash_val: str) -> pd.DataFrame:
    """Load full multi-year player data for player cards (all years, all players)."""
    df = read_csv(combined_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df


@st.cache_data(show_spinner=False, ttl=86400)
def cached_war_reliability(combined_path: str, file_hash_val: str) -> dict:
    """Compute WAR reliability grades from multi-year history (for consistency badges)."""
    df = read_csv(combined_path, low_memory=False)
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


@st.cache_data(show_spinner=False, ttl=86400)
def cached_razzball(razzball_path: str) -> pd.DataFrame:
    """Load razzball MLBAM ID lookup table (local file or R2 URL)."""
    if not razzball_path.startswith("http") and not os.path.exists(razzball_path):
        return pd.DataFrame()
    try:
        df = read_csv(razzball_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=86400)
def cached_mlbam_lookup(razzball_path: str) -> dict[str, str]:
    """Build a dict mapping normalised player name -> MLBAM ID string.

    Uses First+Last columns if available, otherwise the Name column.
    """
    rz = cached_razzball(razzball_path)
    if rz.empty or "MLBAMID" not in rz.columns:
        return {}
    lookup: dict[str, str] = {}
    if "First" in rz.columns and "Last" in rz.columns:
        for _, row in rz.iterrows():
            first = str(row.get("First", "")).strip()
            last  = str(row.get("Last", "")).strip()
            mid   = str(row["MLBAMID"]).strip()
            if first and last and mid.isdigit():
                key = fix_player_name(f"{first} {last}")
                lookup[key] = mid
    if "Name" in rz.columns:
        for _, row in rz.iterrows():
            name = str(row.get("Name", "")).strip()
            mid  = str(row["MLBAMID"]).strip()
            if name and mid.isdigit():
                key = fix_player_name(name)
                lookup.setdefault(key, mid)
    return lookup


@st.cache_data(show_spinner=False, ttl=86400)
def cached_40man_roster(roster_path: str, fhash: str) -> pd.DataFrame:
    """Load the 40-man roster CSV (from local file or R2 URL)."""
    try:
        df = read_csv(roster_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=86400)
def load_enriched_roster() -> pd.DataFrame:
    """Load the enriched 2026 roster+payroll dataset (single source of truth).

    Contains 1,248 players across 30 teams with salary, contract stage,
    status, position, and year-by-year commitments through 2032.
    """
    path = data_url("roster_payroll_2026_enriched.csv")
    try:
        df = read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]

    # Primary position: first before any "/"
    if "position" in df.columns:
        df["position_primary"] = df["position"].astype(str).str.split("/").str[0].str.strip()
        df.loc[df["position_primary"].isin(["nan", "None", ""]), "position_primary"] = pd.NA

    # Clean stage display
    _stg_map = {"Guaranteed": "Free Agent", "Arb-Eligible": "Arb",
                "Arb": "Arb", "Pre-Arb": "Pre-Arb", "FA": "Free Agent", "Off 40-Man": "Off 40-Man"}
    if "contract_stage" in df.columns:
        df["stage_display"] = df["contract_stage"].map(_stg_map).fillna("\u2014")

    # FA year: first year where status == FREE AGENT
    def _fa_year(row):
        for yr in range(2026, 2033):
            if str(row.get(f"status_{yr}", "")).upper() == "FREE AGENT":
                return yr
        return None
    df["fa_year"] = df.apply(_fa_year, axis=1)

    # Contract years remaining (Signed years only)
    def _yrs_rem(row):
        return sum(1 for yr in range(2026, 2033) if str(row.get(f"status_{yr}", "")).upper() == "SIGNED")
    df["contract_years_remaining"] = df.apply(_yrs_rem, axis=1)

    # Ensure salary columns are numeric
    for yr in range(2026, 2033):
        col = f"salary_{yr}_M"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False, ttl=86400)
def build_carousel_players(combined_path: str) -> list[str]:
    """Return top-3 WAR players per team from 2025, ensuring all 30 teams are represented.

    Skips players whose headshot cannot be resolved (no MLBAM ID).
    """
    try:
        df = read_csv(combined_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["WAR_Total"] = pd.to_numeric(df.get("WAR_Total", pd.Series(dtype=float)), errors="coerce")
        df["Salary_M"] = pd.to_numeric(df.get("Salary_M", pd.Series(dtype=float)), errors="coerce")
        df25 = df[df["Year"] == 2025].dropna(subset=["WAR_Total", "Team", "Player"])
        df25["_eff"] = df25["WAR_Total"] / df25["Salary_M"].clip(lower=0.01)

        mlbam = cached_mlbam_lookup(RAZZBALL_PATH)

        # 1 best efficiency player per stage per team (3 per team)
        _stages = {"FA": "Free Agent", "Arb": "Arbitration", "Pre-Arb": "Pre-Arbitration"}
        result = []
        for _tm, grp in df25.groupby("Team"):
            for stg_key in _stages:
                stg_players = grp[grp.get("Stage_Clean", pd.Series()).str.contains(stg_key, case=False, na=False)]
                if stg_players.empty:
                    continue
                best = stg_players.sort_values("_eff", ascending=False)
                for _, row in best.iterrows():
                    pname = row["Player"]
                    if pname in mlbam:
                        result.append(pname)
                        break
        return result
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=86400)
def cached_carousel_images(headshots_dir: str, n: int = 90, seed: int = 42,
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
        mlbam = cached_mlbam_lookup(RAZZBALL_PATH)
        players = list(player_list)
        rng.shuffle(players)

        # Build (index, url) pairs so we can restore order after parallel fetch
        candidates = []
        for name in players[:n]:
            mid = mlbam.get(name)
            if mid:
                candidates.append((name, headshot_url(mid, width=213)))

        def _fetch(item):
            _, url = item
            try:
                resp = _requests.get(url, timeout=4)
                if resp.status_code == 200 and len(resp.content) > 8000:
                    return base64.b64encode(resp.content).decode()
            except Exception:
                pass
            return None

        result = []
        with ThreadPoolExecutor(max_workers=12) as executor:
            for img in executor.map(_fetch, candidates):
                if img is not None:
                    result.append(img)
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
# MLB Stats API fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_2026_standings() -> dict[str, tuple[int, int]]:
    """Fetch current 2026 W-L records from MLB Stats API. Cached for 24 hours."""
    if not _requests_available:
        return {}
    try:
        resp = _requests.get(
            "https://statsapi.mlb.com/api/v1/standings"
            "?leagueId=103,104&season=2026&standingsTypes=regularSeason",
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        records: dict[str, tuple[int, int]] = {}
        for division in data.get("records", []):
            for tr in division.get("teamRecords", []):
                tid = tr.get("team", {}).get("id")
                abbr = MLB_TEAM_ID_MAP.get(tid)
                if abbr:
                    records[abbr] = (int(tr.get("wins", 0)), int(tr.get("losses", 0)))
        return records
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_2026_standings_full() -> dict[str, dict]:
    """Fetch full 2026 standings with division rank, GB, league rank. Cached 24h."""
    if not _requests_available:
        return {}
    _DIV_NAMES = {
        103: {"E": "AL East", "C": "AL Central", "W": "AL West"},
        104: {"E": "NL East", "C": "NL Central", "W": "NL West"},
    }
    try:
        resp = _requests.get(
            "https://statsapi.mlb.com/api/v1/standings"
            "?leagueId=103,104&season=2026&standingsTypes=regularSeason",
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        # Build overall league rank
        all_teams = []
        for division in data.get("records", []):
            lg_id = division.get("league", {}).get("id", 0)
            for tr in division.get("teamRecords", []):
                tid = tr.get("team", {}).get("id")
                abbr = MLB_TEAM_ID_MAP.get(tid)
                if abbr:
                    w = int(tr.get("wins", 0))
                    l = int(tr.get("losses", 0))
                    all_teams.append((abbr, w, l, lg_id))
        # Sort by wins desc for league ranking
        all_sorted = sorted(all_teams, key=lambda x: (-x[1], x[2]))
        _lg_rank = {t[0]: i + 1 for i, t in enumerate(all_sorted)}

        result: dict[str, dict] = {}
        for division in data.get("records", []):
            lg_id = division.get("league", {}).get("id", 0)
            for tr in division.get("teamRecords", []):
                tid = tr.get("team", {}).get("id")
                abbr = MLB_TEAM_ID_MAP.get(tid)
                if not abbr:
                    continue
                div_rank = tr.get("divisionRank", "?")
                gb = tr.get("divisionGamesBack", "-")
                w = int(tr.get("wins", 0))
                l = int(tr.get("losses", 0))
                # Determine division name from league + rank position
                div_name = "?"
                for dk, dv in _DIV_NAMES.get(lg_id, {}).items():
                    # We don't have direct division ID mapping, use the division object
                    pass
                result[abbr] = {
                    "wins": w, "losses": l,
                    "div_rank": div_rank, "gb": gb,
                    "league_rank": _lg_rank.get(abbr, 0),
                    "league_id": lg_id,
                }
        return result
    except Exception:
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_2026_team_stats(team_id: int) -> pd.DataFrame:
    """Fetch 2026 roster stats from MLB Stats API. Cached 24 hours.

    Returns DataFrame with player_id, name, and key batting/pitching stats.
    """
    if not _requests_available:
        return pd.DataFrame()
    try:
        url = (f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
               f"?rosterType=active&season=2026&hydrate=person(stats(type=season,season=2026))")
        resp = _requests.get(url, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        rows = []
        for entry in data.get("roster", []):
            person = entry.get("person", {})
            pid = person.get("id")
            name = person.get("fullName", "")
            pos = entry.get("position", {}).get("abbreviation", "")
            # Extract stats
            stats_groups = person.get("stats", [])
            batting = {}
            pitching = {}
            for sg in stats_groups:
                if sg.get("type", {}).get("displayName") == "season":
                    for split in sg.get("splits", []):
                        stat = split.get("stat", {})
                        if "era" in stat:
                            pitching = stat
                        elif "avg" in stat:
                            batting = stat
            rows.append({
                "player_id": pid, "name": name, "pos_2026": pos,
                "g_2026": batting.get("gamesPlayed") or pitching.get("gamesPlayed"),
                "avg_2026": batting.get("avg"), "hr_2026": batting.get("homeRuns"),
                "rbi_2026": batting.get("rbi"), "obp_2026": batting.get("obp"),
                "era_2026": pitching.get("era"), "ip_2026": pitching.get("inningsPitched"),
                "whip_2026": pitching.get("whip"), "so_2026": pitching.get("strikeOuts"),
                "w_2026": pitching.get("wins"), "l_2026": pitching.get("losses"),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()
