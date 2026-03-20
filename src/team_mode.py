"""
team_mode.py
------------
Team-based offseason analysis utilities.

Given a selected team, this module:
  1. Identifies which players have committed contracts into 2026+ (locked),
     which are arbitration-controlled (arb), and which are expiring.
  2. Calculates historical payrolls to inform a realistic 2026 budget.
  3. Determines which roster slots still need to be filled.
  4. Returns an offseason scenario dict the Streamlit UI can use to
     run the existing optimizer on just the open slots.

Public API
----------
get_all_teams(data_dir)                              -> list[str]
get_team_payroll_history(data_dir)                   -> pd.DataFrame
get_team_roster_status(data_dir, team, combined_df)  -> pd.DataFrame
build_offseason_scenario(...)                         -> dict
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Position-group & slot maps (mirrors projections.py)
# ---------------------------------------------------------------------------
_POS_GROUP_MAP = {
    "C":   "C",
    "1B":  "CI",
    "3B":  "CI",
    "IF":  "CI",
    "2B":  "MI",
    "SS":  "MI",
    "LF":  "OF",
    "CF":  "CF",
    "RF":  "OF",
    "OF":  "OF",
    "SP":  "SP",
    "RP":  "RP",
    "TWP": "SP",
    "DH":  "DH",
    "P":   "SP",   # generic pitcher in lookup file
}

_ELIGIBLE_SLOTS_MAP = {
    "C":   ["C"],
    "CI":  ["1B", "3B"],
    "MI":  ["2B", "SS"],
    "CF":  ["CF", "LF", "RF"],
    "OF":  ["LF", "RF"],
    "SP":  ["SP"],
    "RP":  ["RP"],
    "DH":  ["DH"],
}

# Primary (most specific) slot to assign each pos_group to
_PRIMARY_SLOT = {
    "C":  "C",
    "CI": "1B",
    "MI": "2B",
    "CF": "CF",
    "OF": "LF",
    "SP": "SP",
    "RP": "RP",
    "DH": "DH",
}

# Year-specific individual data files
_YEAR_FILES = {
    2021: "2021mlbshared.csv",
    2022: "2022mlbshared.csv",
    2023: "2023mlbshared.csv",
    2024: "2024mlbshared.csv",
    2025: "2025mlbshared.csv",
}

_CURRENT_SEASON = 2025
_NEXT_SEASON    = 2026

# Estimated arb raise multiplier per arb year (conservative)
_ARB_RAISE = {"ARB 1": 1.20, "ARB 2": 1.25, "ARB 3": 1.30}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _parse_dollar(val) -> float | None:
    """Parse a year-column value like '$14,000,000' → 14.0 (millions).
    Returns None for 'ARB X', 'FREE AGENT', NaN, 'TBD', etc."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.startswith("$"):
        try:
            return float(s.replace("$", "").replace(",", "")) / 1_000_000
        except ValueError:
            return None
    return None


def _classify_2026(val) -> str:
    """Classify a player's 2026 column as 'locked', 'arb', or 'expiring'."""
    if pd.isna(val):
        return "expiring"
    s = str(val).strip().upper()
    if s.startswith("$"):
        return "locked"
    if "ARB" in s:
        return "arb"
    return "expiring"   # FREE AGENT, TBD, blank, etc.


def _arb_key(val) -> str | None:
    """Extract arb tier string ('ARB 1' / 'ARB 2' / 'ARB 3') or None."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    for k in _ARB_RAISE:
        if k in s:
            return k
    return None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_all_teams(data_dir: str) -> list[str]:
    """Return sorted list of all teams present in the 2025 data file."""
    path = os.path.join(data_dir, _YEAR_FILES[_CURRENT_SEASON])
    df = pd.read_csv(path, low_memory=False, usecols=["Team"])
    return sorted(df["Team"].dropna().unique().tolist())


def get_team_payroll_history(data_dir: str) -> pd.DataFrame:
    """
    Compute team payrolls for each available season by summing committed
    dollar-amount salaries from each year's individual CSV.

    Returns
    -------
    DataFrame with columns: Team, Year, payroll_M
    """
    records = []
    for year, fname in _YEAR_FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        year_col = str(year)
        if year_col not in df.columns:
            continue
        df["_sal_M"] = df[year_col].apply(_parse_dollar)
        # Deduplicate traded players (keep highest salary row per player)
        df = (
            df.sort_values("_sal_M", ascending=False, na_position="last", kind="mergesort")
              .drop_duplicates(subset=["Player"], keep="first")
              .reset_index(drop=True)
        )
        team_pay = (
            df.dropna(subset=["_sal_M"])
              .groupby("Team")["_sal_M"]
              .sum()
              .reset_index()
              .rename(columns={"_sal_M": "payroll_M"})
        )
        team_pay["Year"] = year
        records.append(team_pay)

    if not records:
        return pd.DataFrame(columns=["Team", "Year", "payroll_M"])
    return pd.concat(records, ignore_index=True)


def get_team_roster_status(
    data_dir: str,
    team: str,
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return all players on the selected team's 2025 roster, annotated with
    their 2026 contract status and projected WAR.

    Uses the 2025 individual CSV for contract/salary columns and the
    combined multi-year DataFrame for position and projected WAR.

    Parameters
    ----------
    data_dir    : absolute path to the Data/ directory
    team        : team abbreviation (e.g. 'ARI')
    combined_df : the full multi-year raw DataFrame (used for projections)

    Returns
    -------
    DataFrame with columns:
        Player, pos_group, eligible_slots, age,
        war_2025, sal_2025_M, sal_2026_M, arb_tier,
        contract_status, contract_2026_raw, Contract,
        proj_arb_cost_M
    """
    # --- Load 2025 individual file (has contract/year columns) ---
    ind_path = os.path.join(data_dir, _YEAR_FILES[_CURRENT_SEASON])
    ind = pd.read_csv(ind_path, low_memory=False)
    ind.columns = [c.strip() for c in ind.columns]
    team_ind = ind[ind["Team"] == team].copy()

    # --- Get positions from combined file (2025 rows) ---
    comb_2025 = combined_df[combined_df["Year"] == _CURRENT_SEASON][
        ["Player", "Position", "WAR_Total", "Stage_Clean"]
    ].copy()

    # Merge to bring in position data
    merged = team_ind.merge(comb_2025, on="Player", how="left")

    # --- Parse salary/contract columns ---
    merged["sal_2025_M"]         = merged["2025"].apply(_parse_dollar)
    merged["sal_2026_M"]         = merged["2026"].apply(_parse_dollar)
    merged["contract_2026_raw"]  = merged["2026"].fillna("").astype(str)
    merged["contract_status"]    = merged["2026"].apply(_classify_2026)
    merged["arb_tier"]           = merged["2026"].apply(_arb_key)

    # --- Projected arb cost for 2026 ---
    def _proj_arb_cost(row) -> float | None:
        if row["contract_status"] != "arb":
            return None
        base = row["sal_2025_M"]
        if pd.isna(base) or base <= 0:
            base = 0.7
        mult = _ARB_RAISE.get(row["arb_tier"], 1.20)
        return round(base * mult, 2)

    merged["proj_arb_cost_M"] = merged.apply(_proj_arb_cost, axis=1)

    # --- Position group ---
    merged["pos_raw"] = merged["Position"].fillna("UNK").astype(str).str.strip()
    merged["pos_group"]      = merged["pos_raw"].map(lambda p: _POS_GROUP_MAP.get(p, "UNK"))
    merged["eligible_slots"] = merged["pos_group"].map(
        lambda g: _ELIGIBLE_SLOTS_MAP.get(g, [])
    )

    # --- WAR: prefer combined file WAR_Total, fall back to ind file ---
    if "WAR_Total" in merged.columns:
        merged["war_2025"] = pd.to_numeric(merged["WAR_Total"], errors="coerce")
    elif "WAR" in merged.columns:
        merged["war_2025"] = pd.to_numeric(merged["WAR"], errors="coerce")
    else:
        merged["war_2025"] = np.nan

    # --- Age ---
    merged["age"] = pd.to_numeric(merged.get("Age", np.nan), errors="coerce")

    # --- Contract text ---
    if "Contract" not in merged.columns:
        merged["Contract"] = ""

    keep = [
        "Player", "pos_group", "eligible_slots", "age",
        "war_2025", "sal_2025_M", "sal_2026_M", "proj_arb_cost_M",
        "arb_tier", "contract_status", "contract_2026_raw", "Contract",
    ]
    keep = [c for c in keep if c in merged.columns]
    result = merged[keep].copy()

    # Drop UNK positions (bench coaches, etc.)
    result = result[result["pos_group"] != "UNK"].reset_index(drop=True)

    return result


def assign_locked_to_slots(
    locked_df: pd.DataFrame,
    base_roster_slots: dict,
    include_arb: bool = True,
) -> tuple[pd.DataFrame, dict, float]:
    """
    Greedily assign locked (and optionally arb) players to roster slots.
    Uses each player's primary position slot first; if taken, tries
    remaining eligible_slots.

    Parameters
    ----------
    locked_df         : rows from get_team_roster_status where
                        contract_status in ('locked', 'arb')
    base_roster_slots : the full roster slot counts from config
    include_arb       : if True, arb players are counted as filling slots

    Returns
    -------
    assignments_df  : locked_df with added 'assigned_slot' column
    remaining_slots : dict of unfilled slot → count
    committed_M     : total committed 2026 payroll for assigned players
    """
    remaining = dict(base_roster_slots)
    committed_M = 0.0
    assignments = []

    statuses = ["locked"]
    if include_arb:
        statuses.append("arb")

    # Sort: locked first, then arb; within each, higher WAR first
    df = locked_df[locked_df["contract_status"].isin(statuses)].copy()
    df = df.sort_values(
        ["contract_status", "war_2025"],
        ascending=[True, False],
        kind="mergesort",
    )

    for _, row in df.iterrows():
        eligible = row["eligible_slots"] if isinstance(row["eligible_slots"], list) else []
        # Try primary slot, then others in eligible list
        primary = _PRIMARY_SLOT.get(row["pos_group"])
        ordered = []
        if primary and primary in eligible:
            ordered.append(primary)
        ordered += [s for s in eligible if s != primary]
        # Also try BENCH as a fallback if no position slot remains
        ordered.append("BENCH")

        assigned = None
        for slot in ordered:
            if remaining.get(slot, 0) > 0:
                remaining[slot] -= 1
                if remaining[slot] == 0:
                    del remaining[slot]
                assigned = slot
                break

        row_dict = row.to_dict()
        row_dict["assigned_slot"] = assigned

        # Committed cost
        if row["contract_status"] == "locked" and not pd.isna(row.get("sal_2026_M")):
            committed_M += float(row["sal_2026_M"])
        elif row["contract_status"] == "arb" and not pd.isna(row.get("proj_arb_cost_M")):
            committed_M += float(row["proj_arb_cost_M"])

        assignments.append(row_dict)

    assignments_df = pd.DataFrame(assignments) if assignments else pd.DataFrame()
    return assignments_df, remaining, round(committed_M, 2)


def suggest_non_tenders(
    arb_df: pd.DataFrame,
    market_dpw_M: float = 5.5,
    non_tender_war_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Flag arb players who might not be worth retaining.
    Heuristic: proj_arb_cost_M > market_dpw_M * war_2025 * 0.85
    (arb players get ~85% of market rate at most; below threshold WAR → non-tender).

    Returns arb_df with added 'recommendation' column ('Keep' / 'Non-tender').
    """
    df = arb_df.copy()
    df["market_value_M"] = df["war_2025"].clip(lower=0) * market_dpw_M * 0.85
    df["recommendation"] = df.apply(
        lambda r: (
            "Non-tender"
            if (
                r["war_2025"] < non_tender_war_threshold
                or (
                    not pd.isna(r["proj_arb_cost_M"])
                    and not pd.isna(r["market_value_M"])
                    and r["proj_arb_cost_M"] > r["market_value_M"]
                )
            )
            else "Keep"
        ),
        axis=1,
    )
    return df


def build_offseason_scenario(
    data_dir: str,
    team: str,
    combined_df: pd.DataFrame,
    base_roster_slots: dict,
    market_dpw_M: float = 5.5,
    include_arb: bool = True,
    budget_override_M: float | None = None,
    history_years: int = 3,
    depth_chart_dir: str | None = None,
    include_minors: bool = False,
) -> dict:
    """
    Build a complete offseason scenario for the selected team.

    Returns a dict with keys:
      team                 : str
      roster_status_df     : full DataFrame from get_team_roster_status
      locked_df            : players under contract for 2026
      arb_df               : arb-eligible players (with non-tender recs)
      expiring_df          : players expiring after 2025
      assignments_df       : locked+arb players assigned to slots
      remaining_slots      : {slot: count} to fill via optimizer
      committed_payroll_M  : already-committed 2026 payroll
      historical_avg_M     : team's 3-yr avg payroll
      recommended_budget_M : suggested total budget (user can override)
      available_budget_M   : recommended_budget_M - committed_payroll_M
      depth_chart_df       : merged depth chart + payroll (if depth_chart_dir given)
      depth_total_M        : total payroll from depth chart view
      depth_available_M    : budget minus depth chart total
      minors_df            : AAA/AA players (if include_minors=True)
    """
    # --- Roster status ---
    roster_df = get_team_roster_status(data_dir, team, combined_df)

    locked_df   = roster_df[roster_df["contract_status"] == "locked"].copy()
    arb_df      = roster_df[roster_df["contract_status"] == "arb"].copy()
    expiring_df = roster_df[roster_df["contract_status"] == "expiring"].copy()

    # Non-tender suggestions for arb players
    if not arb_df.empty:
        arb_df = suggest_non_tenders(arb_df, market_dpw_M=market_dpw_M)

    # --- Slot assignments ---
    assignments_df, remaining_slots, committed_M = assign_locked_to_slots(
        roster_df, base_roster_slots, include_arb=include_arb
    )

    # --- Historical payroll ---
    payroll_hist = get_team_payroll_history(data_dir)
    team_hist = payroll_hist[payroll_hist["Team"] == team].copy()
    recent_years = sorted(team_hist["Year"].unique())[-history_years:]
    recent_pay = team_hist[team_hist["Year"].isin(recent_years)]["payroll_M"]
    hist_avg_M = float(recent_pay.mean()) if not recent_pay.empty else 130.0

    # Recommended budget: slightly above historical avg (5% growth)
    rec_budget_M = round(hist_avg_M * 1.05, 1)
    if budget_override_M is not None:
        rec_budget_M = float(budget_override_M)

    available_M = max(round(rec_budget_M - committed_M, 1), 0.0)

    # --- Depth chart integration ---
    depth_chart_df   = pd.DataFrame()
    depth_total_M    = 0.0
    depth_available_M = available_M
    minors_df        = pd.DataFrame()

    minors_40man_df   = pd.DataFrame()
    minors_40man_cost_M = 0.0

    if depth_chart_dir:
        try:
            from src.depth_chart import (
                load_projected_roster,
                load_minors_players,
                merge_depth_with_payroll,
                LEAGUE_MIN_M,
            )
            dc_df = load_projected_roster(depth_chart_dir, team)
            if not dc_df.empty:
                depth_chart_df = merge_depth_with_payroll(dc_df, roster_df)
                depth_total_M  = float(depth_chart_df["sal_2026_M"].sum())

            # Always load minors (AAA + AA) to detect 40-man optioned players.
            # 40-man players count toward spending at league minimum even when
            # the "include minors" display toggle is off.
            _all_minors = load_minors_players(depth_chart_dir, team, levels=("AAA", "AA"))
            if not _all_minors.empty:
                if "on_40_man" in _all_minors.columns:
                    # Exclude players already in the depth-chart projected roster
                    _dc_players = set(depth_chart_df["Player"].tolist()) if not depth_chart_df.empty else set()
                    _all_minors_new = _all_minors[~_all_minors["Player"].isin(_dc_players)]
                    minors_40man_df  = _all_minors_new[_all_minors_new["on_40_man"]].copy()
                    minors_40man_cost_M = round(len(minors_40man_df) * LEAGUE_MIN_M, 2)
                    depth_total_M += minors_40man_cost_M
                    if include_minors:
                        # Show only non-40-man prospects in the display table
                        minors_df = _all_minors_new[~_all_minors_new["on_40_man"]].copy()
                else:
                    if include_minors:
                        minors_df = _all_minors

            depth_available_M = max(round(rec_budget_M - depth_total_M, 1), 0.0)
        except Exception:
            pass

    return {
        "team":                   team,
        "roster_status_df":       roster_df,
        "locked_df":              locked_df,
        "arb_df":                 arb_df,
        "expiring_df":            expiring_df,
        "assignments_df":         assignments_df,
        "remaining_slots":        remaining_slots,
        "committed_payroll_M":    committed_M,
        "historical_avg_M":       round(hist_avg_M, 1),
        "recommended_budget_M":   rec_budget_M,
        "available_budget_M":     available_M,
        "depth_chart_df":         depth_chart_df,
        "depth_total_M":          depth_total_M,
        "depth_available_M":      depth_available_M,
        "minors_df":              minors_df,
        "minors_40man_df":        minors_40man_df,
        "minors_40man_cost_M":    minors_40man_cost_M,
    }
