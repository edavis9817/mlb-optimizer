"""
projections.py
--------------
Builds weighted multi-year WAR projections from the raw salary/WAR CSV.

Public API
----------
make_projections(raw_df, config) -> pd.DataFrame
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Position-group mapping
# ---------------------------------------------------------------------------
_POS_GROUP_MAP = {
    "C":   "C",
    "1B":  "CI",
    "3B":  "CI",
    "IF":  "CI",
    "2B":  "MI",
    "SS":  "MI",
    "LF":  "OF",        # corner outfielder
    "CF":  "CF",        # center fielder — separate group (premium / flexible)
    "RF":  "OF",        # corner outfielder
    "OF":  "OF",        # generic OF listing → treat as corner
    "SP":  "SP",
    "RP":  "RP",
    "TWP": "SP",
    "DH":  "DH",
}

# Slots each pos_group is eligible to fill
# CF-capable players can play all three OF spots; corner OF is LF/RF only.
_ELIGIBLE_SLOTS_MAP = {
    "C":   ["C"],
    "CI":  ["1B", "3B"],
    "MI":  ["2B", "SS"],
    "CF":  ["CF", "LF", "RF"],   # CF player can fill any OF slot
    "OF":  ["LF", "RF"],          # corner-only player cannot play CF
    "SP":  ["SP"],
    "RP":  ["RP"],
    "DH":  ["DH"],
}


def _assign_pos_group(pos: str) -> str:
    if pd.isna(pos):
        return "UNK"
    return _POS_GROUP_MAP.get(str(pos).strip(), "UNK")


def _eligible_slots(pos_group: str) -> list:
    return _ELIGIBLE_SLOTS_MAP.get(pos_group, [])


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def make_projections(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Parameters
    ----------
    raw_df  : DataFrame loaded from raw_salary_war_path (all years)
    config  : full config dict

    Returns
    -------
    projected_df : one row per player with projection columns
    """
    proj_weights: dict = {int(k): float(v) for k, v in config["projection_weights"].items()}
    target_season: int = int(config["season"])
    clip_neg: bool = bool(config["clip_negative_war"])
    # market_mode: "open" = FA + Arb only (default); "all" = include Pre-Arb
    market_mode: str = str(config.get("market_mode", "open")).lower()
    min_war: float = float(config["min_war_threshold"])
    max_yrs: int   = int(config["max_contract_years"])

    # ------------------------------------------------------------------
    # 1. Normalise column names (strip whitespace)
    # ------------------------------------------------------------------
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]

    # ------------------------------------------------------------------
    # 2. Filter to seasons we care about
    # ------------------------------------------------------------------
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df[df["Year"].isin(proj_weights.keys())].copy()

    # ------------------------------------------------------------------
    # 3. Build cost_raw = Salary if present else AAV_Clean
    # ------------------------------------------------------------------
    for col in ("Salary", "AAV_Clean"):
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cost_raw"] = df["Salary"].fillna(df["AAV_Clean"])
    # Convert dollars → millions
    df["cost_raw_M"] = df["cost_raw"] / 1_000_000

    # ------------------------------------------------------------------
    # 4. Deduplicate traded players: keep highest-salary row per (Player, Year)
    # ------------------------------------------------------------------
    df = (
        df.sort_values("cost_raw_M", ascending=False, kind="mergesort")
          .drop_duplicates(["Player", "Year"], keep="first")
          .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 5. Ensure numeric columns
    # ------------------------------------------------------------------
    df["WAR_Total"] = pd.to_numeric(df.get("WAR_Total", np.nan), errors="coerce").fillna(0.0)
    for col in ("IP", "PA", "GS", "SV", "G"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    # ------------------------------------------------------------------
    # 6. Weighted WAR + workload per player
    # ------------------------------------------------------------------
    records = []
    for player, grp in df.groupby("Player", sort=True):
        grp = grp.copy()
        # weight vector aligned to available years
        weights = np.array([proj_weights.get(int(y), 0.0) for y in grp["Year"]])
        total_w = weights.sum()
        if total_w == 0:
            continue

        proj_war = float((weights * grp["WAR_Total"].values).sum() / total_w)

        # WAR std: actual std across years; floor 0.5 for single-year players
        if len(grp) > 1:
            war_sd = float(np.std(grp["WAR_Total"].values, ddof=1))
            war_sd = max(war_sd, 0.3)
        else:
            war_sd = 0.5

        # Cost: prefer target-season salary, then most-recent
        target_row = grp[grp["Year"] == target_season]
        if not target_row.empty and not pd.isna(target_row.iloc[0]["cost_raw_M"]):
            proj_cost_M = float(target_row.iloc[0]["cost_raw_M"])
        else:
            recent = grp.sort_values("Year", ascending=False, kind="mergesort").iloc[0]
            proj_cost_M = float(recent["cost_raw_M"]) if not pd.isna(recent["cost_raw_M"]) else 1.0

        # Clip free cost floor
        proj_cost_M = max(proj_cost_M, 0.7)   # MLB minimum ~$720k

        # Weighted workload averages (pitchers: IP, hitters: PA)
        ip_vals = grp["IP"].values
        proj_ip  = float((weights * ip_vals).sum() / total_w)

        pa_vals = grp["PA"].values
        proj_pa  = float((weights * pa_vals).sum() / total_w)

        # Position info from most recent row
        # Support both "Position" (actual CSV) and legacy "Pos" column name
        recent_row = grp.sort_values("Year", ascending=False, kind="mergesort").iloc[0]
        pos_col = next((c for c in ("Position", "Pos") if c in grp.columns), None)
        pos_raw = str(recent_row[pos_col]) if pos_col else "UNK"
        pos_group = _assign_pos_group(pos_raw)
        eligible = _eligible_slots(pos_group)

        # Stage / contract metadata
        # Support both "Stage_Clean" (actual CSV) and legacy "Stage" column name
        stage_col = next((c for c in ("Stage_Clean", "Stage") if c in grp.columns), None)
        stage_raw = str(recent_row[stage_col]) if stage_col else "FA"
        # Normalise long-form stage names to short codes
        _STAGE_MAP = {
            "Free Agent":       "FA",
            "Arbitration":      "Arb",
            "Pre-Arbitration":  "Pre-Arb",
        }
        stage = _STAGE_MAP.get(stage_raw, stage_raw)

        # Contract years: support "Contract Length" or "Contract_Years_Left"
        cyl_col = next((c for c in ("Contract_Years_Left", "Contract Length") if c in grp.columns), None)
        contract_years_left = recent_row[cyl_col] if cyl_col else np.nan
        if pd.isna(contract_years_left):
            contract_years_left = np.nan
        else:
            contract_years_left = float(contract_years_left)

        age = recent_row.get("Age", np.nan)
        age = float(age) if not pd.isna(age) else np.nan

        records.append({
            "Player":              player,
            "sub_pos":             pos_raw,           # raw position (CF, SS, LF, etc.)
            "pos_group":           pos_group,
            "eligible_slots":      eligible,
            "age":                 age,
            "proj_WAR":            proj_war,
            "proj_WAR_sd":         war_sd,
            "proj_cost_M":         proj_cost_M,
            "proj_ip":             proj_ip,           # weighted avg innings pitched
            "proj_pa":             proj_pa,           # weighted avg plate appearances
            "stage":               stage,
            "contract_years_left": contract_years_left,
        })

    proj_df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 7. Optional filters
    # ------------------------------------------------------------------
    if clip_neg:
        proj_df["proj_WAR"] = proj_df["proj_WAR"].clip(lower=0.0)

    # Market mode filter:
    #   "fa"   = free agents only (stage == "FA")  — highest cost, most realistic budget spend
    #   "open" = FA + Arb (exclude Pre-Arb only)  — includes arbitration players
    #   "all"  = no filter (includes Pre-Arb)
    if market_mode == "fa":
        proj_df = proj_df[proj_df["stage"] == "FA"].copy()
    elif market_mode == "open":
        proj_df = proj_df[proj_df["stage"] != "Pre-Arb"].copy()

    proj_df = proj_df[proj_df["proj_WAR"] >= min_war].copy()

    if max_yrs < 99:
        proj_df = proj_df[
            proj_df["contract_years_left"].isna() | (proj_df["contract_years_left"] <= max_yrs)
        ].copy()

    # Drop unknown positions (no eligible slots)
    proj_df = proj_df[proj_df["pos_group"] != "UNK"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 8. Write processed output
    # ------------------------------------------------------------------
    processed_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed"
    )
    os.makedirs(processed_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(processed_dir, f"projections_{ts}.csv")
    proj_df.to_csv(out_path, index=False)

    return proj_df
