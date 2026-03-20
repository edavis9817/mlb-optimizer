"""
archetypes.py
-------------
Stage + WAR-tier archetype system.  Deterministic — no clustering.

archetype_id = f"{pos_group}_{stage}_{war_tier}"

  pos_group : C, CI, MI, CF, OF, SP, RP, DH
              CF = center-field capable (flexible, premium)
              OF = corner outfield only (LF/RF)
  stage     : FA (free agent), Arb (arbitration), Pre-Arb (pre-arbitration)
  war_tier  : Elite (>=4 WAR), Solid (2-4), Average (0.5-2), Depth (<0.5)

Slot eligibility:
  CF archetypes → eligible for CF, LF, RF  (true CF player can play anywhere)
  OF archetypes → eligible for LF, RF only (corner-only player cannot play CF)

Role column (display / diagnostics — not used in optimization directly):
  SP:  Ace (Elite), Rotation (Solid), Depth (Average/Depth)
  RP:  Leverage (Solid+), Middle (Average), Mop (Depth)
  C:   Defensive, Offensive
  Others: derived from WAR tier

Stage carries a WAR uncertainty multiplier:
  FA      1.00 — proven track record, most stable
  Arb     1.25 — 2-3 seasons of data, moderate uncertainty
  Pre-Arb 1.50 — limited data, high bust / breakout variance

Public API
----------
build_archetype_definitions(projected_df, war_player_cap, min_players) -> archetype_df
assign_archetypes(projected_df)                                         -> projected_df  (with archetype_id column)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# WAR tiers
# ---------------------------------------------------------------------------

def _war_tier(war: float) -> str:
    if war >= 4.0:
        return "Elite"
    elif war >= 2.0:
        return "Solid"
    elif war >= 0.5:
        return "Average"
    else:
        return "Depth"


# ---------------------------------------------------------------------------
# Stage SD multipliers
# ---------------------------------------------------------------------------
_STAGE_SD_MULTIPLIER: dict[str, float] = {
    "FA":      1.00,   # Proven free agents — most stable
    "Arb":     1.25,   # Arbitration-eligible — moderate uncertainty
    "Pre-Arb": 1.50,   # Pre-arbitration — high bust / breakout variance
}


# ---------------------------------------------------------------------------
# Slot-eligibility rules
# CF-capable players can fill any OF slot; corner OF cannot play CF.
# ---------------------------------------------------------------------------
_ELIGIBLE_SLOTS: dict[str, list[str]] = {
    "C":  ["C"],
    "CI": ["1B", "3B"],
    "MI": ["2B", "SS"],
    "CF": ["CF", "LF", "RF"],   # CF player is flexible — fills any OF slot
    "OF": ["LF", "RF"],          # corner OF cannot play CF
    "SP": ["SP"],
    "RP": ["RP"],
    "DH": ["DH"],
}

# All hitter pos_groups (eligible for DH slot)
_HITTER_POS_GROUPS = {"C", "CI", "MI", "CF", "OF", "DH"}


def _slots_for_archetype(pos_group: str) -> list[str]:
    base = _ELIGIBLE_SLOTS.get(pos_group, [])
    # Any hitter archetype is also eligible for DH
    if pos_group in _HITTER_POS_GROUPS and "DH" not in base:
        base = base + ["DH"]
    return base


def _archetype_id(pos_group: str, stage: str, war: float) -> str:
    return f"{pos_group}_{stage}_{_war_tier(war)}"


# ---------------------------------------------------------------------------
# Role assignment (display / diagnostics label)
# ---------------------------------------------------------------------------

def _assign_role(pos_group: str, war_tier: str, stage: str) -> str:
    """Return a human-readable role for display in diagnostics."""
    if pos_group == "SP":
        return {"Elite": "Ace", "Solid": "Rotation", "Average": "Depth", "Depth": "Depth"}.get(war_tier, "Depth")
    if pos_group == "RP":
        return {"Elite": "Leverage", "Solid": "Leverage", "Average": "Middle", "Depth": "Mop"}.get(war_tier, "Mop")
    if pos_group == "C":
        return "Defensive" if (stage in ("Pre-Arb", "Arb") and war_tier in ("Elite", "Solid")) else "Offensive"
    if pos_group == "MI":
        return "Premium" if war_tier in ("Elite", "Solid") else "Utility"
    if pos_group == "CF":
        return "Premium" if war_tier in ("Elite", "Solid") else "Backup"
    return war_tier


# ---------------------------------------------------------------------------
# assign_archetypes
# ---------------------------------------------------------------------------

def assign_archetypes(projected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an `archetype_id` column to *projected_df*.
    Returns a copy — does not mutate the input.
    """
    df = projected_df.copy()
    df["war_tier"] = df["proj_WAR"].apply(_war_tier)
    df["archetype_id"] = df.apply(
        lambda r: _archetype_id(r["pos_group"], r["stage"], r["proj_WAR"]),
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# build_archetype_definitions
# ---------------------------------------------------------------------------

def build_archetype_definitions(
    projected_df: pd.DataFrame,
    war_player_cap: float = 6.5,
    min_players: int = 5,
) -> pd.DataFrame:
    """
    Aggregate players into one-row-per-archetype summary.

    Parameters
    ----------
    projected_df   : output of make_projections()
    war_player_cap : cap each player's WAR contribution to the archetype mean
                     at this value (prevents single-outlier inflation).
                     Set to np.inf to disable.  Default 6.5.
    min_players    : drop archetypes with fewer than this many real players.
                     Prevents single-player outlier buckets from distorting the model.
                     Default 5.

    Columns returned
    ----------------
    archetype_id, pos_group, stage, war_tier, role, eligible_slots,
    war_mean, war_sd, cost_mean, cost_sd, ip_mean, pa_mean, n_players
    """
    df = assign_archetypes(projected_df)

    records = []
    for arch_id, grp in df.groupby("archetype_id", sort=True):
        row0      = grp.iloc[0]
        pos_group = row0["pos_group"]
        stage     = row0["stage"]
        war_tier  = row0["war_tier"]
        eligible  = _slots_for_archetype(pos_group)
        role      = _assign_role(pos_group, war_tier, stage)

        # Cap per-player WAR before computing archetype mean
        war_vals_raw = grp["proj_WAR"].values
        war_vals     = np.minimum(war_vals_raw, war_player_cap)
        cost_vals    = grp["proj_cost_M"].values
        war_sd_vals  = grp["proj_WAR_sd"].values

        # Workload means
        ip_vals = grp["proj_ip"].values  if "proj_ip" in grp.columns else np.zeros(len(grp))
        pa_vals = grp["proj_pa"].values  if "proj_pa" in grp.columns else np.zeros(len(grp))

        # Stage SD multiplier: pre-arb players carry more real-world variance
        sd_mult = _STAGE_SD_MULTIPLIER.get(stage, 1.0)

        # Pooled SD = mean of individual SDs (avoids zero when n==1), then scaled
        pooled_war_sd = float(np.mean(war_sd_vals)) * sd_mult
        if len(grp) > 1:
            between_sd    = float(np.std(war_vals, ddof=1))
            pooled_war_sd = max(pooled_war_sd, between_sd * sd_mult)

        records.append({
            "archetype_id":   arch_id,
            "pos_group":      pos_group,
            "stage":          stage,
            "war_tier":       war_tier,
            "role":           role,
            "eligible_slots": eligible,
            "war_mean":       float(np.mean(war_vals)),
            "war_sd":         pooled_war_sd,
            "cost_mean":      float(np.mean(cost_vals)),
            "cost_sd":        float(np.std(cost_vals, ddof=1)) if len(grp) > 1 else 0.0,
            "ip_mean":        float(np.mean(ip_vals)),
            "pa_mean":        float(np.mean(pa_vals)),
            "n_players":      len(grp),
        })

    arch_df = pd.DataFrame(records)

    # Drop sparse archetypes (statistical noise, not real market buckets)
    if min_players > 1:
        arch_df = arch_df[arch_df["n_players"] >= min_players].reset_index(drop=True)

    return arch_df
