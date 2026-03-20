"""
diagnostics.py
--------------
Budget frontier and marginal-value analysis.

Public API
----------
budget_frontier(archetype_df, config, budget_range=None) -> frontier_df
marginal_analysis(roster_df, archetype_df, config)       -> (upgrades_df, cuts_df)
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .optimizer   import run_optimizer
from .simulation  import monte_carlo


# ---------------------------------------------------------------------------
# Budget frontier
# ---------------------------------------------------------------------------

def budget_frontier(
    archetype_df: pd.DataFrame,
    config: dict,
    budget_range: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run the optimizer at N budget points and return a summary table.

    Parameters
    ----------
    archetype_df : from build_archetype_definitions()
    config       : base config (budget_M will be overridden per step)
    budget_range : list of budget values in $M; default 10 pts from 60 to 300

    Returns
    -------
    frontier_df : columns = budget_M, expected_wins, archetype_mix (dict as str),
                  total_cost_M, status
    """
    if budget_range is None:
        budget_range = list(np.linspace(60, 300, 10))

    rows = []
    for b in budget_range:
        cfg = copy.deepcopy(config)
        cfg["budget_M"] = float(b)
        result = run_optimizer(archetype_df, cfg)
        if result.roster_df.empty:
            rows.append({
                "budget_M":       b,
                "expected_wins":  float("nan"),
                "archetype_mix":  "{}",
                "total_cost_M":   float("nan"),
                "status":         result.status,
            })
            continue

        sim = monte_carlo(result.roster_df, cfg)
        total_cost = float(result.roster_df["cost_mean"].sum())

        rows.append({
            "budget_M":       b,
            "expected_wins":  sim.mean_wins,
            "p10":            sim.p10,
            "p90":            sim.p90,
            "playoff_odds":   sim.playoff_odds,
            "archetype_mix":  str(result.archetype_mix),
            "total_cost_M":   total_cost,
            "status":         result.status,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Marginal analysis
# ---------------------------------------------------------------------------

_WAR_TIERS_ORDERED  = ["Depth", "Average", "Solid", "Elite"]
_COST_TIERS_ORDERED = ["Cheap", "Mid", "Premium"]


def _next_tier_up(war_tier: str, cost_tier: str) -> tuple[str, str]:
    wi = _WAR_TIERS_ORDERED.index(war_tier)  if war_tier  in _WAR_TIERS_ORDERED  else -1
    ci = _COST_TIERS_ORDERED.index(cost_tier) if cost_tier in _COST_TIERS_ORDERED else -1
    # Upgrade WAR tier if possible, otherwise upgrade cost tier
    if wi < len(_WAR_TIERS_ORDERED) - 1:
        return _WAR_TIERS_ORDERED[wi + 1], cost_tier
    if ci < len(_COST_TIERS_ORDERED) - 1:
        return war_tier, _COST_TIERS_ORDERED[ci + 1]
    return war_tier, cost_tier  # already at top


def _next_tier_down(war_tier: str, cost_tier: str) -> tuple[str, str]:
    wi = _WAR_TIERS_ORDERED.index(war_tier)  if war_tier  in _WAR_TIERS_ORDERED  else 0
    ci = _COST_TIERS_ORDERED.index(cost_tier) if cost_tier in _COST_TIERS_ORDERED else 0
    if wi > 0:
        return _WAR_TIERS_ORDERED[wi - 1], cost_tier
    if ci > 0:
        return war_tier, _COST_TIERS_ORDERED[ci - 1]
    return war_tier, cost_tier


def marginal_analysis(
    roster_df: pd.DataFrame,
    archetype_df: pd.DataFrame,
    config: dict,
    upgrade_delta_M: float = 5.0,
    cut_delta_M: float     = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each slot in roster_df, evaluate swapping the archetype to the
    next tier up (upgrade) or down (cut) and compute Δwins / Δcost.

    Returns
    -------
    upgrades_df : top 10 upgrades sorted by delta_wins / delta_cost_M
    cuts_df     : top 10 cuts sorted by delta_wins / delta_cost_M (negative wins saved)
    """
    if roster_df.empty:
        empty = pd.DataFrame(columns=[
            "slot", "current_archetype", "alt_archetype",
            "delta_war", "delta_cost_M", "delta_wins_per_M"
        ])
        return empty, empty

    arch_lookup = archetype_df.set_index("archetype_id")

    # Baseline sim
    base_sim = monte_carlo(roster_df, config)
    base_wins = base_sim.mean_wins

    upgrade_rows = []
    cut_rows     = []

    for _, slot_row in roster_df.iterrows():
        slot      = slot_row["slot"]
        curr_arch = slot_row["archetype_id"]

        if curr_arch not in arch_lookup.index:
            continue

        curr_rec  = arch_lookup.loc[curr_arch]
        war_tier  = curr_rec.get("war_tier",  "Average")
        cost_tier = curr_rec.get("cost_tier", "Mid")
        pos_group = curr_rec["pos_group"]

        # --- Upgrade ---
        new_wt_u, new_ct_u = _next_tier_up(war_tier, cost_tier)
        alt_arch_u = f"{pos_group}_{new_wt_u}_{new_ct_u}"
        if alt_arch_u != curr_arch and alt_arch_u in arch_lookup.index:
            alt_rec = arch_lookup.loc[alt_arch_u]
            swap_roster = roster_df.copy()
            swap_roster.loc[swap_roster["slot"] == slot, "war_mean"]  = alt_rec["war_mean"]
            swap_roster.loc[swap_roster["slot"] == slot, "war_sd"]    = alt_rec["war_sd"]
            swap_roster.loc[swap_roster["slot"] == slot, "cost_mean"] = alt_rec["cost_mean"]
            swap_roster.loc[swap_roster["slot"] == slot, "archetype_id"] = alt_arch_u
            new_sim   = monte_carlo(swap_roster, config)
            delta_war  = float(alt_rec["war_mean"]) - float(curr_rec["war_mean"])
            delta_cost = float(alt_rec["cost_mean"]) - float(curr_rec["cost_mean"])
            delta_wins = new_sim.mean_wins - base_wins
            eff        = delta_wins / delta_cost if delta_cost > 0.01 else float("nan")
            upgrade_rows.append({
                "slot":              slot,
                "current_archetype": curr_arch,
                "alt_archetype":     alt_arch_u,
                "delta_war":         round(delta_war, 2),
                "delta_cost_M":      round(delta_cost, 1),
                "delta_wins":        round(delta_wins, 2),
                "delta_wins_per_M":  round(eff, 3) if not np.isnan(eff) else None,
            })

        # --- Cut (downgrade) ---
        new_wt_d, new_ct_d = _next_tier_down(war_tier, cost_tier)
        alt_arch_d = f"{pos_group}_{new_wt_d}_{new_ct_d}"
        if alt_arch_d != curr_arch and alt_arch_d in arch_lookup.index:
            alt_rec = arch_lookup.loc[alt_arch_d]
            swap_roster = roster_df.copy()
            swap_roster.loc[swap_roster["slot"] == slot, "war_mean"]  = alt_rec["war_mean"]
            swap_roster.loc[swap_roster["slot"] == slot, "war_sd"]    = alt_rec["war_sd"]
            swap_roster.loc[swap_roster["slot"] == slot, "cost_mean"] = alt_rec["cost_mean"]
            swap_roster.loc[swap_roster["slot"] == slot, "archetype_id"] = alt_arch_d
            new_sim   = monte_carlo(swap_roster, config)
            delta_war  = float(alt_rec["war_mean"]) - float(curr_rec["war_mean"])
            delta_cost = float(alt_rec["cost_mean"]) - float(curr_rec["cost_mean"])
            delta_wins = new_sim.mean_wins - base_wins
            savings    = -delta_cost  # positive = money saved
            eff        = delta_wins / savings if savings > 0.01 else float("nan")
            cut_rows.append({
                "slot":              slot,
                "current_archetype": curr_arch,
                "alt_archetype":     alt_arch_d,
                "delta_war":         round(delta_war, 2),
                "delta_cost_M":      round(delta_cost, 1),
                "savings_M":         round(savings, 1),
                "delta_wins":        round(delta_wins, 2),
                "delta_wins_per_M_saved": round(eff, 3) if not np.isnan(eff) else None,
            })

    upgrades_df = (
        pd.DataFrame(upgrade_rows)
          .sort_values("delta_wins_per_M", ascending=False, kind="mergesort")
          .head(10)
          .reset_index(drop=True)
        if upgrade_rows else
        pd.DataFrame(columns=["slot","current_archetype","alt_archetype",
                               "delta_war","delta_cost_M","delta_wins","delta_wins_per_M"])
    )

    cuts_df = (
        pd.DataFrame(cut_rows)
          .sort_values("savings_M", ascending=False, kind="mergesort")
          .head(10)
          .reset_index(drop=True)
        if cut_rows else
        pd.DataFrame(columns=["slot","current_archetype","alt_archetype",
                               "delta_war","delta_cost_M","savings_M",
                               "delta_wins","delta_wins_per_M_saved"])
    )

    return upgrades_df, cuts_df
