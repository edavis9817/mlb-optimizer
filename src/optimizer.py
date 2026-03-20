"""
optimizer.py
------------
MILP roster optimizer using PuLP (CBC backend).
Falls back to a greedy heuristic if PuLP is not installed.

Public API
----------
run_optimizer(archetype_df, config) -> OptimizerResult
    .status          : str  ("Optimal" | "Infeasible" | "Greedy-fallback" …)
    .roster_df       : DataFrame — one row per slot filled
    .objective_value : float
    .tight_constraints: list[str]
    .archetype_mix   : dict {archetype_id: count}
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional PuLP import
# ---------------------------------------------------------------------------
try:
    import pulp
    _PULP_AVAILABLE = True
except ImportError:
    _PULP_AVAILABLE = False
    warnings.warn(
        "PuLP not found — optimizer falls back to greedy selection.\n"
        "Install with:  pip install pulp",
        RuntimeWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Pos-group sets for slot eligibility
# CF is now a separate pos_group (center-field capable).
# ---------------------------------------------------------------------------
_HITTER_POS_GROUPS = {"C", "CI", "MI", "CF", "OF", "DH"}
# BENCH accepts utility/corner/middle IF, OF, and CF — but NOT catchers or DH-only.
# Catchers fill their own C slots; DH archetypes are not bench-typical.
_BENCH_POS_GROUPS  = {"CI", "MI", "CF", "OF"}


def _arch_eligible_for_slot(arch_slots: list[str], slot: str) -> bool:
    """Return True if this archetype can fill *slot*."""
    if slot in arch_slots:
        return True
    return False


@dataclass
class OptimizerResult:
    status: str
    roster_df: pd.DataFrame
    objective_value: float
    tight_constraints: list[str] = field(default_factory=list)
    archetype_mix: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coefficient helper
# ---------------------------------------------------------------------------

def _effective_war_coef(
    war_mean: float,
    cost_mean: float,
    war_sd: float,
    config: dict,
) -> float:
    ow  = config["objective_weights"]
    w_w = float(ow.get("wins",         1.0))
    w_s = float(ow.get("surplus",      0.0))
    w_r = float(ow.get("risk_penalty", 0.0))
    dpw = float(config["market_dpw_M"])

    return (w_w + w_s * dpw) * war_mean - w_s * cost_mean - w_r * war_sd


# ---------------------------------------------------------------------------
# MILP — archetype mode
# ---------------------------------------------------------------------------

def _run_milp_archetype(archetype_df: pd.DataFrame, config: dict) -> OptimizerResult:
    slots_cfg: dict = config["roster_slots"]
    budget_M: float = float(config["budget_M"])
    arch_caps: dict = config.get("archetype_caps", {})

    # Expand slots (e.g., SP: 5 → ["SP_0","SP_1","SP_2","SP_3","SP_4"])
    slot_list = []
    for slot_name, count in slots_cfg.items():
        for i in range(int(count)):
            slot_list.append(f"{slot_name}_{i}" if count > 1 else slot_name)

    # Build list of archetypes + compute coefficients
    archs = archetype_df.reset_index(drop=True)
    archs = archs[archs["eligible_slots"].apply(lambda s: len(s) > 0)].copy()

    coefs = {}
    for _, row in archs.iterrows():
        aid = row["archetype_id"]
        coefs[aid] = _effective_war_coef(
            row["war_mean"], row["cost_mean"], row["war_sd"], config
        )

    # PuLP model
    prob = pulp.LpProblem("MLB_Roster_Archetype", pulp.LpMaximize)

    # Decision variables: x[archetype_id, slot_label] ∈ {0,1}
    x = {}
    for _, row in archs.iterrows():
        aid = row["archetype_id"]
        eligible = row["eligible_slots"]
        for slot in slot_list:
            base_slot = slot.split("_")[0] if "_" in slot else slot
            # DH accepts any hitter; BENCH accepts utility positions (CI/MI/CF/OF, no C or DH)
            if base_slot == "DH":
                pos_grp = row["pos_group"]
                ok = pos_grp in _HITTER_POS_GROUPS
            elif base_slot == "BENCH":
                pos_grp = row["pos_group"]
                ok = pos_grp in _BENCH_POS_GROUPS
            else:
                ok = base_slot in eligible
            if ok:
                x[(aid, slot)] = pulp.LpVariable(
                    f"x_{aid}_{slot}", cat="Binary"
                )

    # Objective
    prob += pulp.lpSum(
        coefs[aid] * var for (aid, slot), var in x.items()
    )

    arch_lookup_pre = archs.set_index("archetype_id")

    # Constraint 1: budget
    prob += (
        pulp.lpSum(
            arch_lookup_pre.loc[aid, "cost_mean"] * var
            for (aid, slot), var in x.items()
        )
        <= budget_M,
        "Budget",
    )

    # Constraint 2: exactly one archetype per slot
    for slot in slot_list:
        slot_vars = [var for (aid, s), var in x.items() if s == slot]
        if slot_vars:
            prob += pulp.lpSum(slot_vars) == 1, f"Fill_{slot}"

    # Constraint 3: archetype caps
    # Explicit config caps override; auto-cap at n_players prevents the optimizer
    # from using an archetype more times than real players exist in that bucket.
    auto_cap = config.get("auto_cap_archetypes", True)
    for aid in archs["archetype_id"].unique():
        cap_vars = [var for (a, s), var in x.items() if a == aid]
        if not cap_vars:
            continue
        if aid in arch_caps:
            effective_cap = int(arch_caps[aid])
        elif auto_cap:
            n_players = int(arch_lookup_pre.loc[aid, "n_players"]) if "n_players" in arch_lookup_pre.columns else 99
            effective_cap = n_players
        else:
            continue
        prob += pulp.lpSum(cap_vars) <= effective_cap, f"Cap_{aid}"

    # Constraint 4: total roster WAR cap (optional safety guardrail)
    max_war = config.get("max_total_roster_war", None)
    if max_war is not None:
        prob += (
            pulp.lpSum(
                arch_lookup_pre.loc[aid, "war_mean"] * var
                for (aid, slot), var in x.items()
            )
            <= float(max_war),
            "MaxTotalWAR",
        )

    # Constraint 5: minimum spend floor
    min_spend_pct: float = float(config.get("min_spend_pct", 0.0))
    if min_spend_pct > 0.0:
        prob += (
            pulp.lpSum(
                arch_lookup_pre.loc[aid, "cost_mean"] * var
                for (aid, slot), var in x.items()
            )
            >= min_spend_pct * budget_M,
            "MinSpend",
        )

    # Constraint 6: stage mix minimums
    stage_mix_min: dict = config.get("stage_mix_min", {})
    if stage_mix_min and "stage" in arch_lookup_pre.columns:
        for req_stage, min_count in stage_mix_min.items():
            stage_vars = [
                var for (aid, slot), var in x.items()
                if arch_lookup_pre.loc[aid, "stage"] == req_stage
            ]
            if stage_vars:
                prob += (
                    pulp.lpSum(stage_vars) >= int(min_count),
                    f"StageMin_{req_stage}",
                )

    # Constraint 7: SP innings pitched floor (workload realism)
    # Ensures rotation isn't stacked entirely with Depth-tier spot starters.
    # Only applied if ip_mean column is present (from updated archetypes).
    sp_ip_floor: float = float(config.get("sp_ip_floor", 0.0))
    if sp_ip_floor > 0 and "ip_mean" in arch_lookup_pre.columns:
        sp_ip_vars = [
            (arch_lookup_pre.loc[aid, "ip_mean"], var)
            for (aid, slot), var in x.items()
            if (slot.split("_")[0] if "_" in slot else slot) == "SP"
        ]
        if sp_ip_vars:
            prob += (
                pulp.lpSum(ip * var for ip, var in sp_ip_vars) >= sp_ip_floor,
                "SP_IP_Floor",
            )

    # Constraint 8: RP innings pitched floor (bullpen coverage)
    rp_ip_floor: float = float(config.get("rp_ip_floor", 0.0))
    if rp_ip_floor > 0 and "ip_mean" in arch_lookup_pre.columns:
        rp_ip_vars = [
            (arch_lookup_pre.loc[aid, "ip_mean"], var)
            for (aid, slot), var in x.items()
            if (slot.split("_")[0] if "_" in slot else slot) == "RP"
        ]
        if rp_ip_vars:
            prob += (
                pulp.lpSum(ip * var for ip, var in rp_ip_vars) >= rp_ip_floor,
                "RP_IP_Floor",
            )

    # Constraint 9: minimum leverage relievers (Solid+ WAR tier)
    # Prevents filling all RP slots with mop-up arms.
    min_leverage_rp: int = int(config.get("min_leverage_rp", 0))
    if min_leverage_rp > 0 and "war_tier" in arch_lookup_pre.columns:
        leverage_vars = [
            var for (aid, slot), var in x.items()
            if (slot.split("_")[0] if "_" in slot else slot) == "RP"
            and arch_lookup_pre.loc[aid, "war_tier"] in ("Solid", "Elite")
        ]
        if leverage_vars:
            prob += (
                pulp.lpSum(leverage_vars) >= min_leverage_rp,
                "MinLeverageRP",
            )

    # Solve (silent, 30-second time limit per solve call)
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    status_str = pulp.LpStatus[prob.status]

    if prob.status not in (1, -1):
        if prob.status == 0:
            return OptimizerResult(
                status="MILP-Timeout",
                roster_df=pd.DataFrame(),
                objective_value=float("nan"),
            )
        return OptimizerResult(
            status=f"MILP-{status_str}",
            roster_df=pd.DataFrame(),
            objective_value=float("nan"),
        )
    if prob.status == -1 and all(
        pulp.value(v) is None for v in prob.variables()
    ):
        return OptimizerResult(
            status="MILP-Infeasible",
            roster_df=pd.DataFrame(),
            objective_value=float("nan"),
        )

    # Extract solution
    rows = []
    tight = []
    arch_mix: dict = {}
    total_cost = 0.0

    arch_lookup = archs.set_index("archetype_id")

    for slot in slot_list:
        chosen_aid = None
        for (aid, s), var in x.items():
            if s == slot and pulp.value(var) is not None and pulp.value(var) > 0.5:
                chosen_aid = aid
                break
        if chosen_aid is None:
            continue
        row = arch_lookup.loc[chosen_aid]
        base_slot = slot.split("_")[0] if "_" in slot else slot
        total_cost += row["cost_mean"]
        arch_mix[chosen_aid] = arch_mix.get(chosen_aid, 0) + 1

        row_dict = {
            "slot":         base_slot,
            "archetype_id": chosen_aid,
            "pos_group":    row["pos_group"],
            "war_mean":     row["war_mean"],
            "war_sd":       row["war_sd"],
            "cost_mean":    row["cost_mean"],
            "n_players":    row["n_players"],
        }
        # Include role and workload if present
        if "role" in row.index:
            row_dict["role"] = row["role"]
        if "ip_mean" in row.index:
            row_dict["ip_mean"] = row["ip_mean"]
        if "pa_mean" in row.index:
            row_dict["pa_mean"] = row["pa_mean"]
        rows.append(row_dict)

    # Identify tight constraints
    for name, con in prob.constraints.items():
        if abs(pulp.value(con)) < 1e-4:
            tight.append(name)

    roster_df = pd.DataFrame(rows)
    obj_val   = float(pulp.value(prob.objective) or 0.0)

    return OptimizerResult(
        status=f"MILP-{status_str}",
        roster_df=roster_df,
        objective_value=obj_val,
        tight_constraints=tight,
        archetype_mix=arch_mix,
    )


# ---------------------------------------------------------------------------
# MILP — player mode
# ---------------------------------------------------------------------------

def _run_milp_player(projected_df: pd.DataFrame, config: dict) -> OptimizerResult:
    """Binary LP: one variable per (player, slot)."""
    slots_cfg: dict = config["roster_slots"]
    budget_M: float = float(config["budget_M"])

    slot_list = []
    for slot_name, count in slots_cfg.items():
        for i in range(int(count)):
            slot_list.append(f"{slot_name}_{i}" if count > 1 else slot_name)

    players = projected_df.reset_index(drop=True)

    coefs = {}
    for _, row in players.iterrows():
        coefs[row["Player"]] = _effective_war_coef(
            row["proj_WAR"], row["proj_cost_M"], row["proj_WAR_sd"], config
        )

    prob = pulp.LpProblem("MLB_Roster_Player", pulp.LpMaximize)

    y = {}
    for _, prow in players.iterrows():
        pname   = prow["Player"]
        elig    = prow["eligible_slots"] if isinstance(prow["eligible_slots"], list) else []
        pos_grp = prow["pos_group"]
        for slot in slot_list:
            base_slot = slot.split("_")[0] if "_" in slot else slot
            if base_slot == "DH":
                ok = pos_grp in _HITTER_POS_GROUPS
            elif base_slot == "BENCH":
                ok = pos_grp in _BENCH_POS_GROUPS
            else:
                ok = base_slot in elig
            if ok:
                safe_name = pname.replace(" ", "_").replace("'", "").replace(".", "")
                y[(pname, slot)] = pulp.LpVariable(
                    f"y_{safe_name}_{slot}", cat="Binary"
                )

    # Objective
    prob += pulp.lpSum(coefs[p] * var for (p, s), var in y.items())

    # Budget
    p_lookup = players.set_index("Player")
    prob += (
        pulp.lpSum(
            p_lookup.loc[p, "proj_cost_M"] * var for (p, s), var in y.items()
        )
        <= budget_M,
        "Budget",
    )

    # Fill each slot
    for slot in slot_list:
        slot_vars = [var for (p, s), var in y.items() if s == slot]
        if slot_vars:
            prob += pulp.lpSum(slot_vars) == 1, f"Fill_{slot}"

    # Each player at most once
    for pname in players["Player"].unique():
        p_vars = [var for (p, s), var in y.items() if p == pname]
        if p_vars:
            prob += pulp.lpSum(p_vars) <= 1, f"Unique_{pname.replace(' ','_')}"

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)
    status_str = pulp.LpStatus[prob.status]

    if prob.status not in (1, -1):
        return OptimizerResult(
            status=f"MILP-{status_str}",
            roster_df=pd.DataFrame(),
            objective_value=float("nan"),
        )

    rows = []
    arch_mix: dict = {}
    tight = []

    for slot in slot_list:
        chosen_p = None
        for (p, s), var in y.items():
            if s == slot and pulp.value(var) is not None and pulp.value(var) > 0.5:
                chosen_p = p
                break
        if chosen_p is None:
            continue
        prow = p_lookup.loc[chosen_p]
        base_slot = slot.split("_")[0] if "_" in slot else slot
        arch_id = prow.get("archetype_id", "unknown")
        arch_mix[arch_id] = arch_mix.get(arch_id, 0) + 1
        rows.append({
            "slot":         base_slot,
            "Player":       chosen_p,
            "archetype_id": arch_id,
            "pos_group":    prow["pos_group"],
            "war_mean":     prow["proj_WAR"],
            "war_sd":       prow["proj_WAR_sd"],
            "cost_mean":    prow["proj_cost_M"],
        })

    for name, con in prob.constraints.items():
        if abs(pulp.value(con)) < 1e-4:
            tight.append(name)

    roster_df  = pd.DataFrame(rows)
    obj_val    = float(pulp.value(prob.objective) or 0.0)

    return OptimizerResult(
        status=f"MILP-{status_str}",
        roster_df=roster_df,
        objective_value=obj_val,
        tight_constraints=tight,
        archetype_mix=arch_mix,
    )


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------

def _run_greedy(archetype_df: pd.DataFrame, config: dict) -> OptimizerResult:
    """Simple greedy: pick highest WAR/$ archetype per slot within budget."""
    slots_cfg: dict = config["roster_slots"]
    budget_M: float = float(config["budget_M"])

    archs = archetype_df.copy()
    archs["war_per_M"] = archs["war_mean"] / archs["cost_mean"].clip(lower=0.01)
    archs = archs.sort_values("war_per_M", ascending=False, kind="mergesort")

    remaining = budget_M
    rows = []
    arch_mix: dict = {}

    slot_list = []
    for slot_name, count in slots_cfg.items():
        for _ in range(int(count)):
            slot_list.append(slot_name)

    for slot in slot_list:
        best = None
        for _, row in archs.iterrows():
            eligible = row["eligible_slots"]
            if isinstance(eligible, str):
                eligible = [s for s in eligible.split("|") if s]
            pos_grp = row["pos_group"]
            if slot == "DH":
                ok = pos_grp in _HITTER_POS_GROUPS
            elif slot == "BENCH":
                ok = pos_grp in _BENCH_POS_GROUPS
            else:
                ok = slot in eligible
            if not ok:
                continue
            if row["cost_mean"] > remaining:
                continue
            best = row
            break
        if best is None:
            for _, row in archs.sort_values("cost_mean", kind="mergesort").iterrows():
                eligible = row["eligible_slots"]
                if isinstance(eligible, str):
                    eligible = [s for s in eligible.split("|") if s]
                pos_grp = row["pos_group"]
                if slot == "DH":
                    ok = pos_grp in _HITTER_POS_GROUPS
                else:
                    ok = slot in eligible
                if ok:
                    best = row
                    break

        if best is not None:
            remaining -= best["cost_mean"]
            aid = best["archetype_id"]
            arch_mix[aid] = arch_mix.get(aid, 0) + 1
            rows.append({
                "slot":         slot,
                "archetype_id": aid,
                "pos_group":    best["pos_group"],
                "war_mean":     best["war_mean"],
                "war_sd":       best["war_sd"],
                "cost_mean":    best["cost_mean"],
                "n_players":    best["n_players"],
            })

    roster_df = pd.DataFrame(rows)
    obj_val   = float(roster_df["war_mean"].sum()) if not roster_df.empty else 0.0

    return OptimizerResult(
        status="Greedy-fallback",
        roster_df=roster_df,
        objective_value=obj_val,
        tight_constraints=[],
        archetype_mix=arch_mix,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_optimizer(
    archetype_df: pd.DataFrame,
    config: dict,
    projected_df: pd.DataFrame | None = None,
) -> OptimizerResult:
    """
    Parameters
    ----------
    archetype_df : from build_archetype_definitions()
    config       : full config dict
    projected_df : only used when optimizer_mode == "player"

    Returns
    -------
    OptimizerResult
    """
    mode = config.get("optimizer_mode", "archetype")

    if not _PULP_AVAILABLE:
        return _run_greedy(archetype_df, config)

    if mode == "player":
        if projected_df is None:
            raise ValueError("projected_df required for player mode")
        return _run_milp_player(projected_df, config)

    return _run_milp_archetype(archetype_df, config)
