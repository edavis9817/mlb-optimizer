"""
pipeline.py
-----------
Orchestrates the full end-to-end run:

  load config -> load raw data -> projections -> archetypes ->
  optimizer -> simulation -> backtest -> diagnostics -> artifacts

Public API
----------
run_pipeline(config_path) -> run_dir : str
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime

import pandas as pd

from .projections  import make_projections
from .archetypes   import build_archetype_definitions, assign_archetypes
from .gold         import write_gold
from .optimizer    import run_optimizer
from .simulation   import monte_carlo
from .backtest     import run_backtest
from .diagnostics  import budget_frontier, marginal_analysis
from .artifacts    import write_run_artifacts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    with open(config_path) as fh:
        cfg = json.load(fh)
    return cfg


def _resolve_path(raw_path: str, config_path: str) -> str:
    """
    Resolve a possibly-relative path from the config against the config file's
    directory so users can write paths like "../../Data/file.csv".
    """
    if os.path.isabs(raw_path):
        return raw_path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(config_dir, raw_path))


def _hash_config(cfg: dict) -> str:
    return hashlib.md5(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str) -> str:
    """
    Run the full pipeline and return the run directory.

    Parameters
    ----------
    config_path : path to a JSON config file (relative or absolute)

    Returns
    -------
    run_dir : absolute path of the run's output folder
    """
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    cfg = _load_config(config_path)

    # Resolve data paths relative to config file
    salary_path = _resolve_path(cfg["raw_salary_war_path"], config_path)
    wins_path   = _resolve_path(cfg["raw_wins_path"],       config_path)

    # ------------------------------------------------------------------
    # 2. Prepare run directory
    # ------------------------------------------------------------------
    # run root = mlb_optimizer/ = parent of configs/
    run_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = os.path.join(run_root, "runs", f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"[pipeline] Run dir  : {run_dir}")
    print(f"[pipeline] Config   : {config_path}")
    print(f"[pipeline] WAR data : {salary_path}")

    # ------------------------------------------------------------------
    # 3. Load raw data
    # ------------------------------------------------------------------
    raw_df = pd.read_csv(salary_path, low_memory=False)
    print(f"[pipeline] Loaded salary/WAR data: {len(raw_df):,} rows")

    wins_df = pd.DataFrame()
    if os.path.exists(wins_path):
        wins_df = pd.read_csv(wins_path, low_memory=False)
        print(f"[pipeline] Loaded wins data: {len(wins_df):,} rows")
    else:
        print(f"[pipeline] Wins file not found - backtest will be skipped: {wins_path}")

    # ------------------------------------------------------------------
    # 4. Projections
    # ------------------------------------------------------------------
    print("[pipeline] Building projections ...")
    proj_df = make_projections(raw_df, cfg)
    print(f"[pipeline] Projections: {len(proj_df)} players")

    # Show CF vs corner OF split
    if "pos_group" in proj_df.columns:
        cf_n = (proj_df["pos_group"] == "CF").sum()
        of_n = (proj_df["pos_group"] == "OF").sum()
        if cf_n > 0 or of_n > 0:
            print(f"[pipeline]   OF split: {cf_n} CF-capable  |  {of_n} corner-OF")

    # ------------------------------------------------------------------
    # 5. Archetypes
    # ------------------------------------------------------------------
    print("[pipeline] Building archetypes ...")
    war_cap     = float(cfg.get("archetype_war_player_cap", 6.5))
    min_players = int(cfg.get("min_archetype_players", 5))
    proj_with_arch = assign_archetypes(proj_df)
    arch_df        = build_archetype_definitions(proj_df, war_player_cap=war_cap, min_players=min_players)
    print(f"[pipeline] Archetypes: {len(arch_df)} distinct archetypes")

    gold_dir = write_gold(arch_df, proj_with_arch, run_root)
    print(f"[pipeline] Gold written -> {gold_dir}")

    # ------------------------------------------------------------------
    # 6. Optimize
    # ------------------------------------------------------------------
    print(f"[pipeline] Running optimizer (mode={cfg.get('optimizer_mode','archetype')}) ...")
    opt_result = run_optimizer(arch_df, cfg, projected_df=proj_with_arch)
    print(f"[pipeline] Solver: {opt_result.status}  |  obj={opt_result.objective_value:.2f}")

    if opt_result.roster_df.empty:
        print("[pipeline] WARNING: optimizer returned empty roster - check constraints.")
    else:
        _print_efficiency(opt_result.roster_df, cfg)
        _print_binding_constraints(opt_result.tight_constraints)

    # ------------------------------------------------------------------
    # 7. Backtest
    # ------------------------------------------------------------------
    print("[pipeline] Running backtest ...")
    bt_result = run_backtest(raw_df, wins_df, cfg)
    adj = bt_result.adjustment_factor
    print(
        f"[pipeline] Backtest RMSE={bt_result.rmse:.2f}  bias={bt_result.bias:.2f}  "
        f"cal_slope={bt_result.cal_slope:.3f}  cal_intercept={bt_result.cal_intercept:.1f}"
    )

    # ------------------------------------------------------------------
    # 8. Simulation (calibrated)
    # ------------------------------------------------------------------
    print("[pipeline] Running Monte Carlo (calibrated) ...")
    sim_result = monte_carlo(
        opt_result.roster_df,
        cfg,
        backtest_adjustment=adj,
        cal_slope=bt_result.cal_slope,
        cal_intercept=bt_result.cal_intercept,
    )
    print(
        f"[pipeline] Wins: {sim_result.mean_wins:.1f} "
        f"(P10={sim_result.p10:.1f}, P90={sim_result.p90:.1f})  "
        f"Playoff odds: {sim_result.playoff_odds:.1%}"
    )

    # ------------------------------------------------------------------
    # 9. Diagnostics
    # ------------------------------------------------------------------
    print("[pipeline] Running diagnostics ...")
    upgrades_df, cuts_df = marginal_analysis(opt_result.roster_df, arch_df, cfg)

    print("[pipeline] Budget frontier ...")
    frontier_df = budget_frontier(arch_df, cfg)

    # ------------------------------------------------------------------
    # 10. Figures (basic matplotlib)
    # ------------------------------------------------------------------
    figures = _make_figures(opt_result, sim_result, frontier_df, arch_df, cfg)

    # ------------------------------------------------------------------
    # 11. Artifacts
    # ------------------------------------------------------------------
    diag_bundle = {
        "frontier_df":  frontier_df,
        "upgrades_df":  upgrades_df,
        "cuts_df":      cuts_df,
        "backtest":     bt_result,
        "sim_result":   sim_result,
    }

    write_run_artifacts(run_dir, cfg, arch_df, opt_result, diag_bundle, figures)
    print(f"[pipeline] Artifacts written -> {run_dir}")

    return run_dir


# ---------------------------------------------------------------------------
# Efficiency reporting
# ---------------------------------------------------------------------------

def _print_efficiency(roster_df, cfg: dict) -> None:
    """Print a concise money-efficiency breakdown after the optimizer runs."""
    import pandas as pd

    rdf = roster_df.copy()

    total_war  = rdf["war_mean"].sum()
    total_cost = rdf["cost_mean"].sum()
    budget     = float(cfg.get("budget_M", 130))
    mkt_dpw    = float(cfg.get("market_dpw_M", 5.5))   # $/WAR market rate

    actual_dpw   = total_cost / max(total_war, 0.01)    # $/WAR this roster
    market_value = total_war * mkt_dpw                  # what FA market pays for this WAR
    savings      = market_value - total_cost            # surplus vs market

    # Per-slot efficiency
    rdf["war_per_M"] = rdf["war_mean"] / rdf["cost_mean"].clip(lower=0.1)
    best  = rdf.loc[rdf["war_per_M"].idxmax()]
    worst = rdf.loc[rdf["war_per_M"].idxmin()]

    # Stage breakdown
    from collections import Counter
    stages = [a.split("_")[1] for a in rdf["archetype_id"] if len(a.split("_")) >= 2]
    stage_counts = Counter(stages)

    fa_war  = rdf[[a.split("_")[1] == "FA"      for a in rdf["archetype_id"]]]["war_mean"].sum()
    arb_war = rdf[[a.split("_")[1] == "Arb"     for a in rdf["archetype_id"]]]["war_mean"].sum()
    pre_war = rdf[[a.split("_")[1] == "Pre-Arb" for a in rdf["archetype_id"]]]["war_mean"].sum()

    fa_cost  = rdf[[a.split("_")[1] == "FA"      for a in rdf["archetype_id"]]]["cost_mean"].sum()
    arb_cost = rdf[[a.split("_")[1] == "Arb"     for a in rdf["archetype_id"]]]["cost_mean"].sum()
    pre_cost = rdf[[a.split("_")[1] == "Pre-Arb" for a in rdf["archetype_id"]]]["cost_mean"].sum()

    # Role breakdown (if available)
    role_summary = ""
    if "role" in rdf.columns:
        role_counts = Counter(rdf["role"].tolist())
        top_roles = sorted(role_counts.items(), key=lambda x: -x[1])[:5]
        role_summary = "  ".join(f"{r}={n}" for r, n in top_roles)

    # IP coverage (if available)
    ip_summary = ""
    if "ip_mean" in rdf.columns and "slot" in rdf.columns:
        sp_ip = rdf[rdf["slot"] == "SP"]["ip_mean"].sum()
        rp_ip = rdf[rdf["slot"] == "RP"]["ip_mean"].sum()
        if sp_ip > 0 or rp_ip > 0:
            ip_summary = f"SP={sp_ip:.0f} IP  RP={rp_ip:.0f} IP"

    print(f"[pipeline] --- Efficiency Report ---")
    print(f"[pipeline] Budget used:   ${total_cost:.1f}M / ${budget:.0f}M  ({total_cost/budget:.0%})")
    print(f"[pipeline] Total WAR:     {total_war:.1f}  |  ${actual_dpw:.2f}M/WAR actual  vs  ${mkt_dpw:.1f}M/WAR market")
    print(f"[pipeline] Surplus value: ${savings:.1f}M saved vs signing equivalent FA talent")
    print(f"[pipeline] Stage mix:     FA={stage_counts.get('FA',0)} (${fa_cost:.0f}M, {fa_war:.1f} WAR)  "
          f"Arb={stage_counts.get('Arb',0)} (${arb_cost:.0f}M, {arb_war:.1f} WAR)  "
          f"Pre-Arb={stage_counts.get('Pre-Arb',0)} (${pre_cost:.0f}M, {pre_war:.1f} WAR)")
    if role_summary:
        print(f"[pipeline] Role mix:      {role_summary}")
    if ip_summary:
        print(f"[pipeline] Pitcher IP:    {ip_summary}")
    print(f"[pipeline] Best value:    {best['archetype_id']}  ({best['war_mean']:.2f} WAR / ${best['cost_mean']:.1f}M = {best['war_per_M']:.2f} WAR/$M)")
    print(f"[pipeline] Least value:   {worst['archetype_id']}  ({worst['war_mean']:.2f} WAR / ${worst['cost_mean']:.1f}M = {worst['war_per_M']:.2f} WAR/$M)")
    print(f"[pipeline] ----------------------------")


def _print_binding_constraints(tight: list[str]) -> None:
    """Report which constraints were binding (at their limit) after solving."""
    if not tight:
        return
    # Filter to informative ones (skip slot-fill constraints)
    notable = [c for c in tight if not c.startswith("Fill_")]
    if not notable:
        return
    print(f"[pipeline] Binding constraints: {', '.join(notable[:10])}")


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _make_figures(opt_result, sim_result, frontier_df, arch_df, cfg) -> dict:
    """Return a dict of {stem: Figure}.  Safe to call even without display."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    figs = {}

    # --- Fig 1: WAR bar chart by slot ---
    if not opt_result.roster_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        rdf = opt_result.roster_df.copy()
        colors = plt.cm.RdYlGn(
            [(v - rdf["war_mean"].min()) / max(rdf["war_mean"].max() - rdf["war_mean"].min(), 0.01)
             for v in rdf["war_mean"]]
        )
        ax.bar(range(len(rdf)), rdf["war_mean"], color=colors)
        ax.set_xticks(range(len(rdf)))
        ax.set_xticklabels(
            [f"{r['slot']}\n{r['archetype_id'].split('_',1)[1] if '_' in r['archetype_id'] else ''}"
             for _, r in rdf.iterrows()],
            fontsize=8, rotation=45, ha="right"
        )
        ax.set_ylabel("Projected WAR")
        ax.set_title(f"Optimal Roster - ${cfg['budget_M']:.0f}M budget")
        budget_val = cfg['budget_M']
        total_cost = rdf['cost_mean'].sum()
        ax.set_xlabel(f"Total cost: ${total_cost:.1f}M / ${budget_val:.0f}M budget")
        plt.tight_layout()
        figs["roster_war"] = fig

    # --- Fig 2: Win distribution ---
    if sim_result.wins_array is not None and len(sim_result.wins_array) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sim_result.wins_array, bins=40, color="#2980b9", alpha=0.7, edgecolor="white")
        threshold = float(cfg.get("playoff_threshold_wins", 88))
        ax.axvline(sim_result.p10,         color="#e74c3c", lw=1.5, linestyle="--", label=f"P10={sim_result.p10:.0f}")
        ax.axvline(sim_result.median_wins,  color="white",  lw=2,   linestyle="-",  label=f"P50={sim_result.median_wins:.0f}")
        ax.axvline(sim_result.p90,         color="#27ae60", lw=1.5, linestyle="--", label=f"P90={sim_result.p90:.0f}")
        ax.axvline(threshold, color="#f39c12", lw=2, linestyle=":", label=f"Playoff>={threshold:.0f}")
        ax.set_xlabel("Season wins")
        ax.set_ylabel("Simulations")
        ax.set_title(f"Win Distribution  (playoff odds {sim_result.playoff_odds:.1%})")
        ax.legend(fontsize=9)
        plt.tight_layout()
        figs["win_distribution"] = fig

    # --- Fig 3: Budget frontier ---
    if not frontier_df.empty and "expected_wins" in frontier_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        fdf = frontier_df.dropna(subset=["expected_wins"])
        ax.plot(fdf["budget_M"], fdf["expected_wins"], "o-", color="#2980b9", lw=2)
        if "p10" in fdf.columns and "p90" in fdf.columns:
            ax.fill_between(fdf["budget_M"], fdf["p10"], fdf["p90"], alpha=0.2, color="#2980b9")
        ax.set_xlabel("Budget ($M)")
        ax.set_ylabel("Expected wins")
        ax.set_title("Budget Frontier")
        plt.tight_layout()
        figs["budget_frontier"] = fig

    plt.close("all")
    return figs
