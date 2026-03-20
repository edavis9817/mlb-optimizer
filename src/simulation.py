"""
simulation.py
-------------
Monte Carlo win-distribution simulator.

Upgrades over naive model:
  1. Calibrated wins formula: cal_intercept + cal_slope * total_WAR
     (from OLS regression in backtest.py — accounts for non-unit WAR→wins conversion)
  2. Correlated group shocks: SP / RP / hitters each share a team-level shock
     (captures injury waves, bullpen blowups, lineup-wide slumps)

Public API
----------
monte_carlo(roster_df, config, backtest_adjustment, cal_slope, cal_intercept) -> SimResult
    .mean_wins       : float
    .median_wins     : float
    .p10             : float
    .p90             : float
    .std_wins        : float
    .playoff_odds    : float   (fraction of sims >= playoff_threshold_wins)
    .wins_array      : np.ndarray  (shape: [n_sims])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    mean_wins:    float
    median_wins:  float
    p10:          float
    p90:          float
    std_wins:     float
    playoff_odds: float
    wins_array:   np.ndarray

    def summary(self) -> dict:
        return {
            "mean_wins":    round(self.mean_wins, 1),
            "median_wins":  round(self.median_wins, 1),
            "p10":          round(self.p10, 1),
            "p90":          round(self.p90, 1),
            "std_wins":     round(self.std_wins, 1),
            "playoff_odds": round(self.playoff_odds, 3),
        }


def monte_carlo(
    roster_df: pd.DataFrame,
    config: dict,
    backtest_adjustment: float = 0.0,
    cal_slope: float = 1.0,
    cal_intercept: float | None = None,
) -> SimResult:
    """
    Parameters
    ----------
    roster_df           : output of run_optimizer()  (must have war_mean, war_sd, slot columns)
    config              : full config dict
    backtest_adjustment : mean residual bias from run_backtest() (kept for backwards compat;
                          when cal_intercept is provided, this is no longer added separately)
    cal_slope           : OLS slope from backtest regression (default 1.0 = no rescaling)
    cal_intercept       : OLS intercept from backtest regression.
                          If None, falls back to wins_intercept + backtest_adjustment.

    Returns
    -------
    SimResult
    """
    n_sims: int    = int(config.get("mc_simulations", 1000))
    seed:   int    = int(config.get("mc_seed", 42))
    intercept: float = float(config.get("wins_intercept", 48.0))
    threshold: float = float(config.get("playoff_threshold_wins", 88.0))
    bench_bp_war: float = float(config.get("bench_bullpen_war", 0.0))

    # Calibrated wins model parameters
    # If OLS calibration is available, use it; otherwise fall back to simple intercept
    if cal_intercept is None:
        cal_intercept = intercept + backtest_adjustment
    effective_slope = float(cal_slope)
    effective_intercept = float(cal_intercept)

    # Group shock standard deviations (captures correlated team-level variance)
    # These add realistic downside spread: a bad SP injury year or bullpen collapse
    sp_shock_sd:  float = float(config.get("sp_group_shock_sd",  1.5))
    rp_shock_sd:  float = float(config.get("rp_group_shock_sd",  1.0))
    hit_shock_sd: float = float(config.get("hit_group_shock_sd", 1.0))

    rng = np.random.default_rng(seed)

    if roster_df.empty:
        base = effective_intercept + bench_bp_war
        empty = np.full(n_sims, base)
        return SimResult(
            mean_wins=base,
            median_wins=base,
            p10=base,
            p90=base,
            std_wins=0.0,
            playoff_odds=float(base >= threshold),
            wins_array=empty,
        )

    war_means = roster_df["war_mean"].values.astype(float)
    war_sds   = roster_df["war_sd"].values.astype(float)
    war_sds   = np.maximum(war_sds, 0.1)    # floor SD at 0.1

    n_slots = len(war_means)

    # Identify slot groups for correlated shocks
    slots = roster_df["slot"].values if "slot" in roster_df.columns else np.full(n_slots, "")
    is_sp  = np.array([str(s).startswith("SP") for s in slots])
    is_rp  = np.array([str(s).startswith("RP") for s in slots])
    is_hit = ~is_sp & ~is_rp

    # shape: (n_sims, n_slots) — sample each slot's WAR independently
    sampled_war = rng.normal(
        loc=war_means[np.newaxis, :],
        scale=war_sds[np.newaxis, :],
        size=(n_sims, n_slots),
    )

    # Clip negative WAR samples to 0 (replacement-level floor)
    sampled_war = np.maximum(sampled_war, 0.0)

    # Apply correlated group shocks (shared injury/performance effects per group)
    # Shape: (n_sims,) — same shock applied to total WAR of that group
    if sp_shock_sd > 0 and is_sp.any():
        sp_shock = rng.normal(0, sp_shock_sd, size=n_sims)
        sp_group_war = sampled_war[:, is_sp].sum(axis=1) + sp_shock
        sp_group_war = np.maximum(sp_group_war, 0.0)
    else:
        sp_group_war = sampled_war[:, is_sp].sum(axis=1) if is_sp.any() else np.zeros(n_sims)

    if rp_shock_sd > 0 and is_rp.any():
        rp_shock = rng.normal(0, rp_shock_sd, size=n_sims)
        rp_group_war = sampled_war[:, is_rp].sum(axis=1) + rp_shock
        rp_group_war = np.maximum(rp_group_war, 0.0)
    else:
        rp_group_war = sampled_war[:, is_rp].sum(axis=1) if is_rp.any() else np.zeros(n_sims)

    if hit_shock_sd > 0 and is_hit.any():
        hit_shock = rng.normal(0, hit_shock_sd, size=n_sims)
        hit_group_war = sampled_war[:, is_hit].sum(axis=1) + hit_shock
        hit_group_war = np.maximum(hit_group_war, 0.0)
    else:
        hit_group_war = sampled_war[:, is_hit].sum(axis=1) if is_hit.any() else np.zeros(n_sims)

    total_war  = sp_group_war + rp_group_war + hit_group_war
    # Calibrated wins formula: intercept + slope * WAR  (slope near 1.0 typically)
    wins_array = effective_intercept + effective_slope * total_war + bench_bp_war

    mean_wins    = float(np.mean(wins_array))
    median_wins  = float(np.median(wins_array))
    p10          = float(np.percentile(wins_array, 10))
    p90          = float(np.percentile(wins_array, 90))
    std_wins     = float(np.std(wins_array, ddof=1))
    playoff_odds = float(np.mean(wins_array >= threshold))

    return SimResult(
        mean_wins=mean_wins,
        median_wins=median_wins,
        p10=p10,
        p90=p90,
        std_wins=std_wins,
        playoff_odds=playoff_odds,
        wins_array=wins_array,
    )
