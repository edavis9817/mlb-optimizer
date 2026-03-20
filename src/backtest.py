"""
backtest.py
-----------
Validates the WAR → wins linear model against historical data.
Also fits a calibrated OLS regression: actual_wins = cal_intercept + cal_slope * team_WAR

Public API
----------
run_backtest(raw_df, wins_df, config) -> BacktestResult
    .rmse              : float
    .bias              : float   (mean residual = actual - predicted)
    .adjustment_factor : float   (same as bias — added to all win forecasts)
    .cal_slope         : float   (OLS slope for WAR→wins; typically near 1.0)
    .cal_intercept     : float   (OLS intercept; calibrated replacement-level wins)
    .per_team_df       : DataFrame  (Team, Year, predicted_wins, actual_wins, residual)
    .n_team_years      : int
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    rmse: float
    bias: float
    adjustment_factor: float
    cal_slope: float          # OLS slope (WAR → wins conversion rate)
    cal_intercept: float      # OLS intercept (calibrated replacement-level wins)
    per_team_df: pd.DataFrame
    n_team_years: int

    def summary(self) -> dict:
        return {
            "rmse":              round(self.rmse, 2),
            "bias":              round(self.bias, 2),
            "adjustment_factor": round(self.adjustment_factor, 2),
            "cal_slope":         round(self.cal_slope, 4),
            "cal_intercept":     round(self.cal_intercept, 2),
            "n_team_years":      self.n_team_years,
        }


def run_backtest(
    raw_df: pd.DataFrame,
    wins_df: pd.DataFrame,
    config: dict,
) -> BacktestResult:
    """
    Parameters
    ----------
    raw_df   : raw salary/WAR file (Player, Year, Team, WAR_Total …)
    wins_df  : sportsref team wins file (expected cols: Team/Tm, Year/Season, W)
    config   : full config dict

    Returns
    -------
    BacktestResult
    """
    intercept: float = float(config.get("wins_intercept", 48.0))

    # ------------------------------------------------------------------
    # 1. Normalise raw_df columns
    # ------------------------------------------------------------------
    raw = raw_df.copy()
    raw.columns = [c.strip() for c in raw.columns]
    raw["Year"]      = pd.to_numeric(raw.get("Year",      np.nan), errors="coerce")
    raw["WAR_Total"] = pd.to_numeric(raw.get("WAR_Total", 0),      errors="coerce").fillna(0)

    # Find team column
    team_col_raw = next((c for c in raw.columns if c.lower() in ("team", "tm")), None)
    if team_col_raw is None:
        return _empty_backtest(intercept)
    raw = raw.rename(columns={team_col_raw: "Team"})

    # Aggregate WAR per team-year
    team_war = (
        raw.groupby(["Team", "Year"], sort=True)["WAR_Total"]
           .sum()
           .reset_index()
           .rename(columns={"WAR_Total": "total_WAR"})
    )

    # ------------------------------------------------------------------
    # 2. Normalise wins_df columns
    # Supports two layouts:
    #   (a) Long format: Team/Tm, Year/Season, W columns
    #   (b) Wide format: Year column + one column per team abbreviation
    #       (sportsref style: Year, G, ARI, ATH, ATL, ...)
    # ------------------------------------------------------------------
    wins = wins_df.copy()
    wins.columns = [c.strip() for c in wins.columns]

    year_col_w = next((c for c in wins.columns if c.lower() in ("year", "season")), None)
    team_col_w = next((c for c in wins.columns if c.lower() in ("team", "tm")), None)
    wins_col_w = next((c for c in wins.columns if c.lower() == "w"), None)

    if year_col_w and team_col_w and wins_col_w:
        # (a) Long format — already tidy
        wins = wins.rename(columns={team_col_w: "Team", year_col_w: "Year", wins_col_w: "W"})
        wins["Year"] = pd.to_numeric(wins["Year"], errors="coerce")
        wins["W"]    = pd.to_numeric(wins["W"],    errors="coerce")
        wins = wins.dropna(subset=["Team", "Year", "W"])
    elif year_col_w:
        # (b) Wide format — melt team columns into rows
        non_team = {year_col_w.lower(), "g", "lg", "year", "season"}
        team_cols = [c for c in wins.columns if c.lower() not in non_team]
        wins = wins.rename(columns={year_col_w: "Year"})
        wins["Year"] = pd.to_numeric(wins["Year"], errors="coerce")
        wins = wins.dropna(subset=["Year"])
        wins = wins.melt(id_vars=["Year"], value_vars=team_cols, var_name="Team", value_name="W")
        wins["W"] = pd.to_numeric(wins["W"], errors="coerce")
        wins = wins.dropna(subset=["Team", "Year", "W"])
    else:
        return _empty_backtest(intercept)

    # ------------------------------------------------------------------
    # 3. Merge on (Team, Year)
    # ------------------------------------------------------------------
    merged = pd.merge(team_war, wins[["Team", "Year", "W"]], on=["Team", "Year"], how="inner")

    if merged.empty:
        return _empty_backtest(intercept)

    merged["predicted_wins"] = intercept + merged["total_WAR"]
    merged["residual"]       = merged["W"] - merged["predicted_wins"]

    rmse = float(np.sqrt(np.mean(merged["residual"].values ** 2)))
    bias = float(np.mean(merged["residual"].values))

    # ------------------------------------------------------------------
    # 4. Calibrated OLS regression: actual_wins = cal_intercept + cal_slope * total_WAR
    # A slope near 1.0 confirms the WAR→wins conversion.
    # The intercept captures average replacement-level win total.
    # Requires at least 3 data points for a meaningful regression.
    # ------------------------------------------------------------------
    n = len(merged)
    if n >= 3:
        x = merged["total_WAR"].values.astype(float)
        y = merged["W"].values.astype(float)
        # OLS via numpy polyfit (degree 1)
        coeffs = np.polyfit(x, y, 1)   # [slope, intercept]
        cal_slope     = float(coeffs[0])
        cal_intercept = float(coeffs[1])
    else:
        # Not enough data — fall back to config intercept, slope=1
        cal_slope     = 1.0
        cal_intercept = intercept

    per_team_df = merged.rename(columns={"W": "actual_wins"})[
        ["Team", "Year", "predicted_wins", "actual_wins", "residual"]
    ].reset_index(drop=True)

    return BacktestResult(
        rmse=rmse,
        bias=bias,
        adjustment_factor=bias,
        cal_slope=cal_slope,
        cal_intercept=cal_intercept,
        per_team_df=per_team_df,
        n_team_years=n,
    )


def _empty_backtest(intercept: float = 48.0) -> BacktestResult:
    return BacktestResult(
        rmse=0.0,
        bias=0.0,
        adjustment_factor=0.0,
        cal_slope=1.0,
        cal_intercept=intercept,
        per_team_df=pd.DataFrame(
            columns=["Team", "Year", "predicted_wins", "actual_wins", "residual"]
        ),
        n_team_years=0,
    )
