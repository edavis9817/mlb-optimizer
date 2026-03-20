"""
artifacts.py
------------
Writes all run artifacts to runs/run_{timestamp}/.

Public API
----------
write_run_artifacts(run_dir, config, gold_df, results, diagnostics, figures) -> run_dir
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd


def _hash_df(df: pd.DataFrame) -> str:
    """Quick MD5 of a DataFrame's CSV representation."""
    return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()


def _hash_file(path: str) -> str:
    """MD5 of a file on disk (or 'file_not_found')."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "file_not_found"


def _json_safe(obj: Any) -> Any:
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:   # nan
            return None
        return obj
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj


def write_run_artifacts(
    run_dir: str,
    config: dict,
    gold_df: pd.DataFrame,
    results: Any,            # OptimizerResult
    diagnostics: dict,       # keys: frontier_df, upgrades_df, cuts_df, backtest
    figures: dict[str, Any], # key = filename stem, value = matplotlib Figure
) -> str:
    """
    Parameters
    ----------
    run_dir      : path of this run's folder (already created by pipeline.py)
    config       : full config dict used for this run
    gold_df      : archetype_df used as optimizer input
    results      : OptimizerResult (has .roster_df, .status, .objective_value, .archetype_mix)
    diagnostics  : dict with optional keys:
                     frontier_df, upgrades_df, cuts_df, backtest (BacktestResult)
    figures      : dict mapping filename stem → matplotlib Figure object

    Returns
    -------
    run_dir : same path passed in
    """
    os.makedirs(run_dir, exist_ok=True)
    figs_dir = os.path.join(run_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    ts = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # config.json
    # ------------------------------------------------------------------
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(_json_safe(config), fh, indent=2)

    # ------------------------------------------------------------------
    # gold_inputs.csv
    # ------------------------------------------------------------------
    gold_out = gold_df.copy()
    if "eligible_slots" in gold_out.columns:
        gold_out["eligible_slots"] = gold_out["eligible_slots"].apply(
            lambda v: "|".join(v) if isinstance(v, list) else str(v)
        )
    gold_out.to_csv(os.path.join(run_dir, "gold_inputs.csv"), index=False)

    # ------------------------------------------------------------------
    # results_roster.csv
    # ------------------------------------------------------------------
    if hasattr(results, "roster_df") and not results.roster_df.empty:
        results.roster_df.to_csv(os.path.join(run_dir, "results_roster.csv"), index=False)

    # ------------------------------------------------------------------
    # results_archetypes.csv  (archetype mix summary)
    # ------------------------------------------------------------------
    if hasattr(results, "archetype_mix") and results.archetype_mix:
        mix_df = pd.DataFrame(
            [{"archetype_id": k, "count": v} for k, v in results.archetype_mix.items()]
        )
        mix_df.to_csv(os.path.join(run_dir, "results_archetypes.csv"), index=False)

    # ------------------------------------------------------------------
    # diagnostics.json
    # ------------------------------------------------------------------
    diag_out: dict = {}

    if "frontier_df" in diagnostics and not diagnostics["frontier_df"].empty:
        diag_out["frontier"] = diagnostics["frontier_df"].to_dict(orient="records")

    if "upgrades_df" in diagnostics and not diagnostics["upgrades_df"].empty:
        diag_out["upgrades"] = diagnostics["upgrades_df"].to_dict(orient="records")

    if "cuts_df" in diagnostics and not diagnostics["cuts_df"].empty:
        diag_out["cuts"] = diagnostics["cuts_df"].to_dict(orient="records")

    if "backtest" in diagnostics and diagnostics["backtest"] is not None:
        bt = diagnostics["backtest"]
        diag_out["backtest"] = bt.summary() if hasattr(bt, "summary") else {}

    if "sim_result" in diagnostics and diagnostics["sim_result"] is not None:
        sr = diagnostics["sim_result"]
        diag_out["simulation"] = sr.summary() if hasattr(sr, "summary") else {}

    with open(os.path.join(run_dir, "diagnostics.json"), "w") as fh:
        json.dump(_json_safe(diag_out), fh, indent=2)

    # ------------------------------------------------------------------
    # figures/*.png
    # ------------------------------------------------------------------
    for stem, fig in figures.items():
        if fig is not None:
            try:
                fig.savefig(os.path.join(figs_dir, f"{stem}.png"), dpi=150, bbox_inches="tight")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # run_metadata.json
    # ------------------------------------------------------------------
    config_hash = hashlib.md5(
        json.dumps(_json_safe(config), sort_keys=True).encode()
    ).hexdigest()

    input_hashes: dict = {}
    for key in ("raw_salary_war_path", "raw_wins_path"):
        raw_path = config.get(key, "")
        if raw_path:
            input_hashes[key] = _hash_file(raw_path)

    metadata = {
        "timestamp":    ts,
        "config_hash":  config_hash,
        "input_hashes": input_hashes,
        "solver_status": getattr(results, "status", "unknown"),
        "objective_value": getattr(results, "objective_value", None),
    }

    with open(os.path.join(run_dir, "run_metadata.json"), "w") as fh:
        json.dump(_json_safe(metadata), fh, indent=2)

    return run_dir
