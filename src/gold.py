"""
gold.py
-------
Writes the "gold" inputs used by the optimizer (archetype definitions)
to data/gold/ for reproducibility.

Public API
----------
write_gold(archetype_df, projected_df, run_root) -> gold_path
load_gold(gold_path)                              -> (archetype_df, projected_df)
"""

import os
from datetime import datetime

import pandas as pd


def write_gold(
    archetype_df: pd.DataFrame,
    projected_df: pd.DataFrame,
    run_root: str,
) -> str:
    """
    Persist archetype + projection tables to data/gold/.

    Parameters
    ----------
    archetype_df : output of build_archetype_definitions()
    projected_df : output of assign_archetypes()
    run_root     : root of the mlb_optimizer project (contains data/)

    Returns
    -------
    gold_dir : path of the directory written
    """
    gold_dir = os.path.join(run_root, "data", "gold")
    os.makedirs(gold_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    arch_path = os.path.join(gold_dir, f"archetypes_{ts}.csv")
    proj_path = os.path.join(gold_dir, f"projections_with_arch_{ts}.csv")

    # eligible_slots is a list — serialise as pipe-separated string
    arch_out = archetype_df.copy()
    if "eligible_slots" in arch_out.columns:
        arch_out["eligible_slots"] = arch_out["eligible_slots"].apply(
            lambda v: "|".join(v) if isinstance(v, list) else str(v)
        )
    arch_out.to_csv(arch_path, index=False)

    proj_out = projected_df.copy()
    if "eligible_slots" in proj_out.columns:
        proj_out["eligible_slots"] = proj_out["eligible_slots"].apply(
            lambda v: "|".join(v) if isinstance(v, list) else str(v)
        )
    proj_out.to_csv(proj_path, index=False)

    return gold_dir


def load_gold(gold_dir: str):
    """
    Load the most-recently-written gold files from gold_dir.

    Returns
    -------
    (archetype_df, projected_df) : DataFrames with eligible_slots as list[str]
    """
    def _latest(prefix: str) -> str:
        files = sorted(
            [f for f in os.listdir(gold_dir) if f.startswith(prefix) and f.endswith(".csv")],
            reverse=True,
        )
        if not files:
            raise FileNotFoundError(f"No {prefix}*.csv found in {gold_dir}")
        return os.path.join(gold_dir, files[0])

    arch_df = pd.read_csv(_latest("archetypes_"))
    proj_df = pd.read_csv(_latest("projections_with_arch_"))

    for df in (arch_df, proj_df):
        df["eligible_slots"] = df["eligible_slots"].apply(
            lambda v: [s for s in str(v).split("|") if s] if pd.notna(v) else []
        )

    return arch_df, proj_df
