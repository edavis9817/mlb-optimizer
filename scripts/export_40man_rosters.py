#!/usr/bin/env python3
"""
export_40man_rosters.py
-----------------------
Fetch all 30 MLB 40-man rosters and save to data/40man_rosters_2026.csv.

Usage:
    python scripts/export_40man_rosters.py          # default season=2025
    python scripts/export_40man_rosters.py --season 2026
"""

import argparse
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mlb_rosters import fetch_all_40man_rosters, TEAM_IDS, _ttl_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 40-man rosters to CSV")
    parser.add_argument("--season", type=int, default=2025, help="MLB season year (default: 2025)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: data/40man_rosters_{season}.csv)")
    args = parser.parse_args()

    season = args.season
    out_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "data", f"40man_rosters_{season}.csv"
    )
    out_path = os.path.abspath(out_path)

    print(f"Fetching 40-man rosters for {season} season...")
    print(f"Output: {out_path}\n")

    df = fetch_all_40man_rosters(season=season, _ttl=_ttl_hash())

    if df.empty:
        print("ERROR: No roster data returned. Check your internet connection and MLB-StatsAPI installation.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"{'Team':<6} {'Players':>8}  {'Status'}")
    print("-" * 36)

    teams_found = set()
    for abbr in sorted(TEAM_IDS.keys()):
        team_df = df[df["team"] == abbr]
        count = len(team_df)
        teams_found.add(abbr)

        flag = ""
        if count == 0:
            flag = "  *** WARNING: 0 players ***"
        elif count > 42:
            flag = f"  *** WARNING: {count} players (>42) ***"
        elif count < 25:
            flag = f"  (low — {count})"

        print(f"{abbr:<6} {count:>8}  {flag}")

    missing = set(TEAM_IDS.keys()) - teams_found
    if missing:
        print(f"\nMISSING TEAMS: {', '.join(sorted(missing))}")

    print(f"\nTotal: {len(df)} players across {len(df['team'].unique())} teams")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
