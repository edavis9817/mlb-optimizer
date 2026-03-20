"""
check_team_payrolls.py
----------------------
Exports team payroll totals for each year (2021-2025) to a CSV and
prints a formatted summary to the console.

Run from anywhere:
    python check_team_payrolls.py

Output file:
    team_payrolls_by_year.csv  (same folder as this script)
"""

import os
import re
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

YEAR_FILES = {
    2021: "2021mlbshared.csv",
    2022: "2022mlbshared.csv",
    2023: "2023mlbshared.csv",
    2024: "2024mlbshared.csv",
    2025: "2025mlbshared.csv",
}

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_payrolls_by_year.csv")


# ---------------------------------------------------------------------------
# Parse a salary string → float millions
# ---------------------------------------------------------------------------
def parse_salary(val) -> float:
    """
    Convert a salary cell to millions of dollars.
    Handles: '$14,000,000', '$8,125,000.00', '925000', NaN, 'FREE AGENT', etc.
    Returns 0.0 for anything that isn't a committed dollar amount.
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip()

    # Remove leading $
    s = s.replace("$", "").replace(",", "").strip()

    # Skip non-numeric markers
    if not s or not re.match(r"^\d", s):
        return 0.0

    try:
        raw = float(s)
        # Values already look like they're in dollars (>= 100k)
        # Convert to millions
        return raw / 1_000_000 if raw >= 1_000 else raw  # guard against already-M values
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_records = []

    for year, fname in sorted(YEAR_FILES.items()):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {fname} not found")
            continue

        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]

        year_col = str(year)
        if year_col not in df.columns:
            print(f"  [WARN] {fname}: no '{year_col}' column — skipping year")
            continue

        # Also collect AAV as a cross-check column
        aav_col = next((c for c in ("AAV", " AAV ") if c in df.columns), None)

        df["_sal_M"]  = df[year_col].apply(parse_salary)
        df["_aav_M"]  = df[aav_col].apply(parse_salary) if aav_col else 0.0

        # Deduplicate traded players: a player who changed teams mid-season
        # appears once per team; keep only their highest-salary row so they
        # aren't double-counted in league-wide totals.
        df = (
            df.sort_values("_sal_M", ascending=False, kind="mergesort")
              .drop_duplicates(subset=["Player"], keep="first")
              .reset_index(drop=True)
        )

        for team, grp in df.groupby("Team"):
            committed = grp["_sal_M"].sum()
            aav_total = grp["_aav_M"].sum()
            n_players = len(grp)
            n_locked  = (grp["_sal_M"] > 0).sum()
            all_records.append({
                "Year":           year,
                "Team":           team,
                "Committed_M":    round(committed, 2),
                "AAV_Total_M":    round(aav_total, 2),
                "N_Players":      n_players,
                "N_With_Salary":  n_locked,
            })

        print(f"  Loaded {year}: {len(df)} players across {df['Team'].nunique()} teams")

    result = pd.DataFrame(all_records).sort_values(["Year", "Team"])

    # -----------------------------------------------------------------------
    # Console summary — pivot: teams as rows, years as columns
    # -----------------------------------------------------------------------
    pivot = result.pivot_table(
        index="Team", columns="Year", values="Committed_M", aggfunc="sum"
    ).round(1)

    pivot["3yr_avg"] = pivot[[2023, 2024, 2025]].mean(axis=1).round(1)
    pivot = pivot.sort_values("3yr_avg", ascending=False)

    print("\n" + "=" * 80)
    print("TEAM PAYROLLS BY YEAR ($M committed salary from year-specific columns)")
    print("=" * 80)
    print(pivot.to_string())
    print()

    # -----------------------------------------------------------------------
    # Save to CSV
    # -----------------------------------------------------------------------
    result.to_csv(OUTPUT_CSV, index=False)
    pivot.reset_index().to_csv(
        OUTPUT_CSV.replace(".csv", "_pivot.csv"), index=False
    )
    print(f"Saved detail rows  -> {OUTPUT_CSV}")
    print(f"Saved pivot table  -> {OUTPUT_CSV.replace('.csv', '_pivot.csv')}")

    # -----------------------------------------------------------------------
    # Quick sanity check: show top/bottom spenders per year
    # -----------------------------------------------------------------------
    print("\n--- Top 5 spenders each year ---")
    for yr in sorted(result["Year"].unique()):
        top5 = (
            result[result["Year"] == yr]
            .nlargest(5, "Committed_M")[["Team", "Committed_M"]]
        )
        teams_str = "  ".join(f"{r['Team']} ${r['Committed_M']:.0f}M" for _, r in top5.iterrows())
        print(f"  {yr}: {teams_str}")

    print("\n--- Bottom 5 spenders each year ---")
    for yr in sorted(result["Year"].unique()):
        bot5 = (
            result[result["Year"] == yr]
            .nsmallest(5, "Committed_M")[["Team", "Committed_M"]]
        )
        teams_str = "  ".join(f"{r['Team']} ${r['Committed_M']:.0f}M" for _, r in bot5.iterrows())
        print(f"  {yr}: {teams_str}")


if __name__ == "__main__":
    main()
