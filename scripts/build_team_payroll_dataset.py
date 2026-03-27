"""
build_team_payroll_dataset.py
-----------------------------
Reads 2026 Payroll Excel files to produce two team-level summary CSVs:
  1. team_payroll_summary.csv  — one row per team per year (2026-2032)
  2. team_payroll_by_stage.csv — one row per team per year per contract stage

Uses the Luxury Tax Payroll Estimate sheet as the authoritative CBT number.
"""

import os
import sys
import re
import pandas as pd
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAYROLL_DIR = r"C:\Users\Ethan Davis\Downloads\Payrolls"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = list(range(2026, 2033))  # 2026-2032
YEAR_STRS = [str(y) for y in YEARS]

CBT_THRESHOLDS = {
    2026: 244, 2027: 251, 2028: 258, 2029: 265,
    2030: 272, 2031: 279, 2032: 286,
}

TEAM_MAP = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH",
    "Blue Jays": "TOR", "Braves": "ATL", "Brewers": "MIL",
    "Cardinals": "STL", "Cubs": "CHC", "Diamondbacks": "ARI",
    "Dodgers": "LAD", "Giants": "SFG", "Guardians": "CLE",
    "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM",
    "Nationals": "WSN", "Orioles": "BAL", "Padres": "SDP",
    "Phillies": "PHI", "Pirates": "PIT", "Rangers": "TEX",
    "Rays": "TBR", "Red Sox": "BOS", "Reds": "CIN",
    "Rockies": "COL", "Royals": "KCR", "Tigers": "DET",
    "Twins": "MIN", "White Sox": "CHW", "Yankees": "NYY",
}

SHEETS = {
    "Guaranteed": "Guaranteed",
    "Eligible For Arb": "Arb-Eligible",
    "Not Yet Eligible For Arb": "Pre-Arb",
}


def parse_dollar(val):
    """Convert dollar string like '$35,541,667' or '-$3,000,000' to float in $M."""
    if pd.isna(val):
        return None
    s = str(val).replace(",", "").replace("$", "").strip()
    if not s or s.lower() in ("nan", "none", "tbd", ""):
        return None
    try:
        v = float(s)
        if abs(v) > 1000:
            v /= 1_000_000
        return round(v, 3)
    except ValueError:
        return None


def is_signed_status(val):
    """Check if a year column value represents a signed/committed salary."""
    if pd.isna(val):
        return False
    s = str(val).strip()
    # It's signed if it looks like a dollar amount, not "ARB N", "FREE AGENT", "Pre-ARB", "TBD"
    if any(kw in s.upper() for kw in ["ARB", "FREE", "PRE-ARB", "TBD"]):
        return False
    # Must contain a digit to be a salary
    return bool(re.search(r"\d", s))


def classify_year_status(val):
    """Classify what a player's status is for a given year column."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    if "FREE AGENT" in s:
        return "FREE AGENT"
    if "PRE-ARB" in s or "PRE ARB" in s:
        return "Pre-ARB"
    if re.match(r"ARB\s*\d", s):
        return s.replace(" ", " ")  # "ARB 1", "ARB 2", etc.
    if "TBD" in s:
        return "Pre-ARB"
    if re.search(r"\d", s):
        return "Signed"
    return None


# ---------------------------------------------------------------------------
# Process each team
# ---------------------------------------------------------------------------
summary_rows = []
stage_rows = []
team_lux = {}  # team -> {year: luxury_tax_M}

files = sorted(f for f in os.listdir(PAYROLL_DIR) if f.endswith(".xlsx"))
print(f"Found {len(files)} payroll files in {PAYROLL_DIR}\n")

for fname in files:
    team_name = fname.replace("-Payroll-2026.xlsx", "")
    team_abbr = TEAM_MAP.get(team_name, team_name[:3].upper())
    fpath = os.path.join(PAYROLL_DIR, fname)

    # --- Read Luxury Tax Payroll Estimate ---
    lux_values = {}
    try:
        lux_df = pd.read_excel(fpath, sheet_name="Luxury Tax Payroll Estimate", header=None)
        # Find the "Estimated Luxury Tax Payroll" row
        for idx, row in lux_df.iterrows():
            desc = str(row.iloc[0]).strip().lower()
            if "estimated luxury tax payroll" in desc:
                for ci, yr in enumerate(YEARS):
                    col_idx = ci + 1  # columns 1-7 = 2026-2032
                    if col_idx < len(row):
                        lux_values[yr] = parse_dollar(row.iloc[col_idx])
                break
    except Exception as e:
        print(f"  WARNING: Could not read Luxury Tax sheet for {team_abbr}: {e}")
    team_lux[team_abbr] = lux_values

    # --- Read player sheets ---
    # Track per-year data
    year_data = {yr: {
        "guaranteed_salary": 0.0,
        "arb_count": 0,
        "pre_arb_count": 0,
        "fa_count": 0,
        "players_under_contract": 0,
        "stage_salary": {"Guaranteed": 0.0, "Arb-Eligible": 0.0, "Pre-Arb": 0.0},
        "stage_count": {"Guaranteed": 0, "Arb-Eligible": 0, "Pre-Arb": 0},
    } for yr in YEARS}

    for sheet_name, stage_label in SHEETS.items():
        try:
            df = pd.read_excel(fpath, sheet_name=sheet_name, header=0)
        except Exception:
            continue

        # Find year columns
        yr_col_map = {}  # year_int -> column_name
        for col in df.columns:
            col_s = str(col).strip()
            if col_s.isdigit() and int(col_s) in YEARS:
                yr_col_map[int(col_s)] = col
            elif col_s.replace(".0", "").isdigit() and int(float(col_s)) in YEARS:
                yr_col_map[int(float(col_s))] = col

        for _, player_row in df.iterrows():
            if pd.isna(player_row.get("Player")):
                continue

            for yr in YEARS:
                if yr not in yr_col_map:
                    continue
                cell = player_row.get(yr_col_map[yr])
                status = classify_year_status(cell)

                if status is None:
                    continue
                elif status == "Signed":
                    sal = parse_dollar(cell)
                    if sal is not None:
                        year_data[yr]["guaranteed_salary"] += sal
                        year_data[yr]["players_under_contract"] += 1
                        year_data[yr]["stage_salary"][stage_label] += sal
                        year_data[yr]["stage_count"][stage_label] += 1
                elif "ARB" in status and "PRE" not in status:
                    year_data[yr]["arb_count"] += 1
                    year_data[yr]["stage_count"]["Arb-Eligible"] += 1
                elif status == "Pre-ARB":
                    year_data[yr]["pre_arb_count"] += 1
                    year_data[yr]["stage_count"]["Pre-Arb"] += 1
                elif status == "FREE AGENT":
                    year_data[yr]["fa_count"] += 1

    # --- Read Other Payments ---
    other_payments = {yr: 0.0 for yr in YEARS}
    try:
        op_df = pd.read_excel(fpath, sheet_name="Other Payments", header=0)
        for col in op_df.columns:
            col_s = str(col).strip().replace(".0", "")
            if col_s.isdigit() and int(col_s) in YEARS:
                yr_int = int(col_s)
                for val in op_df[col].dropna():
                    p = parse_dollar(val)
                    if p is not None:
                        other_payments[yr_int] += p
    except Exception:
        pass

    # --- Build summary rows ---
    for yr in YEARS:
        yd = year_data[yr]
        lux = lux_values.get(yr)
        cbt = CBT_THRESHOLDS[yr]
        over_cbt = lux > cbt if lux is not None else None
        cbt_overage = round(lux - cbt, 1) if lux is not None else None

        summary_rows.append({
            "team": team_abbr,
            "year": yr,
            "guaranteed_salary": round(yd["guaranteed_salary"], 2),
            "arb_players_count": yd["arb_count"],
            "pre_arb_count": yd["pre_arb_count"],
            "fa_count": yd["fa_count"],
            "players_under_contract": yd["players_under_contract"],
            "other_payments_net": round(other_payments[yr], 2),
            "luxury_tax_estimate": round(lux, 2) if lux is not None else None,
            "cbt_threshold": cbt,
            "over_cbt": over_cbt,
            "cbt_overage": cbt_overage,
        })

        # Stage breakdown rows
        for stg in ["Guaranteed", "Arb-Eligible", "Pre-Arb"]:
            stage_rows.append({
                "team": team_abbr,
                "year": yr,
                "stage": stg,
                "total_salary": round(yd["stage_salary"][stg], 2),
                "player_count": yd["stage_count"][stg],
            })
        # Other payments as a stage
        if other_payments[yr] != 0:
            stage_rows.append({
                "team": team_abbr,
                "year": yr,
                "stage": "Other",
                "total_salary": round(other_payments[yr], 2),
                "player_count": 0,
            })

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "team_payroll_summary.csv"), index=False)

stage_df = pd.DataFrame(stage_rows)
stage_df.to_csv(os.path.join(OUT_DIR, "team_payroll_by_stage.csv"), index=False)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("2026 LUXURY TAX vs CBT THRESHOLD ($244M)")
print("=" * 60)

s26 = summary_df[summary_df["year"] == 2026].sort_values("luxury_tax_estimate", ascending=False)
for _, row in s26.iterrows():
    lux = row["luxury_tax_estimate"]
    if pd.isna(lux):
        print(f"  {row['team']}: No luxury tax estimate available")
        continue
    cbt = row["cbt_threshold"]
    diff = lux - cbt
    if diff > 0:
        print(f"  {row['team']}: ${lux:.0f}M luxury tax | ${diff:.0f}M OVER CBT (${cbt}M threshold)")
    else:
        print(f"  {row['team']}: ${lux:.0f}M luxury tax | ${abs(diff):.0f}M UNDER CBT (${cbt}M threshold) ✓")

# Stats
print(f"\n{'=' * 60}")
over_26 = s26[s26["over_cbt"] == True]
print(f"Teams over CBT in 2026: {len(over_26)}")

s27 = summary_df[summary_df["year"] == 2027]
over_27 = s27[s27["luxury_tax_estimate"].notna() & (s27["luxury_tax_estimate"] > 251)]
print(f"Teams projected over CBT in 2027: {len(over_27)}")

print(f"Largest 2026 luxury tax: {s26.iloc[0]['team']} ${s26.iloc[0]['luxury_tax_estimate']:.0f}M")
print(f"Smallest 2026 luxury tax: {s26.iloc[-1]['team']} ${s26.iloc[-1]['luxury_tax_estimate']:.0f}M")

total_26 = summary_df[summary_df["year"] == 2026]["guaranteed_salary"].sum()
total_27 = summary_df[summary_df["year"] == 2027]["guaranteed_salary"].sum()
print(f"League total committed salary 2026: ${total_26:.0f}M")
print(f"League total committed salary 2027: ${total_27:.0f}M (guaranteed only)")

print(f"\nSaved: {os.path.join(OUT_DIR, 'team_payroll_summary.csv')} ({len(summary_df)} rows)")
print(f"Saved: {os.path.join(OUT_DIR, 'team_payroll_by_stage.csv')} ({len(stage_df)} rows)")
