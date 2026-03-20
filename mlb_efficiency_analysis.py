"""
mlb_efficiency_analysis.py
---------------------------
Generates three charts and two CSVs analyzing team spending efficiency
relative to the wins frontier.

Questions answered:
  1. How many $M is each team spending above/below the efficiency frontier?
  2. Which positions/players drive a team above or below the line?
  3. Is there a correlation between frontier efficiency and winning success?

Output files (same folder as this script):
  efficiency_scatter.png       – Payroll vs Wins scatter with frontier by year
  efficiency_ranking.png       – Teams ranked by avg $ above/below frontier
  position_breakdown.png       – WAR by position for efficient vs inefficient teams
  al_nl_ranking_table.csv      – Full AL/NL ranking table for printing
  efficiency_detail.csv        – Year-by-year team detail

Run:
  python mlb_efficiency_analysis.py
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Data")

COMBINED_PATH = os.path.join(DATA_DIR, "mlb_combined_2021_2025.csv")
WINS_PATH     = os.path.join(DATA_DIR, "sportsref_download.csv")

YEARS = [2021, 2022, 2023, 2024, 2025]

# ---------------------------------------------------------------------------
# League / division metadata
# ---------------------------------------------------------------------------
AL = {"BAL","BOS","NYY","TBR","TOR",          # AL East
      "CHW","CLE","DET","KCR","MIN",          # AL Central
      "HOU","LAA","ATH","SEA","TEX"}          # AL West

NL = {"ATL","MIA","NYM","PHI","WSN",          # NL East
      "CHC","CIN","MIL","PIT","STL",          # NL Central
      "ARI","COL","LAD","SDP","SFG"}          # NL West

DIVISION = {
    "BAL":"AL East","BOS":"AL East","NYY":"AL East","TBR":"AL East","TOR":"AL East",
    "CHW":"AL Central","CLE":"AL Central","DET":"AL Central","KCR":"AL Central","MIN":"AL Central",
    "HOU":"AL West","LAA":"AL West","ATH":"AL West","SEA":"AL West","TEX":"AL West",
    "ATL":"NL East","MIA":"NL East","NYM":"NL East","PHI":"NL East","WSN":"NL East",
    "CHC":"NL Central","CIN":"NL Central","MIL":"NL Central","PIT":"NL Central","STL":"NL Central",
    "ARI":"NL West","COL":"NL West","LAD":"NL West","SDP":"NL West","SFG":"NL West",
}

# ---------------------------------------------------------------------------
# Playoff / World Series data  (2021–2025)
# 2025: LAD def. TOR in World Series (4-3)
#   ALCS: TOR def. SEA  |  NLCS: LAD def. MIL
#   ALDS: TOR def. NYY, SEA def. DET  |  NLDS: LAD def. PHI, MIL def. CHC
#   ALWCS: DET def. CLE, NYY def. BOS  |  NLWCS: CHC def. SDP, LAD def. CIN
# ---------------------------------------------------------------------------
PLAYOFF_TEAMS = {
    2021: {"ATL","LAD","MIL","SFG","STL",
           "HOU","TBR","CHW","BOS","NYY"},
    2022: {"LAD","ATL","STL","SDP","NYM","PHI",
           "HOU","NYY","CLE","SEA","TOR","TBR"},
    2023: {"ATL","LAD","MIL","PHI","ARI","MIA",
           "BAL","HOU","TEX","MIN","TOR","TBR"},
    2024: {"LAD","PHI","MIL","SDP","ARI","NYM",
           "NYY","CLE","HOU","KCR","BAL","DET"},
    2025: {"LAD","TOR","SEA","MIL",          # WS + LCS
           "NYY","DET","PHI","CHC",          # DS
           "CLE","BOS","SDP","CIN"},         # Wild Card
}

WS_CHAMP   = {2021:"ATL", 2022:"HOU", 2023:"TEX", 2024:"LAD", 2025:"LAD"}
WS_RUNNERUP= {2021:"HOU", 2022:"PHI", 2023:"ARI", 2024:"NYY", 2025:"TOR"}

# ---------------------------------------------------------------------------
# Position-group map (mirrors projections.py)
# ---------------------------------------------------------------------------
_PG = {
    "C":"C","1B":"CI","3B":"CI","2B":"MI","SS":"MI",
    "LF":"OF","RF":"OF","OF":"OF","CF":"CF",
    "SP":"SP","RP":"RP","TWP":"SP","DH":"DH",
}
def pos_group(p):
    return _PG.get(str(p).strip(), "Other")

# ---------------------------------------------------------------------------
# 1. Load & aggregate data
# ---------------------------------------------------------------------------
print("Loading data …")
comb = pd.read_csv(COMBINED_PATH, low_memory=False)
comb.columns = [c.strip() for c in comb.columns]
comb["Year"] = pd.to_numeric(comb["Year"], errors="coerce").astype("Int64")
comb = comb[comb["Year"].isin(YEARS)].copy()
comb["Salary_M"] = pd.to_numeric(comb["Salary"], errors="coerce") / 1e6
comb["WAR_Total"] = pd.to_numeric(comb["WAR_Total"], errors="coerce").fillna(0)
comb["pos_group"] = comb["Position"].apply(pos_group)

# Deduplicate traded players per team-year (keep highest-salary row)
comb = (
    comb.sort_values("Salary_M", ascending=False, kind="mergesort")
        .drop_duplicates(["Player", "Year"], keep="first")
        .reset_index(drop=True)
)

# Team-level aggregation
team_yr = (
    comb.groupby(["Year","Team"])
        .agg(
            payroll_M = ("Salary_M",   "sum"),
            team_WAR  = ("WAR_Total",  "sum"),
            n_players = ("Player",     "count"),
        )
        .reset_index()
)

# Position-level WAR per team-year
pos_yr = (
    comb.groupby(["Year","Team","pos_group"])
        .agg(pos_WAR=("WAR_Total","sum"), pos_pay_M=("Salary_M","sum"))
        .reset_index()
)

# Win records
wins_wide = pd.read_csv(WINS_PATH)
wins_wide.columns = [c.strip() for c in wins_wide.columns]
wins_wide["Year"] = pd.to_numeric(wins_wide["Year"], errors="coerce").astype("Int64")
wins_wide = wins_wide[wins_wide["Year"].isin(YEARS)].copy()

# Melt wins to long format
id_cols = ["Year","G"]
team_cols = [c for c in wins_wide.columns if c not in id_cols]
wins_long = wins_wide.melt(id_vars=id_cols, value_vars=team_cols,
                           var_name="Team", value_name="Wins")
wins_long["Wins"] = pd.to_numeric(wins_long["Wins"], errors="coerce")
wins_long = wins_long.dropna(subset=["Wins"])

# Merge
df = team_yr.merge(wins_long[["Year","Team","Wins"]], on=["Year","Team"], how="inner")
df["league"]   = df["Team"].map(lambda t: "AL" if t in AL else "NL")
df["division"] = df["Team"].map(DIVISION)
df["in_playoffs"] = df.apply(
    lambda r: r["Team"] in PLAYOFF_TEAMS.get(int(r["Year"]), set()), axis=1
)
df["ws_champ"]    = df.apply(lambda r: WS_CHAMP.get(int(r["Year"])) == r["Team"], axis=1)
df["ws_runnerup"] = df.apply(lambda r: WS_RUNNERUP.get(int(r["Year"])) == r["Team"], axis=1)

print(f"  {len(df)} team-year rows across {df['Team'].nunique()} teams, "
      f"{df['Year'].nunique()} seasons\n")

# ---------------------------------------------------------------------------
# 2. Compute efficiency frontier (OLS: Wins ~ Payroll per year)
#    "$ to frontier" = actual_payroll - payroll_that_would_predict_actual_wins
# ---------------------------------------------------------------------------
residuals = []
frontier_lines = {}

for year in YEARS:
    sub = df[df["Year"] == year].dropna(subset=["payroll_M","Wins"])
    if len(sub) < 5:
        continue
    slope, intercept, r, p, se = stats.linregress(sub["payroll_M"], sub["Wins"])
    frontier_lines[year] = (slope, intercept, r**2)
    # Predicted payroll needed to achieve actual wins
    for _, row in sub.iterrows():
        pred_wins     = intercept + slope * row["payroll_M"]
        # payroll needed to get actual wins via the regression line
        pay_needed    = (row["Wins"] - intercept) / slope if slope != 0 else row["payroll_M"]
        dollar_gap_M  = row["payroll_M"] - pay_needed      # + = overpaying; - = bargain
        residuals.append({
            "Year":           int(year),
            "Team":           row["Team"],
            "payroll_M":      row["payroll_M"],
            "team_WAR":       row["team_WAR"],
            "Wins":           row["Wins"],
            "pred_wins":      round(pred_wins, 1),
            "wins_vs_pred":   round(row["Wins"] - pred_wins, 1),
            "dollar_gap_M":   round(dollar_gap_M, 1),
            "league":         row["league"],
            "division":       row["division"],
            "in_playoffs":    row["in_playoffs"],
            "ws_champ":       row["ws_champ"],
            "ws_runnerup":    row["ws_runnerup"],
            "r2":             round(r**2, 3),
        })

res_df = pd.DataFrame(residuals)

# ---------------------------------------------------------------------------
# 3. Team summary across all years
# ---------------------------------------------------------------------------
team_sum = (
    res_df.groupby("Team")
          .agg(
              league         = ("league",       "first"),
              division       = ("division",     "first"),
              avg_payroll_M  = ("payroll_M",    "mean"),
              avg_WAR        = ("team_WAR",     "mean"),
              avg_wins       = ("Wins",         "mean"),
              avg_gap_M      = ("dollar_gap_M", "mean"),    # + = overspending
              avg_wins_pred  = ("wins_vs_pred", "mean"),    # + = outperforming
              playoff_apps   = ("in_playoffs",  "sum"),
              ws_champs      = ("ws_champ",     "sum"),
              ws_runnerups   = ("ws_runnerup",  "sum"),
          )
          .reset_index()
)

# Add per-year wins columns
for yr in YEARS:
    yr_wins = df[df["Year"] == yr].set_index("Team")["Wins"]
    team_sum[f"W{yr}"] = team_sum["Team"].map(yr_wins)

team_sum["ws_appearances"] = team_sum["ws_champs"] + team_sum["ws_runnerups"]
team_sum = team_sum.sort_values("avg_gap_M").reset_index(drop=True)   # most efficient first
team_sum["eff_rank"] = range(1, len(team_sum) + 1)

# ---------------------------------------------------------------------------
# 4. CHART 1 — Payroll vs Wins scatter (2×3 grid)
# ---------------------------------------------------------------------------
print("Drawing Chart 1: Payroll vs Wins scatter …")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor("#0e1117")
axes_flat = axes.flatten()

al_color = "#3498db"
nl_color = "#e74c3c"

for ax_i, year in enumerate(YEARS):
    ax = axes_flat[ax_i]
    ax.set_facecolor("#1a1a2e")

    sub = res_df[res_df["Year"] == year]
    if sub.empty:
        ax.set_visible(False)
        continue

    # Regression line
    if year in frontier_lines:
        slope, intercept, r2 = frontier_lines[year]
        x_range = np.linspace(sub["payroll_M"].min() * 0.9, sub["payroll_M"].max() * 1.05, 200)
        ax.plot(x_range, intercept + slope * x_range,
                color="#f0c040", lw=2, alpha=0.85, label=f"Frontier (R²={r2:.2f})", zorder=2)

    # Scatter
    for _, row in sub.iterrows():
        color = al_color if row["league"] == "AL" else nl_color
        marker = "*" if row["ws_champ"] else ("D" if row["ws_runnerup"] else "o")
        ms = 14 if row["ws_champ"] else (10 if row["ws_runnerup"] else 7)
        zorder = 6 if row["ws_champ"] else (5 if row["in_playoffs"] else 3)
        alpha = 1.0 if row["in_playoffs"] else 0.55
        ax.scatter(row["payroll_M"], row["Wins"],
                   c=color, marker=marker, s=ms**2, alpha=alpha,
                   edgecolors="white" if row["ws_champ"] else "none",
                   linewidths=1.2, zorder=zorder)
        # Label
        ax.text(row["payroll_M"] + 1.5, row["Wins"] + 0.5,
                row["Team"], fontsize=6.5, color="white", alpha=0.88, zorder=7)

    ax.set_xlabel("Payroll ($M)", color="white", fontsize=9)
    ax.set_ylabel("Wins", color="white", fontsize=9)
    ax.set_title(f"{year}", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.legend(fontsize=7, loc="lower right",
              facecolor="#0e1117", labelcolor="white")

# Hide unused 6th panel, use it as legend
axes_flat[5].set_facecolor("#0e1117")
axes_flat[5].axis("off")
legend_elements = [
    Line2D([0], [0], marker="*",  color="w", markerfacecolor="#3498db", markersize=12, label="AL – WS Champion"),
    Line2D([0], [0], marker="*",  color="w", markerfacecolor="#e74c3c", markersize=12, label="NL – WS Champion"),
    Line2D([0], [0], marker="D",  color="w", markerfacecolor="#3498db", markersize=8,  label="AL – WS Runner-up"),
    Line2D([0], [0], marker="D",  color="w", markerfacecolor="#e74c3c", markersize=8,  label="NL – WS Runner-up"),
    Line2D([0], [0], marker="o",  color="w", markerfacecolor="#3498db", markersize=7,  label="AL – Playoff team"),
    Line2D([0], [0], marker="o",  color="w", markerfacecolor="#e74c3c", markersize=7,  label="NL – Playoff team"),
    Line2D([0], [0], marker="o",  color="w", markerfacecolor="#3498db", markersize=7, alpha=0.45, label="AL – Missed playoffs"),
    Line2D([0], [0], marker="o",  color="w", markerfacecolor="#e74c3c", markersize=7, alpha=0.45, label="NL – Missed playoffs"),
    Line2D([0], [0], color="#f0c040", lw=2, label="Efficiency Frontier"),
]
axes_flat[5].legend(handles=legend_elements, loc="center", fontsize=9,
                    facecolor="#0e1117", labelcolor="white", framealpha=0.9)
axes_flat[5].set_title("Legend", color="white", fontsize=10)

fig.suptitle("MLB Team Payroll vs Wins — Efficiency Frontier (2021–2025)",
             color="white", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out1 = os.path.join(HERE, "efficiency_scatter.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved -> {out1}")

# ---------------------------------------------------------------------------
# 5. CHART 2 — Team ranking: avg $M above/below frontier, split AL/NL
# ---------------------------------------------------------------------------
print("Drawing Chart 2: Efficiency ranking bar chart …")

fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=False)
fig.patch.set_facecolor("#0e1117")

for ax, league in zip(axes, ["AL", "NL"]):
    ax.set_facecolor("#1a1a2e")
    sub = team_sum[team_sum["league"] == league].sort_values("avg_gap_M")

    colors = []
    for _, row in sub.iterrows():
        if row["ws_champs"] > 0:
            colors.append("#FFD700")          # gold = WS winner
        elif row["ws_appearances"] > 0:
            colors.append("#C0C0C0")          # silver = WS appearance
        elif row["playoff_apps"] >= 3:
            colors.append("#27ae60")           # green = consistent playoff
        elif row["playoff_apps"] >= 1:
            colors.append("#2980b9")           # blue = some playoffs
        else:
            colors.append("#7f8c8d")           # gray = no playoffs

    bars = ax.barh(sub["Team"], sub["avg_gap_M"], color=colors, edgecolor="#555", height=0.7)

    # Annotate each bar
    for bar, (_, row) in zip(bars, sub.iterrows()):
        x = bar.get_width()
        label_x = x + (1 if x >= 0 else -1)
        ha = "left" if x >= 0 else "right"
        stars = "★" * int(row["ws_champs"])
        playoff_str = f"P{int(row['playoff_apps'])}"
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f"  {x:+.0f}M  {playoff_str}{stars}",
                va="center", ha=ha, fontsize=7.5, color="white")

    ax.axvline(0, color="#f0c040", lw=1.5, linestyle="--", alpha=0.7)
    ax.set_title(f"{league} — Avg $ Above/Below Frontier (2021–2025)\n"
                 f"← More Efficient  |  Less Efficient →",
                 color="white", fontsize=11)
    ax.set_xlabel("Avg $M Over (+) / Under (−) Expected Payroll for Wins", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    # Footnote legend
    legend_els = [
        Line2D([0],[0], color="#FFD700", lw=6, label="WS Champion season"),
        Line2D([0],[0], color="#C0C0C0", lw=6, label="WS Runner-up season"),
        Line2D([0],[0], color="#27ae60", lw=6, label="3+ playoff appearances"),
        Line2D([0],[0], color="#2980b9", lw=6, label="1–2 playoff appearances"),
        Line2D([0],[0], color="#7f8c8d", lw=6, label="No playoff appearances"),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, loc="lower right",
              facecolor="#0e1117", labelcolor="white")

fig.suptitle("Spending Efficiency vs Playoff Success — AL & NL",
             color="white", fontsize=14, fontweight="bold")
plt.tight_layout()
out2 = os.path.join(HERE, "efficiency_ranking.png")
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved -> {out2}")

# ---------------------------------------------------------------------------
# 6. CHART 3 — Position WAR breakdown for top-5 efficient vs top-5 inefficient
# ---------------------------------------------------------------------------
print("Drawing Chart 3: Position breakdown …")

top5_eff   = team_sum.nsmallest(5, "avg_gap_M")["Team"].tolist()    # most efficient (negative gap)
top5_ineff = team_sum.nlargest(5, "avg_gap_M")["Team"].tolist()     # most inefficient

focus_teams = top5_eff + top5_ineff
pos_focus = pos_yr[pos_yr["Team"].isin(focus_teams)].copy()

# Average across years
pos_avg = (
    pos_focus.groupby(["Team","pos_group"])
             .agg(avg_WAR=("pos_WAR","mean"), avg_pay_M=("pos_pay_M","mean"))
             .reset_index()
)

POS_ORDER = ["SP","RP","C","CI","MI","CF","OF","DH","Other"]
POS_COLORS = {
    "SP":"#e74c3c","RP":"#e67e22","C":"#f1c40f","CI":"#2ecc71",
    "MI":"#1abc9c","CF":"#3498db","OF":"#9b59b6","DH":"#e91e63","Other":"#95a5a6",
}

fig, axes = plt.subplots(2, 5, figsize=(22, 10))
fig.patch.set_facecolor("#0e1117")

for ax_i, team in enumerate(top5_eff + top5_ineff):
    row_idx = 0 if ax_i < 5 else 1
    col_idx = ax_i % 5
    ax = axes[row_idx][col_idx]
    ax.set_facecolor("#1a1a2e")

    sub = pos_avg[pos_avg["Team"] == team].copy()
    sub = sub.set_index("pos_group").reindex(POS_ORDER).dropna(subset=["avg_WAR"])

    bar_colors = [POS_COLORS.get(pg, "#aaa") for pg in sub.index]
    bars = ax.bar(sub.index, sub["avg_WAR"], color=bar_colors, edgecolor="#444", width=0.7)

    gap_row = team_sum[team_sum["Team"] == team].iloc[0]
    gap_val  = gap_row["avg_gap_M"]
    eff_label = f"${gap_val:+.0f}M vs frontier"
    color_label = "#e74c3c" if gap_val > 0 else "#27ae60"

    ax.set_title(
        f"{team}\n{eff_label}",
        color=color_label if ax_i >= 5 else "#27ae60",
        fontsize=9, fontweight="bold",
    )
    ax.set_ylabel("Avg WAR" if col_idx == 0 else "", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7.5, axis="x", rotation=30)
    ax.tick_params(colors="white", labelsize=7.5, axis="y")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    # Annotate bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                f"{h:.1f}", ha="center", va="bottom", fontsize=6.5, color="white")

# Row labels
axes[0][0].set_ylabel("MOST EFFICIENT\n\nAvg WAR", color="#27ae60", fontsize=9)
axes[1][0].set_ylabel("MOST INEFFICIENT\n\nAvg WAR", color="#e74c3c", fontsize=9)

fig.suptitle("WAR Contribution by Position Group — Top 5 Efficient vs Inefficient Teams",
             color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
out3 = os.path.join(HERE, "position_breakdown.png")
fig.savefig(out3, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Saved -> {out3}")

# ---------------------------------------------------------------------------
# 7. Build the full AL/NL ranking table for printing / CSV
# ---------------------------------------------------------------------------
print("\nBuilding ranking table …")

TEAM_NAMES = {
    "ARI":"Arizona Diamondbacks","ATH":"Oakland/Sacramento Athletics","ATL":"Atlanta Braves",
    "BAL":"Baltimore Orioles","BOS":"Boston Red Sox","CHC":"Chicago Cubs",
    "CHW":"Chicago White Sox","CIN":"Cincinnati Reds","CLE":"Cleveland Guardians",
    "COL":"Colorado Rockies","DET":"Detroit Tigers","HOU":"Houston Astros",
    "KCR":"Kansas City Royals","LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers",
    "MIA":"Miami Marlins","MIL":"Milwaukee Brewers","MIN":"Minnesota Twins",
    "NYM":"New York Mets","NYY":"New York Yankees","PHI":"Philadelphia Phillies",
    "PIT":"Pittsburgh Pirates","SDP":"San Diego Padres","SFG":"San Francisco Giants",
    "SEA":"Seattle Mariners","STL":"St. Louis Cardinals","TBR":"Tampa Bay Rays",
    "TEX":"Texas Rangers","TOR":"Toronto Blue Jays","WSN":"Washington Nationals",
}

table_rows = []
for league in ["AL", "NL"]:
    sub = team_sum[team_sum["league"] == league].sort_values("avg_gap_M")
    league_rank = 1
    for _, row in sub.iterrows():
        team = row["Team"]
        # Win/loss record each year from df
        yr_wins = {}
        yr_losses = {}
        for yr in YEARS:
            w_row = df[(df["Team"] == team) & (df["Year"] == yr)]
            w = int(w_row["Wins"].values[0]) if not w_row.empty else None
            yr_wins[yr] = w
            yr_losses[yr] = (162 - w) if w is not None else None

        # Playoff wins: just count round appearances
        # (We'll use rough tier: WS win=3 rounds won, WS appearance=2, LCS=1, else 0)
        ws_wins = int(row["ws_champs"])
        ws_apps = int(row["ws_appearances"])
        playoff_apps = int(row["playoff_apps"])
        # Rough playoff rounds won heuristic
        playoff_rnd = ws_wins * 3 + (ws_apps - ws_wins) * 2 + max(0, playoff_apps - ws_apps) * 1

        table_rows.append({
            "League":        league,
            "Division":      row["division"],
            "Team":          TEAM_NAMES.get(team, team),
            "Abbr":          team,
            "Eff_Rank":      int(row["eff_rank"]),
            "Leag_Rank":     league_rank,
            f"W2021":        yr_wins.get(2021),
            f"W2022":        yr_wins.get(2022),
            f"W2023":        yr_wins.get(2023),
            f"W2024":        yr_wins.get(2024),
            f"W2025":        yr_wins.get(2025),
            "Avg_Wins":      round(float(row["avg_wins"]), 1),
            "Avg_Pay_M":     round(float(row["avg_payroll_M"]), 1),
            "Avg_$/WAR_M":   round(float(row["avg_payroll_M"]) / max(float(row["avg_WAR"]), 0.1), 2),
            "Avg_Gap_M":     round(float(row["avg_gap_M"]), 1),
            "Playoff_Apps":  playoff_apps,
            "WS_Apps":       ws_apps,
            "WS_Wins":       ws_wins,
            "Playoff_Rnds":  playoff_rnd,
        })
        league_rank += 1

rank_df = pd.DataFrame(table_rows)

# Save CSV
out_csv = os.path.join(HERE, "al_nl_ranking_table.csv")
rank_df.to_csv(out_csv, index=False)

# Save detail per year
detail_out = os.path.join(HERE, "efficiency_detail.csv")
res_df.to_csv(detail_out, index=False)

# ---------------------------------------------------------------------------
# 8. Print formatted tables to console
# ---------------------------------------------------------------------------
def print_league_table(league_df: pd.DataFrame, header: str):
    print(f"\n{'='*130}")
    print(f"  {header}")
    print(f"{'='*130}")
    h = (f"{'Rank':>4}  {'Team':<32} {'Div':<12} "
         f"{'2021':>4} {'2022':>4} {'2023':>4} {'2024':>4} {'2025':>4}  "
         f"{'AvgW':>5}  {'Payroll':>8}  {'$/WAR':>5}  "
         f"{'$vsLine':>8}  {'Playoffs':>8}  {'WS App':>6}  {'WS Wins':>7}  {'PO Rnds':>7}")
    print(h)
    print("-" * 130)
    for _, r in league_df.iterrows():
        ws_star = " *" if r["WS_Wins"] > 0 else ("  " if r["WS_Apps"] == 0 else " ^")
        gap_str = f"${r['Avg_Gap_M']:+.0f}M"
        print(
            f"{r['Leag_Rank']:>4}  {r['Team']:<32} {r['Division']:<12} "
            f"{str(r['W2021']) if r['W2021'] else '--':>4} "
            f"{str(r['W2022']) if r['W2022'] else '--':>4} "
            f"{str(r['W2023']) if r['W2023'] else '--':>4} "
            f"{str(r['W2024']) if r['W2024'] else '--':>4} "
            f"{str(r['W2025']) if r['W2025'] else '--':>4}  "
            f"{r['Avg_Wins']:>5.1f}  "
            f"${r['Avg_Pay_M']:>6.0f}M  "
            f"${r['Avg_$/WAR_M']:>4.1f}M  "
            f"{gap_str:>8}  "
            f"{r['Playoff_Apps']:>8}  "
            f"{r['WS_Apps']:>6}  "
            f"{r['WS_Wins']:>6}{ws_star}  "
            f"{r['Playoff_Rnds']:>7}"
        )
    print(f"{'='*130}")
    print("  Rank = by spending efficiency (most efficient first = spent least ABOVE frontier)")
    print("  $vsLine = avg $M team spent above (+) or below (-) the wins frontier")
    print("  $/WAR = avg payroll / avg team WAR")
    print("  PO Rnds = est. cumulative playoff rounds won (WS win=3, runner-up=2, div=1)")
    print("  * = WS Champion  ^ = WS Runner-up")

al_table = rank_df[rank_df["League"] == "AL"].sort_values("Leag_Rank")
nl_table = rank_df[rank_df["League"] == "NL"].sort_values("Leag_Rank")

print_league_table(al_table, "AMERICAN LEAGUE -- Team Efficiency Ranking (Most Efficient first)")
print_league_table(nl_table, "NATIONAL LEAGUE -- Team Efficiency Ranking (Most Efficient first)")

# ---------------------------------------------------------------------------
# 9. Print frontier model stats
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("  EFFICIENCY FRONTIER MODEL — Year-by-Year R² (Payroll vs Wins OLS)")
print(f"{'='*70}")
for yr, (slope, intercept, r2) in frontier_lines.items():
    print(f"  {yr}:  slope={slope:.4f} wins/$M   intercept={intercept:.1f}   R²={r2:.3f}")
    corr_row = res_df[res_df["Year"] == yr]
    above = (corr_row["dollar_gap_M"] > 0).sum()
    below = (corr_row["dollar_gap_M"] < 0).sum()
    print(f"         {above} teams overspent | {below} teams underspent relative to frontier")

# Correlation: efficiency gap vs wins, playoffs, etc.
print(f"\n{'='*70}")
print("  CORRELATION: $ Gap vs Outcomes (full dataset, all years)")
print(f"{'='*70}")
from scipy.stats import pearsonr, pointbiserialr
r_wins, p_wins = pearsonr(res_df["dollar_gap_M"], res_df["Wins"])
print(f"  $ gap vs Wins:     r = {r_wins:+.3f}  p = {p_wins:.4f}  "
      f"{'(significant)' if p_wins < 0.05 else '(not significant)'}")

playoff_bin = res_df["in_playoffs"].astype(int)
r_po, p_po = pointbiserialr(res_df["dollar_gap_M"], playoff_bin)
print(f"  $ gap vs Playoffs: r = {r_po:+.3f}  p = {p_po:.4f}  "
      f"{'(significant)' if p_po < 0.05 else '(not significant)'}")

print(f"\nFiles saved:")
print(f"  {out1}")
print(f"  {out2}")
print(f"  {out3}")
print(f"  {out_csv}")
print(f"  {detail_out}")
