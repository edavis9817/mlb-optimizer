# MLB Optimizer — Project Context for RAG Planning

## What This Project Is

A data-driven MLB roster optimization and analysis platform. It combines integer linear programming, Monte Carlo simulation, and interactive data visualization to answer questions like:

- Given a fixed payroll budget, what mix of players maximizes expected wins?
- Which players are underpaid or overpaid relative to their production?
- How efficiently is each team converting payroll into wins?
- How does player production change with age?
- Which pre-arbitration players are improving vs. declining?

The platform has two entry points:
- **CLI** (`run_pipeline.py`) — batch optimization runs
- **Web UI** (`app/streamlit_app.py`) — interactive Streamlit app deployed on Render.com

---

## Data Available

### Primary Dataset
**`mlb_combined_2021_2025.csv`** — the core dataset, ~4,000+ player-seasons

| Column | Description |
|--------|-------------|
| `Player` | Player name |
| `Year` | Season (2021–2025) |
| `Team` | Team abbreviation (e.g. LAD, NYY) |
| `Position` | Position code (SP, RP, C, 1B, 2B, 3B, SS, LF, CF, RF, DH, TWP) |
| `Salary` | Salary in dollars |
| `WAR_Total` | Fangraphs WAR (fWAR) for that season |
| `Age` | Player age |
| `Stage_Clean` | Contract stage: `Free Agent`, `Arbitration`, `Pre-Arbitration` |
| `League` | AL or NL (derived from team) |

### Supporting Data
- **`sportsref_download.csv`** — actual team win totals by season (2021–2025)
- **Individual payroll CSVs** (`2021mlbshared.csv` … `2025mlbshared.csv`) — team-level payroll breakdowns by position
- **Roster data** — fetched live from MLB StatsAPI (`mlb_rosters.py`)

### Derived / Computed Fields (in-app)
- `PPR` (Pay-Performance Ratio) — actual salary / expected salary from regression line. <1.0 = underpaid, >1.0 = overpaid
- `residual` — actual salary minus regression-predicted salary ($M)
- `fWAR/$M` — production efficiency ratio
- `cost_per_WAR` — dollars paid per unit of WAR
- `WAR_delta` — year-over-year WAR change (used in Pre-Arb explorer)
- `predicted` — salary predicted by OLS/LOESS/Poly regression on WAR vs Salary

---

## Application Architecture

```
mlb-optimizer/
├── app/
│   └── streamlit_app.py        # Entire frontend UI (~10,000 lines)
├── src/
│   ├── pipeline.py             # Orchestrates end-to-end optimization
│   ├── projections.py          # Multi-year WAR blending (weighted avg)
│   ├── archetypes.py           # Player classification into 72+ archetypes
│   ├── optimizer.py            # MILP roster optimizer (PuLP)
│   ├── simulation.py           # Monte Carlo win distribution
│   ├── backtest.py             # Historical WAR→wins model calibration
│   ├── diagnostics.py          # Budget frontier & marginal value analysis
│   ├── team_mode.py            # Team-specific offseason scenario planning
│   └── mlb_rosters.py          # Live MLB roster fetching via StatsAPI
├── configs/
│   └── default_config.json     # 47 optimizer parameters
├── Data/                       # Raw CSV files
└── upload_to_r2.py             # Deploys data files to Cloudflare R2
```

---

## UI Structure (Streamlit App)

The app is organized into pages/tabs. Key sections relevant to RAG:

### Player Analysis Page (`_render_efficiency_frontier()`)
The richest page for RAG — filters by year, team, stage, position, age, WAR, salary.

**Tabs:**
1. **Cost Effective Line** — WAR vs Salary scatter with regression. Shows top 25 underpaid players table.
2. **PPEL** — Pay-Performance ratio analysis
3. **Age Trajectory** — Average WAR/salary/efficiency by age. Shows top 25 value-by-age standouts table.
4. **Efficient Players** — Top 100 players ranked by PPR (lowest = most underpaid). Scrollable table.
5. **Residual Analysis** — Distribution of overpaid/underpaid players
6. **Pre-Arb Explorer** — Pre-arbitration players only. WAR trajectory chart (filterable by player), trend classification (Improving/Neutral/Declining), ranked summary table.
7. **WAR Stability** — fWAR mean vs std dev scatter (reliability quadrants)

### Other Pages
- **Team Rankings** — Team efficiency ($/win above/below frontier)
- **Optimizer** — Configure and run the MILP roster optimizer
- **Team Analysis** — Team-specific offseason scenario planning

---

## Core Domain Concepts

### Archetypes
Players are classified as `{position_group}_{contract_stage}_{WAR_tier}`:
- **Position groups:** C, CI (1B/3B), MI (2B/SS), CF, OF (corners), SP, RP, DH
- **Stages:** FA (Free Agent), Arb (Arbitration), Pre-Arb (Pre-Arbitration)
- **WAR tiers:** Elite (≥4.0), Solid (2.0–4.0), Average (0.5–2.0), Depth (<0.5)
- Examples: `SP_FA_Elite`, `CF_Arb_Solid`, `CI_Pre-Arb_Average`

### Contract Stages
- **Pre-Arbitration** — Years 1–3 service time, near league minimum (~$740K–$780K). Highest WAR/$ efficiency.
- **Arbitration** — Years 3–6, salary negotiated annually. Moderate efficiency.
- **Free Agent** — Open market. Typically overpaid relative to production.

### WAR (Wins Above Replacement)
- Fangraphs fWAR is the primary production metric throughout
- Multi-year projection blends: 50% current year, 30% prior year, 20% two years ago (configurable)
- Calibrated wins conversion: fitted via OLS from historical data (not naive 1 WAR = 1 win)
- Stage-based uncertainty: Pre-Arb players get 1.5× WAR standard deviation in simulations

### Optimizer Constraints
- Roster slots: 2C, 1×1B, 1×2B, 1×3B, 1×SS, 1×LF, 1×CF, 1×RF, 1×DH, 5×SP, 8×RP
- Budget ceiling (default $130M)
- Archetype caps (e.g. max 2 Elite SP)
- Stage mix minimums (e.g. at least 5 FA, 3 Arb)
- SP innings floor (750 IP), RP innings floor (320 IP)

---

## Deployment

- **Production:** Render.com (Python 3.11, auto-deploy from `main` branch)
- **Data CDN:** Cloudflare R2 object storage (`R2_BASE_URL` env var)
- **R2 Mode:** In production, all CSV/image data is fetched from R2 URLs with ETag-based caching
- **Local Mode:** Falls back to local `/Data` folder if R2 unavailable

---

## Potential RAG Use Cases for This Project

### Natural Language Querying
- "Who are the most underpaid shortstops in 2024?"
- "Show me pre-arb outfielders with improving WAR trends"
- "Which teams overspend relative to the cost effective line?"
- "What does a $150M roster look like optimized for wins?"

### Glossary / Explainer Layer
- The app already has inline glossary components (`_render_glossary()`) for terms like fWAR, PPR, WSR, Pre-Arb, etc.
- A RAG system could answer follow-up questions about methodology

### Player Lookup
- Retrieve a specific player's salary, WAR, PPR, stage, team across years
- Compare two players head-to-head

### Optimizer Context
- Explain why a player was or wasn't selected in an optimizer run
- Describe what archetype a player falls into and why

---

## Key Limitations / Considerations for RAG

1. **Data freshness** — Dataset covers 2021–2025; no real-time stats
2. **Data size** — ~4,000 player-season rows; small enough to embed fully or chunk by player/year
3. **Computed fields** — PPR, residuals, etc. are computed at runtime from raw data; a RAG system would need to either pre-compute these or invoke the calculation logic
4. **Ambiguous player names** — Some players share names or have name formatting inconsistencies across seasons
5. **Position flexibility** — CF players can play corner OF; CI covers 1B and 3B; this needs to be captured in retrieval
6. **Stage changes** — A player's stage (Pre-Arb → Arb → FA) changes year to year; queries need year context
