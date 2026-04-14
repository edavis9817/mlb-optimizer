"""MLB Toolbox -- Rankings page (extracted from streamlit_app.py)."""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.components import (
    render_feedback_widget as _render_feedback_widget,
    render_glossary as _render_glossary,
)
from utils.player_utils import (
    fix_player_name as _fix_player_name,
    headshot_url as _headshot_url,
)
from utils.theme import plotly_theme as _pt
from utils.data_loading import (
    data_url,
    read_csv,
    team_logo_url,
    cached_mlbam_lookup,
    R2_MODE as r2_mode,
    RAZZBALL_PATH as razzball_path,
)


_STAGE_COLORS = {"Pre-Arb": "#5dc9a5", "Arb": "#ef9f27", "FA": "#97c459"}
_STAGE_LABELS = {"Pre-Arb": "Pre-Arbitration", "Arb": "Arbitration", "FA": "Free Agent"}


def _stage_card(stage_key: str, big_val: str, sub_text: str, unit: str = ""):
    """Render a stage summary card with accent border."""
    color = _STAGE_COLORS.get(stage_key, "#4a687e")
    label = _STAGE_LABELS.get(stage_key, stage_key)
    return (
        f"<div style='background:#1c2a42;border:1px solid #1e3050;"
        f"border-left:3px solid {color};border-radius:8px;"
        f"padding:0.7rem 0.9rem;margin-bottom:0.5rem;text-align:center;'>"
        f"<div style='font-size:0.7rem;color:#93b8d8;font-weight:600;"
        f"letter-spacing:0.05em;'>{label}</div>"
        f"<div style='font-size:1.4rem;font-weight:800;color:{color};"
        f"margin:0.2rem 0;'>{big_val}</div>"
        f"<div style='font-size:0.68rem;color:#7a9ebc;'>{unit}</div>"
        f"<div style='font-size:0.65rem;color:#5a7a94;margin-top:0.15rem;'>{sub_text}</div>"
        f"</div>"
    )


def render(*_args, **_kwargs):
    """Rankings page entry point."""

    # ── Query-param driven box + tab selection ──────────────────────────────────
    _qp_box = st.query_params.get("rk_box")
    _qp_yr  = st.query_params.get("rk_year")
    _qp_tab = st.query_params.get("rk_tab")
    if _qp_box:
        st.session_state["rk_selected_box"] = _qp_box
        st.session_state["rk_box_clicked"] = True
    if _qp_tab:
        st.session_state["rk_active_tab"] = _qp_tab

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown("""<style>
.rk-hdr{background:linear-gradient(135deg,#18243a 0%,#111927 100%);
  border:1px solid #1e3250;border-radius:12px;padding:0.9rem 1.3rem;margin-bottom:0.8rem;}
.rk-hdr h2{margin:0;font-size:1.25rem;color:#d6e8f8;font-weight:700;}
.rk-hdr .rk-sub{font-size:0.72rem;color:#7a9ebc;margin-top:0.15rem;}
.rk-answer{background:#1c2a42;border:1px solid #1e3250;border-radius:10px;
  padding:0.6rem 0.8rem;text-align:center;height:165px;min-height:165px;max-height:165px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  transition:border-color 0.2s,box-shadow 0.2s;box-sizing:border-box;overflow:hidden;
  width:100%;}
.rk-answer:hover{border-color:#2a4060;box-shadow:0 0 8px rgba(93,202,165,0.1);}
.rk-answer .rk-q{font-size:0.72rem;color:#93b8d8;
  letter-spacing:0.05em;margin-bottom:0.2rem;font-weight:600;}
.rk-answer .rk-team{font-size:1.25rem;font-weight:800;color:#d6e8f8;line-height:1.1;}
.rk-answer .rk-val{font-size:0.82rem;color:#93b8d8;margin-top:0.2rem;}
.rk-answer .rk-icon{font-size:1.3rem;margin-bottom:0.15rem;line-height:1;}
.rk-box-sel{border:2px solid #5dc9a5 !important;box-shadow:0 0 15px rgba(93,202,165,0.3),0 0 30px rgba(93,202,165,0.15) !important;}
.rk-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.7rem;margin-bottom:0.7rem;}
.rk-grid a{text-decoration:none;color:inherit;}
[data-testid="column"] a{text-decoration:none;color:inherit;display:block;}
[data-testid="column"] .rk-answer{width:100%;}
@keyframes gentle-pulse{0%,100%{opacity:0.4;}50%{opacity:1;}}
.rk-hint{text-align:center;color:#4a687e;font-size:0.82rem;margin:0.6rem 0;
  animation:gentle-pulse 2.5s ease-in-out infinite;}
.rk-tabs{display:flex;flex-wrap:wrap;gap:0.3rem;margin-bottom:0.8rem;
  border-bottom:1px solid #1e3250;padding-bottom:0.4rem;}
.rk-tab{padding:0.35rem 0.7rem;border-radius:6px 6px 0 0;font-size:0.75rem;
  color:#7a9ebc;text-decoration:none;transition:all 0.2s;border:1px solid transparent;
  border-bottom:none;font-weight:500;}
.rk-tab:hover{color:#d6e8f8;background:#1c2a42;}
.rk-tab-active{color:#d6e8f8;background:#1c2a42;border-color:#1e3250;
  border-bottom:2px solid #5dc9a5;font-weight:700;}
</style>""", unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    detail_csv  = data_url("efficiency_detail.csv")
    ranking_csv = data_url("al_nl_ranking_table.csv")

    _files_local = not r2_mode
    if _files_local and (not os.path.exists(detail_csv) or not os.path.exists(ranking_csv)):
        st.warning(
            "Rankings data not yet generated. Go to **League Analysis** → "
            "**Regenerate Analysis** first."
        )
        return

    try:
        detail_df  = read_csv(detail_csv)
        ranking_df = read_csv(ranking_csv)
    except Exception as _e:
        st.error(f"Could not load rankings data: {_e}")
        return

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='rk-hdr'>"
        "<h2>\U0001f3c6 MLB Toolbox Rankings</h2>"
        "<div class='rk-sub' style='font-size:0.88rem;color:#93b8d8;'>The 30-team report card on spending vs. winning.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Instruction + year selector on same row ──────────────────────────────
    years_avail = sorted(detail_df["Year"].dropna().unique().astype(int), reverse=True)
    _hint_col, _yr_col = st.columns([8, 2])
    with _hint_col:
        st.markdown(
            "<div style='padding-top:0.35rem;font-size:1rem;color:#93b8d8;font-weight:700;'>"
            "Click a box to get started, and start exploring more in-depth</div>",
            unsafe_allow_html=True,
        )
    with _yr_col:
        _yr_default = years_avail.index(int(_qp_yr)) if _qp_yr and int(_qp_yr) in years_avail else 0
        sel_year = st.selectbox("Season", years_avail, key="rk_year_sel", index=_yr_default, label_visibility="collapsed")

    yr_df = detail_df[detail_df["Year"] == sel_year].copy()
    if yr_df.empty:
        st.warning(f"No data for {sel_year}.")
        return

    # Merge division/league metadata from the ranking table
    if {"Abbr", "Division", "League"}.issubset(ranking_df.columns):
        _meta = ranking_df[["Abbr", "Team", "Division", "League",
                             "Playoff_Apps", "WS_Wins"]].copy()
        _meta = _meta.rename(columns={"Abbr": "_abbr", "Team": "Team_Full"})
        yr_df = yr_df.merge(_meta, left_on="Team", right_on="_abbr", how="left")
        yr_df.drop(columns=["_abbr"], errors="ignore", inplace=True)

    # Derived column: $/WAR for the selected season
    yr_df["DPW"] = (yr_df["payroll_M"] / yr_df["team_WAR"].clip(lower=0.1)).round(2)

    # ── Terms glossary ────────────────────────────────────────────────────────
    _render_glossary([
        ("CEL",            "Cost Effective Line",
         "The league regression of team payroll vs wins. Shows the expected win total for any given payroll. "
         "Teams on the line get average value; teams below it win more than their spending predicts."),
        ("Efficiency Gap", "Dollars Above/Below the CEL",
         "How many $M a team's payroll sits above or below what their win total 'should' cost at market rate. "
         "Negative (green) = efficient \u2014 winning more per dollar. Positive (red) = overspending."),
        ("$/fWAR",         "Dollars per fWAR",
         "Average cost of one fWAR for this team's roster in the selected season. "
         "League average free-agent rate is ~$7\u20139M/fWAR; Pre-Arb players cost far less."),
        ("Wins vs Pred.",  "Wins Above Payroll Prediction",
         "Actual wins minus the wins predicted by the payroll regression. "
         "Positive = team beats their expected win total given their spending (e.g. strong coaching, "
         "player development). Negative = underperforming relative to payroll."),
        ("Avg Gap $M",     "5-Year Average Efficiency Gap",
         "Average dollars above or below the CEL per season across 2021\u20132025. "
         "Consistently negative teams (Tampa Bay, Cleveland) build wins efficiently year over year."),
        ("fWAR",           "FanGraphs Wins Above Replacement (Team)",
         "Sum of all player fWAR on the 40-man roster for the season. "
         "fWAR measures extra wins a player provides vs a replacement-level minor leaguer. "
         "Average team fWAR is ~34; playoff teams typically need 27+; World Series champions average 42+."),
    ], title="\U0001f4d6 Terms & Definitions")

    _best_eff  = yr_df.loc[yr_df["dollar_gap_M"].idxmin()]
    _worst_eff = yr_df.loc[yr_df["dollar_gap_M"].idxmax()]
    _top_war   = yr_df.loc[yr_df["team_WAR"].idxmax()]
    _top_wins  = yr_df.loc[yr_df["Wins"].idxmax()]
    _overperf  = yr_df.loc[yr_df["wins_vs_pred"].idxmax()]
    _best_dpw  = yr_df.loc[yr_df["DPW"].idxmin()]

    # Use full team names if available
    def _full(row):
        return row.get("Team_Full") or row["Team"]

    # ── Clickable highlight boxes → control which tab shows below ───────────
    # Box → tab mapping
    _BOX_TAB = {
        "most_efficient": "efficiency", "least_efficient": "efficiency",
        "top_overperformer": "winperf", "most_wins": "winperf",
        "best_dpw": "salary",
        "top_fwar": "fwar",
        "p_top_fwar": "fwar",
        "p_contract_val": "contract_value",
        "p_stability": "stability",
        "best_marginal": "salary",
        "fwar_wins_link": "fwar",
        "eff_playoffs": "playoff_success",
        "fwar_cost": "fwar_cost",
        "best_position": "position_value",
        "best_age": "age_value",
    }
    _TAB_ORDER = [
        "efficiency", "fwar", "salary", "winperf",
        "contract_value", "stability",
        "playoff_success", "fwar_cost", "position_value", "age_value",
    ]
    if "rk_active_tab" not in st.session_state:
        st.session_state["rk_active_tab"] = "efficiency"
    if "rk_selected_box" not in st.session_state:
        st.session_state["rk_selected_box"] = "most_efficient"
    if "rk_box_clicked" not in st.session_state:
        st.session_state["rk_box_clicked"] = False

    # Apply query-param box → active tab
    if _qp_box and _qp_box in _BOX_TAB:
        st.session_state["rk_active_tab"] = _BOX_TAB[_qp_box]

    _sel_box = st.session_state["rk_selected_box"]
    _act_tab = st.session_state["rk_active_tab"]

    # ── Load player data for boxes & new tabs ─────────────────────────────────
    _paw_df = None
    _paw_25 = pd.DataFrame()
    _wsr_grp = pd.DataFrame()
    _p_top_war = None
    _p_best_val = None
    _p_best_wsr = None
    _mlbam = {}
    try:
        _paw_csv = data_url("data/mlb_combined_2021_2025.csv")
        _paw_df = read_csv(_paw_csv, low_memory=False)
        _paw_df.columns = [c.strip() for c in _paw_df.columns]
        _paw_df["Year"] = pd.to_numeric(_paw_df["Year"], errors="coerce")
        _paw_df["WAR_Total"] = pd.to_numeric(_paw_df["WAR_Total"], errors="coerce")
        _paw_df["Salary_M"] = pd.to_numeric(_paw_df["Salary_M"], errors="coerce")
        _paw_25 = _paw_df[_paw_df["Year"] == sel_year].dropna(subset=["WAR_Total"])

        # Top fWAR player
        _p_top_war = _paw_25.sort_values("WAR_Total", ascending=False).iloc[0] if not _paw_25.empty else None

        # Best contract value (highest WAR/$M for players earning > $1M)
        _paw_val = _paw_25[_paw_25["Salary_M"] > 1.0].copy()
        _paw_val["_wpm"] = _paw_val["WAR_Total"] / _paw_val["Salary_M"].clip(lower=0.1)
        _p_best_val = _paw_val.sort_values("_wpm", ascending=False).iloc[0] if not _paw_val.empty else None

        # Best fWAR stability (WSR) — need multi-year data
        _paw_all = _paw_df.dropna(subset=["WAR_Total"])
        _wsr_grp = _paw_all.groupby("Player").agg(
            _mean=("WAR_Total", "mean"), _std=("WAR_Total", "std"), _n=("Year", "nunique"),
            Team=("Team", "last"),
        ).reset_index()
        _wsr_grp = _wsr_grp[_wsr_grp["_n"] >= 3].copy()
        _wsr_grp["_std"] = _wsr_grp["_std"].fillna(0)
        _wsr_grp["WSR"] = (_wsr_grp["_mean"] / (1 + _wsr_grp["_std"])).round(2)
        _p_best_wsr = _wsr_grp.sort_values("WSR", ascending=False).iloc[0] if not _wsr_grp.empty else None

        _mlbam = cached_mlbam_lookup(razzball_path)
    except Exception:
        pass

    # ── Compute row-4 box values ──────────────────────────────────────────────
    # Best Marginal Spending — tier with highest slope
    _best_marginal_name = "N/A"
    _best_marginal_val = ""
    if {"payroll_M", "Wins"}.issubset(detail_df.columns):
        _f2_pre = detail_df.dropna(subset=["payroll_M", "Wins"]).copy()
        _f2_pre["payroll_M"] = pd.to_numeric(_f2_pre["payroll_M"], errors="coerce")
        _f2_pre["Wins"] = pd.to_numeric(_f2_pre["Wins"], errors="coerce")
        _TIERS_PRE = [
            ("Budget (&#36;0-100M)", 0, 100), ("Mid-Market (&#36;100-175M)", 100, 175),
            ("Contender (&#36;175-244M)", 175, 244), ("Big Market (&#36;244M+)", 244, 999),
        ]
        _best_slope = -999
        for _tn, _tlo, _thi in _TIERS_PRE:
            _tdf = _f2_pre[(_f2_pre["payroll_M"] >= _tlo) & (_f2_pre["payroll_M"] < _thi)]
            if len(_tdf) >= 5:
                _tc = np.polyfit(_tdf["payroll_M"], _tdf["Wins"], 1)
                _tw10 = _tc[0] * 10
                if _tw10 > _best_slope:
                    _best_slope = _tw10
                    _best_marginal_name = _tn
                    _best_marginal_val = f"+{_tw10:.2f} wins/&#36;10M"

    # Strongest fWAR-Wins Link — R² value
    _fwar_r2_str = ""
    _fwar_r2 = 0.0
    if "team_WAR" in detail_df.columns and "Wins" in detail_df.columns:
        _f1_pre = detail_df.dropna(subset=["team_WAR", "Wins"]).copy()
        _f1_pre["team_WAR"] = pd.to_numeric(_f1_pre["team_WAR"], errors="coerce")
        _f1_pre["Wins"] = pd.to_numeric(_f1_pre["Wins"], errors="coerce")
        _f1_pre = _f1_pre.dropna(subset=["team_WAR", "Wins"])
        if len(_f1_pre) > 5:
            _xp = _f1_pre["team_WAR"].values
            _yp = _f1_pre["Wins"].values
            _cp = np.polyfit(_xp, _yp, 1)
            _pp = np.polyval(_cp, _xp)
            _ss_res_p = np.sum((_yp - _pp) ** 2)
            _ss_tot_p = np.sum((_yp - _yp.mean()) ** 2)
            _fwar_r2 = 1 - (_ss_res_p / _ss_tot_p) if _ss_tot_p > 0 else 0
            _fwar_r2_str = f"R\u00b2 = {_fwar_r2:.3f}"

    # ── Compute new insight box values ──────────────────────────────────────────
    # 1. Efficiency → Playoffs rate
    _eff_playoff_str = ""
    if {"dollar_gap_M", "in_playoffs"}.issubset(detail_df.columns):
        _ep_df = detail_df.dropna(subset=["dollar_gap_M"])
        _ep_df["dollar_gap_M"] = pd.to_numeric(_ep_df["dollar_gap_M"], errors="coerce")
        _ep_eff = _ep_df.nsmallest(len(_ep_df) // 3, "dollar_gap_M")
        _ep_ineff = _ep_df.nlargest(len(_ep_df) // 3, "dollar_gap_M")
        _ep_eff_rate = _ep_eff["in_playoffs"].mean() * 100 if not _ep_eff.empty else 0
        _ep_ineff_rate = _ep_ineff["in_playoffs"].mean() * 100 if not _ep_ineff.empty else 0
        _eff_playoff_str = f"{_ep_eff_rate:.0f}% vs {_ep_ineff_rate:.0f}%"

    # 2. League avg $/fWAR
    _avg_dpw_str = ""
    if not yr_df.empty and "DPW" in yr_df.columns:
        _avg_dpw = yr_df["DPW"].median()
        _avg_dpw_str = f"&#36;{_avg_dpw:.1f}M per fWAR"

    # 3. Most efficient position
    _best_pos_name = ""
    _best_pos_val = ""
    if _paw_df is not None and not _paw_25.empty:
        _pos_eff = _paw_25[(_paw_25["Salary_M"] > 0.5) & (_paw_25["WAR_Total"] > 0)].copy()
        if not _pos_eff.empty:
            _pos_eff["_wpm"] = _pos_eff["WAR_Total"] / _pos_eff["Salary_M"].clip(lower=0.1)
            _pos_avg = _pos_eff.groupby("Position")["_wpm"].mean()
            _pos_avg = _pos_avg[_pos_avg.index.isin(["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "SP", "RP"])]
            if not _pos_avg.empty:
                _best_pos_name = _pos_avg.idxmax()
                _best_pos_val = f"{_pos_avg.max():.2f} fWAR/&#36;M"

    # 4. Most efficient age
    _best_age = ""
    _best_age_val = ""
    if _paw_df is not None and not _paw_25.empty:
        _age_eff = _paw_25[(_paw_25["Salary_M"] > 0.5) & (_paw_25["WAR_Total"] >= 1.0)].copy()
        _age_eff["Age"] = pd.to_numeric(_age_eff.get("Age", pd.Series(dtype=float)), errors="coerce")
        _age_eff = _age_eff.dropna(subset=["Age"])
        if not _age_eff.empty:
            _age_eff["_age_int"] = _age_eff["Age"].astype(int)
            _age_eff["_wpm"] = _age_eff["WAR_Total"] / _age_eff["Salary_M"].clip(lower=0.1)
            _age_grp = _age_eff.groupby("_age_int").agg(_wpm=("_wpm", "mean"), _n=("Player", "count"))
            _age_grp = _age_grp[_age_grp["_n"] >= 5]
            if not _age_grp.empty:
                _best_age = str(_age_grp["_wpm"].idxmax())
                _best_age_val = f"{_age_grp['_wpm'].max():.2f} fWAR/&#36;M"

    # ── Build box HTML helper ─────────────────────────────────────────────────
    def _box_html(box_id, label, team_name, val_str, logo_url="", img_html="", team_abbr=""):
        _sel_cls = "rk-box-sel" if box_id == _sel_box else ""
        _box_tab = _BOX_TAB.get(box_id, "efficiency")
        _href = f"?page=rankings&rk_box={box_id}&rk_year={sel_year}&rk_tab={_box_tab}#rk-content"
        _logo_tag = (f"<img src='{logo_url}' width='36' height='36' "
                     f"style='object-fit:contain;margin-bottom:4px;' "
                     f"onerror=\"this.style.display='none'\">") if logo_url else ""
        if img_html:
            _logo_tag = img_html
        return (
            f"<a href='{_href}' target='_self' style='text-decoration:none;color:inherit;display:block;'>"
            f"<div class='rk-answer {_sel_cls}'>"
            f"{_logo_tag}"
            f"<div class='rk-q'>{label}</div>"
            f"<div class='rk-team'>{team_name}</div>"
            f"<div class='rk-val'>{val_str}</div>"
            f"</div></a>"
        )

    def _player_box_html(box_id, title, pname, pteam, pval, psub=""):
        _sel_cls = "rk-box-sel" if box_id == _sel_box else ""
        _box_tab = _BOX_TAB.get(box_id, "efficiency")
        _href = f"?page=rankings&rk_box={box_id}&rk_year={sel_year}&rk_tab={_box_tab}#rk-content"
        mid = _mlbam.get(_fix_player_name(pname), "") if _mlbam else ""
        _img = (f"<img src='{_headshot_url(mid, 120)}' width='50' height='50' loading='lazy' "
                f"style='border-radius:50%;object-fit:cover;margin-bottom:4px;' "
                f"onerror=\"this.style.display='none'\">") if mid else ""
        _sub_html = f"<div style='font-size:0.62rem;color:#4a687e;'>{psub}</div>" if psub else ""
        return (
            f"<a href='{_href}' target='_self' style='text-decoration:none;color:inherit;display:block;'>"
            f"<div class='rk-answer {_sel_cls}'>"
            f"<div class='rk-q'>{title}</div>{_img}"
            f"<div class='rk-team'>{pname}</div>"
            f"<div style='font-size:0.75rem;color:#93b8d8;'>{pteam}</div>"
            f"<div class='rk-val'>{pval}</div>"
            f"{_sub_html}</div></a>"
        )

    # ── Row 1 + Row 2: Team boxes ─────────────────────────────────────────────
    _box_defs = [
        ("most_efficient", "MOST EFFICIENT", _full(_best_eff),
         f"&#36;{_best_eff['dollar_gap_M']:.0f}M below the line", team_logo_url(_best_eff["Team"]), _best_eff["Team"]),
        ("top_overperformer", "TOP OVERPERFORMER", _full(_overperf),
         f"+{_overperf['wins_vs_pred']:.1f} wins vs forecast", team_logo_url(_overperf["Team"]), _overperf["Team"]),
        ("best_dpw", "BEST &#36;/fWAR", _full(_best_dpw),
         f"&#36;{_best_dpw['DPW']:.1f}M per fWAR", team_logo_url(_best_dpw["Team"]), _best_dpw["Team"]),
        ("least_efficient", "LEAST EFFICIENT", _full(_worst_eff),
         f"&#36;{_worst_eff['dollar_gap_M']:.0f}M above the line", team_logo_url(_worst_eff["Team"]), _worst_eff["Team"]),
        ("top_fwar", "TOP fWAR", _full(_top_war),
         f"{_top_war['team_WAR']:.1f} total fWAR", team_logo_url(_top_war["Team"]), _top_war["Team"]),
        ("most_wins", "MOST WINS", _full(_top_wins),
         f"{int(_top_wins['Wins'])} wins", team_logo_url(_top_wins["Team"]), _top_wins["Team"]),
    ]

    # ── Box grid using st.columns(3) per row ─────────────────────────────────
    def _render_box(html):
        st.markdown(html, unsafe_allow_html=True)

    # Row 1
    _r1c1, _r1c2, _r1c3 = st.columns(3)
    with _r1c1:
        _render_box(_box_html(*_box_defs[0][:4], logo_url=_box_defs[0][4], team_abbr=_box_defs[0][5]))
    with _r1c2:
        _render_box(_box_html(*_box_defs[1][:4], logo_url=_box_defs[1][4], team_abbr=_box_defs[1][5]))
    with _r1c3:
        _render_box(_box_html(*_box_defs[2][:4], logo_url=_box_defs[2][4], team_abbr=_box_defs[2][5]))

    # Row 2
    _r2c1, _r2c2, _r2c3 = st.columns(3)
    with _r2c1:
        _render_box(_box_html(*_box_defs[3][:4], logo_url=_box_defs[3][4], team_abbr=_box_defs[3][5]))
    with _r2c2:
        _render_box(_box_html(*_box_defs[4][:4], logo_url=_box_defs[4][4], team_abbr=_box_defs[4][5]))
    with _r2c3:
        _render_box(_box_html(*_box_defs[5][:4], logo_url=_box_defs[5][4], team_abbr=_box_defs[5][5]))

    # Row 3: Player boxes
    _r3c1, _r3c2, _r3c3 = st.columns(3)
    with _r3c1:
        if _p_top_war is not None:
            _render_box(_player_box_html(
                "p_top_fwar", f"#1 fWAR ({sel_year})", str(_p_top_war["Player"]),
                str(_p_top_war["Team"]), f"{_p_top_war['WAR_Total']:.1f} fWAR"))
    with _r3c2:
        if _p_best_val is not None:
            _render_box(_player_box_html(
                "p_contract_val", "TOP CONTRACT VALUE", str(_p_best_val["Player"]),
                str(_p_best_val["Team"]), f"{_p_best_val['_wpm']:.2f} fWAR/&#36;M",
                f"{_p_best_val['WAR_Total']:.1f} fWAR \u00b7 &#36;{_p_best_val['Salary_M']:.1f}M"))
    with _r3c3:
        if _p_best_wsr is not None:
            _render_box(_player_box_html(
                "p_stability", "BEST fWAR STABILITY", str(_p_best_wsr["Player"]),
                str(_p_best_wsr["Team"]), f"{_p_best_wsr['WSR']:.2f} WSR",
                f"Avg {_p_best_wsr['_mean']:.1f} fWAR \u00b7 {int(_p_best_wsr['_n'])} seasons"))

    # Row 4: Analysis boxes
    _r4c1, _r4c2, _r4c3 = st.columns(3)
    with _r4c1:
        _render_box(_box_html("best_marginal", "BEST MARGINAL SPENDING", _best_marginal_name, _best_marginal_val))
    with _r4c2:
        _render_box(_box_html("fwar_wins_link", "STRONGEST fWAR-WINS LINK", "All Teams", _fwar_r2_str))
    with _r4c3:
        _render_box(_box_html("eff_playoffs", "EFFICIENCY \u2192 PLAYOFFS?",
                               _eff_playoff_str, "Efficient vs Inefficient",
                               img_html="<div style='font-size:1.3rem;margin-bottom:4px;'>\U0001f4ca</div>"))

    # Row 5: Insight boxes
    _r5c1, _r5c2, _r5c3 = st.columns(3)
    with _r5c1:
        _render_box(_box_html("fwar_cost", "HOW MUCH DOES 1 fWAR COST?",
                               _avg_dpw_str, "",
                               img_html="<div style='font-size:1.3rem;margin-bottom:4px;'>\U0001f4b5</div>"))
    with _r5c2:
        _render_box(_box_html("best_position", "MOST EFFICIENT POSITION",
                               _best_pos_name, _best_pos_val,
                               img_html="<div style='font-size:1.3rem;margin-bottom:4px;'>\U0001f3df\ufe0f</div>"))
    with _r5c3:
        _render_box(_box_html("best_age", "MOST EFFICIENT AGE",
                               _best_age, _best_age_val,
                               img_html="<div style='font-size:1.3rem;margin-bottom:4px;'>\U0001f4c5</div>"))

    # Animated hint (disappears after first click)
    if not st.session_state.get("rk_box_clicked"):
        st.markdown("<div class='rk-hint'>Click a box above to explore</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)

    def _hbar(df_in, x_col, color_fn, title, x_label, text_fn=None, zero_line=False):
        """Render a themed horizontal bar chart for rankings."""
        vals   = df_in[x_col].tolist()
        teams  = df_in["Team"].tolist()
        colors = [color_fn(v) for v in vals]
        texts  = [text_fn(v) for v in vals] if text_fn else [f"{v:.1f}" for v in vals]
        fig = go.Figure(go.Bar(
            y=teams, x=vals, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=texts, textposition="outside",
            textfont=dict(color="#d6e8f8", size=9),
            hovertemplate="%{y}: %{text}<extra></extra>",
        ))
        _xaxis = dict(title=x_label, zeroline=zero_line,
                       zerolinecolor="#4a687e", zerolinewidth=1)
        # Center 0 on charts with diverging values
        if zero_line and vals:
            _abs_max = max(abs(v) for v in vals) * 1.15
            _xaxis["range"] = [-_abs_max, _abs_max]
        fig.update_layout(**_pt(
            title=title,
            xaxis=_xaxis,
            yaxis=dict(autorange="reversed"),
            height=max(340, len(df_in) * 22),
            margin=dict(l=60, r=80, t=42, b=30),
        ))
        return fig

    # ── Tabs ─────────────────────────────────────────────────────────────────
    # Scroll anchor for box clicks
    st.markdown("<div id='rk-content'></div>", unsafe_allow_html=True)

    # ── Custom tab bar ─────────────────────────────────────────────────────────
    _TAB_NAMES = {
        "efficiency": "\U0001f3c6 Efficiency",
        "fwar": "\u2b50 fWAR",
        "salary": "\U0001f4b0 Salary",
        "winperf": "\U0001f4c8 Win Performance",
        "contract_value": "\U0001f48e Contract Value",
        "stability": "\U0001f512 Stability",
        "playoff_success": "📊 Playoff Success",
        "fwar_cost": "💵 fWAR Cost",
        "position_value": "🏟️ Position Value",
        "age_value": "📅 Age Value",
    }
    _tab_html = "<div class='rk-tabs'>"
    for _tk in _TAB_ORDER:
        _cls = "rk-tab rk-tab-active" if _tk == _act_tab else "rk-tab"
        _tab_html += (f"<a href='?page=rankings&rk_box={_sel_box}&rk_year={sel_year}&rk_tab={_tk}#rk-content' "
                      f"target='_self' class='{_cls}'>{_TAB_NAMES[_tk]}</a>")
    _tab_html += "</div>"
    st.markdown(_tab_html, unsafe_allow_html=True)


    # ── Tab 1: Efficiency ─────────────────────────────────────────────────────
    if _act_tab == "efficiency":
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "Dollar gap from the <b>Cost Effective Line</b> \u2014 how much more or less each team "
            "spends relative to what their wins cost at league-average market rates. "
            "<span style='color:#22c55e;font-weight:600;'>Green</span> = below the line (efficient). "
            "<span style='color:#ef4444;font-weight:600;'>Red</span> = above the line (overspending).</div>",
            unsafe_allow_html=True,
        )
        _eff = yr_df.sort_values("dollar_gap_M", ascending=True).reset_index(drop=True)
        _eff["Rank"] = range(1, len(_eff) + 1)

        ch1, tb1 = st.columns([3, 2])
        with ch1:
            st.plotly_chart(_hbar(
                _eff, "dollar_gap_M",
                color_fn=lambda v: "#22c55e" if v <= 0 else "#ef4444",
                title=f"{sel_year} \u2014 Spending Efficiency Ranking of MLB Teams Based on Regular Season Wins",
                x_label="$ Gap ($M) \u2014 negative = efficient",
                text_fn=lambda v: f"${v:+.0f}M",
                zero_line=True,
            ), use_container_width=True, config={"displayModeBar": False})
        with tb1:
            _e = _eff[["Rank", "Team", "dollar_gap_M", "payroll_M", "Wins", "in_playoffs"]].copy()
            _e.columns = ["#", "Team", "Gap $M", "Payroll $M", "Wins", "Postseason"]
            _e["Gap $M"]     = _e["Gap $M"].round(0).astype(int)
            _e["Payroll $M"] = _e["Payroll $M"].round(0).astype(int)
            _e["Wins"]       = _e["Wins"].round(0).astype(int)
            _e["Postseason"]  = _e["Postseason"].map({True: "\u2713", False: ""})

            def _eff_clr(row):
                g = row["Gap $M"]
                if g < -80:  return ["background-color:#0c2218"] * len(row)
                if g < 0:    return ["background-color:#14532d55"] * len(row)
                if g > 120:  return ["background-color:#2d0c0c"] * len(row)
                if g > 0:    return ["background-color:#2d150c55"] * len(row)
                return [""] * len(row)

            st.dataframe(
                _e.style.apply(_eff_clr, axis=1).format(
                    {"Gap $M": "{:+d}", "Payroll $M": "{:d}", "Wins": "{:d}"}, na_rep="\u2014"),
                hide_index=True, use_container_width=True,
                height=min(60 + len(_e) * 35, 720),
            )

        # Multi-year summary
        with st.expander("\U0001f4ca Multi-Year Summary (2025-2021)", expanded=False):
            st.caption(
                "5-year averages from the efficiency analysis. "
                "**Avg Gap** = average dollars above/below the cost-effective line per season \u2014 "
                "negative means the team consistently spends below market rate for their wins."
            )
            _rk = ranking_df.sort_values("Leag_Rank").reset_index(drop=True).copy()
            _rk.insert(0, "#", range(1, len(_rk) + 1))
            _rk_cols = [c for c in ["#", "Abbr", "Team", "Division",
                                     "Avg_Wins", "Avg_Pay_M", "Avg_$/WAR_M", "Avg_Gap_M",
                                     "Playoff_Apps", "WS_Wins"] if c in _rk.columns]
            _rk = _rk[_rk_cols].rename(columns={
                "Abbr": "Abbr", "Team": "Team", "Division": "Division",
                "Avg_Wins": "Avg Wins", "Avg_Pay_M": "Avg Pay $M",
                "Avg_$/WAR_M": "$/fWAR M", "Avg_Gap_M": "Avg Gap $M",
                "Playoff_Apps": "Playoffs (5yr)", "WS_Wins": "WS",
            })

            def _rk_clr(row):
                try:
                    g = row["Avg Gap $M"]
                    if g < -100: return ["background-color:#0c2218"] * len(row)
                    if g < 0:    return ["background-color:#14532d55"] * len(row)
                    if g > 100:  return ["background-color:#2d0c0c"] * len(row)
                except Exception:
                    pass
                return [""] * len(row)

            _rk["Avg Pay $M"] = _rk["Avg Pay $M"].round(0).astype(int)
            _rk["Avg Gap $M"] = _rk["Avg Gap $M"].round(0).astype(int)
            _rk["$/fWAR M"]    = _rk["$/fWAR M"].round(2)
            st.dataframe(
                _rk.style.apply(_rk_clr, axis=1),
                hide_index=True, use_container_width=True, height=680,
            )

    # ── Tab 2: fWAR ───────────────────────────────────────────────────────────
    if _act_tab == "fwar":
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "Total team WAR for the selected season across all pitchers and position players. "
            "Higher WAR = more total roster talent. Top-5 teams highlighted in green.</div>",
            unsafe_allow_html=True,
        )
        _war = yr_df.sort_values("team_WAR", ascending=False).reset_index(drop=True)
        _war["Rank"] = range(1, len(_war) + 1)
        _war_max = _war["team_WAR"].max() or 1

        ch2, tb2 = st.columns([3, 2])
        with ch2:
            st.plotly_chart(_hbar(
                _war, "team_WAR",
                color_fn=lambda v: (
                    "#22c55e" if v >= _war_max * 0.85 else
                    "#4873b8" if v >= _war_max * 0.60 else "#4a687e"
                ),
                title=f"{sel_year} \u2014 Total Team fWAR",
                x_label="Total fWAR",
                text_fn=lambda v: f"{v:.1f}",
            ), use_container_width=True, config={"displayModeBar": False})
        with tb2:
            _w = _war[["Rank", "Team", "team_WAR", "payroll_M", "DPW", "Wins", "in_playoffs"]].copy()
            _w.columns = ["#", "Team", "fWAR", "Payroll $M", "$/fWAR M", "Wins", "Postseason"]
            _w["Payroll $M"] = _w["Payroll $M"].round(0).astype(int)
            _w["Wins"]       = _w["Wins"].round(0).astype(int)
            _w["Postseason"]  = _w["Postseason"].map({True: "\u2713", False: ""})

            def _war_clr(row):
                if row["#"] <= 5:  return ["background-color:#0c2218"] * len(row)
                if row["#"] <= 10: return ["background-color:#14532d55"] * len(row)
                return [""] * len(row)

            st.dataframe(
                _w.style.apply(_war_clr, axis=1).format(
                    {"fWAR": "{:.1f}", "Payroll $M": "{:d}", "$/fWAR M": "{:.1f}", "Wins": "{:d}"}, na_rep="\u2014"),
                hide_index=True, use_container_width=True,
                height=min(60 + len(_w) * 35, 720),
            )

        # Player Rankings — Top 25 by fWAR
        st.markdown("---")
        st.markdown("### Player Rankings")
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
            f"Top 25 individual players by fWAR for the <b>{sel_year}</b> season.</div>",
            unsafe_allow_html=True,
        )
        try:
            _pr_df = _paw_df if _paw_df is not None else read_csv(data_url("data/mlb_combined_2021_2025.csv"))
            _pr_df_c = _pr_df.copy()
            _pr_df_c["Year"] = pd.to_numeric(_pr_df_c["Year"], errors="coerce")
            _pr_df_c["WAR_Total"] = pd.to_numeric(_pr_df_c["WAR_Total"], errors="coerce")
            _pr_df_c["Salary_M"] = pd.to_numeric(_pr_df_c["Salary_M"], errors="coerce")
            _pr_yr = _pr_df_c[_pr_df_c["Year"] == sel_year].copy()
            _pr_yr = _pr_yr.dropna(subset=["WAR_Total"]).sort_values("WAR_Total", ascending=False).head(25)
            if not _pr_yr.empty:
                _pr_show = _pr_yr[["Player", "Team", "Position", "WAR_Total", "Salary_M"]].reset_index(drop=True)
                _pr_show.insert(0, "#", range(1, len(_pr_show) + 1))
                _pr_show = _pr_show.rename(columns={
                    "WAR_Total": "fWAR", "Salary_M": "Salary $M",
                })
                _pr_show["Salary $M"] = _pr_show["Salary $M"].round(1)
                _pr_show["fWAR"] = _pr_show["fWAR"].round(1)
                st.dataframe(_pr_show, hide_index=True, width="stretch", height=500)
            else:
                st.info(f"No player data available for {sel_year}.")
        except Exception as _pr_e:
            st.warning(f"Could not load player rankings: {_pr_e}")

        # fWAR-to-Wins Relationship
        st.markdown("---")
        st.markdown("### fWAR-to-Wins Relationship")
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
            "Team fWAR (total roster talent) is one of the strongest predictors of regular-season wins. "
            "Each dot below is one team-season. <span style='color:#22c55e;font-weight:600;'>Green</span> = "
            "made the playoffs. The orange regression line shows the expected wins for a given fWAR level. "
            "The vertical blue dashed line marks the ~30 fWAR threshold \u2014 teams consistently above it are "
            "postseason contenders.</div>",
            unsafe_allow_html=True,
        )

        if "team_WAR" in detail_df.columns and "Wins" in detail_df.columns:
            _f1 = detail_df.dropna(subset=["team_WAR", "Wins"]).copy()
            _f1["team_WAR"] = pd.to_numeric(_f1["team_WAR"], errors="coerce")
            _f1["Wins"] = pd.to_numeric(_f1["Wins"], errors="coerce")
            _f1 = _f1.dropna(subset=["team_WAR", "Wins"])

            if len(_f1) > 5:
                _x = _f1["team_WAR"].values
                _y = _f1["Wins"].values
                _coef = np.polyfit(_x, _y, 1)
                _pred = np.polyval(_coef, _x)
                _ss_res = np.sum((_y - _pred) ** 2)
                _ss_tot = np.sum((_y - _y.mean()) ** 2)
                _r2 = 1 - (_ss_res / _ss_tot) if _ss_tot > 0 else 0

                st.markdown(
                    f"<div style='font-size:0.88rem;color:#d6e8f8;font-weight:600;margin-bottom:0.5rem;'>"
                    f"fWAR explains <span style='color:#60a5fa;'>{_r2 * 100:.1f}%</span> of win variation "
                    f"across {len(_f1)} team-seasons (2025-2021). Each additional fWAR is worth roughly "
                    f"<span style='color:#60a5fa;'>{_coef[0]:.2f}</span> wins.</div>",
                    unsafe_allow_html=True,
                )

                _f1_playoff = _f1.get("in_playoffs", pd.Series([False] * len(_f1)))
                _f1_colors = ["#22c55e" if p else "#4a687e" for p in _f1_playoff]
                _f1_hover = _f1.apply(lambda r: (
                    f"<b>{r['Team']}</b> {int(r['Year'])}<br>"
                    + f"fWAR: {r['team_WAR']:.1f} \u00b7 Wins: {int(r['Wins'])}<br>"
                    + f"Playoff: {'Yes' if r.get('in_playoffs') else 'No'}"
                ), axis=1)

                fig_f1 = go.Figure()
                fig_f1.add_trace(go.Scatter(
                    x=_f1["team_WAR"], y=_f1["Wins"], mode="markers",
                    marker=dict(color=_f1_colors, size=8, opacity=0.8),
                    text=_f1_hover, hovertemplate="%{text}<extra></extra>",
                    name="Teams",
                ))
                _xr = np.linspace(_x.min(), _x.max(), 100)
                fig_f1.add_trace(go.Scatter(
                    x=_xr, y=np.polyval(_coef, _xr), mode="lines",
                    line=dict(color="#f4a261", width=2), name=f"OLS (R\u00b2={_r2:.3f})",
                ))
                fig_f1.add_vline(x=30, line_dash="dash", line_color="#3b6fd4", opacity=0.5,
                                 annotation_text="Avg contender floor", annotation_position="top right",
                                 annotation_font_color="#3b6fd4")

                fig_f1.update_layout(**_pt(
                    title="Team fWAR vs Actual Wins (2025-2021)",
                    xaxis=dict(title="Total Team fWAR"),
                    yaxis=dict(title="Actual Wins"),
                    height=440, showlegend=True,
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                    hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                    font=dict(color="#dbeafe", size=12)),
                ))
                _f1_col, _ = st.columns([3, 1])
                with _f1_col:
                    st.plotly_chart(fig_f1, width="stretch")

            # Efficiency vs Postseason table
            if {"dollar_gap_M", "in_playoffs", "ws_champ"}.issubset(detail_df.columns):
                _eff_tbl = detail_df.groupby("Team").agg(
                    Avg_Gap=("dollar_gap_M", "mean"),
                    Playoff_Apps=("in_playoffs", "sum"),
                    WS_Apps=("ws_champ", lambda x: int(x.sum()) + int(detail_df.loc[x.index, "ws_runnerup"].sum()) if "ws_runnerup" in detail_df.columns else int(x.sum())),
                    WS_Wins=("ws_champ", "sum"),
                ).reset_index()
                _eff_tbl["Avg_Gap"] = _eff_tbl["Avg_Gap"].round(1)
                _eff_tbl["Playoff_Apps"] = _eff_tbl["Playoff_Apps"].astype(int)
                _eff_tbl["WS_Wins"] = _eff_tbl["WS_Wins"].astype(int)

                def _eff_tier(gap):
                    if gap < -20: return ("Elite Value", "#22c55e")
                    if gap < -5:  return ("Efficient", "#14b8a6")
                    if gap <= 5:  return ("Market Rate", "#6b7280")
                    return ("Overpaying", "#ef4444")

                _eff_tbl["Tier"] = _eff_tbl["Avg_Gap"].apply(lambda g: _eff_tier(g)[0])
                _eff_tbl = _eff_tbl.sort_values("Avg_Gap")

                st.markdown("#### Efficiency vs Postseason Outcomes (2025-2021)")

                _n_teams = len(_eff_tbl)
                _tier_size = max(1, _n_teams // 5)
                def _rank_tier(idx):
                    if idx < _tier_size:     return "Top Tier"
                    if idx < _tier_size * 2: return "Above Average"
                    if idx < _tier_size * 3: return "Average"
                    if idx < _tier_size * 4: return "Below Average"
                    return "Bottom"
                _eff_tbl["Efficiency Tier"] = [_rank_tier(i) for i in range(len(_eff_tbl))]
                _eff_tbl.insert(0, "#", range(1, len(_eff_tbl) + 1))

                _RANK_CLR = {"Top Tier": "#14532d", "Above Average": "#1a3a20",
                             "Average": "", "Below Average": "#2d1f0c", "Bottom": "#2d0c0c"}
                def _tier_clr(row):
                    bg = _RANK_CLR.get(row.get("Efficiency Tier", ""), "")
                    return [f"background-color:{bg}"] * len(row) if bg else [""] * len(row)

                _eff_disp = _eff_tbl.drop(columns=["Tier"], errors="ignore").rename(columns={
                    "Avg_Gap": "Avg Gap ($M)", "Playoff_Apps": "Playoff Apps",
                    "WS_Apps": "WS Appearances", "WS_Wins": "WS Wins",
                })
                st.dataframe(
                    _eff_disp.style.apply(_tier_clr, axis=1).format(
                        {"Avg Gap ($M)": "{:.1f}"}, na_rep="\u2014"),
                    hide_index=True, use_container_width=True, height=400,
                )

                _eff_good = _eff_tbl[_eff_tbl["Avg_Gap"] < -10]
                _eff_bad  = _eff_tbl[_eff_tbl["Avg_Gap"] > 10]
                _n_seasons = len(detail_df["Year"].unique())
                if not _eff_good.empty and not _eff_bad.empty:
                    _good_pct = _eff_good["Playoff_Apps"].sum() / (len(_eff_good) * _n_seasons) * 100
                    _bad_pct  = _eff_bad["Playoff_Apps"].sum() / (len(_eff_bad) * _n_seasons) * 100
                    st.info(
                        f"Among the {len(_eff_good)} most efficient teams (gap < -$10M), "
                        f"{_eff_good['Playoff_Apps'].sum()} made the playoffs "
                        f"({_good_pct:.0f}% of seasons) vs {_bad_pct:.0f}% for the "
                        f"{len(_eff_bad)} least efficient teams."
                    )

                st.markdown(
                    "<div style='background:#0d1e35;border-left:3px solid #3b6fd4;border-radius:0 8px 8px 0;"
                    "padding:0.7rem 1rem;margin-top:0.6rem;font-size:0.82rem;color:#93b8d8;line-height:1.6;'>"
                    "<b style='color:#60a5fa;'>Research context:</b> Over the past decade, World Series "
                    "champions averaged 8th in payroll \u2014 efficient roster construction matters more than "
                    "total spend once you reach the postseason.</div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 3: Salary ─────────────────────────────────────────────────────────
    if _act_tab == "salary":
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "Team payroll for the selected season. Use $/WAR to compare how much each team "
            "paid per unit of performance. Top-5 spenders in gold, bottom-5 in blue.</div>",
            unsafe_allow_html=True,
        )
        _sal = yr_df.sort_values("payroll_M", ascending=False).reset_index(drop=True)
        _sal["Rank"]   = range(1, len(_sal) + 1)
        _sal_max = _sal["payroll_M"].max() or 1

        ch3, tb3 = st.columns([3, 2])
        with ch3:
            st.plotly_chart(_hbar(
                _sal, "payroll_M",
                color_fn=lambda v: (
                    "#b88840" if v >= _sal_max * 0.80 else
                    "#4873b8" if v >= _sal_max * 0.45 else "#2e4a62"
                ),
                title=f"{sel_year} \u2014 Team Payroll",
                x_label="Payroll ($M)",
                text_fn=lambda v: f"${v:.0f}M",
            ), use_container_width=True, config={"displayModeBar": False})
        with tb3:
            _s = _sal[["Rank", "Team", "payroll_M", "team_WAR", "DPW", "Wins", "in_playoffs"]].copy()
            _s.columns = ["#", "Team", "Payroll $M", "WAR", "$/fWAR M", "Wins", "Postseason"]
            _s["Payroll $M"] = _s["Payroll $M"].round(0).astype(int)
            _s["WAR"]        = _s["WAR"].round(1)
            _s["$/fWAR M"]    = _s["$/fWAR M"].round(1)
            _s["Wins"]       = _s["Wins"].round(0).astype(int)
            _s["Postseason"]  = _s["Postseason"].map({True: "\u2713", False: ""})

            def _sal_clr(row):
                if row["#"] <= 5:   return ["background-color:#2d1f0c"] * len(row)
                if row["#"] >= 26:  return ["background-color:#0c1a2d"] * len(row)
                return [""] * len(row)

            st.dataframe(
                _s.style.apply(_sal_clr, axis=1).format(
                    {"Payroll $M": "{:d}", "WAR": "{:.1f}", "$/fWAR M": "{:.1f}", "Wins": "{:d}"}, na_rep="\u2014"),
                hide_index=True, use_container_width=True,
                height=min(60 + len(_s) * 35, 720),
            )

        # Marginal Spending Impact
        st.markdown("---")
        st.markdown("### Marginal Spending Impact")
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
            "Not all payroll dollars spent are equal. The first $100M spent gains the most wins per $ "
            "than the next tier $244M to $300M+. Based on this nonlinear relationship, each bar below "
            "shows how many additional wins a team can expect per $10M spent within that spending tier, "
            "based on 2021-2025 data. "
            "<span style='color:#22c55e;font-weight:600;'>Green</span> = strong return, "
            "<span style='color:#f59e0b;font-weight:600;'>amber</span> = moderate, "
            "<span style='color:#ef4444;font-weight:600;'>red</span> = diminishing returns.</div>",
            unsafe_allow_html=True,
        )

        if {"payroll_M", "Wins"}.issubset(detail_df.columns):
            _f2 = detail_df.dropna(subset=["payroll_M", "Wins"]).copy()
            _f2["payroll_M"] = pd.to_numeric(_f2["payroll_M"], errors="coerce")
            _f2["Wins"]      = pd.to_numeric(_f2["Wins"], errors="coerce")

            _TIERS = [
                ("Budget ($0-100M)",       0,   100),
                ("Mid-Market ($100-175M)", 100, 175),
                ("Contender ($175-244M)",  175, 244),
                ("Big Market ($244M+)",    244, 999),
            ]

            _slopes = []
            for name, lo, hi in _TIERS:
                tier_df = _f2[(_f2["payroll_M"] >= lo) & (_f2["payroll_M"] < hi)]
                if len(tier_df) >= 5:
                    c = np.polyfit(tier_df["payroll_M"], tier_df["Wins"], 1)
                    wins_per_10m = c[0] * 10
                else:
                    wins_per_10m = 0
                _slopes.append((name, round(wins_per_10m, 2)))

            _sl_names  = [s[0] for s in _slopes]
            _sl_vals   = [s[1] for s in _slopes]
            _sl_colors = ["#22c55e" if v >= 1.0 else "#f59e0b" if v >= 0.3 else "#ef4444" for v in _sl_vals]

            fig_f2 = go.Figure(go.Bar(
                y=_sl_names, x=_sl_vals, orientation="h",
                marker_color=_sl_colors,
                text=[f"{v:+.2f} wins" for v in _sl_vals],
                textposition="outside", textfont=dict(color="#dbeafe", size=10),
                hovertemplate="%{y}: %{x:.2f} wins per $10M<extra></extra>",
            ))
            _xmax_f2 = max(abs(v) for v in _sl_vals) * 1.5 if _sl_vals else 3
            fig_f2.update_layout(**_pt(
                title="Marginal Wins per $10M by Spending Tier",
                xaxis=dict(title="Wins per $10M spent",
                           zeroline=True, zerolinecolor="#4a687e", zerolinewidth=1,
                           range=[-_xmax_f2, _xmax_f2]),
                yaxis=dict(autorange="reversed"),
                height=300,
                margin=dict(l=180, r=80, t=40, b=40),
            ))
            _f2_col, _ = st.columns([3, 1])
            with _f2_col:
                st.plotly_chart(fig_f2, width="stretch")

            _spend_add = st.slider("Add spending ($M)", 0, 50, 10, step=5,
                                    key="v2_spend_slider")
            if _spend_add > 0 and _slopes:
                _cur_tier_idx = 1
                _cur_slope = _slopes[_cur_tier_idx][1]
                _add_wins = _cur_slope * (_spend_add / 10)
                st.markdown(
                    f"<div style='background:#0d1e35;border:1px solid #1e3250;border-radius:8px;"
                    f"padding:0.8rem 1rem;font-size:0.85rem;color:#d6e8f8;'>"
                    f"Adding <b>${_spend_add}M</b> at the <b>{_slopes[_cur_tier_idx][0]}</b> tier "
                    f"buys ~<b>{_add_wins:.1f}</b> additional wins "
                    f"(slope: {_cur_slope:.2f} wins/$10M)."
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 4: Win Performance ────────────────────────────────────────────────
    if _act_tab == "winperf":
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "Actual wins minus the wins <b>predicted by payroll</b> from the league regression. "
            "<span style='color:#22c55e;font-weight:600;'>Green</span> = team wins more than their "
            "spending predicts (efficient roster building). "
            "<span style='color:#ef4444;font-weight:600;'>Red</span> = team underperforms their payroll.</div>",
            unsafe_allow_html=True,
        )
        _wvp = yr_df.sort_values("wins_vs_pred", ascending=False).reset_index(drop=True)
        _wvp["Rank"] = range(1, len(_wvp) + 1)

        ch4, tb4 = st.columns([3, 2])
        with ch4:
            st.plotly_chart(_hbar(
                _wvp, "wins_vs_pred",
                color_fn=lambda v: "#22c55e" if v >= 0 else "#ef4444",
                title=f"{sel_year} \u2014 Wins Above/Below Payroll Prediction",
                x_label="Wins vs Predicted",
                text_fn=lambda v: f"{v:+.1f}W",
                zero_line=True,
            ), use_container_width=True, config={"displayModeBar": False})
        with tb4:
            _vp = _wvp[["Rank", "Team", "Wins", "pred_wins", "wins_vs_pred",
                         "payroll_M", "in_playoffs"]].copy()
            _vp.columns = ["#", "Team", "Wins", "Predicted", "\u0394 Wins", "Payroll $M", "Postseason"]
            _vp["Wins"]       = _vp["Wins"].round(0).astype(int)
            _vp["Predicted"]  = _vp["Predicted"].round(1)
            _vp["\u0394 Wins"]     = _vp["\u0394 Wins"].round(1)
            _vp["Payroll $M"] = _vp["Payroll $M"].round(0).astype(int)
            _vp["Postseason"]  = _vp["Postseason"].map({True: "\u2713", False: ""})

            def _vp_clr(row):
                d = row["\u0394 Wins"]
                if d > 10:  return ["background-color:#0c2218"] * len(row)
                if d > 0:   return ["background-color:#14532d55"] * len(row)
                if d < -10: return ["background-color:#2d0c0c"] * len(row)
                if d < 0:   return ["background-color:#2d150c55"] * len(row)
                return [""] * len(row)

            st.dataframe(
                _vp.style.apply(_vp_clr, axis=1).format(
                    {"Wins": "{:d}", "Predicted": "{:.1f}", "\u0394 Wins": "{:+.1f}", "Payroll $M": "{:d}"}, na_rep="\u2014"),
                hide_index=True, use_container_width=True,
                height=min(60 + len(_vp) * 35, 720),
            )

    # ── Tab 5: Contract Value ─────────────────────────────────────────────────
    if _act_tab == "contract_value":
        # Explainer
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;line-height:1.6;'>"
            "<b style='color:#d6e8f8;'>Contract Value</b> measures how much on-field production "
            "a player delivers per dollar of salary. <b>Higher is better</b> — it means the player "
            "produces more fWAR for every million spent. "
            "<span style='color:#5dc9a5;'>Pre-Arb</span> players typically offer the best value "
            "because they earn near league minimum."
            "<br><span style='font-size:0.75rem;color:#5a7a94;'>Formula: "
            "<code style='color:#60a5fa;'>Contract Value = fWAR ÷ Salary ($M)</code></span></div>",
            unsafe_allow_html=True,
        )
        try:
            if _paw_df is not None and not _paw_25.empty:
                _cv_df = _paw_25[_paw_25["Salary_M"] > 1.0].copy()
                _cv_df["fWAR_per_M"] = (_cv_df["WAR_Total"] / _cv_df["Salary_M"].clip(lower=0.1)).round(2)
                _cv_top = _cv_df.sort_values("fWAR_per_M", ascending=False).head(15)

                _cv_left, _cv_right = st.columns([7, 3])
                with _cv_left:
                    if not _cv_top.empty:
                        _cv_bar_colors = [
                            _STAGE_COLORS.get(s, "#4a687e")
                            for s in _cv_top.get("Stage_Clean", pd.Series([""] * len(_cv_top)))
                        ]
                        _cv_labels = [
                            f"{p} ({t})" for p, t in zip(_cv_top["Player"], _cv_top["Team"])
                        ]
                        fig_cv = go.Figure(go.Bar(
                            y=_cv_labels, x=_cv_top["fWAR_per_M"].tolist(), orientation="h",
                            marker=dict(color=_cv_bar_colors, line=dict(width=0)),
                            text=[f"{v:.2f}" for v in _cv_top["fWAR_per_M"]],
                            textposition="outside",
                            textfont=dict(color="#d6e8f8", size=9),
                            hovertemplate="%{y}: %{x:.2f} fWAR/$M<extra></extra>",
                        ))
                        fig_cv.update_layout(**_pt(
                            title=f"{sel_year} — Top 15 Players by Contract Value (fWAR/$M)",
                            xaxis=dict(title="fWAR per $M"),
                            yaxis=dict(autorange="reversed"),
                            height=max(380, len(_cv_top) * 30),
                            margin=dict(l=160, r=80, t=42, b=30),
                        ))
                        st.plotly_chart(fig_cv, use_container_width=True, config={"displayModeBar": False})
                with _cv_right:
                    if "Stage_Clean" in _cv_df.columns:
                        for _stg_key in ["FA", "Arb", "Pre-Arb"]:
                            _stg_sub = _cv_df[_cv_df["Stage_Clean"] == _stg_key]
                            _stg_val = _stg_sub["fWAR_per_M"].mean() if not _stg_sub.empty else 0
                            _stg_best = _stg_sub.sort_values("fWAR_per_M", ascending=False).iloc[0] if not _stg_sub.empty else None
                            _best_txt = f"Best: {_stg_best['Player']} — {_stg_best['fWAR_per_M']:.2f}" if _stg_best is not None else ""
                            st.markdown(
                                _stage_card(_stg_key, f"{_stg_val:.2f}", _best_txt, "avg fWAR/$M"),
                                unsafe_allow_html=True,
                            )

                # Historical line plot by stage
                st.markdown("---")
                st.markdown("##### Average Contract Value by Career Stage (2021–2025)")
                _cv_all = _paw_df[_paw_df["Salary_M"] > 1.0].copy()
                _cv_all["fWAR_per_M"] = (_cv_all["WAR_Total"] / _cv_all["Salary_M"].clip(lower=0.1)).round(3)
                _cv_hist = _cv_all.groupby(["Year", "Stage_Clean"])["fWAR_per_M"].mean().reset_index()

                fig_cv_hist = go.Figure()
                for _sk in ["Pre-Arb", "Arb", "FA"]:
                    _sk_df = _cv_hist[_cv_hist["Stage_Clean"] == _sk].sort_values("Year")
                    if not _sk_df.empty:
                        fig_cv_hist.add_trace(go.Scatter(
                            x=_sk_df["Year"].astype(int), y=_sk_df["fWAR_per_M"],
                            mode="lines+markers", name=_STAGE_LABELS[_sk],
                            line=dict(color=_STAGE_COLORS[_sk], width=2),
                            marker=dict(size=7),
                        ))
                fig_cv_hist.update_layout(**_pt(
                    xaxis=dict(title="Season", dtick=1),
                    yaxis=dict(title="Avg fWAR/$M"),
                    height=350, showlegend=True,
                    legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                    margin=dict(l=50, r=30, t=30, b=40),
                ))
                _cv_hist_col, _ = st.columns([3, 1])
                with _cv_hist_col:
                    st.plotly_chart(fig_cv_hist, use_container_width=True, config={"displayModeBar": False})

                # PPEL reference
                with st.expander("📐 PPEL Formula Reference", expanded=False):
                    st.markdown(
                        "The **Pay-to-Performance Efficiency Line (PPEL)** is the league-wide regression "
                        "of salary vs fWAR. Players above the line deliver more value than their salary "
                        "predicts; players below it are overpaid relative to production.\n\n"
                        "- **Contract Value** = fWAR ÷ Salary ($M) — single-season snapshot\n"
                        "- **PPR** (Pay-Performance Ratio) = Σ actual fWAR across contract years ÷ total contract $M — "
                        "multi-year view of whether the full contract delivered value"
                    )
            else:
                st.info("Player data not available for contract value analysis.")
        except Exception as _cv_e:
            st.warning(f"Could not compute contract value data: {_cv_e}")

    # ── Tab 6: Stability ──────────────────────────────────────────────────────
    if _act_tab == "stability":
        # Explainer
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;line-height:1.6;'>"
            "<b style='color:#d6e8f8;'>WAR Stability Rating (WSR)</b> measures how consistently "
            "a player produces fWAR year over year. <b>Higher is better</b> — a high WSR means "
            "you can count on reliable production, not a one-year outlier."
            "<br><span style='font-size:0.75rem;color:#5a7a94;'>Formula: "
            "<code style='color:#60a5fa;'>WSR = Mean fWAR ÷ (1 + Std Dev fWAR)</code> · "
            "Players with 3+ seasons included.</span></div>",
            unsafe_allow_html=True,
        )
        try:
            if _paw_df is not None and not _wsr_grp.empty:
                # Controls
                _, _, _ws_c1, _ws_c2 = st.columns([4, 2, 2, 2])
                with _ws_c1:
                    _ws_n = st.selectbox("# Counted", [5, 10, 15, 25], index=2, key="ws_n_count")
                with _ws_c2:
                    _ws_yr_opt = st.selectbox("Years", ["5 yr (2021–2025)", "3 yr (2023–2025)"], key="ws_yr_range")
                _ws_min_yr = 2023 if "3 yr" in _ws_yr_opt else 2021

                # Recompute WSR with selected year range
                _ws_paw = _paw_df[_paw_df["Year"] >= _ws_min_yr].dropna(subset=["WAR_Total"])
                _ws_grp2 = _ws_paw.groupby("Player").agg(
                    _mean=("WAR_Total", "mean"), _std=("WAR_Total", "std"),
                    _n=("Year", "nunique"), Team=("Team", "last"),
                    Stage_Clean=("Stage_Clean", "last"),
                ).reset_index()
                _min_seasons = 3 if _ws_min_yr == 2021 else 2
                _ws_grp2 = _ws_grp2[_ws_grp2["_n"] >= _min_seasons].copy()
                _ws_grp2["_std"] = _ws_grp2["_std"].fillna(0)
                _ws_grp2["WSR"] = (_ws_grp2["_mean"] / (1 + _ws_grp2["_std"])).round(2)
                _wsr_top2 = _ws_grp2.sort_values("WSR", ascending=False).head(_ws_n)

                _ws_left, _ws_right = st.columns([7, 3])
                with _ws_left:
                    # fWAR over time for top N players
                    _ws_players = _wsr_top2["Player"].tolist()
                    _ws_ts = _ws_paw[_ws_paw["Player"].isin(_ws_players)].copy()
                    fig_ws = go.Figure()
                    for _, _row in _wsr_top2.iterrows():
                        _p = _row["Player"]
                        _pdata = _ws_ts[_ws_ts["Player"] == _p].sort_values("Year")
                        _stg = str(_row.get("Stage_Clean", ""))
                        _clr = _STAGE_COLORS.get(_stg, "#4a687e")
                        fig_ws.add_trace(go.Scatter(
                            x=_pdata["Year"].astype(int), y=_pdata["WAR_Total"],
                            mode="lines+markers", name=_p,
                            line=dict(color=_clr, width=2), marker=dict(size=6),
                            hovertemplate=f"<b>{_p}</b> ({_row['Team']})<br>"
                                          f"Year: %{{x}}<br>fWAR: %{{y:.1f}}<extra></extra>",
                        ))
                    fig_ws.update_layout(**_pt(
                        title=f"fWAR Stability — Top {_ws_n} Players",
                        xaxis=dict(title="Season", dtick=1),
                        yaxis=dict(title="fWAR"),
                        height=420, showlegend=True,
                        legend=dict(font=dict(size=9), y=0.5),
                        margin=dict(l=50, r=10, t=40, b=40),
                    ))
                    st.plotly_chart(fig_ws, use_container_width=True, config={"displayModeBar": False})
                with _ws_right:
                    for _stg_key in ["FA", "Arb", "Pre-Arb"]:
                        _stg_sub = _ws_grp2[_ws_grp2["Stage_Clean"] == _stg_key]
                        _stg_val = _stg_sub["WSR"].mean() if not _stg_sub.empty else 0
                        _stg_best = _stg_sub.sort_values("WSR", ascending=False).iloc[0] if not _stg_sub.empty else None
                        _best_txt = f"Best: {_stg_best['Player']} — {_stg_best['WSR']:.2f}" if _stg_best is not None else ""
                        st.markdown(
                            _stage_card(_stg_key, f"{_stg_val:.2f}", _best_txt, "avg WSR"),
                            unsafe_allow_html=True,
                        )

                # ── Who's Trending Up ─────────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Who's Trending Up")
                st.markdown(
                    "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
                    "Players whose fWAR has been increasing over the selected time window "
                    "(positive slope). Steeper lines = faster improvement.</div>",
                    unsafe_allow_html=True,
                )

                # Calculate per-player fWAR trend (slope)
                _trend_players = _ws_paw.groupby("Player").filter(lambda g: len(g) >= _min_seasons)
                _slopes_list = []
                for _p, _grp in _trend_players.groupby("Player"):
                    if len(_grp) >= _min_seasons:
                        _yrs = _grp["Year"].values.astype(float)
                        _wars = _grp["WAR_Total"].values.astype(float)
                        _slope = np.polyfit(_yrs, _wars, 1)[0]
                        _last = _grp.sort_values("Year").iloc[-1]
                        _slopes_list.append({
                            "Player": _p, "slope": round(_slope, 3),
                            "Team": _last["Team"],
                            "Stage_Clean": _last.get("Stage_Clean", ""),
                            "last_fWAR": round(_last["WAR_Total"], 1),
                        })
                if _slopes_list:
                    _slope_df = pd.DataFrame(_slopes_list)
                    _trending = _slope_df[_slope_df["slope"] > 0].sort_values("slope", ascending=False).head(_ws_n)

                    if not _trending.empty:
                        _tu_left, _tu_right = st.columns([7, 3])
                        with _tu_left:
                            _tu_players = _trending["Player"].tolist()
                            _tu_ts = _ws_paw[_ws_paw["Player"].isin(_tu_players)].copy()
                            fig_tu = go.Figure()
                            for _, _row in _trending.iterrows():
                                _p = _row["Player"]
                                _pdata = _tu_ts[_tu_ts["Player"] == _p].sort_values("Year")
                                _clr = _STAGE_COLORS.get(str(_row.get("Stage_Clean", "")), "#4a687e")
                                fig_tu.add_trace(go.Scatter(
                                    x=_pdata["Year"].astype(int), y=_pdata["WAR_Total"],
                                    mode="lines+markers", name=_p,
                                    line=dict(color=_clr, width=2), marker=dict(size=6),
                                    hovertemplate=f"<b>{_p}</b> ({_row['Team']})<br>"
                                                  f"Year: %{{x}}<br>fWAR: %{{y:.1f}}<extra></extra>",
                                ))
                            fig_tu.update_layout(**_pt(
                                title=f"Trending Up — Top {_ws_n} by fWAR Slope",
                                xaxis=dict(title="Season", dtick=1),
                                yaxis=dict(title="fWAR"),
                                height=400, showlegend=True,
                                legend=dict(font=dict(size=9), y=0.5),
                                margin=dict(l=50, r=10, t=40, b=40),
                            ))
                            st.plotly_chart(fig_tu, use_container_width=True, config={"displayModeBar": False})
                        with _tu_right:
                            for _stg_key in ["FA", "Arb", "Pre-Arb"]:
                                _stg_sub = _trending[_trending["Stage_Clean"] == _stg_key]
                                _stg_avg = _stg_sub["slope"].mean() if not _stg_sub.empty else 0
                                _stg_best = _stg_sub.sort_values("slope", ascending=False).iloc[0] if not _stg_sub.empty else None
                                _best_txt = f"Best: {_stg_best['Player']} — +{_stg_best['slope']:.2f}/yr" if _stg_best is not None else "—"
                                st.markdown(
                                    _stage_card(_stg_key, f"+{_stg_avg:.2f}", _best_txt, "avg fWAR slope/yr"),
                                    unsafe_allow_html=True,
                                )
            else:
                st.info("Not enough multi-year player data for stability analysis.")
        except Exception as _ws_e:
            st.warning(f"Could not compute stability data: {_ws_e}")

    # ── Tab 7: Playoff Success ───────────────────────────────────────────────
    if _act_tab == "playoff_success":
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;line-height:1.6;'>"
            "Do teams that spend efficiently actually win more in October? We compare playoff "
            "appearance rates between the most and least efficient teams across 2021–2025.</div>",
            unsafe_allow_html=True,
        )
        try:
            if {"dollar_gap_M", "in_playoffs"}.issubset(detail_df.columns):
                _ps_df = detail_df.dropna(subset=["dollar_gap_M"]).copy()
                _ps_df["dollar_gap_M"] = pd.to_numeric(_ps_df["dollar_gap_M"], errors="coerce")
                _n3 = max(1, len(_ps_df["Team"].unique()) // 3)

                # Rank teams by avg efficiency
                _ps_team = _ps_df.groupby("Team").agg(
                    avg_gap=("dollar_gap_M", "mean"),
                    playoff_apps=("in_playoffs", "sum"),
                    seasons=("Year", "nunique"),
                ).reset_index()
                _ps_team["playoff_rate"] = (_ps_team["playoff_apps"] / _ps_team["seasons"] * 100).round(1)
                _ps_team = _ps_team.sort_values("avg_gap")

                _top_tier = _ps_team.head(_n3)
                _mid_tier = _ps_team.iloc[_n3:2 * _n3]
                _bot_tier = _ps_team.tail(_n3)

                _ps_left, _ps_right = st.columns([7, 3])
                with _ps_left:
                    fig_ps = go.Figure()
                    for _tier_name, _tier_df, _color in [
                        ("Top Efficiency", _top_tier, "#22c55e"),
                        ("Mid Efficiency", _mid_tier, "#f59e0b"),
                        ("Low Efficiency", _bot_tier, "#ef4444"),
                    ]:
                        fig_ps.add_trace(go.Bar(
                            x=_tier_df["Team"], y=_tier_df["playoff_rate"],
                            name=_tier_name, marker_color=_color,
                            text=[f"{v:.0f}%" for v in _tier_df["playoff_rate"]],
                            textposition="outside", textfont=dict(size=9, color="#d6e8f8"),
                            hovertemplate="%{x}: %{y:.0f}% playoff rate<extra></extra>",
                        ))
                    fig_ps.update_layout(**_pt(
                        title="Playoff Appearance Rate by Efficiency Tier (2021–2025)",
                        xaxis=dict(title="", tickangle=-45),
                        yaxis=dict(title="Playoff Rate (%)", range=[0, 110]),
                        height=420, showlegend=True, barmode="group",
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                        margin=dict(l=50, r=20, t=40, b=80),
                    ))
                    st.plotly_chart(fig_ps, use_container_width=True, config={"displayModeBar": False})
                with _ps_right:
                    for _label, _tier, _clr in [
                        ("Top Tier", _top_tier, "#22c55e"),
                        ("Mid Tier", _mid_tier, "#f59e0b"),
                        ("Bottom Tier", _bot_tier, "#ef4444"),
                    ]:
                        _rate = _tier["playoff_rate"].mean()
                        _apps = int(_tier["playoff_apps"].sum())
                        st.markdown(
                            f"<div style='background:#1c2a42;border:1px solid #1e3050;"
                            f"border-left:3px solid {_clr};border-radius:8px;"
                            f"padding:0.7rem 0.9rem;margin-bottom:0.5rem;text-align:center;'>"
                            f"<div style='font-size:0.7rem;color:#93b8d8;font-weight:600;'>{_label} Efficiency</div>"
                            f"<div style='font-size:1.4rem;font-weight:800;color:{_clr};margin:0.2rem 0;'>"
                            f"{_rate:.0f}%</div>"
                            f"<div style='font-size:0.68rem;color:#7a9ebc;'>playoff rate</div>"
                            f"<div style='font-size:0.65rem;color:#5a7a94;'>{_apps} appearances over 5 yr</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                # Historical trend
                st.markdown("---")
                st.markdown("##### Playoff Rate by Efficiency Tier Over Time")
                _ps_yr_data = []
                for yr in sorted(_ps_df["Year"].unique()):
                    _yr_teams = _ps_df[_ps_df["Year"] == yr].copy()
                    _yr_teams = _yr_teams.sort_values("dollar_gap_M")
                    _n_t = max(1, len(_yr_teams) // 3)
                    for _tname, _tslice in [("Top", _yr_teams.head(_n_t)),
                                              ("Mid", _yr_teams.iloc[_n_t:2*_n_t]),
                                              ("Bottom", _yr_teams.tail(_n_t))]:
                        _ps_yr_data.append({
                            "Year": int(yr), "Tier": _tname,
                            "Rate": _tslice["in_playoffs"].mean() * 100,
                        })
                if _ps_yr_data:
                    _ps_yr_df = pd.DataFrame(_ps_yr_data)
                    _tier_colors = {"Top": "#22c55e", "Mid": "#f59e0b", "Bottom": "#ef4444"}
                    fig_ps_hist = go.Figure()
                    for _t in ["Top", "Mid", "Bottom"]:
                        _td = _ps_yr_df[_ps_yr_df["Tier"] == _t].sort_values("Year")
                        fig_ps_hist.add_trace(go.Scatter(
                            x=_td["Year"], y=_td["Rate"], mode="lines+markers",
                            name=f"{_t} Tier", line=dict(color=_tier_colors[_t], width=2),
                            marker=dict(size=7),
                        ))
                    fig_ps_hist.update_layout(**_pt(
                        xaxis=dict(title="Season", dtick=1),
                        yaxis=dict(title="Playoff Rate (%)", range=[0, 100]),
                        height=340, showlegend=True,
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                        margin=dict(l=50, r=30, t=30, b=40),
                    ))
                    _ps_h_col, _ = st.columns([3, 1])
                    with _ps_h_col:
                        st.plotly_chart(fig_ps_hist, use_container_width=True, config={"displayModeBar": False})
        except Exception as _ps_e:
            st.warning(f"Could not compute playoff success data: {_ps_e}")

    # ── Tab 8: fWAR Cost ─────────────────────────────────────────────────────
    if _act_tab == "fwar_cost":
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;line-height:1.6;'>"
            "The market price of one win fluctuates each year. Here's what teams pay per fWAR "
            "across career stages — Pre-Arb players cost a fraction of free agents.</div>",
            unsafe_allow_html=True,
        )
        try:
            if _paw_df is not None:
                _fc_all = _paw_df[(_paw_df["Salary_M"] > 0.3) & (_paw_df["WAR_Total"] > 0.5)].copy()
                _fc_all["cost_per_fwar"] = (_fc_all["Salary_M"] / _fc_all["WAR_Total"].clip(lower=0.1)).round(2)

                # Historical $/fWAR by stage
                _fc_hist = _fc_all.groupby(["Year", "Stage_Clean"])["cost_per_fwar"].median().reset_index()

                _fc_left, _fc_right = st.columns([7, 3])
                with _fc_left:
                    fig_fc = go.Figure()
                    for _sk in ["Pre-Arb", "Arb", "FA"]:
                        _sk_df = _fc_hist[_fc_hist["Stage_Clean"] == _sk].sort_values("Year")
                        if not _sk_df.empty:
                            fig_fc.add_trace(go.Scatter(
                                x=_sk_df["Year"].astype(int), y=_sk_df["cost_per_fwar"],
                                mode="lines+markers", name=_STAGE_LABELS[_sk],
                                line=dict(color=_STAGE_COLORS[_sk], width=2.5),
                                marker=dict(size=8),
                            ))
                    fig_fc.update_layout(**_pt(
                        title="Median $/fWAR by Career Stage (2021–2025)",
                        xaxis=dict(title="Season", dtick=1),
                        yaxis=dict(title="$/fWAR ($M)"),
                        height=400, showlegend=True,
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                        margin=dict(l=50, r=30, t=40, b=40),
                    ))
                    st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})
                with _fc_right:
                    _fc_cur = _fc_all[_fc_all["Year"] == sel_year]
                    for _stg_key in ["FA", "Arb", "Pre-Arb"]:
                        _stg_sub = _fc_cur[_fc_cur["Stage_Clean"] == _stg_key]
                        _stg_val = _stg_sub["cost_per_fwar"].median() if not _stg_sub.empty else 0
                        _stg_best = _stg_sub.sort_values("cost_per_fwar").iloc[0] if not _stg_sub.empty else None
                        _best_txt = f"Best: {_stg_best['Player']} — &#36;{_stg_best['cost_per_fwar']:.1f}M" if _stg_best is not None else ""
                        st.markdown(
                            _stage_card(_stg_key, f"&#36;{_stg_val:.1f}M", _best_txt, "median &#36;/fWAR"),
                            unsafe_allow_html=True,
                        )

                # Top 10 best-value players (min 2.0 fWAR)
                st.markdown("---")
                st.markdown(f"##### Best Value Players {sel_year} (min 2.0 fWAR)")
                _fc_best = _fc_cur[_fc_cur["WAR_Total"] >= 2.0].sort_values("cost_per_fwar").head(10)
                if not _fc_best.empty:
                    _fb = _fc_best[["Player", "Team", "Position", "WAR_Total", "Salary_M", "cost_per_fwar", "Stage_Clean"]].copy()
                    _fb.insert(0, "#", range(1, len(_fb) + 1))
                    _fb = _fb.rename(columns={"WAR_Total": "fWAR", "Salary_M": "Salary $M", "cost_per_fwar": "$/fWAR $M", "Stage_Clean": "Stage"})
                    st.dataframe(_fb, hide_index=True, use_container_width=True, height=400)
        except Exception as _fc_e:
            st.warning(f"Could not compute fWAR cost data: {_fc_e}")

    # ── Tab 9: Position Value ────────────────────────────────────────────────
    if _act_tab == "position_value":
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;line-height:1.6;'>"
            "Which positions give teams the most production per dollar? Position scarcity "
            "and market dynamics create big differences in value.</div>",
            unsafe_allow_html=True,
        )
        try:
            if _paw_df is not None:
                _pv_pos = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "SP", "RP"]
                _pv_all = _paw_df[
                    (_paw_df["Position"].isin(_pv_pos)) &
                    (_paw_df["Salary_M"] > 0.3) &
                    (_paw_df["WAR_Total"].notna())
                ].copy()
                _pv_all["fWAR_per_M"] = (_pv_all["WAR_Total"] / _pv_all["Salary_M"].clip(lower=0.1)).round(3)

                # Current year position avg
                _pv_cur = _pv_all[_pv_all["Year"] == sel_year]
                _pv_avg = _pv_cur.groupby("Position")["fWAR_per_M"].mean().reindex(_pv_pos).dropna().sort_values(ascending=True)

                _pv_left, _pv_right = st.columns([7, 3])
                with _pv_left:
                    _pv_colors = ["#22c55e" if v >= _pv_avg.quantile(0.75) else
                                  "#4873b8" if v >= _pv_avg.quantile(0.25) else "#4a687e"
                                  for v in _pv_avg.values]
                    fig_pv = go.Figure(go.Bar(
                        y=_pv_avg.index.tolist(), x=_pv_avg.values.tolist(), orientation="h",
                        marker=dict(color=_pv_colors, line=dict(width=0)),
                        text=[f"{v:.2f}" for v in _pv_avg.values],
                        textposition="outside", textfont=dict(color="#d6e8f8", size=10),
                        hovertemplate="%{y}: %{x:.2f} fWAR/$M<extra></extra>",
                    ))
                    fig_pv.update_layout(**_pt(
                        title=f"{sel_year} — Average fWAR/$M by Position",
                        xaxis=dict(title="fWAR per $M"),
                        yaxis=dict(autorange="reversed"),
                        height=max(340, len(_pv_avg) * 32),
                        margin=dict(l=60, r=80, t=42, b=30),
                    ))
                    st.plotly_chart(fig_pv, use_container_width=True, config={"displayModeBar": False})
                with _pv_right:
                    _best_p = _pv_avg.idxmax() if not _pv_avg.empty else "—"
                    _worst_p = _pv_avg.idxmin() if not _pv_avg.empty else "—"

                    # Biggest year-over-year change
                    _pv_prev = _pv_all[_pv_all["Year"] == sel_year - 1].groupby("Position")["fWAR_per_M"].mean()
                    _pv_cur_avg = _pv_all[_pv_all["Year"] == sel_year].groupby("Position")["fWAR_per_M"].mean()
                    _pv_delta = (_pv_cur_avg - _pv_prev).dropna()
                    _biggest_change = _pv_delta.idxmax() if not _pv_delta.empty else "—"
                    _biggest_val = _pv_delta.max() if not _pv_delta.empty else 0

                    st.markdown(
                        f"<div style='background:#1c2a42;border:1px solid #1e3050;"
                        f"border-left:3px solid #22c55e;border-radius:8px;"
                        f"padding:0.7rem 0.9rem;margin-bottom:0.5rem;text-align:center;'>"
                        f"<div style='font-size:0.7rem;color:#93b8d8;font-weight:600;'>Most Efficient</div>"
                        f"<div style='font-size:1.4rem;font-weight:800;color:#22c55e;margin:0.2rem 0;'>{_best_p}</div>"
                        f"<div style='font-size:0.68rem;color:#7a9ebc;'>{_pv_avg.max():.2f} fWAR/$M</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='background:#1c2a42;border:1px solid #1e3050;"
                        f"border-left:3px solid #ef4444;border-radius:8px;"
                        f"padding:0.7rem 0.9rem;margin-bottom:0.5rem;text-align:center;'>"
                        f"<div style='font-size:0.7rem;color:#93b8d8;font-weight:600;'>Least Efficient</div>"
                        f"<div style='font-size:1.4rem;font-weight:800;color:#ef4444;margin:0.2rem 0;'>{_worst_p}</div>"
                        f"<div style='font-size:0.68rem;color:#7a9ebc;'>{_pv_avg.min():.2f} fWAR/$M</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='background:#1c2a42;border:1px solid #1e3050;"
                        f"border-left:3px solid #60a5fa;border-radius:8px;"
                        f"padding:0.7rem 0.9rem;margin-bottom:0.5rem;text-align:center;'>"
                        f"<div style='font-size:0.7rem;color:#93b8d8;font-weight:600;'>Biggest Change YoY</div>"
                        f"<div style='font-size:1.4rem;font-weight:800;color:#60a5fa;margin:0.2rem 0;'>{_biggest_change}</div>"
                        f"<div style='font-size:0.68rem;color:#7a9ebc;'>+{_biggest_val:.2f} fWAR/$M vs prior year</div></div>",
                        unsafe_allow_html=True,
                    )

                # Historical top 3 positions
                st.markdown("---")
                st.markdown("##### Position Value Trends (2021–2025)")
                _pv_hist = _pv_all.groupby(["Year", "Position"])["fWAR_per_M"].mean().reset_index()
                _top3_pos = _pv_avg.nlargest(3).index.tolist() if not _pv_avg.empty else []
                if _top3_pos:
                    fig_pv_hist = go.Figure()
                    _pos_colors = ["#22c55e", "#60a5fa", "#f59e0b"]
                    for i, _pos in enumerate(_top3_pos):
                        _pd = _pv_hist[_pv_hist["Position"] == _pos].sort_values("Year")
                        fig_pv_hist.add_trace(go.Scatter(
                            x=_pd["Year"].astype(int), y=_pd["fWAR_per_M"],
                            mode="lines+markers", name=_pos,
                            line=dict(color=_pos_colors[i % 3], width=2),
                            marker=dict(size=7),
                        ))
                    fig_pv_hist.update_layout(**_pt(
                        title="Top 3 Positions — fWAR/$M Over Time",
                        xaxis=dict(title="Season", dtick=1),
                        yaxis=dict(title="Avg fWAR/$M"),
                        height=340, showlegend=True,
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                        margin=dict(l=50, r=30, t=40, b=40),
                    ))
                    _pv_h_col, _ = st.columns([3, 1])
                    with _pv_h_col:
                        st.plotly_chart(fig_pv_hist, use_container_width=True, config={"displayModeBar": False})
        except Exception as _pv_e:
            st.warning(f"Could not compute position value data: {_pv_e}")

    # ── Tab 10: Age Value ───────────────────────────────────────────────────
    if _act_tab == "age_value":
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;line-height:1.6;'>"
            "At what age do players deliver the most production per dollar? Young players on "
            "pre-arb deals offer extreme value, but peak performance often comes later. "
            "This analysis finds the sweet spot."
            "<br><span style='font-size:0.75rem;color:#5a7a94;'>Formula: "
            "<code style='color:#60a5fa;'>Age Efficiency = Avg fWAR \u00f7 Avg Salary (&#36;M)</code>"
            "</span></div>",
            unsafe_allow_html=True,
        )
        try:
            if _paw_df is not None:
                _, _, _av_c1 = st.columns([6, 2, 2])
                with _av_c1:
                    _av_min_war = st.selectbox("Min fWAR", [0.5, 1.0, 2.0, 3.0], index=1, key="av_min_war")

                _av_all = _paw_df[
                    (_paw_df["Salary_M"] > 0.5) & (_paw_df["WAR_Total"] >= _av_min_war)
                ].copy()
                _av_all["Age"] = pd.to_numeric(_av_all.get("Age", pd.Series(dtype=float)), errors="coerce")
                _av_all = _av_all.dropna(subset=["Age"])
                _av_all["_age_int"] = _av_all["Age"].astype(int)
                _av_all["_wpm"] = _av_all["WAR_Total"] / _av_all["Salary_M"].clip(lower=0.1)

                _av_cur = _av_all[_av_all["Year"] == sel_year]
                _av_grp = _av_cur.groupby("_age_int").agg(
                    _wpm=("_wpm", "mean"), _n=("Player", "count"),
                    _war=("WAR_Total", "mean"), _sal=("Salary_M", "mean"),
                ).reset_index()
                _av_grp = _av_grp[_av_grp["_n"] >= 3].sort_values("_age_int")

                _av_left, _av_right = st.columns([7, 3])
                with _av_left:
                    if not _av_grp.empty:
                        _peak_age = _av_grp.loc[_av_grp["_wpm"].idxmax(), "_age_int"]
                        # Color by typical stage at each age
                        _av_colors = []
                        for age in _av_grp["_age_int"]:
                            if age <= 24:
                                _av_colors.append(_STAGE_COLORS["Pre-Arb"])
                            elif age <= 27:
                                _av_colors.append(_STAGE_COLORS["Arb"])
                            else:
                                _av_colors.append(_STAGE_COLORS["FA"])
                        fig_av = go.Figure(go.Bar(
                            x=_av_grp["_age_int"], y=_av_grp["_wpm"],
                            marker=dict(color=_av_colors, line=dict(width=0)),
                            text=[f"{v:.2f}" for v in _av_grp["_wpm"]],
                            textposition="outside", textfont=dict(color="#d6e8f8", size=9),
                            hovertemplate=(
                                "Age %{x}<br>fWAR/&#36;M: %{y:.2f}<br>"
                                "Players: %{customdata[0]}<br>"
                                "Avg fWAR: %{customdata[1]:.1f}<br>"
                                "Avg Salary: &#36;%{customdata[2]:.1f}M<extra></extra>"
                            ),
                            customdata=_av_grp[["_n", "_war", "_sal"]].values,
                        ))
                        fig_av.update_layout(**_pt(
                            title=f"{sel_year} — Efficiency by Age (min {_av_min_war} fWAR)",
                            xaxis=dict(title="Age", dtick=1),
                            yaxis=dict(title="Avg fWAR/&#36;M"),
                            height=400,
                            margin=dict(l=50, r=30, t=40, b=40),
                        ))
                        st.plotly_chart(fig_av, use_container_width=True, config={"displayModeBar": False})
                with _av_right:
                    for _stg_key in ["Pre-Arb", "Arb", "FA"]:
                        _stg_sub = _av_cur[_av_cur["Stage_Clean"] == _stg_key]
                        _stg_avg_age = _stg_sub["Age"].mean() if not _stg_sub.empty else 0
                        _stg_val = _stg_sub["_wpm"].mean() if not _stg_sub.empty else 0
                        _stg_best = _stg_sub.sort_values("_wpm", ascending=False).iloc[0] if not _stg_sub.empty else None
                        _best_txt = (f"Best: {_stg_best['Player']} age {int(_stg_best['Age'])} \u2014 "
                                     f"{_stg_best['_wpm']:.2f}") if _stg_best is not None else ""
                        st.markdown(
                            _stage_card(_stg_key, f"{_stg_val:.2f}",
                                        _best_txt, f"avg fWAR/&#36;M \u00b7 avg age {_stg_avg_age:.0f}"),
                            unsafe_allow_html=True,
                        )

                # Historical line chart by age group
                st.markdown("---")
                st.markdown("##### Efficiency by Age Group Over Time (2021\u20132025)")
                _av_hist_data = []
                for yr in sorted(_av_all["Year"].unique()):
                    _yr_d = _av_all[_av_all["Year"] == yr]
                    for _label, _lo, _hi, _clr in [
                        ("Under 25", 0, 25, _STAGE_COLORS["Pre-Arb"]),
                        ("25\u201329", 25, 30, _STAGE_COLORS["Arb"]),
                        ("30+", 30, 99, _STAGE_COLORS["FA"]),
                    ]:
                        _grp = _yr_d[(_yr_d["_age_int"] >= _lo) & (_yr_d["_age_int"] < _hi)]
                        if len(_grp) >= 3:
                            _av_hist_data.append({
                                "Year": int(yr), "Group": _label,
                                "wpm": _grp["_wpm"].mean(), "color": _clr,
                            })
                if _av_hist_data:
                    _av_hist_df = pd.DataFrame(_av_hist_data)
                    fig_av_hist = go.Figure()
                    for _g in ["Under 25", "25\u201329", "30+"]:
                        _gd = _av_hist_df[_av_hist_df["Group"] == _g].sort_values("Year")
                        if not _gd.empty:
                            fig_av_hist.add_trace(go.Scatter(
                                x=_gd["Year"], y=_gd["wpm"], mode="lines+markers",
                                name=_g, line=dict(color=_gd["color"].iloc[0], width=2),
                                marker=dict(size=7),
                            ))
                    fig_av_hist.update_layout(**_pt(
                        xaxis=dict(title="Season", dtick=1),
                        yaxis=dict(title="Avg fWAR/&#36;M"),
                        height=340, showlegend=True,
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
                        margin=dict(l=50, r=30, t=30, b=40),
                    ))
                    _av_h_col, _ = st.columns([3, 1])
                    with _av_h_col:
                        st.plotly_chart(fig_av_hist, use_container_width=True, config={"displayModeBar": False})

                # Peak age insight
                if not _av_grp.empty:
                    _peak_wpm = _av_grp.loc[_av_grp["_wpm"].idxmax()]
                    _lg_avg_wpm = _av_cur["_wpm"].mean() if not _av_cur.empty else 1
                    _pct_above = ((_peak_wpm["_wpm"] / max(_lg_avg_wpm, 0.01)) - 1) * 100
                    st.markdown(
                        f"<div style='background:#0d1e35;border-left:3px solid #5dc9a5;border-radius:0 8px 8px 0;"
                        f"padding:0.7rem 1rem;margin-top:0.6rem;font-size:0.82rem;color:#93b8d8;line-height:1.6;'>"
                        f"Players aged <b style='color:#5dc9a5;'>{int(_peak_wpm['_age_int'])}</b> produced the "
                        f"best value in {sel_year}, averaging <b style='color:#5dc9a5;'>"
                        f"{_peak_wpm['_wpm']:.2f} fWAR/&#36;M</b>. "
                        f"That's <b>{_pct_above:.0f}%</b> above the league average of "
                        f"{_lg_avg_wpm:.2f} fWAR/&#36;M.</div>",
                        unsafe_allow_html=True,
                    )
        except Exception as _av_e:
            st.warning(f"Could not compute age value data: {_av_e}")

    _render_feedback_widget("rankings")
