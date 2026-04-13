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


def render(*_args, **_kwargs):
    """Rankings page entry point."""

    # ── Query-param driven box selection ──────────────────────────────────────
    _qp_box = st.query_params.get("rk_box")
    _qp_yr  = st.query_params.get("rk_year")
    if _qp_box:
        st.session_state["rk_selected_box"] = _qp_box
        st.session_state["rk_box_clicked"] = True

    # ── CSS ──────────────────────────────────────────────────────────────────
    st.markdown("""<style>
.rk-hdr{background:linear-gradient(135deg,#18243a 0%,#111927 100%);
  border:1px solid #1e3250;border-radius:12px;padding:0.9rem 1.3rem;margin-bottom:0.8rem;}
.rk-hdr h2{margin:0;font-size:1.25rem;color:#d6e8f8;font-weight:700;}
.rk-hdr .rk-sub{font-size:0.72rem;color:#7a9ebc;margin-top:0.15rem;}
.rk-answer{background:#1c2a42;border:1px solid #1e3250;border-radius:10px;
  padding:0.8rem 1rem;text-align:center;min-height:120px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  transition:border-color 0.2s,box-shadow 0.2s;}
.rk-answer:hover{border-color:#f59e0b;box-shadow:0 0 10px rgba(245,158,11,0.25);}
.rk-answer .rk-q{font-size:0.72rem;color:#93b8d8;
  letter-spacing:0.05em;margin-bottom:0.2rem;font-weight:600;}
.rk-answer .rk-team{font-size:1.25rem;font-weight:800;color:#d6e8f8;line-height:1.1;}
.rk-answer .rk-val{font-size:0.82rem;color:#93b8d8;margin-top:0.2rem;}
.rk-answer .rk-icon{font-size:1.3rem;margin-bottom:0.15rem;line-height:1;}
.rk-box-sel{border-color:#f59e0b !important;box-shadow:0 0 12px rgba(245,158,11,0.3) !important;}
.rk-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.7rem;margin-bottom:0.7rem;}
.rk-grid a{text-decoration:none;color:inherit;}
@keyframes gentle-pulse{0%,100%{opacity:0.4;}50%{opacity:1;}}
.rk-hint{text-align:center;color:#4a687e;font-size:0.82rem;margin:0.6rem 0;
  animation:gentle-pulse 2.5s ease-in-out infinite;}
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
        "<h2>\U0001f3c6 MLB Spending Efficiency Rankings</h2>"
        "<div class='rk-sub'>All 30 MLB teams ranked by spending efficiency, fWAR production, "
        "payroll, and win performance. Efficiency measures how far above or below the "
        "cost-effective line each team sits \u2014 negative means winning more per dollar.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Year selector (right) + instruction (left) ───────────────────────────
    years_avail = sorted(detail_df["Year"].dropna().unique().astype(int), reverse=True)
    _hint_col, _, _yr_col = st.columns([5, 3, 2])
    with _hint_col:
        st.markdown(
            "<div style='padding-top:0.45rem;font-size:0.88rem;color:#7a9ebc;'>"
            "Click a box to get started</div>",
            unsafe_allow_html=True,
        )
    with _yr_col:
        _yr_default = years_avail.index(int(_qp_yr)) if _qp_yr and int(_qp_yr) in years_avail else 0
        sel_year = st.selectbox("Season", years_avail, key="rk_year_sel", index=_yr_default)

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
        "p_top_fwar": "player_rankings",
        "p_contract_val": "contract_value",
        "p_stability": "stability",
        "best_marginal": "marginal",
        "top_rss": "rss",
        "fwar_wins_link": "fwar",
    }
    _TAB_ORDER = [
        "efficiency", "fwar", "salary", "winperf", "contract_value", "stability",
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
            ("Budget ($0-100M)", 0, 100), ("Mid-Market ($100-175M)", 100, 175),
            ("Contender ($175-244M)", 175, 244), ("Big Market ($244M+)", 244, 999),
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
                    _best_marginal_val = f"+{_tw10:.2f} wins/$10M"

    # Top Roster Stability — best RSS team for the selected year
    _top_rss_team = "N/A"
    _top_rss_val = ""
    _rss_records_box = []
    try:
        if _paw_df is not None:
            _comb_box = _paw_df.copy()
            for _nc in ["PA", "IP"]:
                if _nc in _comb_box.columns:
                    _comb_box[_nc] = pd.to_numeric(_comb_box[_nc], errors="coerce")
            _is_pit_box = _comb_box["Position"].isin(["SP", "RP", "P", "TWP"])
            _comb_q_box = _comb_box[
                (_is_pit_box & (_comb_box["IP"].fillna(0) >= 150)) |
                (~_is_pit_box & (_comb_box["PA"].fillna(0) >= 150))
            ].copy()
            _yrs_box = sorted(_comb_q_box["Year"].dropna().unique().astype(int))
            if sel_year in _yrs_box and sel_year > min(_yrs_box):
                for _tm in _comb_q_box["Team"].unique():
                    _curr = set(_comb_q_box[(_comb_q_box["Year"] == sel_year) & (_comb_q_box["Team"] == _tm)]["Player"])
                    _prev = set(_comb_q_box[(_comb_q_box["Year"] == sel_year - 1) & (_comb_q_box["Team"] == _tm)]["Player"])
                    if _curr:
                        _rss_val = len(_curr & _prev) / len(_curr) * 100
                        _rss_records_box.append({"Team": _tm, "RSS": round(_rss_val, 1)})
            if _rss_records_box:
                _rss_box_df = pd.DataFrame(_rss_records_box).sort_values("RSS", ascending=False)
                _top_rss_row = _rss_box_df.iloc[0]
                _top_rss_team = str(_top_rss_row["Team"])
                _top_rss_val = f"{_top_rss_row['RSS']:.1f}% returning"
    except Exception:
        pass

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

    # ── Build box HTML helper ─────────────────────────────────────────────────
    def _box_html(box_id, label, team_name, val_str, logo_url="", img_html=""):
        _sel_cls = "rk-box-sel" if box_id == _sel_box else ""
        _href = f"?page=rankings&rk_box={box_id}&rk_year={sel_year}"
        _logo_tag = (f"<img src='{logo_url}' width='36' height='36' "
                     f"style='object-fit:contain;margin-bottom:4px;' "
                     f"onerror=\"this.style.display='none'\">") if logo_url else ""
        if img_html:
            _logo_tag = img_html
        return (
            f"<a href='{_href}' target='_self' style='text-decoration:none;color:inherit;'>"
            f"<div class='rk-answer {_sel_cls}'>"
            f"{_logo_tag}"
            f"<div class='rk-q'>{label}</div>"
            f"<div class='rk-team'>{team_name}</div>"
            f"<div class='rk-val'>{val_str}</div>"
            f"</div></a>"
        )

    def _player_box_html(box_id, title, pname, pteam, pval, psub=""):
        _sel_cls = "rk-box-sel" if box_id == _sel_box else ""
        _href = f"?page=rankings&rk_box={box_id}&rk_year={sel_year}"
        mid = _mlbam.get(_fix_player_name(pname), "") if _mlbam else ""
        _img = (f"<img src='{_headshot_url(mid, 120)}' width='60' height='60' loading='lazy' "
                f"style='border-radius:50%;object-fit:cover;margin-bottom:4px;' "
                f"onerror=\"this.style.display='none'\">") if mid else ""
        _sub_html = f"<div style='font-size:0.62rem;color:#4a687e;'>{psub}</div>" if psub else ""
        return (
            f"<a href='{_href}' target='_self' style='text-decoration:none;color:inherit;'>"
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
         f"${_best_eff['dollar_gap_M']:.0f}M below the line", team_logo_url(_best_eff["Team"])),
        ("top_overperformer", "TOP OVERPERFORMER", _full(_overperf),
         f"+{_overperf['wins_vs_pred']:.1f} wins vs forecast", team_logo_url(_overperf["Team"])),
        ("best_dpw", "BEST $/fWAR", _full(_best_dpw),
         f"${_best_dpw['DPW']:.1f}M per fWAR", team_logo_url(_best_dpw["Team"])),
        ("least_efficient", "LEAST EFFICIENT", _full(_worst_eff),
         f"${_worst_eff['dollar_gap_M']:.0f}M above the line", team_logo_url(_worst_eff["Team"])),
        ("top_fwar", "TOP fWAR", _full(_top_war),
         f"{_top_war['team_WAR']:.1f} total fWAR", team_logo_url(_top_war["Team"])),
        ("most_wins", "MOST WINS", _full(_top_wins),
         f"{int(_top_wins['Wins'])} wins", team_logo_url(_top_wins["Team"])),
    ]

    _grid_html = "<div class='rk-grid'>"
    for box_id, label, team_name, val_str, logo_url in _box_defs:
        _grid_html += _box_html(box_id, label, team_name, val_str, logo_url=logo_url)
    _grid_html += "</div>"
    st.markdown(_grid_html, unsafe_allow_html=True)

    # ── Row 3: Player award boxes ─────────────────────────────────────────────
    _p_grid_html = "<div class='rk-grid'>"
    _p_count = 0
    if _p_top_war is not None:
        _p_grid_html += _player_box_html(
            "p_top_fwar", f"#1 fWAR ({sel_year})", str(_p_top_war["Player"]),
            str(_p_top_war["Team"]), f"{_p_top_war['WAR_Total']:.1f} fWAR")
        _p_count += 1
    if _p_best_val is not None:
        _p_grid_html += _player_box_html(
            "p_contract_val", "TOP CONTRACT VALUE", str(_p_best_val["Player"]),
            str(_p_best_val["Team"]), f"{_p_best_val['_wpm']:.2f} fWAR/$M",
            f"{_p_best_val['WAR_Total']:.1f} fWAR \u00b7 ${_p_best_val['Salary_M']:.1f}M")
        _p_count += 1
    if _p_best_wsr is not None:
        _p_grid_html += _player_box_html(
            "p_stability", "BEST fWAR STABILITY", str(_p_best_wsr["Player"]),
            str(_p_best_wsr["Team"]), f"{_p_best_wsr['WSR']:.2f} WSR",
            f"Avg {_p_best_wsr['_mean']:.1f} fWAR \u00b7 {int(_p_best_wsr['_n'])} seasons")
        _p_count += 1
    _p_grid_html += "</div>"
    if _p_count > 0:
        st.markdown(_p_grid_html, unsafe_allow_html=True)

    # ── Row 4: New analysis boxes ─────────────────────────────────────────────
    _r4_html = "<div class='rk-grid'>"
    _r4_html += _box_html("best_marginal", "BEST MARGINAL SPENDING", _best_marginal_name, _best_marginal_val)
    _top_rss_logo = team_logo_url(_top_rss_team) if _top_rss_team != "N/A" else ""
    _r4_html += _box_html("top_rss", "TOP ROSTER STABILITY", _top_rss_team, _top_rss_val, logo_url=_top_rss_logo)
    _r4_html += _box_html("fwar_wins_link", "STRONGEST fWAR-WINS LINK", "All Teams", _fwar_r2_str)
    _r4_html += "</div>"
    st.markdown(_r4_html, unsafe_allow_html=True)

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
    _tab_default = _TAB_ORDER.index(_act_tab) if _act_tab in _TAB_ORDER else 0
    rt1, rt2, rt3, rt4, rt5, rt6 = st.tabs([
        "\U0001f3c6 Efficiency", "\u2b50 fWAR", "\U0001f4b0 Salary", "\U0001f4c8 Win Performance",
        "\U0001f48e Contract Value", "\U0001f512 Stability",
    ])

    # ── Tab 1: Efficiency ─────────────────────────────────────────────────────
    with rt1:
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
    with rt2:
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
    with rt3:
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
    with rt4:
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
    with rt5:
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "Player contract value measured as fWAR per $M of salary. Higher values mean "
            "more on-field production per dollar spent. Players earning &gt;$1M included.</div>",
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
                        _cv_vals = _cv_top["fWAR_per_M"].tolist()
                        _cv_names = _cv_top["Player"].tolist()
                        _cv_colors = ["#22c55e" if v >= 1.0 else "#4873b8" if v >= 0.5 else "#4a687e" for v in _cv_vals]
                        fig_cv = go.Figure(go.Bar(
                            y=_cv_names, x=_cv_vals, orientation="h",
                            marker=dict(color=_cv_colors, line=dict(width=0)),
                            text=[f"{v:.2f}" for v in _cv_vals],
                            textposition="outside",
                            textfont=dict(color="#d6e8f8", size=9),
                            hovertemplate="%{y}: %{x:.2f} fWAR/$M<extra></extra>",
                        ))
                        fig_cv.update_layout(**_pt(
                            title=f"{sel_year} \u2014 Top 15 Players by fWAR/$M",
                            xaxis=dict(title="fWAR per $M"),
                            yaxis=dict(autorange="reversed"),
                            height=max(340, len(_cv_top) * 28),
                            margin=dict(l=120, r=80, t=42, b=30),
                        ))
                        st.plotly_chart(fig_cv, use_container_width=True, config={"displayModeBar": False})
                with _cv_right:
                    # Avg fWAR/$M by contract stage
                    if "Stage_Clean" in _cv_df.columns:
                        _stage_avg = _cv_df.groupby("Stage_Clean")["fWAR_per_M"].mean().round(2)
                        _stage_map = {"Pre-Arb": "Pre-Arb", "Arb": "Arbitration", "FA": "Free Agent"}
                        for _stg_key in ["Pre-Arb", "Arb", "FA"]:
                            _stg_val = _stage_avg.get(_stg_key, 0)
                            _stg_label = _stage_map.get(_stg_key, _stg_key)
                            _stg_color = "#22c55e" if _stg_val >= 0.8 else "#f59e0b" if _stg_val >= 0.4 else "#ef4444"
                            st.markdown(
                                f"<div class='rk-answer' style='min-height:90px;margin-bottom:0.5rem;'>"
                                f"<div class='rk-q'>{_stg_label}</div>"
                                f"<div class='rk-team' style='color:{_stg_color};'>{_stg_val:.2f}</div>"
                                f"<div class='rk-val'>avg fWAR/$M</div></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("Stage data not available for breakdown.")
            else:
                st.info("Player data not available for contract value analysis.")
        except Exception as _cv_e:
            st.warning(f"Could not compute contract value data: {_cv_e}")

    # ── Tab 6: Stability ──────────────────────────────────────────────────────
    with rt6:
        st.markdown(
            "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.6rem;'>"
            "WAR Stability Rating (WSR) measures how consistently a player produces fWAR across "
            "seasons. Calculated as mean fWAR / (1 + std dev). Players with 3+ seasons included.</div>",
            unsafe_allow_html=True,
        )
        try:
            if not _wsr_grp.empty:
                _wsr_top = _wsr_grp.sort_values("WSR", ascending=False).head(15)

                _ws_left, _ws_right = st.columns([7, 3])
                with _ws_left:
                    _ws_vals = _wsr_top["WSR"].tolist()
                    _ws_names = _wsr_top["Player"].tolist()
                    _ws_colors = ["#22c55e" if v >= 3.0 else "#4873b8" if v >= 1.5 else "#4a687e" for v in _ws_vals]
                    fig_ws = go.Figure(go.Bar(
                        y=_ws_names, x=_ws_vals, orientation="h",
                        marker=dict(color=_ws_colors, line=dict(width=0)),
                        text=[f"{v:.2f}" for v in _ws_vals],
                        textposition="outside",
                        textfont=dict(color="#d6e8f8", size=9),
                        hovertemplate="%{y}: %{x:.2f} WSR<extra></extra>",
                    ))
                    fig_ws.update_layout(**_pt(
                        title="Top 15 Players by WAR Stability Rating (WSR)",
                        xaxis=dict(title="WSR (mean fWAR / (1 + std))"),
                        yaxis=dict(autorange="reversed"),
                        height=max(340, len(_wsr_top) * 28),
                        margin=dict(l=120, r=80, t=42, b=30),
                    ))
                    st.plotly_chart(fig_ws, use_container_width=True, config={"displayModeBar": False})
                with _ws_right:
                    # Avg WSR by contract stage
                    if _paw_df is not None and "Stage_Clean" in _paw_df.columns:
                        _paw_stg = _paw_df.dropna(subset=["WAR_Total"])
                        _stg_wsr = _paw_stg.groupby(["Player", "Stage_Clean"]).agg(
                            _mean=("WAR_Total", "mean"), _std=("WAR_Total", "std"),
                            _n=("Year", "nunique"),
                        ).reset_index()
                        _stg_wsr = _stg_wsr[_stg_wsr["_n"] >= 2].copy()
                        _stg_wsr["_std"] = _stg_wsr["_std"].fillna(0)
                        _stg_wsr["WSR"] = (_stg_wsr["_mean"] / (1 + _stg_wsr["_std"])).round(2)
                        _stg_avg_wsr = _stg_wsr.groupby("Stage_Clean")["WSR"].mean().round(2)
                        _stage_map = {"Pre-Arb": "Pre-Arb", "Arb": "Arbitration", "FA": "Free Agent"}
                        for _stg_key in ["Pre-Arb", "Arb", "FA"]:
                            _stg_val = _stg_avg_wsr.get(_stg_key, 0)
                            _stg_label = _stage_map.get(_stg_key, _stg_key)
                            _stg_color = "#22c55e" if _stg_val >= 1.5 else "#f59e0b" if _stg_val >= 0.8 else "#ef4444"
                            st.markdown(
                                f"<div class='rk-answer' style='min-height:90px;margin-bottom:0.5rem;'>"
                                f"<div class='rk-q'>{_stg_label}</div>"
                                f"<div class='rk-team' style='color:{_stg_color};'>{_stg_val:.2f}</div>"
                                f"<div class='rk-val'>avg WSR</div></div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("Stage data not available for breakdown.")

                # RSS section
                st.markdown("---")
                st.markdown("### Roster Stability and Win Correlation")
                st.markdown(
                    "<div style='font-size:0.82rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
                    "Roster Stability Score (RSS) measures what percentage of a team's qualifying players "
                    "returned from the prior season. Higher RSS means more continuity. "
                    "<span style='color:#22c55e;font-weight:600;'>Green</span> dots made the postseason.</div>",
                    unsafe_allow_html=True,
                )

                _comb_path_4 = data_url("data/mlb_combined_2021_2025.csv")
                try:
                    _comb_4 = read_csv(_comb_path_4, low_memory=False)
                    _comb_4.columns = [c.strip() for c in _comb_4.columns]
                    for _nc in ["Year", "PA", "IP", "WAR_Total"]:
                        if _nc in _comb_4.columns:
                            _comb_4[_nc] = pd.to_numeric(_comb_4[_nc], errors="coerce")

                    _rss_thresh = st.slider("Min PA (hitters) / IP (pitchers)", 50, 300, 150,
                                             key="v2_rss_min_pa")

                    _is_pit4 = _comb_4["Position"].isin(["SP", "RP", "P", "TWP"])
                    _comb_q = _comb_4[
                        (_is_pit4 & (_comb_4["IP"].fillna(0) >= _rss_thresh)) |
                        (~_is_pit4 & (_comb_4["PA"].fillna(0) >= _rss_thresh))
                    ].copy()

                    _years4 = sorted(_comb_q["Year"].dropna().unique().astype(int))
                    _rss_records = []
                    for yr in _years4:
                        if yr == min(_years4):
                            continue
                        for tm in _comb_q["Team"].unique():
                            curr = set(_comb_q[(_comb_q["Year"] == yr) & (_comb_q["Team"] == tm)]["Player"])
                            prev = set(_comb_q[(_comb_q["Year"] == yr - 1) & (_comb_q["Team"] == tm)]["Player"])
                            if not curr:
                                continue
                            returning = curr & prev
                            rss = len(returning) / len(curr) * 100
                            _wins_row = detail_df[(detail_df["Year"] == yr) & (detail_df["Team"] == tm)]
                            wins = float(_wins_row["Wins"].iloc[0]) if not _wins_row.empty and "Wins" in _wins_row.columns else None
                            playoff = bool(_wins_row["in_playoffs"].iloc[0]) if not _wins_row.empty and "in_playoffs" in _wins_row.columns else False
                            _rss_records.append({"Team": tm, "Year": yr, "RSS": round(rss, 1),
                                                 "Wins": wins, "Playoff": playoff,
                                                 "Returning": len(returning), "Total": len(curr)})

                    if _rss_records:
                        _rss_df = pd.DataFrame(_rss_records).dropna(subset=["Wins"])

                        if len(_rss_df) > 10:
                            _rx = _rss_df["RSS"].values
                            _ry = _rss_df["Wins"].values
                            _rc = np.polyfit(_rx, _ry, 1)
                            _rp = np.polyval(_rc, _rx)
                            _rss_r2 = 1 - np.sum((_ry - _rp) ** 2) / max(np.sum((_ry - _ry.mean()) ** 2), 1e-9)

                            _rss_colors = ["#22c55e" if p else "#4a687e" for p in _rss_df["Playoff"]]
                            _rss_hover = _rss_df.apply(lambda r: (
                                f"<b>{r['Team']}</b> {int(r['Year'])}<br>"
                                + f"RSS: {r['RSS']:.1f}% \u00b7 Wins: {int(r['Wins'])}<br>"
                                + f"Returning: {int(r['Returning'])}/{int(r['Total'])}"
                            ), axis=1)

                            fig_rss = go.Figure()
                            fig_rss.add_trace(go.Scatter(
                                x=_rss_df["RSS"], y=_rss_df["Wins"], mode="markers",
                                marker=dict(color=_rss_colors, size=8, opacity=0.8),
                                text=_rss_hover, hovertemplate="%{text}<extra></extra>",
                                name="Teams",
                            ))
                            _xr4 = np.linspace(_rx.min(), _rx.max(), 100)
                            fig_rss.add_trace(go.Scatter(
                                x=_xr4, y=np.polyval(_rc, _xr4), mode="lines",
                                line=dict(color="#f4a261", width=2),
                                name=f"OLS (R\u00b2={_rss_r2:.3f})",
                            ))
                            fig_rss.update_layout(**_pt(
                                title="Roster Stability vs Win Total",
                                xaxis=dict(title="Roster Stability Score (%)"),
                                yaxis=dict(title="Wins"), height=440, showlegend=True,
                                legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                                hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                                font=dict(color="#dbeafe", size=12)),
                            ))
                            _rss_col, _ = st.columns([3, 1])
                            with _rss_col:
                                st.plotly_chart(fig_rss, width="stretch")

                            _med_rss = float(_rss_df["RSS"].median())
                            _high = _rss_df[_rss_df["RSS"] >= _med_rss]
                            _low  = _rss_df[_rss_df["RSS"] < _med_rss]
                            if not _high.empty and not _low.empty:
                                st.info(
                                    f"Teams with RSS above {_med_rss:.0f}% won an average of "
                                    f"{_high['Wins'].mean():.1f} games vs {_low['Wins'].mean():.1f} "
                                    f"games for teams below that threshold."
                                )
                except Exception:
                    st.caption("Could not load combined data for RSS analysis.")
            else:
                st.info("Not enough multi-year player data for stability analysis.")
        except Exception as _ws_e:
            st.warning(f"Could not compute stability data: {_ws_e}")

    _render_feedback_widget("rankings")
