"""MLB Toolbox -- Team Analysis page (extracted from streamlit_app.py)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.components import (
    render_feedback_widget as _render_feedback_widget,
)
from utils.player_utils import (
    fix_player_name as _fix_player_name,
)
from utils.team_utils import cbt_info as _cbt_info, ordinal as _ordinal
from utils.theme import plotly_theme as _pt
from utils.constants import (
    ABBR_TO_FULL as _ABBR_TO_FULL,
    TEAM_CITIES as _TEAM_CITIES,
    TEAM_COLORS as _TEAM_COLORS,
)
from utils.data_loading import (
    data_url,
    read_csv,
    load_enriched_roster,
    team_logo_url,
    fetch_2026_standings as fetch_standings,
    fetch_2026_standings_full as fetch_standings_full,
    fetch_2026_team_stats as fetch_team_stats,
    R2_MODE as r2_mode,
)


def render(*_args, **_kwargs):
    """Team analysis page entry point.

    All data functions are now imported directly from utils.data_loading.
    Legacy parameters are accepted but ignored.
    """

    # ── Team picker CSS ──────────────────────────────────────────────────
    pass

    # ── Data loading (enriched roster = single source of truth) ─────────
    _enriched = load_enriched_roster()
    try:
        detail_df = read_csv(data_url("efficiency_detail.csv"))
    except Exception:
        detail_df = pd.DataFrame()

    # ── Team selector — square card grid: AL left, NL right ─────────────
    _AL_DIVS = [
        ("East",    ["BAL", "BOS", "NYY", "TBR", "TOR"]),
        ("Central", ["CHW", "CLE", "DET", "KCR", "MIN"]),
        ("West",    ["ATH", "HOU", "LAA", "SEA", "TEX"]),
    ]
    _NL_DIVS = [
        ("East",    ["ATL", "MIA", "NYM", "PHI", "WSN"]),
        ("Central", ["CHC", "CIN", "MIL", "PIT", "STL"]),
        ("West",    ["ARI", "COL", "LAD", "SDP", "SFG"]),
    ]

    # Handle incoming team selection from rankings page link
    _qp_sel = st.query_params.get("sel_team")
    if _qp_sel and _qp_sel in [t for d in _AL_DIVS + _NL_DIVS for t in d[1]]:
        st.session_state["team_analysis_sel"] = _qp_sel
    if "team_analysis_sel" not in st.session_state:
        st.session_state["team_analysis_sel"] = "NYY"
    sel_team = st.session_state.get("team_analysis_sel", "NYY")

    # Inject CSS to hide team picker button chrome
    st.markdown("""<style>
    /* Team picker — pure HTML, no buttons */
    </style>""", unsafe_allow_html=True)

    def _render_league_grid(league_name, divs):
        st.markdown(
            f"<div style='font-size:0.82rem;font-weight:700;color:#d6e8f8;text-align:center;"
            f"margin-bottom:0.3rem;letter-spacing:0.1em;'>{league_name}</div>",
            unsafe_allow_html=True,
        )
        for div_name, teams in divs:
            st.markdown(
                f"<div style='font-size:0.62rem;color:#d6e8f8;font-weight:600;"
                f"margin:0.2rem 0 0.1rem;'>{div_name}</div>",
                unsafe_allow_html=True,
            )
            # All 5 logos as pure HTML clickable cards — NO st.button
            _cards = ""
            for tm in teams:
                _url = team_logo_url(tm)
                is_active = tm == sel_team
                _bdr = "2px solid #3b82f6" if is_active else "1px solid transparent"
                _bg = "rgba(59,130,246,0.1)" if is_active else "transparent"
                _shadow = "box-shadow:0 0 12px #3b82f644;" if is_active else ""
                _name = _ABBR_TO_FULL.get(tm, tm)
                _cards += (
                    f"<a href='?page=team&sel_team={tm}' target='_self' style='text-decoration:none;'>"
                    f"<div class='tpick-logo-card' style='background:{_bg};border:{_bdr};border-radius:8px;"
                    f"padding:6px 4px 4px;text-align:center;{_shadow}cursor:pointer;'>"
                    f"<img src='{_url}' width='55' height='55' style='object-fit:contain;'>"
                    f"<div style='font-size:0.75rem;font-weight:700;color:#e8f4ff;margin-top:3px;'>{_name}</div>"
                    f"</div></a>"
                )
            st.markdown(
                f"<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:6px;'>"
                f"{_cards}</div>",
                unsafe_allow_html=True,
            )

    # Team picker grid (pure HTML, no st.button)
    al_col, nl_col = st.columns(2, gap="medium")
    with al_col:
        _render_league_grid("American League", _AL_DIVS)
    with nl_col:
        _render_league_grid("National League", _NL_DIVS)

    # Spacer between team picker and content
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # If no team selected yet, show prompt and stop
    if not sel_team:
        st.markdown(
            "<div style='text-align:center;padding:3rem 0;color:#7a9ebc;font-size:1rem;'>"
            "Select a team above to view their analysis.</div>",
            unsafe_allow_html=True,
        )
        _render_feedback_widget("team")
        return

    _full_name = f"{_TEAM_CITIES.get(sel_team, '')} {_ABBR_TO_FULL.get(sel_team, sel_team)}"

    # ── Filter enriched data for selected team ───────────────────────────
    team_data = _enriched[_enriched["team"] == sel_team].copy() if not _enriched.empty else pd.DataFrame()
    team_pay = team_data  # alias for compatibility

    # Roster counts by status
    n_active = int((team_data["status"] == "Active").sum()) if not team_data.empty else 0
    _il_statuses = ["Injured 10-Day", "Injured 15-Day", "Injured 60-Day"]
    n_il = int(team_data["status"].isin(_il_statuses).sum()) if not team_data.empty else 0
    n_restricted = int((team_data["status"] == "Restricted").sum()) if not team_data.empty else 0
    n_total = len(team_data)

    # Team payroll: sum only Signed salaries for accurate total
    _signed_mask = team_data["status_2026"].astype(str).str.upper() == "SIGNED" if not team_data.empty else pd.Series(dtype=bool)
    _team_payroll_m = float(team_data.loc[_signed_mask, "salary_2026_M"].sum()) if _signed_mask.any() else 0
    # Add known arb salaries
    _arb_mask = team_data["status_2026"].astype(str).str.contains("ARB|TBD", case=False, na=False)
    _team_payroll_m += float(team_data.loc[_arb_mask, "salary_2026_M"].sum()) if _arb_mask.any() else 0

    # ── Efficiency data for this team ────────────────────────────────────
    team_eff = detail_df[detail_df["Team"] == sel_team].copy() if not detail_df.empty else pd.DataFrame()
    all_eff_2025 = detail_df[detail_df["Year"] == 2025] if not detail_df.empty else pd.DataFrame()

    # ── Latest season stats ──────────────────────────────────────────────
    _latest = team_eff[team_eff["Year"] == 2025].iloc[0] if not team_eff.empty and 2025 in team_eff["Year"].values else None

    # ══════════════════════════════════════════════════════════════════════
    # HEADER CARD
    # ══════════════════════════════════════════════════════════════════════
    _payroll_m = _team_payroll_m if _team_payroll_m > 0 else (float(_latest["payroll_M"]) if _latest is not None else 0)
    _wins      = int(_latest["Wins"]) if _latest is not None else 0
    _war       = float(_latest["team_WAR"]) if _latest is not None else 0
    _gap       = float(_latest["dollar_gap_M"]) if _latest is not None else 0

    # Compute rankings among all teams (2025)
    _eff_rank = int((all_eff_2025["dollar_gap_M"].rank(ascending=True) == all_eff_2025.loc[all_eff_2025["Team"] == sel_team, "dollar_gap_M"].rank(ascending=True).values[0]).sum()) if not all_eff_2025.empty and sel_team in all_eff_2025["Team"].values else 0
    if not all_eff_2025.empty and sel_team in all_eff_2025["Team"].values:
        _eff_rank = int(all_eff_2025["dollar_gap_M"].rank().loc[all_eff_2025["Team"] == sel_team].values[0])
        _war_rank = int(all_eff_2025["team_WAR"].rank(ascending=False).loc[all_eff_2025["Team"] == sel_team].values[0])
        _pay_rank = int(all_eff_2025["payroll_M"].rank(ascending=False).loc[all_eff_2025["Team"] == sel_team].values[0])
        # League-specific efficiency rank
        _team_league = all_eff_2025.loc[all_eff_2025["Team"] == sel_team, "league"].values
        _team_lg = str(_team_league[0]) if len(_team_league) > 0 else "?"
        _lg_df = all_eff_2025[all_eff_2025["league"] == _team_lg] if _team_lg != "?" else pd.DataFrame()
        _lg_rank = int(_lg_df["dollar_gap_M"].rank().loc[_lg_df["Team"] == sel_team].values[0]) if not _lg_df.empty and sel_team in _lg_df["Team"].values else 0
        _lg_total = len(_lg_df) if not _lg_df.empty else 15
    else:
        _eff_rank = _war_rank = _pay_rank = _lg_rank = 0
        _team_lg = "?"
        _lg_total = 15

    _dpw = round(_payroll_m / max(_war, 0.1), 1)

    # Team colors
    _tc_primary, _tc_accent, _tc_dark = _TEAM_COLORS.get(sel_team, ("#3b82f6", "#93c5fd", "#081420"))

    # Team logo URL from R2
    _logo_url = team_logo_url(sel_team)

    # Live 2026 standings from MLB API (cached 24h)
    _standings = fetch_standings()
    _standings_full = fetch_standings_full()
    _w26, _l26 = _standings.get(sel_team, (0, 0))
    _record_26 = f"{_w26}\u2013{_l26}" if (_w26 + _l26) > 0 else "\u2014"

    # Division standing
    _stf = _standings_full.get(sel_team, {})
    _div_rank = _stf.get("div_rank", "?")
    _div_gb = _stf.get("gb", "-")
    _mlb_rank = _stf.get("league_rank", 0)
    # Division name from efficiency data
    _team_div = ""
    if not all_eff_2025.empty and "division" in all_eff_2025.columns and sel_team in all_eff_2025["Team"].values:
        _team_div = str(all_eff_2025.loc[all_eff_2025["Team"] == sel_team, "division"].values[0])

    # KPI box style — consistent background, white border
    _kpi = ("background:#0d1b2a;border:1px solid #ffffff33;border-radius:8px;"
            "padding:12px 16px;text-align:center;min-width:100px;")

    # Compute league avg fWAR and $/fWAR for context
    _lg_avg_war = float(all_eff_2025["team_WAR"].mean()) if not all_eff_2025.empty else 0
    _lg_avg_dpw = float((all_eff_2025["payroll_M"] / all_eff_2025["team_WAR"].clip(lower=0.1)).mean()) if not all_eff_2025.empty else 0

    # Estimate luxury tax (payroll + ~15% for benefits/bonuses)
    _lux_tax_est = round(_payroll_m * 1.15)

    # fWAR concentration — how much of team WAR is in top 5 players
    _conc_pct = 0.0
    _conc_rank = 0
    _conc_avg = 0.0
    try:
        _cc_csv = data_url("data/mlb_combined_2021_2025.csv")
        _cc_df = read_csv(_cc_csv, low_memory=False)
        _cc_df.columns = [c.strip() for c in _cc_df.columns]
        _cc_df["Year"] = pd.to_numeric(_cc_df["Year"], errors="coerce")
        _cc_df["WAR_Total"] = pd.to_numeric(_cc_df["WAR_Total"], errors="coerce")
        _cc_25 = _cc_df[_cc_df["Year"] == 2025].dropna(subset=["WAR_Total"])
        _conc_all = {}
        for _ctm, _cgrp in _cc_25.groupby("Team"):
            _ct = _cgrp["WAR_Total"].sum()
            _c5 = _cgrp.nlargest(5, "WAR_Total")["WAR_Total"].sum()
            _conc_all[_ctm] = round((_c5 / max(_ct, 0.1)) * 100, 1)
        _conc_pct = _conc_all.get(sel_team, 0)
        _conc_avg = round(sum(_conc_all.values()) / max(len(_conc_all), 1), 1)
        _conc_sorted = sorted(_conc_all.items(), key=lambda x: x[1])
        _conc_rank = next((i + 1 for i, (t, _) in enumerate(_conc_sorted) if t == sel_team), 0)
    except Exception:
        pass

    st.markdown(
        f"<div style='background:linear-gradient(135deg,{_tc_dark},{_tc_dark}cc);border:1px solid {_tc_primary}44;"
        f"border-left:4px solid {_tc_primary};border-radius:10px;padding:18px 22px;margin-bottom:14px;"
        f"animation:slideUp 0.4s ease-out;'>"
        f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:12px;'>"
        f"<img src='{_logo_url}' width='52' height='52' loading='lazy' style='object-fit:contain;' onerror=\"this.style.display='none'\">"
        f"<div style='font-size:1.5rem;font-weight:800;color:#e8f4ff;'>{_full_name}</div>"
        f"</div>"
        f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;'>"
        # Row 1: Records + Payroll + fWAR
        f"<div style='{_kpi}'>"
        f"<div style='font-size:11px;color:{_tc_accent};letter-spacing:0.05em;'>2025 RECORD</div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:#e8f4ff;'>{_wins}\u2013{162 - _wins}</div>"
        f"<div style='font-size:0.75rem;color:#93b8d8;'>"
        + (_team_div if _team_div else "") +
        f"</div></div>"
        f"<div style='{_kpi}'>"
        f"<div style='font-size:11px;color:{_tc_accent};letter-spacing:0.05em;'>2026 RECORD</div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:#e8f4ff;'>{_record_26}</div>"
        f"<div style='font-size:0.75rem;color:#93b8d8;'>"
        + (f"{_ordinal(int(_div_rank))} {_team_div}" if _div_rank != "?" else "")
        + f"</div></div>"
        f"<div style='{_kpi}'>"
        f"<div style='font-size:11px;color:{_tc_accent};letter-spacing:0.05em;'>2026 PAYROLL</div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:#e8f4ff;'>${_payroll_m:.0f}M</div>"
        f"<div style='font-size:0.75rem;color:#93b8d8;'>#{_pay_rank}/30</div></div>"
        f"<div style='{_kpi}'>"
        f"<div style='font-size:11px;color:{_tc_accent};letter-spacing:0.05em;'>LUX TAX EST.</div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:#e8f4ff;'>~${_lux_tax_est}M</div>"
        f"<div style='font-size:0.75rem;color:#93b8d8;'>CBT: $244M</div></div>"
        f"<div style='{_kpi}'>"
        f"<div style='font-size:11px;color:{_tc_accent};letter-spacing:0.05em;'>TEAM fWAR</div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:#e8f4ff;'>{_war:.1f}</div>"
        f"<div style='font-size:0.75rem;color:#93b8d8;'>#{_war_rank}/30 \u00b7 avg {_lg_avg_war:.0f}</div></div>"
        f"<div style='{_kpi}'>"
        f"<div style='font-size:11px;color:{_tc_accent};letter-spacing:0.05em;'>{'SURPLUS VALUE' if _gap < 0 else 'LOST VALUE'}</div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:{'#22c55e' if _gap < 0 else '#f59e0b'};'>"
        f"${int(_gap):+d}M</div>"
        f"<div style='font-size:0.75rem;color:#93b8d8;'>#{_eff_rank}/30</div></div>"
        f"<div style='{_kpi}'>"
        f"<div style='font-size:11px;color:{_tc_accent};letter-spacing:0.05em;'>SPEND EFFICIENCY</div>"
        f"<div style='font-size:1.1rem;font-weight:700;color:#e8f4ff;'>"
        f"#{_eff_rank} <span style='font-size:0.72rem;color:#7a9ebc;'>MLB</span></div>"
        f"<div style='font-size:0.75rem;color:#93b8d8;'>#{_lg_rank}/{_lg_total} {_team_lg}</div></div>"
        f"<div style='{_kpi}'>"
        f"<div style='font-size:11px;color:{_tc_accent};letter-spacing:0.05em;'>$/fWAR</div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:#e8f4ff;'>${_dpw:.1f}M</div>"
        f"<div style='font-size:0.75rem;color:#93b8d8;'>avg ${_lg_avg_dpw:.1f}M</div></div>"
        f"<div style='{_kpi}'>"
        f"<div style='font-size:11px;color:{_tc_accent};letter-spacing:0.05em;'>fWAR CONCENTRATION</div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:#e8f4ff;'>{_conc_pct:.0f}%</div>"
        f"<div style='font-size:0.75rem;color:#93b8d8;'>#{_conc_rank}/30 \u00b7 avg {_conc_avg:.0f}%</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════════════════════════
    tt1, tt2, tt3, tt4, tt5 = st.tabs(["📋 Roster", "📊 Rankings", "💰 Salary & Payroll", "📈 Payroll Efficiency", "📉 History"])

    # ── Tab 1 — Roster ───────────────────────────────────────────────────
    with tt1:
        if not team_data.empty:
            # ── Merge 2025 fWAR from combined stats ──────────────────────
            _td = team_data.copy()
            # Merge 2025 stats (fWAR + batting/pitching stats)
            try:
                _s_csv = data_url("data/mlb_combined_2021_2025.csv")
                _s_all = read_csv(_s_csv, low_memory=False)
                _s_all.columns = [c.strip() for c in _s_all.columns]
                _s_all["Year"] = pd.to_numeric(_s_all["Year"], errors="coerce")
                for _sc in ["WAR_Total", "HR", "AVG", "OBP", "SLG", "ERA", "WHIP", "IP", "K9"]:
                    if _sc in _s_all.columns:
                        _s_all[_sc] = pd.to_numeric(_s_all[_sc], errors="coerce")
                _s25_cols = ["Player", "WAR_Total", "HR", "AVG", "OBP", "SLG", "ERA", "WHIP", "IP", "K9"]
                _s25_cols = [c for c in _s25_cols if c in _s_all.columns]
                _s25 = _s_all[_s_all["Year"] == 2025][_s25_cols].drop_duplicates("Player", keep="first")
                _s25["_jk"] = _s25["Player"].str.lower().str.strip()
                _td["_jk"] = _td["full_name"].apply(_fix_player_name).str.lower().str.strip()
                _td = _td.merge(_s25.drop(columns=["Player"]), on="_jk", how="left")
                _td = _td.drop(columns=["_jk"])
            except Exception:
                _td["WAR_Total"] = None

            # Merge 2026 live stats from MLB API
            _team_id_val = int(team_data["team_id"].iloc[0]) if "team_id" in team_data.columns and not team_data["team_id"].isna().all() else None
            if _team_id_val:
                _live_stats = fetch_team_stats(_team_id_val)
                if not _live_stats.empty:
                    _live_stats["_pid"] = _live_stats["player_id"].astype(str)
                    _td["_pid"] = _td["player_id"].astype(str)
                    _live_cols = [c for c in _live_stats.columns if c.endswith("_2026") or c == "_pid"]
                    _td = _td.merge(_live_stats[_live_cols], on="_pid", how="left")
                    _td = _td.drop(columns=["_pid"], errors="ignore")

            # ── Safe column getter ────────────────────────────────────────
            def _col(df, name, default=None):
                """Get column values safely, returning NaN series if missing."""
                if name in df.columns:
                    return df[name].values
                return pd.array([default] * len(df))

            # ── Build position player table ──────────────────────────────
            def _build_hitter_tbl(src_df):
                src = src_df.reset_index(drop=True)
                tbl = pd.DataFrame({
                    "Player": _col(src, "full_name"),
                    "Pos": _col(src, "position_primary", _col(src, "position")),
                    "Bats": _col(src, "bats"),
                    "Age": _col(src, "age"),
                    "Stage": src["stage_display"].replace({"FA": "Free Agent"}).values if "stage_display" in src.columns else _col(src, "contract_stage"),
                    "'26 Salary": _col(src, "salary_2026_M"),
                    "AVG": _col(src, "AVG"),
                    "OBP": _col(src, "OBP"),
                    "HR": _col(src, "HR"),
                })
                if "avg_2026" in src.columns:
                    tbl["'26 AVG"] = _col(src, "avg_2026")
                    tbl["'26 HR"] = _col(src, "hr_2026")
                tbl["'25 fWAR"] = _col(src, "WAR_Total")
                tbl["Contract"] = _col(src, "pay_contract")
                tbl = tbl.sort_values("'26 Salary", ascending=False).reset_index(drop=True)
                tbl.insert(0, "#", range(1, len(tbl) + 1))
                return tbl

            def _build_pitcher_tbl(src_df):
                src = src_df.reset_index(drop=True)
                tbl = pd.DataFrame({
                    "Player": _col(src, "full_name"),
                    "Pos": _col(src, "position_primary", _col(src, "position")),
                    "Throws": _col(src, "throws"),
                    "Age": _col(src, "age"),
                    "Stage": src["stage_display"].replace({"FA": "Free Agent"}).values if "stage_display" in src.columns else _col(src, "contract_stage"),
                    "'26 Salary": _col(src, "salary_2026_M"),
                    "ERA": _col(src, "ERA"),
                    "WHIP": _col(src, "WHIP"),
                    "IP": _col(src, "IP"),
                })
                if "era_2026" in src.columns:
                    tbl["'26 ERA"] = _col(src, "era_2026")
                    tbl["'26 W-L"] = (src["w_2026"].fillna("").astype(str) + "-" + src["l_2026"].fillna("").astype(str)).values if "w_2026" in src.columns else None
                    tbl["'26 IP"] = _col(src, "ip_2026")
                tbl["'25 fWAR"] = _col(src, "WAR_Total")
                tbl["Contract"] = _col(src, "pay_contract")
                tbl = tbl.sort_values("'26 Salary", ascending=False).reset_index(drop=True)
                tbl.insert(0, "#", range(1, len(tbl) + 1))
                return tbl

            _STG_CLR = {"Pre-Arb": "#1a6b3a", "Arb": "#0c2a2a", "Guaranteed": "#0c1a2d", "Free Agent": "#0c1a2d"}
            def _stage_clr(row):
                stg = str(row.get("Stage", ""))
                bg = _STG_CLR.get(stg, "")
                return [f"background-color:{bg}55"] * len(row) if bg else [""] * len(row)

            _fmt = {"Age": "{:.0f}", "'26 Salary $M": "${:.2f}M", "'25 fWAR": "{:.1f}"}

            # Split by status
            _td["_status_cat"] = _td["status"].apply(
                lambda s: "IL" if "Injured" in str(s) else "Restricted" if "Restrict" in str(s) else "Active"
            )
            _active_df = _td[_td["_status_cat"] == "Active"]
            _il_df = _td[_td["_status_cat"] == "IL"]
            _restricted_df = _td[_td["_status_cat"] == "Restricted"]

            # ── Active Roster — split into Position Players and Pitchers ──
            _PITCHER_POS = {"SP", "RP", "P", "CL"}
            if not _active_df.empty:
                _act_pos = _active_df[~_active_df.get("position_primary", _active_df.get("position", pd.Series())).isin(_PITCHER_POS)]
                _act_pit = _active_df[_active_df.get("position_primary", _active_df.get("position", pd.Series())).isin(_PITCHER_POS)]

                st.markdown(
                    "<div style='font-size:0.82rem;color:#7a9ebc;margin-bottom:0.3rem;'>"
                    "Color: <span style='color:#4ade80;'>Pre-Arb</span> \u00b7 "
                    "<span style='color:#14b8a6;'>Arb</span> \u00b7 "
                    "<span style='color:#60a5fa;'>Free Agent</span></div>",
                    unsafe_allow_html=True,
                )

                _fmt_h = {"Age": "{:.0f}", "'26 Salary": "${:.2f}M", "'25 fWAR": "{:.1f}",
                          "AVG": "{:.3f}", "OBP": "{:.3f}", "HR": "{:.0f}",
                          "'26 HR": "{:.0f}"}
                _fmt_p = {"Age": "{:.0f}", "'26 Salary": "${:.2f}M", "'25 fWAR": "{:.1f}",
                          "ERA": "{:.2f}", "WHIP": "{:.2f}", "IP": "{:.1f}"}

                # Position Players
                st.markdown(f"##### Position Players ({len(_act_pos)})")
                if not _act_pos.empty:
                    _h_tbl = _build_hitter_tbl(_act_pos)
                    st.dataframe(
                        _h_tbl.style.apply(_stage_clr, axis=1).format(
                            {k: v for k, v in _fmt_h.items() if k in _h_tbl.columns}, na_rep="\u2014"),
                        hide_index=True, use_container_width=True,
                        height=min(60 + len(_h_tbl) * 35, 500),
                    )

                # Pitchers
                st.markdown(f"##### Pitchers ({len(_act_pit)})")
                if not _act_pit.empty:
                    _p_tbl = _build_pitcher_tbl(_act_pit)
                    st.dataframe(
                        _p_tbl.style.apply(_stage_clr, axis=1).format(
                            {k: v for k, v in _fmt_p.items() if k in _p_tbl.columns}, na_rep="\u2014"),
                        hide_index=True, use_container_width=True,
                        height=min(60 + len(_p_tbl) * 35, 500),
                    )

            # ── Simple table builder for IL/Restricted ────────────────────
            def _build_simple_tbl(src_df):
                tbl = pd.DataFrame()
                tbl["Player"] = src_df["full_name"].values
                tbl["Pos"] = src_df.get("position_primary", src_df.get("position", pd.Series())).values
                tbl["Age"] = src_df["age"].values
                _stg = src_df.get("stage_display", src_df.get("contract_stage", pd.Series())).copy()
                _stg = _stg.replace({"FA": "Free Agent"})
                tbl["Stage"] = _stg.values
                tbl["'26 Salary"] = src_df["salary_2026_M"].values
                tbl["'25 fWAR"] = src_df.get("WAR_Total", pd.Series(dtype=float)).values
                tbl = tbl.sort_values("'26 Salary", ascending=False).reset_index(drop=True)
                tbl.insert(0, "#", range(1, len(tbl) + 1))
                return tbl
            _fmt_s = {"Age": "{:.0f}", "'26 Salary": "${:.2f}M", "'25 fWAR": "{:.1f}"}

            # ── Injured List ─────────────────────────────────────────────
            if not _il_df.empty:
                _il_tbl = _build_simple_tbl(_il_df)
                _il_tbl.insert(2, "IL Type", _il_df["status"].apply(
                    lambda s: "60-Day" if "60" in str(s) else "15-Day" if "15" in str(s) else "10-Day"
                ).values)
                st.markdown(f"##### 🏥 Injured List ({len(_il_df)} players)")
                st.dataframe(
                    _il_tbl.style.apply(
                        lambda row: ["background-color:#2d0c0c66;color:#fca5a5"] * len(row), axis=1
                    ).format({k: v for k, v in _fmt_s.items() if k in _il_tbl.columns}, na_rep="\u2014"),
                    hide_index=True, use_container_width=True,
                    height=min(60 + len(_il_tbl) * 35, 400),
                )

            # ── Restricted (Minor League 40-Man) ─────────────────────────
            if not _restricted_df.empty:
                _res_tbl = _build_simple_tbl(_restricted_df)
                st.markdown(f"##### Minor League / Restricted ({len(_restricted_df)} players)")
                st.markdown(
                    "<div style='font-size:0.75rem;color:#4a687e;margin-bottom:0.3rem;'>"
                    "Players on the 40-man roster assigned to minor leagues. "
                    "Still occupy a 40-man spot but are not on the active MLB roster.</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    _res_tbl.style.apply(
                        lambda row: ["background-color:#1a1a2e44"] * len(row), axis=1
                    ).format({k: v for k, v in _fmt_s.items() if k in _res_tbl.columns}, na_rep="\u2014"),
                    hide_index=True, use_container_width=True,
                    height=min(60 + len(_res_tbl) * 35, 500),
                )
        else:
            st.info(f"No roster data available for {sel_team}.")

    # ── Tab 2 — Rankings Position ────────────────────────────────────────
    with tt2:
        if not all_eff_2025.empty:
            st.markdown(
                "<div style='font-size:0.85rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
                "Where this team ranks among all 30 MLB teams in 2025. "
                "<span style='color:#f59e0b;font-weight:600;'>Gold</span> highlights the selected team.</div>",
                unsafe_allow_html=True,
            )

            _rk = all_eff_2025.sort_values("dollar_gap_M").reset_index(drop=True)
            _rk["Rank"] = range(1, len(_rk) + 1)
            _colors = ["#f59e0b" if t == sel_team else ("#22c55e" if g < 0 else "#ef4444")
                        for t, g in zip(_rk["Team"], _rk["dollar_gap_M"])]

            fig_rk = go.Figure(go.Bar(
                y=_rk["Team"], x=_rk["dollar_gap_M"], orientation="h",
                marker_color=_colors,
                text=[f"${g:+.0f}M" for g in _rk["dollar_gap_M"]],
                textposition="outside", textfont=dict(color="#d6e8f8", size=9),
                hovertemplate="%{y}: $%{x:+.0f}M<extra></extra>",
            ))
            _abs_max = max(abs(_rk["dollar_gap_M"].max()), abs(_rk["dollar_gap_M"].min())) * 1.15
            fig_rk.update_layout(**_pt(
                title=f"2025 Spend Efficiency Ranking \u2014 {_full_name} is #{_eff_rank}",
                xaxis=dict(title="Surplus Value ($M) \u2014 negative = surplus, positive = lost value",
                           zeroline=True, zerolinecolor="#4a687e", zerolinewidth=1,
                           range=[-_abs_max, _abs_max]),
                yaxis=dict(autorange="reversed"),
                height=max(400, len(_rk) * 22),
                margin=dict(l=60, r=80, t=42, b=30),
            ))
            st.plotly_chart(fig_rk, width="stretch", config={"displayModeBar": False})

            # fWAR ranking bar
            _wrk = all_eff_2025.sort_values("team_WAR", ascending=False).reset_index(drop=True)
            _wrk["Rank"] = range(1, len(_wrk) + 1)
            _wcolors = ["#f59e0b" if t == sel_team else "#3b82f6" for t in _wrk["Team"]]

            fig_wrk = go.Figure(go.Bar(
                y=_wrk["Team"], x=_wrk["team_WAR"], orientation="h",
                marker_color=_wcolors,
                text=[f"{w:.1f}" for w in _wrk["team_WAR"]],
                textposition="outside", textfont=dict(color="#d6e8f8", size=9),
                hovertemplate="%{y}: %{x:.1f} fWAR<extra></extra>",
            ))
            fig_wrk.update_layout(**_pt(
                title=f"2025 fWAR Ranking \u2014 {_full_name} is #{_war_rank}",
                xaxis=dict(title="Total Team fWAR"),
                yaxis=dict(autorange="reversed"),
                height=max(400, len(_wrk) * 22),
                margin=dict(l=60, r=80, t=42, b=30),
            ))
            st.plotly_chart(fig_wrk, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No efficiency ranking data available.")

    # ── Tab 3 — Salary & Payroll ─────────────────────────────────────────
    with tt3:
        if not team_data.empty:
            # Year toggle: compare 2025 vs 2026 salary levels
            _sal_year = st.radio(
                "Salary Year", ["2026", "2025"], horizontal=True,
                key="ta_sal_year", index=0,
            )
            _sal_col = f"salary_{_sal_year}_M"
            if _sal_col not in team_data.columns or team_data[_sal_col].isna().all():
                _sal_col = "salary_2026_M"  # fallback
                st.caption(f"No {_sal_year} salary data available \u2014 showing 2026.")

            # Salary by stage from enriched data
            _stg_col = "stage_display" if "stage_display" in team_data.columns else "contract_stage"
            _stg_sal = team_data.groupby(_stg_col)[_sal_col].sum().reset_index()
            _stg_colors = {"Pre-Arb": "#4ade80", "Arb": "#14b8a6", "Free Agent": "#60a5fa", "Off 40-Man": "#94a3b8"}

            fig_stg = go.Figure(go.Pie(
                labels=_stg_sal[_stg_col],
                values=_stg_sal[_sal_col],
                marker_colors=[_stg_colors.get(s, "#4a687e") for s in _stg_sal[_stg_col]],
                hole=0.45,
                textinfo="label+percent",
                textfont=dict(color="#d6e8f8", size=11),
                hovertemplate="%{label}: $%{value:.1f}M<extra></extra>",
            ))
            fig_stg.update_layout(**_pt(
                title=f"{_sal_year} Payroll by Contract Stage",
                height=360, showlegend=False,
            ))
            st.plotly_chart(fig_stg, width="stretch", config={"displayModeBar": False})

            # Top 10 highest paid
            _top_sal = team_data.nlargest(10, _sal_col)[["full_name", "position_primary", _sal_col, "contract_stage"]].copy()
            _top_sal.insert(0, "#", range(1, len(_top_sal) + 1))
            _top_sal.columns = ["#", "Player", "Pos", "Salary $M", "Stage"]
            st.markdown(f"##### Top 10 Highest-Paid Players ({_sal_year})")
            st.dataframe(
                _top_sal.style.format({"Salary $M": "${:.2f}M"}, na_rep="\u2014"),
                hide_index=True, use_container_width=True,
            )

            # Future payroll commitments (2026–2032)
            st.markdown("##### 📅 Committed Payroll (2026\u20132032)")
            _fut_years = []
            for yr in range(2026, 2033):
                col = f"salary_{yr}_M"
                if col in team_data.columns:
                    _fut_years.append((str(yr), float(team_data[col].sum())))
            _s26 = _fut_years[0][1] if _fut_years else 0
            _s27 = _fut_years[1][1] if len(_fut_years) > 1 else 0
            _s28 = _fut_years[2][1] if len(_fut_years) > 2 else 0

            fig_proj = go.Figure()
            fig_proj.add_trace(go.Bar(
                x=[y[0] for y in _fut_years],
                y=[y[1] for y in _fut_years],
                marker_color=["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe", "#e0e7ff", "#ede9fe"],
                text=[f"${v:.0f}M" for _, v in _fut_years],
                textposition="outside", textfont=dict(color="#d6e8f8"),
                hovertemplate="%{x}: $%{y:.1f}M<extra></extra>",
            ))
            fig_proj.add_hline(y=244, line_dash="dash", line_color="#f59e0b", opacity=0.5,
                               annotation_text="CBT $244M", annotation_font_color="#f59e0b")
            fig_proj.update_layout(**_pt(
                title=f"{_full_name} \u2014 Committed Payroll (2026\u20132032)",
                yaxis=dict(title="Total $M"), height=380,
            ))
            st.plotly_chart(fig_proj, width="stretch", config={"displayModeBar": False})

            st.caption(
                "2026 salaries reflect actual contracts and league minimum estimates. "
                "Future years include guaranteed commitments plus projected arb raises. "
                "Players hitting free agency show no salary for those years."
            )

            # CBT status
            _cbt_lbl, _cbt_bg, _, _, _cbt_note = _cbt_info(_payroll_m)
            st.markdown(
                f"<div style='background:{_cbt_bg};border-radius:8px;padding:10px 14px;"
                f"font-size:0.85rem;color:#d6e8f8;margin-top:0.5rem;'>"
                f"<b>CBT Status:</b> {_cbt_lbl} at ${_s26:.0f}M \u2014 {_cbt_note}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info(f"No 2026 payroll data available for {sel_team}.")

    # ── Tab 4 — Payroll Efficiency (player-level scatter for this team) ──
    with tt4:
        st.markdown(
            "<div style='font-size:0.85rem;color:#93b8d8;margin-bottom:0.8rem;line-height:1.6;'>"
            "Player-level fWAR vs Salary for this team's roster. Dots below the orange market "
            "line are underpaid (good value). Dots above are overpaid relative to production.</div>",
            unsafe_allow_html=True,
        )
        # Merge with 2025 stats for fWAR scatter
        _tp_plot = pd.DataFrame()
        if not team_data.empty and "salary_2026_M" in team_data.columns:
            try:
                _stats_csv = data_url("data/mlb_combined_2021_2025.csv")
                _stats_all = read_csv(_stats_csv, low_memory=False)
                _stats_all.columns = [c.strip() for c in _stats_all.columns]
                _stats_all["Year"] = pd.to_numeric(_stats_all["Year"], errors="coerce")
                _stats_all["WAR_Total"] = pd.to_numeric(_stats_all.get("WAR_Total", pd.Series()), errors="coerce")
                _stats_25 = _stats_all[_stats_all["Year"] == 2025][["Player", "WAR_Total"]].drop_duplicates("Player", keep="first")
                _stats_25["_key"] = _stats_25["Player"].str.lower().str.strip()
                _td = team_data.copy()
                _td["_key"] = _td["full_name"].apply(_fix_player_name).str.lower().str.strip()
                _td = _td.merge(_stats_25[["_key", "WAR_Total"]], on="_key", how="left")
                _tp_plot = _td.dropna(subset=["WAR_Total", "salary_2026_M"]).copy()
                _tp_plot = _tp_plot.rename(columns={"salary_2026_M": "Salary_M", "full_name": "Player",
                                                     "stage_display": "Stage_Clean", "position_primary": "Position"})
            except Exception:
                pass
        if not _tp_plot.empty:
            if len(_tp_plot) >= 3:
                _stg_clrs = {"Free Agent": "#3b82f6", "Arb": "#14b8a6", "Pre-Arb": "#4ade80"}
                _tp_colors = [_stg_clrs.get(s, "#4a687e") for s in _tp_plot.get("Stage_Clean", [])]
                _dash = "\u2014"
                _tp_hover = _tp_plot.apply(lambda r: (
                    f"<b>{r['Player']}</b><br>"
                    + f"fWAR: {r['WAR_Total']:.1f} \u00b7 Salary: ${r['Salary_M']:.1f}M<br>"
                    + f"Stage: {r.get('Stage_Clean', _dash)}"
                ), axis=1)

                fig_eff = go.Figure()
                for stg, clr in _stg_clrs.items():
                    mask = _tp_plot["Stage_Clean"] == stg
                    if mask.any():
                        fig_eff.add_trace(go.Scatter(
                            x=_tp_plot.loc[mask, "WAR_Total"],
                            y=_tp_plot.loc[mask, "Salary_M"],
                            mode="markers+text",
                            text=_tp_plot.loc[mask, "Player"],
                            textposition="top center",
                            textfont=dict(size=8, color="#7aa2c0"),
                            marker=dict(color=clr, size=9, opacity=0.9),
                            name=stg,
                            hovertemplate="%{text}<br>fWAR: %{x:.1f} \u00b7 $%{y:.1f}M<extra></extra>",
                        ))

                # Market line (simple linear fit)
                _xv = _tp_plot["WAR_Total"].values
                _yv = _tp_plot["Salary_M"].values
                try:
                    _cf = np.polyfit(_xv, _yv, 1)
                    _xl = np.linspace(max(_xv.min(), -1), _xv.max(), 50)
                    fig_eff.add_trace(go.Scatter(
                        x=_xl, y=np.polyval(_cf, _xl), mode="lines",
                        line=dict(color="#f4a261", dash="dash", width=2),
                        name="Team market line", showlegend=True,
                    ))
                except Exception:
                    pass

                fig_eff.update_layout(**_pt(
                    title=f"{_full_name} \u2014 Player fWAR vs Salary",
                    xaxis=dict(title="fWAR (2025)"), yaxis=dict(title="2026 Salary ($M)"),
                    height=500, showlegend=True,
                    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                    hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                    font=dict(color="#dbeafe", size=12)),
                ))
                st.plotly_chart(fig_eff, width="stretch", config={"displayModeBar": False})

                # Top value / worst value mini-tables
                _vc1, _vc2 = st.columns(2)
                with _vc1:
                    st.markdown("##### Best Value Players")
                    _bv = _tp_plot.copy()
                    _bv["fWAR/$M"] = (_bv["WAR_Total"] / _bv["Salary_M"].clip(lower=0.01)).round(2)
                    _bv_top = _bv.nlargest(5, "fWAR/$M")[["Player", "WAR_Total", "Salary_M", "fWAR/$M", "Stage_Clean"]].copy()
                    _bv_top.columns = ["Player", "fWAR", "Salary $M", "fWAR/$M", "Stage"]
                    st.dataframe(_bv_top.style.format({"fWAR": "{:.1f}", "Salary $M": "{:.1f}", "fWAR/$M": "{:.2f}"}, na_rep="\u2014"),
                                 hide_index=True, use_container_width=True)
                with _vc2:
                    st.markdown("##### Most Overpaid Players")
                    _ov_top = _bv.nsmallest(5, "fWAR/$M")[["Player", "WAR_Total", "Salary_M", "fWAR/$M", "Stage_Clean"]].copy()
                    _ov_top.columns = ["Player", "fWAR", "Salary $M", "fWAR/$M", "Stage"]
                    st.dataframe(_ov_top.style.format({"fWAR": "{:.1f}", "Salary $M": "{:.1f}", "fWAR/$M": "{:.2f}"}, na_rep="\u2014"),
                                 hide_index=True, use_container_width=True)
            else:
                st.info("Not enough player data for efficiency scatter.")
        else:
            st.info(f"No payroll data available for {sel_team}.")

    # ── Tab 5 — Historical Trends ────────────────────────────────────────
    with tt5:
        if not team_eff.empty and len(team_eff) >= 2:
            st.markdown(
                "<div style='font-size:0.85rem;color:#93b8d8;margin-bottom:0.8rem;'>"
                "Season-by-season trends for this team across key metrics (2021\u20132025).</div>",
                unsafe_allow_html=True,
            )

            # Wins trend
            _te = team_eff.sort_values("Year")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=_te["Year"], y=_te["Wins"], mode="lines+markers+text",
                marker=dict(color="#3b82f6", size=10),
                line=dict(color="#3b82f6", width=2),
                text=[f"{int(w)}" for w in _te["Wins"]],
                textposition="top center", textfont=dict(color="#d6e8f8", size=10),
                name="Wins",
                hovertemplate="%{x}: %{y} wins<extra></extra>",
            ))
            fig_hist.add_trace(go.Scatter(
                x=_te["Year"], y=_te["pred_wins"], mode="lines",
                line=dict(color="#f59e0b", dash="dash", width=1),
                name="Predicted (by payroll)",
                hovertemplate="%{x}: %{y:.0f} predicted<extra></extra>",
            ))
            fig_hist.update_layout(**_pt(
                title=f"{_full_name} \u2014 Wins vs Payroll Prediction",
                xaxis=dict(title="Season", dtick=1),
                yaxis=dict(title="Wins"),
                height=380, showlegend=True,
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
            ))
            st.plotly_chart(fig_hist, width="stretch", config={"displayModeBar": False})

            # Surplus / Lost Value trend (inverted: negative gap = surplus shown as negative/green)
            _gap_vals = _te["dollar_gap_M"].values
            fig_gap = go.Figure(go.Bar(
                x=_te["Year"].astype(int).astype(str),
                y=_gap_vals,
                marker_color=["#22c55e" if g < 0 else "#f59e0b" for g in _gap_vals],
                text=[f"${g:+.0f}M" for g in _gap_vals],
                textposition="outside", textfont=dict(color="#d6e8f8", size=10),
                hovertemplate="%{x}: $%{y:+.0f}M<extra></extra>",
            ))
            _abs_max_g = max(abs(_gap_vals).max(), 10) * 1.3
            fig_gap.update_layout(**_pt(
                title=f"{_full_name} \u2014 Surplus / Lost Value by Season",
                xaxis=dict(title="Season"),
                yaxis=dict(title="Surplus (\u2212) / Lost Value (+) $M", zeroline=True,
                           zerolinecolor="#4a687e", range=[-_abs_max_g, _abs_max_g]),
                height=340,
            ))
            st.markdown(
                "<div style='font-size:0.75rem;color:#7a9ebc;margin-bottom:0.3rem;'>"
                "Negative (green) = surplus value (winning more than payroll predicts). "
                "Positive (orange) = lost value (underperforming relative to spend).</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_gap, width="stretch", config={"displayModeBar": False})

            # Payroll + fWAR trend
            fig_pw = go.Figure()
            fig_pw.add_trace(go.Bar(
                x=_te["Year"].astype(int).astype(str), y=_te["payroll_M"],
                name="Payroll $M", marker_color="#3b82f6", opacity=0.6,
                hovertemplate="%{x}: $%{y:.0f}M<extra></extra>",
            ))
            fig_pw.add_trace(go.Scatter(
                x=_te["Year"].astype(int).astype(str), y=_te["team_WAR"],
                name="Team fWAR", yaxis="y2",
                mode="lines+markers", marker=dict(color="#22c55e", size=8),
                line=dict(color="#22c55e", width=2),
                hovertemplate="%{x}: %{y:.1f} fWAR<extra></extra>",
            ))
            fig_pw.update_layout(**_pt(
                title=f"{_full_name} \u2014 Payroll vs fWAR",
                yaxis=dict(title="Payroll $M"),
                yaxis2=dict(title="fWAR", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#22c55e"),
                            title_font=dict(color="#22c55e")),
                height=380, showlegend=True,
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
            ))
            st.plotly_chart(fig_pw, width="stretch", config={"displayModeBar": False})

            # Summary table
            st.markdown("##### Season-by-Season Summary")
            _sum = _te[["Year", "Wins", "payroll_M", "team_WAR", "dollar_gap_M", "in_playoffs"]].copy()
            _sum.columns = ["Year", "Wins", "Payroll $M", "fWAR", "Value $M", "Postseason"]
            _sum["Year"] = _sum["Year"].astype(int)
            _sum["Postseason"] = _sum["Postseason"].map({True: "\u2713", False: ""})
            st.dataframe(
                _sum.style.format({"Payroll $M": "{:.0f}", "fWAR": "{:.1f}", "Value $M": "{:+.0f}"}, na_rep="\u2014"),
                hide_index=True, use_container_width=True,
            )
            # fWAR Concentration trend (top 5 players' share of total)
            try:
                _cc_hist = _cc_df.dropna(subset=["WAR_Total"]) if "_cc_df" in dir() else pd.DataFrame()
                if _cc_hist.empty:
                    _cc_csv2 = data_url("data/mlb_combined_2021_2025.csv")
                    _cc_hist = read_csv(_cc_csv2, low_memory=False)
                    _cc_hist.columns = [c.strip() for c in _cc_hist.columns]
                    _cc_hist["Year"] = pd.to_numeric(_cc_hist["Year"], errors="coerce")
                    _cc_hist["WAR_Total"] = pd.to_numeric(_cc_hist["WAR_Total"], errors="coerce")
                    _cc_hist = _cc_hist.dropna(subset=["WAR_Total"])

                _conc_years = []
                for yr in sorted(_cc_hist["Year"].dropna().unique()):
                    yr_data = _cc_hist[(_cc_hist["Year"] == yr) & (_cc_hist["Team"] == sel_team)]
                    if yr_data.empty:
                        continue
                    total = yr_data["WAR_Total"].sum()
                    top5 = yr_data.nlargest(5, "WAR_Total")["WAR_Total"].sum()
                    pct = (top5 / max(total, 0.1)) * 100
                    _conc_years.append({"Year": int(yr), "Concentration": round(pct, 1),
                                        "Top 5 fWAR": round(top5, 1), "Total fWAR": round(total, 1)})

                if _conc_years:
                    _conc_ydf = pd.DataFrame(_conc_years)
                    st.markdown("##### fWAR Concentration \u2014 Top 5 Players' Share")
                    st.markdown(
                        "<div style='font-size:0.82rem;color:#7a9ebc;margin-bottom:0.4rem;'>"
                        "What percentage of the team's total fWAR comes from just the top 5 players? "
                        "High concentration means the team is heavily dependent on a few stars.</div>",
                        unsafe_allow_html=True,
                    )
                    fig_conc = go.Figure()
                    fig_conc.add_trace(go.Bar(
                        x=_conc_ydf["Year"].astype(str),
                        y=_conc_ydf["Concentration"],
                        marker_color=["#f59e0b" if c > 65 else "#3b82f6" for c in _conc_ydf["Concentration"]],
                        text=[f"{c:.0f}%" for c in _conc_ydf["Concentration"]],
                        textposition="outside", textfont=dict(color="#d6e8f8", size=10),
                        hovertemplate="%{x}: %{y:.1f}% concentration<extra></extra>",
                    ))
                    fig_conc.add_hline(y=_conc_avg, line_dash="dash", line_color="#4a687e", opacity=0.6,
                                       annotation_text=f"League avg {_conc_avg:.0f}%",
                                       annotation_font_color="#7a9ebc", annotation_position="top right")
                    fig_conc.update_layout(**_pt(
                        title=f"{_full_name} \u2014 fWAR Concentration (Top 5 Share)",
                        yaxis=dict(title="% of Total fWAR", range=[0, 100]),
                        height=340,
                    ))
                    st.plotly_chart(fig_conc, width="stretch", config={"displayModeBar": False})
            except Exception:
                pass

        else:
            st.info(f"Not enough historical data for {sel_team}.")

    _render_feedback_widget("team")
