"""MLB Toolbox -- Player Analysis page (extracted from streamlit_app.py).

Contains the efficiency frontier (WAR vs Salary) scatter, PPEL multi-year
views, age trajectory, efficient-player tables, residual analysis,
pre-arb explorer, and WAR stability rating.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.components import (
    render_feedback_widget as _render_feedback_widget,
    render_glossary as _render_glossary,
)
from utils.constants import C
from utils.theme import plotly_theme as _pt
from utils.data_loading import (
    R2_BASE_URL,
    data_url,
    read_csv,
    R2_MODE as r2_mode,
    cached_mlbam_lookup,
    RAZZBALL_PATH as razzball_path,
    file_hash as file_hash_fn,
    load_base_config,
    resolve_data_path,
    DEFAULT_CONFIG as default_config,
    HEADSHOTS_DIR as _HEADSHOTS_DIR,
)

try:
    import requests as _requests
    _requests_available = True
except ImportError:
    _requests_available = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render(*_args, **_kwargs):
    """Player analysis page entry point.

    All data functions are now imported directly from utils.data_loading.
    Legacy parameters are accepted but ignored.
    """
    _render_league_analysis()


# ---------------------------------------------------------------------------
# Internal: league analysis wrapper
# ---------------------------------------------------------------------------

def _render_league_analysis():
    """Render the League Efficiency Analysis page."""
    import subprocess

    # -- Interactive player-level Cost Effective Line --------------------------
    st.markdown("### Player Analysis")
    st.caption(
        "Interactive WAR vs Salary analysis for every player (2021\u20132025). "
        "Fit the PPEL regression line, colour by career stage or Pay-Performance Ratio, "
        "and identify the most underpaid / overpaid players in the league."
    )
    _render_efficiency_frontier()

    _render_feedback_widget("league")


# ---------------------------------------------------------------------------
# Internal: efficiency frontier
# ---------------------------------------------------------------------------

def _render_efficiency_frontier():
    """WAR vs Salary efficiency frontier -- interactive Streamlit port of app-2-2.R."""
    from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess

    # -- data path -------------------------------------------------------------
    if r2_mode:
        _cpath = data_url("data/mlb_combined_2021_2025.csv")
    else:
        if not os.path.exists(default_config):
            st.warning("Config file not found \u2014 cannot load player data for frontier analysis.")
            return
        _bcfg     = load_base_config(default_config)
        _sal_abs  = resolve_data_path(_bcfg["raw_salary_war_path"], default_config)
        _data_dir = os.path.dirname(_sal_abs)
        _cpath    = os.path.join(_data_dir, "mlb_combined_2021_2025.csv")
        if not os.path.exists(_cpath):
            st.warning(f"Combined data file not found: `{_cpath}`")
            return

    # -- cached load -----------------------------------------------------------
    @st.cache_data(show_spinner=False)
    def _load_frontier(path: str, fhash: str) -> pd.DataFrame:
        _AL = {"BAL","BOS","CHW","CLE","DET","HOU","KCR","LAA","MIN","NYY","ATH","SEA","TBR","TEX","TOR"}
        _NL = {"ARI","ATL","CHC","CIN","COL","LAD","MIA","MIL","NYM","PHI","PIT","SDP","SFG","STL","WSN"}
        _PITCH = {"SP","RP","TWP","P"}
        df = read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["Salary_M"]  = pd.to_numeric(df.get("Salary", pd.Series(dtype=float)), errors="coerce") / 1_000_000
        df["WAR_Total"] = pd.to_numeric(df.get("WAR_Total", 0), errors="coerce").fillna(0.0)
        # 2023 data has WAR_Total=0 for many players where the WAR column has the real value
        _war_raw = pd.to_numeric(df.get("WAR", np.nan), errors="coerce")
        _fix_mask = (df["WAR_Total"] == 0) & _war_raw.notna() & (_war_raw != 0)
        df.loc[_fix_mask, "WAR_Total"] = _war_raw[_fix_mask]
        df["Age"]       = pd.to_numeric(df.get("Age", np.nan), errors="coerce")
        df["Year"]      = pd.to_numeric(df.get("Year", np.nan), errors="coerce").astype("Int64")
        for col in ("Stage_Clean", "Team", "Position"):
            df[col] = df.get(col, pd.Series("Unknown", index=df.index)).fillna("Unknown").replace("", "Unknown")
        df["League"]      = df["Team"].apply(lambda t: "AL" if t in _AL else ("NL" if t in _NL else "Unknown"))
        df["Player_Type"] = df["Position"].apply(lambda p: "Pitcher" if str(p).strip().upper() in _PITCH else "Position Player")
        df = df[df["Salary_M"].notna() & (df["Salary_M"] > 0)].copy()
        # Sort by salary desc, then WAR_Total desc as tiebreaker so artifact duplicate rows
        # (same salary, WAR_Total=0) lose to the real row with actual WAR data
        df = (df.sort_values(["Salary_M", "WAR_Total"], ascending=[False, False], kind="mergesort")
               .drop_duplicates(["Player", "Year"], keep="first")
               .reset_index(drop=True))
        return df

    raw = _load_frontier(_cpath, file_hash_fn(_cpath))

    # -- Data ranges -----------------------------------------------------------
    _all_teams  = ["All Teams"] + sorted(raw["Team"].dropna().unique())
    _all_stages = sorted(raw["Stage_Clean"].dropna().unique())
    _all_pos    = sorted(raw["Position"].dropna().unique())
    _all_years  = sorted(raw["Year"].dropna().unique())
    _age_bounds = (int(raw["Age"].min(skipna=True)), int(raw["Age"].max(skipna=True))) if raw["Age"].notna().any() else (18, 45)
    _war_bounds = (float(raw["WAR_Total"].min()), float(raw["WAR_Total"].max()))
    _sal_bounds = (float(raw["Salary_M"].min()), float(raw["Salary_M"].max()))

    # -- CSS: compact filter card + polished tabs ------------------------------
    st.markdown(
        "<style>"
        "div[data-testid='column']:first-of-type label{"
        "  font-size:0.70rem!important;margin-bottom:0!important;"
        "  line-height:1.2!important;color:" + C.text_muted + "!important;}"
        "div[data-testid='column']:first-of-type .stSelectbox>div,"
        "div[data-testid='column']:first-of-type .stMultiSelect>div{font-size:0.70rem!important;}"
        "div[data-testid='column']:first-of-type p{font-size:0.70rem!important;margin:0.05rem 0!important;}"
        "div[data-testid='column']:first-of-type hr{margin:0.2rem 0!important;border-color:" + C.border_primary + "!important;}"
        "div[data-testid='column']:first-of-type .stMultiSelect [data-baseweb='select']>div{"
        "  padding:1px 4px!important;min-height:26px!important;}"
        "div[data-testid='column']:first-of-type .stSelectbox [data-baseweb='select']>div{"
        "  padding:2px 6px!important;min-height:26px!important;}"
        "div[data-testid='column']:first-of-type .stSlider{"
        "  padding-top:0!important;padding-bottom:0.05rem!important;}"
        "</style>",
        unsafe_allow_html=True,
    )

    col_f, col_m = st.columns([0.82, 4.5])

    # -- Filter card -----------------------------------------------------------
    with col_f:
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;color:{C.text_dim};'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Filters</div>',
            unsafe_allow_html=True,
        )
        _league_sel = st.selectbox("League", ["Both Leagues", "AL", "NL"], key="ef_league")
        _year_options = [str(y) for y in _all_years]
        # Clear stale session state if it has values not in options
        if "ef_years" in st.session_state:
            _stale = [v for v in st.session_state["ef_years"] if v not in _year_options]
            if _stale:
                del st.session_state["ef_years"]
        _year_sel   = st.multiselect(
            "Year(s)", _year_options,
            default=_year_options, key="ef_years",
        )
        _team_sel   = st.multiselect("Team(s)", _all_teams, default=["All Teams"], key="ef_teams")
        _stage_sel  = st.multiselect("Stage", _all_stages, default=_all_stages, key="ef_stages")

        with st.expander("Position / Player type", expanded=False):
            _pos_sel   = st.multiselect("Position(s)", _all_pos, default=_all_pos, key="ef_pos")
            _ptype_sel = st.selectbox("Player Type", ["All", "Position Player", "Pitcher"], key="ef_ptype")

        with st.expander("Range filters", expanded=False):
            _age_sel   = st.slider("Age", _age_bounds[0], _age_bounds[1], _age_bounds, key="ef_age")
            _excl_zero = st.checkbox("Excl. 0 WAR", value=False, key="ef_excl_zero")
            _war_sel   = st.slider("WAR", float(_war_bounds[0]), float(_war_bounds[1]), _war_bounds, step=0.1, key="ef_war")
            _sal_sel   = st.slider("Salary ($M)", float(_sal_bounds[0]), float(_sal_bounds[1]), _sal_bounds, step=0.5, key="ef_sal")

        st.markdown(f"<hr style='border-color:{C.border_primary};margin:8px 0;'>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;color:{C.text_dim};'
            'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Display</div>',
            unsafe_allow_html=True,
        )
        _color_by   = st.selectbox("Color by", ["Career Stage", "Pay-Performance Ratio", "Cost/WAR vs Market"], key="ef_color")
        _reg_method = st.radio("Regression", ["OLS", "LOESS", "Poly"], horizontal=True, key="ef_reg")
        _show_reg   = st.checkbox("Trendline", value=True, key="ef_showreg")
        _shade_ovuv = st.checkbox("OV/UV zones", value=False, key="ef_shade")
        _hi_eff     = st.checkbox("Highlight efficient", value=False, key="ef_hi")
        _eff_thresh = st.slider("Efficiency %", 5, 30, 15, key="ef_thresh")

        st.markdown(f"<hr style='border-color:{C.border_primary};margin:8px 0;'>", unsafe_allow_html=True)
        if st.button("\u21ba  Reset Filters", use_container_width=True, key="ef_reset_btn"):
            _keep = {"ef_reset_btn"}
            for _k in [k for k in st.session_state if k.startswith("ef_") and k not in _keep]:
                del st.session_state[_k]
            st.rerun()
        if st.button("\u2715  Clear All Filters", use_container_width=True, key="ef_clear_btn"):
            for _k in ("ef_years", "ef_teams", "ef_stages"):
                if _k in st.session_state:
                    del st.session_state[_k]
            st.rerun()

    with col_m:
        # -- Apply filters -----------------------------------------------------
        df = raw.copy()
        if _league_sel != "Both Leagues":
            df = df[df["League"] == _league_sel]
        if "All Teams" not in _team_sel and _team_sel:
            df = df[df["Team"].isin(_team_sel)]
        if _stage_sel:
            df = df[df["Stage_Clean"].isin(_stage_sel)]
        if _pos_sel:
            df = df[df["Position"].isin(_pos_sel)]
        if _year_sel:
            df = df[df["Year"].isin([int(y) for y in _year_sel])]
        if _ptype_sel != "All":
            df = df[df["Player_Type"] == _ptype_sel]
        df = df[(df["Age"] >= _age_sel[0]) & (df["Age"] <= _age_sel[1])]
        if _excl_zero:
            df = df[df["WAR_Total"] != 0]
        df = df[(df["WAR_Total"] >= _war_sel[0]) & (df["WAR_Total"] <= _war_sel[1])]
        df = df[(df["Salary_M"] >= _sal_sel[0]) & (df["Salary_M"] <= _sal_sel[1])]
        df = df.reset_index(drop=True)

        if len(df) < 5:
            st.warning(f"Only {len(df)} players match the filters. Relax some filters to see results.")
            return

        # -- Fit regression ----------------------------------------------------
        x = df["WAR_Total"].values.astype(float)
        y = df["Salary_M"].values.astype(float)

        if _reg_method == "OLS":
            import statsmodels.api as _sm_api
            _Xc  = _sm_api.add_constant(x)
            _mod = _sm_api.OLS(y, _Xc).fit()
            predicted = _mod.predict(_Xc)
            _xseq = np.linspace(x.min(), x.max(), 200)
            _yseq = _mod.params[0] + _mod.params[1] * _xseq
            _reg_lbl = f"PPEL (OLS \u00b7 R\u00b2={_mod.rsquared:.3f})"
        elif _reg_method == "LOESS":
            _sm = _sm_lowess(y, x, frac=0.75, return_sorted=True)
            predicted = np.interp(x, _sm[:, 0], _sm[:, 1])
            _xseq, _yseq = _sm[:, 0], _sm[:, 1]
            _reg_lbl = "PPEL (LOESS)"
        else:
            _coeffs = np.polyfit(x, y, 2)
            _pfn    = np.poly1d(_coeffs)
            predicted = _pfn(x)
            _xseq = np.linspace(x.min(), x.max(), 200)
            _yseq = _pfn(_xseq)
            _reg_lbl = "PPEL (Poly deg 2)"

        df = df.copy()
        df["predicted"]          = predicted
        df["residual"]           = df["Salary_M"] - df["predicted"]
        df["residual_pct"]       = df["residual"].rank(pct=True) * 100
        df["PPR"]                = df["Salary_M"] / df["predicted"].clip(lower=0.01)
        df["cost_per_WAR"]       = np.where(df["WAR_Total"] > 0, df["Salary_M"] / df["WAR_Total"], np.nan)
        df["mkt_cost_per_WAR"]   = np.where(df["WAR_Total"] > 0, df["predicted"] / df["WAR_Total"], np.nan)
        df["cost_per_WAR_ratio"] = (df["cost_per_WAR"] / df["mkt_cost_per_WAR"].clip(lower=0.01)).fillna(1.0)
        df["_efficient"]         = df["residual_pct"] <= _eff_thresh

        _STAGE_COLORS = {
            "FA":      "#06d6a0",
            "Arb":     "#fbbf24",
            "Pre-Arb": "#38bdf8",
        }

        # -- KPI values for page header ----------------------------------------
        _n_shown  = len(df)
        _med_dpw  = float((df["Salary_M"] / df["WAR_Total"].replace(0, np.nan)).median())
        _flt_tms  = "All teams" if "All Teams" in _team_sel else f"{len(_team_sel)} team(s)"
        _flt_yrs  = f"{min(_year_sel)}\u2013{max(_year_sel)}" if _year_sel else "All"
        _r2_txt   = f"R\u00b2={_mod.rsquared:.3f}" if _reg_method == "OLS" else _reg_lbl

        # -- Page header card --------------------------------------------------
        st.markdown(f"""<div class="ef-hdr" style="background:linear-gradient(135deg,#0f2035,{C.bg_dark});border:1px solid #1e3a5c;
border-radius:10px;padding:16px 20px;margin-bottom:14px;display:flex;align-items:center;
justify-content:space-between;gap:16px;flex-wrap:wrap;">
  <div style="flex:1;min-width:200px;">
    <div style="font-size:20px;font-weight:800;color:#e8f4ff;margin-bottom:5px;">Player Analysis</div>
    <div style="font-size:13px;color:{C.text_secondary};line-height:1.6;">
      Explore how every MLB player's salary compares to their on-field production (fWAR).
      Use the tabs below to view the market value regression, multi-year efficiency, age curves,
      team breakdowns, and player stability ratings. Use the filters on the left to narrow
      by team, year, position, or career stage.
    </div>
  </div>
  <div class="ef-hdr-stats" style="display:flex;gap:10px;flex-shrink:0;flex-wrap:wrap;">
    <div style="background:{C.bg_dark};border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;">
      <div style="font-size:10px;color:{C.text_muted};text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Players</div>
      <div style="font-size:18px;font-weight:700;color:#e8f4ff;">{_n_shown:,}</div>
    </div>
    <div style="background:{C.bg_dark};border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;">
      <div style="font-size:10px;color:{C.text_muted};letter-spacing:1px;margin-bottom:2px;">MEDIAN $/fWAR</div>
      <div style="font-size:18px;font-weight:700;color:#e8f4ff;">${_med_dpw:.1f}M</div>
    </div>
    <div style="background:{C.bg_dark};border:1px solid #1e3a5c;border-radius:8px;padding:8px 14px;text-align:center;">
      <div style="font-size:10px;color:{C.text_muted};text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;">Active Filters</div>
      <div style="font-size:12px;font-weight:600;color:#a8c8e8;">{_flt_tms} \u00b7 {_flt_yrs}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        # -- MLBAM ID map for headshot hover images ----------------------------
        _mlb_ids = cached_mlbam_lookup(razzball_path)

        # -- Tabs --------------------------------------------------------------
        t1, t2, t3, t5, t6, t7, t8 = st.tabs([
            "Cost Effective Line",
            "PPEL",
            "Age Trajectory",
            "Efficient Players",
            "Residual Analysis",
            "Pre-Arb Explorer",
            "WAR Stability",
        ])

        # -- Tab 1 -- Cost Effective Line (PPEL 1/3/5-year views) -------------
        with t1:
            st.markdown(
                "<div style='background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;"
                "padding:14px 18px;margin-bottom:12px;'>"
                f"<div style='font-size:1rem;font-weight:700;color:{C.text_primary};margin-bottom:6px;'>Cost Effective Line</div>"
                f"<div style='font-size:0.85rem;color:{C.text_secondary};line-height:1.7;'>"
                "This scatter plot compares every player's <b>actual salary</b> (Y-axis) against their "
                "<b>fWAR production</b> (X-axis). The <span style='color:#f4a261;font-weight:600;'>orange trendline</span> "
                "is the market average \u2014 it shows what teams typically pay for a given level of production.<br><br>"
                "<b>How to read it:</b> Dots <span style='color:#22c55e;font-weight:600;'>below</span> the line "
                "are bargains (the player produces more than their salary suggests). "
                "Dots <span style='color:#ef4444;font-weight:600;'>above</span> the line are overpaid relative to output. "
                "The further from the line, the bigger the gap between actual and market salary.<br><br>"
                "<b>What is fWAR?</b> WAR estimates a player's value in terms of extra wins they provide compared to a "
                "replacement-level player \u2014 a low-cost minor leaguer easily found to fill the same position. "
                "In 2025, the average player had an fWAR of 0.7, All-Stars averaged 3.2+, and MVP candidates reached 4.0+.<br><br>"
                "<b>Key terms:</b> <b>PPR</b> = Pay-to-Performance Ratio (career fWAR \u00f7 total contract $M; "
                "below 1.0 = underpaid, above 1.0 = overpaid). <b>Residual</b> = actual salary minus the trendline's "
                "predicted salary (negative = team is getting a deal)."
                "</div></div>",
                unsafe_allow_html=True,
            )
            # -- Frontier Summary card -----------------------------------------
            _stage_ppr     = df.groupby("Stage_Clean")["PPR"].median().sort_values()
            _most_eff_stg  = _stage_ppr.index[0]  if len(_stage_ppr) > 0 else "\u2014"
            _least_eff_stg = _stage_ppr.index[-1] if len(_stage_ppr) > 0 else "\u2014"
            _most_eff_ppr  = float(_stage_ppr.iloc[0])  if len(_stage_ppr) > 0 else 1.0
            _least_eff_ppr = float(_stage_ppr.iloc[-1]) if len(_stage_ppr) > 0 else 1.0
            _underpaid_n   = int((df["PPR"] < 1.0).sum())
            _overpaid_n    = int((df["PPR"] > 1.0).sum())

            st.markdown(f"""<div style="background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;padding:14px 16px;margin-bottom:12px;">
  <div style="font-size:10px;font-weight:700;color:#5a8aaa;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Career Stages Explained</div>
  <div class="ef-summary-grid" style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">
    <div style="background:{C.bg_dark};border:1px solid #14532d;border-top:3px solid #22c55e;border-radius:8px;padding:10px 12px;">
      <div style="font-size:13px;font-weight:700;color:#22c55e;margin-bottom:4px;">Pre-Arbitration</div>
      <div style="font-size:10px;color:{C.text_muted};line-height:1.5;">0\u20133 years service time. Salary near league minimum (~$740K). Teams control rights \u2014 often the best value in baseball.</div>
    </div>
    <div style="background:{C.bg_dark};border:1px solid #14b8a6;border-top:3px solid #14b8a6;border-radius:8px;padding:10px 12px;">
      <div style="font-size:13px;font-weight:700;color:#14b8a6;margin-bottom:4px;">Arbitration</div>
      <div style="font-size:10px;color:{C.text_muted};line-height:1.5;">3\u20136 years service time. Salary negotiated or set by arbitration hearing. Pay rises based on prior performance.</div>
    </div>
    <div style="background:{C.bg_dark};border:1px solid {C.accent_blue};border-top:3px solid #60a5fa;border-radius:8px;padding:10px 12px;">
      <div style="font-size:13px;font-weight:700;color:#60a5fa;margin-bottom:4px;">Free Agent</div>
      <div style="font-size:10px;color:{C.text_muted};line-height:1.5;">6+ years service time. Player signs on the open market. Full market-rate salary \u2014 highest cost per WAR.</div>
    </div>
  </div>
  <div style="display:flex;gap:16px;margin-top:10px;">
    <div style="font-size:10px;color:{C.text_muted};"><span style="color:#22c55e;font-weight:700;">{_underpaid_n}</span> underpaid / <span style="color:#ef4444;font-weight:700;">{_overpaid_n}</span> overpaid (PPR vs 1.0)</div>
    <div style="font-size:10px;color:{C.text_muted};">{_r2_txt} \u00b7 {_reg_method} \u00b7 N={_n_shown:,}</div>
  </div>
</div>""", unsafe_allow_html=True)

            # -- Player search / highlight -------------------------------------
            _hl_col, _hl_rst_col = st.columns([3, 1])
            with _hl_col:
                _player_hl = st.multiselect(
                    "Highlight player(s)",
                    sorted(df["Player"].unique()),
                    placeholder="Type or select player names…",
                    key="ef_player_hl",
                )
            with _hl_rst_col:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("↺  Reset", use_container_width=True, key="ef_player_hl_reset"):
                    if "ef_player_hl" in st.session_state:
                        del st.session_state["ef_player_hl"]
                    st.rerun()
            _has_hl = bool(_player_hl)

            # -- Chart ---------------------------------------------------------
            _hover = df.apply(lambda r: (
                f"<b>{r['Player']}</b><br>"
                + f"{r['Team']}  {r['Year']}<br>"
                + f"WAR: {r['WAR_Total']:.1f}  |  Salary: ${r['Salary_M']:.2f}M<br>"
                + f"Expected: ${r['predicted']:.2f}M<br>"
                + f"PPR: {r['PPR']:.3f}  |  {r['Stage_Clean']}"
            ), axis=1)
            _sizes = np.where(df["_efficient"] & _hi_eff, 13, 7).tolist()
            _base_opacity = 0.12 if _has_hl else 0.78

            fig1 = go.Figure()

            if _shade_ovuv and _show_reg and len(_xseq) > 1:
                _std = float(df["residual"].std())
                fig1.add_trace(go.Scattergl(
                    x=np.concatenate([_xseq, _xseq[::-1]]),
                    y=np.concatenate([_yseq + _std, _yseq[::-1]]),
                    fill="toself", fillcolor="rgba(239,68,68,0.07)",
                    line=dict(color="rgba(0,0,0,0)"), name="Overpaid zone", hoverinfo="skip",
                ))
                fig1.add_trace(go.Scattergl(
                    x=np.concatenate([_xseq, _xseq[::-1]]),
                    y=np.concatenate([_yseq, (_yseq - _std)[::-1]]),
                    fill="toself", fillcolor="rgba(34,197,94,0.07)",
                    line=dict(color="rgba(0,0,0,0)"), name="Underpaid zone", hoverinfo="skip",
                ))

            if _color_by == "Career Stage":
                for _stg, _grp in df.groupby("Stage_Clean"):
                    _c   = _STAGE_COLORS.get(_stg, "#94a3b8")
                    _idx = _grp.index
                    fig1.add_trace(go.Scattergl(
                        x=_grp["WAR_Total"], y=_grp["Salary_M"],
                        mode="markers", name=_stg,
                        marker=dict(
                            color=_c, size=[_sizes[i] for i in _idx], opacity=_base_opacity,
                            line=dict(color="#22c55e" if _hi_eff else "rgba(0,0,0,0)", width=1.2),
                        ),
                        text=_hover[_idx], hovertemplate="%{text}<extra></extra>",
                        customdata=_grp["Player"].values,
                    ))
            else:
                _col_vals = df["PPR"] if _color_by == "Pay-Performance Ratio" else df["cost_per_WAR_ratio"]
                _cscale   = [
                    [0.0, "#22c55e"], [0.33, "#86efac"], [0.5, "#ffffff"],
                    [0.67, "#fca5a5"], [1.0, "#ef4444"],
                ]
                fig1.add_trace(go.Scattergl(
                    x=df["WAR_Total"], y=df["Salary_M"],
                    mode="markers", name="Players", showlegend=False,
                    marker=dict(
                        color=_col_vals, colorscale=_cscale, cmid=1.0,
                        size=_sizes, opacity=_base_opacity,
                        colorbar=dict(title=_color_by, thickness=12, tickfont=dict(color="#7aa2c0")),
                        showscale=True,
                        line=dict(color="#22c55e" if _hi_eff else "rgba(0,0,0,0)", width=1.2),
                    ),
                    text=_hover, hovertemplate="%{text}<extra></extra>",
                    customdata=df["Player"].values,
                ))

            # -- Highlighted player overlay ------------------------------------
            if _has_hl:
                _df_hl = df[df["Player"].isin(_player_hl)]
                if not _df_hl.empty:
                    for _stg, _grp in _df_hl.groupby("Stage_Clean"):
                        _c   = _STAGE_COLORS.get(_stg, "#94a3b8")
                        _idx = _grp.index
                        fig1.add_trace(go.Scattergl(
                            x=_grp["WAR_Total"], y=_grp["Salary_M"],
                            mode="markers+text",
                            name=_stg, showlegend=False,
                            text=_grp["Player"],
                            textposition="top center",
                            textfont=dict(color="#ffffff", size=9),
                            marker=dict(
                                color=_c, size=14, opacity=1.0,
                                line=dict(color="#ffffff", width=2),
                            ),
                            hovertext=_hover[_idx],
                            hovertemplate="%{hovertext}<extra></extra>",
                            customdata=_grp["Player"].values,
                        ))

            if _show_reg:
                fig1.add_trace(go.Scattergl(
                    x=_xseq, y=_yseq, mode="lines",
                    line=dict(color="#f4a261", width=2),
                    name=_reg_lbl,
                ))

            # Quadrant callout annotations
            _xa = float(np.percentile(x[x > 0], 70)) if (x > 0).any() else float(x.mean())
            _yb = float(np.percentile(y, 22))
            _xb = float(np.percentile(x, 20))
            _ya = float(np.percentile(y, 78))
            fig1.add_annotation(
                x=_xa, y=_yb, text="High WAR, low cost \u2014 Efficient",
                showarrow=False, font=dict(color="#22c55e", size=9),
                opacity=0.55,
            )
            fig1.add_annotation(
                x=_xb, y=_ya, text="Low WAR, high cost \u2014 Overpaid",
                showarrow=False, font=dict(color="#ef4444", size=9),
                opacity=0.55,
            )

            fig1.update_layout(**_pt(
                title="",
                xaxis=dict(
                    title=dict(text="WAR (Wins Above Replacement)", font=dict(size=12, color="#c0d8f0")),
                    tickfont=dict(color="#c0d8f0", size=10),
                    gridcolor="#18293c", linecolor="#18293c", zerolinecolor="#18293c",
                ),
                yaxis=dict(
                    title=dict(text="Salary ($M)", font=dict(size=12, color="#c0d8f0")),
                    tickfont=dict(color="#c0d8f0", size=10),
                    gridcolor="#18293c", linecolor="#18293c", zerolinecolor="#18293c",
                ),
                height=580, showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#a8c8e8"),
                    itemsizing="constant",
                ),
                margin=dict(l=50, r=20, t=16, b=50),
                hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                font=dict(color="#dbeafe", size=12), namelength=-1),
            ))
            _ef_sel = st.plotly_chart(
                fig1, use_container_width=True, key="ef_frontier_scatter",
                on_select="rerun", selection_mode="points",
                config={"modeBarButtonsToKeep": ["zoom2d", "pan2d", "resetScale2d", "toImage"],
                        "displaylogo": False},
            )

            # -- Selected player card (click a dot to reveal) ------------------
            _ef_pts = []
            try:
                _sel_obj = getattr(_ef_sel, "selection", None)
                if _sel_obj:
                    _ef_pts = getattr(_sel_obj, "points", []) or []
            except Exception:
                pass
            if _ef_pts:
                _clicked_name = _ef_pts[0].get("customdata", None)
                if isinstance(_clicked_name, list):
                    _clicked_name = _clicked_name[0] if _clicked_name else None
                if _clicked_name:
                    _prow_df = df[df["Player"] == _clicked_name]
                    if not _prow_df.empty:
                        _pr = _prow_df.iloc[0]
                        _hs_bytes = None
                        try:
                            if r2_mode and _requests_available:
                                _hs_r = _requests.get(
                                    f"{R2_BASE_URL}/data/headshots/{_pr['Player']}.png",
                                    timeout=5,
                                )
                                if _hs_r.status_code == 200:
                                    _hs_bytes = _hs_r.content
                            else:
                                _hs_local = os.path.join(_HEADSHOTS_DIR, f"{_pr['Player']}.png")
                                if os.path.exists(_hs_local):
                                    with open(_hs_local, "rb") as _hf:
                                        _hs_bytes = _hf.read()
                            if _hs_bytes is None and _requests_available:
                                _rzb = _cached_razzball(razzball_path)
                                _mlbam_id = None
                                if not _rzb.empty and "Name" in _rzb.columns and "MLBAMID" in _rzb.columns:
                                    _rzb_r = _rzb[
                                        _rzb["Name"].str.strip().str.lower()
                                        == _pr["Player"].strip().lower()
                                    ]
                                    if not _rzb_r.empty:
                                        _mlbam_id = _rzb_r["MLBAMID"].iloc[0]
                                if _mlbam_id and str(_mlbam_id).isdigit():
                                    _hs_url = (
                                        "https://img.mlbstatic.com/mlb-photos/image/upload/"
                                        "d_people:generic:headshot:67:current.png/w_213,q_auto:best"
                                        f"/v1/people/{_mlbam_id}/headshot/67/current"
                                    )
                                    _hs_r2 = _requests.get(_hs_url, timeout=6)
                                    if _hs_r2.status_code == 200:
                                        _hs_bytes = _hs_r2.content
                        except Exception:
                            pass
                        _ppr_clr = "#22c55e" if _pr["PPR"] < 1.0 else "#ef4444"
                        _res_clr = "#22c55e" if _pr["residual"] <= 0 else "#ef4444"
                        _res_sgn = "\u2212" if _pr["residual"] <= 0 else "+"
                        _card_a, _card_b = st.columns([1, 5])
                        with _card_a:
                            if _hs_bytes:
                                st.image(_hs_bytes, width=120)
                            else:
                                st.markdown(
                                    '<div style="width:120px;height:120px;background:#1a2f4a;'
                                    'border:1px solid #2a4a6a;border-radius:8px;display:flex;'
                                    'align-items:center;justify-content:center;'
                                    'color:#3a6080;font-size:32px;">\U0001f464</div>',
                                    unsafe_allow_html=True,
                                )
                        with _card_b:
                            _age_disp = int(_pr["Age"]) if pd.notna(_pr["Age"]) else "\u2014"
                            st.markdown(f"""<div style="background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;padding:12px 16px;">
  <div style="font-size:17px;font-weight:800;color:#e8f4ff;margin-bottom:3px;">{_pr['Player']}</div>
  <div style="font-size:12px;color:#6a9ab8;margin-bottom:8px;">{_pr['Team']} &middot; {int(_pr['Year'])} &middot; {_pr['Stage_Clean']} &middot; Age {_age_disp}</div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
    <div style="background:{C.bg_dark};border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">WAR</div>
      <div style="font-size:15px;font-weight:700;color:#e8f4ff;">{_pr['WAR_Total']:.1f}</div>
    </div>
    <div style="background:{C.bg_dark};border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">Salary</div>
      <div style="font-size:15px;font-weight:700;color:#e8f4ff;">${_pr['Salary_M']:.1f}M</div>
    </div>
    <div style="background:{C.bg_dark};border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">Expected</div>
      <div style="font-size:15px;font-weight:700;color:#e8f4ff;">${_pr['predicted']:.1f}M</div>
    </div>
    <div style="background:{C.bg_dark};border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">PPR</div>
      <div style="font-size:15px;font-weight:700;color:{_ppr_clr};">{_pr['PPR']:.3f}</div>
    </div>
    <div style="background:{C.bg_dark};border:1px solid #1e3a5c;border-radius:6px;padding:7px 10px;text-align:center;">
      <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:2px;">vs Market</div>
      <div style="font-size:15px;font-weight:700;color:{_res_clr};">{_res_sgn}${abs(_pr['residual']):.1f}M</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # -- Footer explanation --------------------------------------------
            st.markdown(f"""<div style="background:#090f1a;border:1px solid #1a3050;border-radius:8px;
padding:12px 16px;margin-top:4px;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;font-size:11px;color:#5a8aaa;">
    <div>
      <span style="color:#f4a261;font-weight:700;">Trendline ({_reg_method})</span><br>
      The orange line shows the market-predicted salary at each WAR level.
      Points below are underpaid relative to market; points above are overpaid.
    </div>
    <div>
      <span style="font-weight:700;color:#a8c8e8;">Dot colours (Career Stage)</span><br>
      <span style="color:#06d6a0;">\u25cf</span> Free Agent<br>
      <span style="color:#fbbf24;">\u25cf</span> Arbitration<br>
      <span style="color:#38bdf8;">\u25cf</span> Pre-Arb.<br>
      Switch colour mode in the Display panel.
    </div>
    <div>
      <span style="font-weight:700;color:#a8c8e8;">PPR (Pay-Performance Ratio)</span><br>
      Actual salary \u00f7 model-predicted salary.<br>
      PPR &lt; 1 = underpaid<br>
      PPR &gt; 1 = overpaid<br>
      PPR = 1 = fair market.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

            # -- Selected sample summary ---------------------------------------
            _yrs_d  = f"{min(_year_sel)}\u2013{max(_year_sel)}" if _year_sel else "All"
            _tms_d  = "All teams" if "All Teams" in _team_sel else (
                ", ".join(_team_sel[:3]) + ("\u2026" if len(_team_sel) > 3 else "")
            )
            _pos_d  = "All" if len(_pos_sel) == len(_all_pos) else f"{len(_pos_sel)} pos"
            _stg_d  = "All" if len(_stage_sel) == len(_all_stages) else ", ".join(_stage_sel)
            st.markdown(f"""<div style="background:#090f1a;border:1px solid #1a3050;border-radius:8px;
padding:9px 16px;margin-top:6px;display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
  <span style="font-size:9px;color:#3a6080;text-transform:uppercase;letter-spacing:1px;white-space:nowrap;">Selected sample</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">Years</b> {_yrs_d}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">Teams</b> {_tms_d}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">League</b> {_league_sel}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">Stages</b> {_stg_d}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">Positions</b> {_pos_d}</span>
  <span style="font-size:11px;color:#7aa2c0;"><b style="color:#a8c8e8">N</b> {_n_shown:,} player-seasons</span>
</div>""", unsafe_allow_html=True)

            with st.expander("Model summary"):
                if _reg_method == "OLS":
                    st.text(
                        f"Method: OLS\nFormula: Salary_M ~ WAR_Total\n"
                        f"R\u00b2: {_mod.rsquared:.4f}  Adj R\u00b2: {_mod.rsquared_adj:.4f}\n"
                        f"Intercept: {_mod.params[0]:.3f}  Slope (WAR): {_mod.params[1]:.3f}\n"
                        f"Residual Std Error: {_mod.mse_resid**0.5:.3f}"
                    )
                elif _reg_method == "LOESS":
                    st.text(f"Method: LOESS  span=0.75  N={len(df)}")
                else:
                    st.text(f"Method: Polynomial (deg 2)\n"
                            f"Coefficients (highest\u2192lowest): {[f'{c:.4f}' for c in _coeffs]}")

            # -- Top 25 Most Underpaid Players (auto-adjusts with filters) -----
            st.markdown(
                f"<div style='margin-top:1rem;font-size:0.92rem;font-weight:700;color:{C.text_primary};'>"
                "Top 25 Most Underpaid Players</div>"
                f"<div style='font-size:0.76rem;color:{C.text_muted};margin-bottom:0.4rem;'>"
                "Ranked by PPR (lowest = most underpaid). Adjusts with all filters above. Excludes Pre-Arbitration players and players under 1 fWAR.</div>",
                unsafe_allow_html=True,
            )
            _top25_pool = df[~df["Stage_Clean"].str.contains("Pre.Arb|Pre-Arb|Pre_Arb|Pre Arb|Pre-Arbitration|PreArb", case=False, na=False, regex=True) & (df["WAR_Total"] >= 1.0)]
            _top25 = _top25_pool.nsmallest(25, "PPR")[["Player","Team","Year","WAR_Total","Salary_M","predicted","PPR","Stage_Clean"]].copy()
            _top25.insert(0, "#", range(1, len(_top25) + 1))
            _top25.columns = ["#", "Player", "Team", "Year", "fWAR", "Salary $M", "Expected $M", "PPR", "Stage"]
            _top25["Year"] = _top25["Year"].astype(int)
            st.dataframe(
                _top25.style.format({
                    "fWAR": "{:.1f}", "Salary $M": "{:.1f}",
                    "Expected $M": "{:.1f}", "PPR": "{:.3f}",
                }).apply(lambda row: ["background-color:#0c221866"] * len(row) if row["#"] <= 5 else [""] * len(row), axis=1),
                hide_index=True, use_container_width=True, height=385,
            )

        # -- Tab 3 -- Age Trajectory -------------------------------------------
        with t3:
            st.markdown(
                "<div style='background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;"
                "padding:14px 18px;margin-bottom:12px;'>"
                f"<div style='font-size:1rem;font-weight:700;color:{C.text_primary};margin-bottom:6px;'>Age Trajectory</div>"
                f"<div style='font-size:0.85rem;color:{C.text_secondary};line-height:1.7;'>"
                "See how player production changes as they age. Most players peak in their late 20s "
                "and decline into their 30s \u2014 but the shape varies by position and player type.<br><br>"
                "<b>How to read it:</b> The line shows the <b>average</b> metric at each age across all "
                "filtered players. Choose fWAR (production), Salary (cost), or Efficiency (production per $) "
                "to see different age curves. Split by Player Type or Position to compare groups."
                "</div></div>",
                unsafe_allow_html=True,
            )
            _c1, _c2, _c3 = st.columns([2, 2, 1])
            with _c1:
                _traj_m = st.selectbox("Metric", ["WAR", "Salary ($M)", "Efficiency (WAR/$M)"], key="ef_traj_m")
            with _c2:
                _traj_s = st.selectbox("Split by", ["None", "Player Type", "Position"], key="ef_traj_s")
            with _c3:
                _show_ind = st.checkbox("Individual points", value=False, key="ef_traj_ind")

            _mcol = {"WAR": "WAR_Total", "Salary ($M)": "Salary_M", "Efficiency (WAR/$M)": "_eff"}[_traj_m]
            df2 = df.copy()
            df2["_eff"] = df2["WAR_Total"] / df2["Salary_M"].clip(lower=0.01)
            df2["_age_i"] = df2["Age"].round().astype("Int64")
            df2 = df2.dropna(subset=["Age", _mcol])

            fig2 = go.Figure()
            _split_col = None if _traj_s == "None" else ("Player_Type" if _traj_s == "Player Type" else "Position")

            def _add_age_trace(grp, name, color=None):
                _agg = grp.groupby("_age_i")[_mcol].agg(["mean","count"]).reset_index()
                _kw = dict(color=color) if color else {}
                fig2.add_trace(go.Scatter(
                    x=_agg["_age_i"].astype(int), y=_agg["mean"],
                    mode="lines+markers", name=name,
                    marker=dict(size=(_agg["count"] / _agg["count"].max() * 14 + 5).clip(5, 20).tolist(), **_kw),
                    line=dict(**_kw),
                    hovertemplate=f"{name} age %{{x}}: avg=%{{y:.2f}}<extra></extra>",
                ))
                if _show_ind:
                    fig2.add_trace(go.Scatter(
                        x=grp["_age_i"].astype(int), y=grp[_mcol],
                        mode="markers", name=f"{name} (pts)", showlegend=False,
                        marker=dict(size=4, opacity=0.3, **_kw),
                        text=grp["Player"], hovertemplate="%{text}: %{y:.2f}<extra></extra>",
                    ))

            if _split_col is None:
                _add_age_trace(df2, "Average", "#2a9d8f")
            else:
                for _gn, _gg in df2.groupby(_split_col):
                    if len(_gg) >= 3:
                        _add_age_trace(_gg, str(_gn))

            fig2.update_layout(**_pt(
                title=f"Average {_traj_m} by Age",
                xaxis=dict(title="Age"), yaxis=dict(title=_traj_m),
                height=640, showlegend=True,
            ))
            st.plotly_chart(fig2, width="stretch")

            # -- Top 25 Standouts by age efficiency ----------------------------
            st.markdown(
                f"<div style='margin-top:1rem;font-size:0.92rem;font-weight:700;color:{C.text_primary};'>"
                "Top 25 Standouts \u2014 Best Value by Age</div>"
                f"<div style='font-size:0.76rem;color:{C.text_muted};margin-bottom:0.4rem;'>"
                "Highest fWAR per $M among filtered players. Adjusts with all filters above.</div>",
                unsafe_allow_html=True,
            )
            _age_eff = df.copy()
            _age_eff["fWAR/$M"] = _age_eff["WAR_Total"] / _age_eff["Salary_M"].clip(lower=0.01)
            _age_top = _age_eff.nlargest(25, "fWAR/$M")[["Player","Team","Year","Age","WAR_Total","Salary_M","Stage_Clean"]].copy()
            _age_top["fWAR/$M"] = (_age_top["WAR_Total"] / _age_top["Salary_M"].clip(lower=0.01)).round(2)
            _age_top.insert(0, "#", range(1, len(_age_top) + 1))
            _age_top.columns = ["#", "Player", "Team", "Year", "Age", "fWAR", "Salary $M", "Stage", "fWAR/$M"]
            _age_top["Year"] = _age_top["Year"].astype(int)
            _age_top["Age"] = _age_top["Age"].round(0).astype(int)
            st.dataframe(
                _age_top.style.format({
                    "fWAR": "{:.1f}", "Salary $M": "{:.1f}", "fWAR/$M": "{:.2f}",
                }).apply(lambda row: ["background-color:#0c221866"] * len(row) if row["#"] <= 5 else [""] * len(row), axis=1),
                hide_index=True, use_container_width=True, height=385,
            )

        # -- Tab 5 -- Efficient Players ----------------------------------------
        with t5:
            _top100 = (
                df.nsmallest(100, "PPR")
                  [["Player","Team","League","Year","Age","Position","Stage_Clean",
                    "WAR_Total","Salary_M","predicted","residual","PPR","cost_per_WAR"]]
                  .rename(columns={
                      "Stage_Clean":"Stage","WAR_Total":"WAR","Salary_M":"Actual $M",
                      "predicted":"Expected $M","residual":"Residual $M","cost_per_WAR":"$/WAR",
                  }).reset_index(drop=True)
            )

            def _clr_ppr(val):
                try:
                    v = float(val)
                    if v < 0.75:  return "background-color:#14532d;color:white"
                    if v < 1.0:   return "background-color:#166534;color:white"
                    if v > 1.5:   return "background-color:#7f1d1d;color:white"
                    if v > 1.0:   return "background-color:#991b1b;color:white"
                except Exception:
                    pass
                return ""

            st.dataframe(
                _top100.style
                    .format({"WAR":"{:.1f}","Actual $M":"${:.2f}M","Expected $M":"${:.2f}M",
                             "Residual $M":"${:+.2f}M","PPR":"{:.3f}","$/WAR":"${:.2f}M","Age":"{:.0f}"}, na_rep="\u2014")
                    .map(_clr_ppr, subset=["PPR"]),
                use_container_width=True, hide_index=True, height=480,
            )

        # -- Tab 6 -- Residual Analysis ----------------------------------------
        with t6:
            _r1, _r2 = st.columns(2)
            with _r1:
                fR1 = go.Figure()
                fR1.add_hline(y=0, line_color="#94a3b8", line_dash="dash", opacity=0.5)
                fR1.add_trace(go.Scatter(
                    x=df["predicted"], y=df["residual"], mode="markers",
                    marker=dict(color="#2a9d8f", size=5, opacity=0.65),
                    text=df["Player"],
                    hovertemplate="%{text}<br>Fitted: $%{x:.2f}M  Residual: $%{y:.2f}M<extra></extra>",
                ))
                fR1.update_layout(**_pt(title="Residuals vs Fitted",
                    xaxis=dict(title="Predicted Salary ($M)"), yaxis=dict(title="Residual ($M)"), height=400))
                st.plotly_chart(fR1, width="stretch")

            with _r2:
                fR2 = go.Figure(go.Histogram(
                    x=df["residual"], nbinsx=30,
                    marker=dict(color="#2a9d8f", line=dict(color="#0e1117", width=0.5)),
                ))
                fR2.add_vline(x=0, line_color="#f4a261", line_dash="dash")
                fR2.update_layout(**_pt(title="Residual Distribution",
                    xaxis=dict(title="Residual ($M)"), yaxis=dict(title="Count"), height=400))
                st.plotly_chart(fR2, width="stretch")

            fR3 = go.Figure()
            fR3.add_hline(y=0, line_color="#94a3b8", line_dash="dash", opacity=0.5)
            fR3.add_trace(go.Scatter(
                x=df["WAR_Total"], y=df["residual"], mode="markers",
                marker=dict(color="#a8dadc", size=5, opacity=0.6),
                text=df["Player"],
                hovertemplate="%{text}<br>WAR: %{x:.1f}  Residual: $%{y:.2f}M<extra></extra>",
            ))
            if len(df) >= 10:
                _rlo = _sm_lowess(df["residual"].values.astype(float),
                                  df["WAR_Total"].values.astype(float), frac=0.5, return_sorted=True)
                fR3.add_trace(go.Scatter(x=_rlo[:, 0], y=_rlo[:, 1], mode="lines",
                                         line=dict(color="#ef4444", width=2), name="LOESS trend"))
            fR3.update_layout(**_pt(title="Residuals vs WAR",
                xaxis=dict(title="WAR"), yaxis=dict(title="Residual ($M)"), height=440, showlegend=True))
            st.plotly_chart(fR3, width="stretch")

        # -- Tab 2 -- PPEL (Pay-Performance Efficiency Line) -------------------
        with t2:
            st.markdown(
                "<div style='background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;"
                "padding:14px 18px;margin-bottom:12px;'>"
                f"<div style='font-size:1rem;font-weight:700;color:{C.text_primary};margin-bottom:6px;'>PPEL \u2014 Multi-Year Value Analysis</div>"
                f"<div style='font-size:0.85rem;color:{C.text_secondary};line-height:1.7;'>"
                "This view extends the Cost Effective Line across <b>multiple seasons</b> to separate "
                "one-year flukes from sustained performance. Use the toggle to switch between 1-year, "
                "3-year, and 5-year windows.<br><br>"
                "<b>Why it matters:</b> A single great season can make a player look like a bargain, but "
                "PPEL3 and PPEL5 show whether that value held up over a full contract. Players who "
                "consistently sit below the line are the truly elite values in baseball.<br><br>"
                "<b>fWAR/$M</b> = fWAR per million dollars spent. Higher is better. "
                "Pre-Arb players often dominate this metric because they produce at near league-minimum salary (~$740K)."
                "</div></div>",
                unsafe_allow_html=True,
            )
            # Controls in a compact horizontal row above the chart
            _tc1, _tc2, _tc3, _tc4 = st.columns([1.4, 1, 1.2, 1])
            with _tc1:
                _pv_mode = st.radio("View", ["PPEL (1-Year)", "PPEL3 (3-Year)", "PPEL5 (5-Year)"],
                                    horizontal=True, key="ef_pvp_mode")
            with _tc2:
                _pv_n = st.slider("Top N players", 10, 50, 25, key="ef_pvp_n")
            with _tc3:
                _pv_sort = st.radio("Sort", ["Best value (high)", "Worst value (low)"],
                                    key="ef_pvp_sort")
            with _tc4:
                _pv_min_war = st.slider("Min WAR", 0.0, 5.0, 0.5, step=0.1, key="ef_pvp_minwar")

            if _pv_mode == "PPEL (1-Year)":
                # Scatter of filtered single-season data with CEL regression line
                _pvp_df = df[df["WAR_Total"] >= _pv_min_war].copy()
                _pvp_df["PVP"] = _pvp_df["WAR_Total"] / _pvp_df["Salary_M"].clip(lower=0.01)
                _pvp_df["_above"] = _pvp_df["residual"] <= 0  # below regression = underpaid

                _pvp_hover = _pvp_df.apply(lambda r: (
                    f"<b>{r['Player']}</b><br>"
                    + f"{r['Team']}  {r['Year']}<br>"
                    + f"WAR: {r['WAR_Total']:.1f}  Salary: ${r['Salary_M']:.2f}M<br>"
                    + f"PVP: {r['PVP']:.3f}  |  {r['Stage_Clean']}"
                ), axis=1)

                _pvp_clrs = ["#22c55e" if v else "#ef4444" for v in _pvp_df["_above"]]
                fig_cel = go.Figure()
                fig_cel.add_trace(go.Scatter(
                    x=_pvp_df["WAR_Total"], y=_pvp_df["Salary_M"],
                    mode="markers",
                    marker=dict(color=_pvp_clrs, size=7, opacity=0.75,
                                line=dict(color="rgba(0,0,0,0)", width=0)),
                    text=_pvp_hover, hovertemplate="%{text}<extra></extra>",
                    name="Players", showlegend=False,
                ))
                fig_cel.add_trace(go.Scatter(
                    x=_xseq, y=_yseq, mode="lines",
                    line=dict(color="#f4a261", width=2.5),
                    name="Cost Effective Line",
                ))
                fig_cel.update_layout(**_pt(
                    title="Cost Effective Line \u2014 WAR vs Salary (1-Year)",
                    xaxis=dict(title="WAR"), yaxis=dict(title="Salary ($M)"),
                    height=640, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                    font=dict(color="#dbeafe", size=12), namelength=-1),
                ))
                st.plotly_chart(fig_cel, width="stretch")

                # Ranked bar below
                _pvp_ranked = (
                    _pvp_df.nlargest(_pv_n, "PVP") if _pv_sort.startswith("Best")
                    else _pvp_df.nsmallest(_pv_n, "PVP")
                ).sort_values("PVP", ascending=_pv_sort.startswith("Worst"))
                _pvp_ranked["_lbl"] = (
                    _pvp_ranked["Player"] + "  (" + _pvp_ranked["Team"]
                    + ", " + _pvp_ranked["Year"].astype(str) + ")"
                )
                fig_bar1 = go.Figure(go.Bar(
                    y=_pvp_ranked["_lbl"], x=_pvp_ranked["PVP"], orientation="h",
                    marker=dict(color=["#22c55e" if v else "#ef4444"
                                       for v in (_pvp_ranked["residual"] <= 0)]),
                    text=_pvp_ranked["PVP"].round(3).astype(str),
                    textposition="outside", textfont=dict(color="#dbeafe", size=9),
                    hovertemplate="%{y}<br>PVP: %{x:.3f}<extra></extra>",
                ))
                fig_bar1.update_layout(**_pt(
                    title="PVP Ranking (WAR / Salary $M)",
                    xaxis=dict(title="PVP"), yaxis=dict(autorange="reversed"),
                    height=max(350, _pv_n * 22), margin=dict(l=230, r=60, t=40, b=40),
                ))
                st.plotly_chart(fig_bar1, width="stretch")
                _league_pvp = df["WAR_Total"].sum() / df["Salary_M"].sum()
                st.caption(f"League-average PVP (filtered data): **{_league_pvp:.3f}** WAR per $M")

            elif _pv_mode == "PPEL3 (3-Year)":  # PPEL3 -- last 3 years, actual contract data only
                _all_yrs_sorted = sorted(raw["Year"].dropna().unique())
                _last3 = _all_yrs_sorted[-3:] if len(_all_yrs_sorted) >= 3 else _all_yrs_sorted

                _raw3 = raw[raw["Year"].isin(_last3) & (raw["Salary_M"] > 0)].copy()
                if _league_sel != "Both Leagues":
                    _raw3 = _raw3[_raw3["League"] == _league_sel]
                if "All Teams" not in _team_sel and _team_sel:
                    _raw3 = _raw3[_raw3["Team"].isin(_team_sel)]
                if _stage_sel:
                    _raw3 = _raw3[_raw3["Stage_Clean"].isin(_stage_sel)]
                if _pos_sel:
                    _raw3 = _raw3[_raw3["Position"].isin(_pos_sel)]
                if _ptype_sel != "All":
                    _raw3 = _raw3[_raw3["Player_Type"] == _ptype_sel]

                _p3 = (
                    _raw3.groupby("Player")
                    .agg(
                        WAR3    =("WAR_Total",   "sum"),
                        Sal3    =("Salary_M",    "sum"),
                        Seasons =("Year",        "nunique"),
                        Team    =("Team",        "last"),
                        Stage   =("Stage_Clean", "last"),
                        Position=("Position",    "last"),
                        Age     =("Age",         "last"),
                    )
                    .reset_index()
                )
                _p3 = _p3[(_p3["Sal3"] > 0) & (_p3["WAR3"] >= _pv_min_war)].copy()
                _p3["PVP3"] = _p3["WAR3"] / _p3["Sal3"].clip(lower=0.01)

                if len(_p3) < 5:
                    st.warning("Not enough players with 3-year data under current filters.")
                else:
                    _x3 = _p3["WAR3"].values.astype(float)
                    _y3 = _p3["Sal3"].values.astype(float)
                    if _reg_method == "OLS":
                        import statsmodels.api as _sm3
                        _m3  = _sm3.OLS(_y3, _sm3.add_constant(_x3)).fit()
                        _p3["_pred3"] = _m3.predict(_sm3.add_constant(_x3))
                        _x3seq = np.linspace(_x3.min(), _x3.max(), 200)
                        _y3seq = _m3.params[0] + _m3.params[1] * _x3seq
                        _cel_lbl = f"Cost Effective Line  R\u00b2={_m3.rsquared:.3f}"
                    elif _reg_method == "LOESS":
                        _lo3 = _sm_lowess(_y3, _x3, frac=0.75, return_sorted=True)
                        _p3["_pred3"] = np.interp(_x3, _lo3[:, 0], _lo3[:, 1])
                        _x3seq, _y3seq = _lo3[:, 0], _lo3[:, 1]
                        _cel_lbl = "Cost Effective Line (LOESS)"
                    else:
                        _cf3 = np.polyfit(_x3, _y3, 2)
                        _pf3 = np.poly1d(_cf3)
                        _p3["_pred3"] = _pf3(_x3)
                        _x3seq = np.linspace(_x3.min(), _x3.max(), 200)
                        _y3seq = _pf3(_x3seq)
                        _cel_lbl = "Cost Effective Line (Poly)"

                    _p3["_resid3"] = _p3["Sal3"] - _p3["_pred3"]
                    _p3["_above3"] = _p3["_resid3"] <= 0

                    _p3_hover = _p3.apply(lambda r: (
                        f"<b>{r['Player']}</b><br>"
                        + f"{r['Team']}  ({int(r['Seasons'])} seasons)<br>"
                        + f"3yr WAR: {r['WAR3']:.1f}  |  3yr Salary: ${r['Sal3']:.1f}M<br>"
                        + f"PVP3: {r['PVP3']:.3f}  |  {r['Stage']}"
                    ), axis=1)

                    _clrs3 = ["#22c55e" if v else "#ef4444" for v in _p3["_above3"]]
                    fig_p3 = go.Figure()
                    fig_p3.add_trace(go.Scatter(
                        x=_p3["WAR3"], y=_p3["Sal3"],
                        mode="markers",
                        marker=dict(color=_clrs3, size=8, opacity=0.8,
                                    line=dict(color="rgba(0,0,0,0)", width=0)),
                        text=_p3_hover, hovertemplate="%{text}<extra></extra>",
                        name="Players", showlegend=False,
                    ))
                    fig_p3.add_trace(go.Scatter(
                        x=_x3seq, y=_y3seq, mode="lines",
                        line=dict(color="#f4a261", width=2.5),
                        name=_cel_lbl,
                    ))
                    fig_p3.update_layout(**_pt(
                        title=f"Cost Effective Line \u2014 3-Year Cumulative WAR vs Salary  ({_last3[0]}\u2013{_last3[-1]})",
                        xaxis=dict(title="3-Year Total WAR"),
                        yaxis=dict(title="3-Year Total Salary ($M)"),
                        height=640, showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                        font=dict(color="#dbeafe", size=12), namelength=-1),
                    ))
                    st.plotly_chart(fig_p3, width="stretch")

                    _p3_ranked = (
                        _p3.nlargest(_pv_n, "PVP3") if _pv_sort.startswith("Best")
                        else _p3.nsmallest(_pv_n, "PVP3")
                    ).sort_values("PVP3", ascending=_pv_sort.startswith("Worst"))
                    _p3_ranked["_lbl"] = (
                        _p3_ranked["Player"] + "  (" + _p3_ranked["Team"] + ")"
                    )
                    fig_bar3 = go.Figure(go.Bar(
                        y=_p3_ranked["_lbl"], x=_p3_ranked["PVP3"], orientation="h",
                        marker=dict(color=["#22c55e" if v else "#ef4444"
                                           for v in (_p3_ranked["_above3"])]),
                        text=_p3_ranked["PVP3"].round(3).astype(str),
                        textposition="outside", textfont=dict(color="#dbeafe", size=9),
                        hovertemplate=(
                            "%{y}<br>PVP3: %{x:.3f}<br>"
                            "3yr WAR: " + _p3_ranked["WAR3"].round(1).astype(str)
                            + "  3yr Salary: $" + _p3_ranked["Sal3"].round(1).astype(str) + "M"
                            + "<extra></extra>"
                        ).tolist(),
                    ))
                    fig_bar3.update_layout(**_pt(
                        title="PVP3 Ranking (3-Year WAR / 3-Year Salary $M)",
                        xaxis=dict(title="PVP3"), yaxis=dict(autorange="reversed"),
                        height=max(350, _pv_n * 22), margin=dict(l=230, r=60, t=40, b=40),
                    ))
                    st.plotly_chart(fig_bar3, width="stretch")
                    _lg_pvp3 = _p3["WAR3"].sum() / _p3["Sal3"].sum()
                    st.caption(
                        f"Seasons used: **{', '.join(str(y) for y in _last3)}** "
                        f"({len(_p3):,} players with actual contract data)  |  "
                        f"League-average PPEL3: **{_lg_pvp3:.3f}** WAR per $M"
                    )

            else:  # PPEL5 -- last 5 years, actual contract data only
                _all_yrs_sorted5 = sorted(raw["Year"].dropna().unique())
                _last5 = _all_yrs_sorted5[-5:] if len(_all_yrs_sorted5) >= 5 else _all_yrs_sorted5

                _raw5 = raw[raw["Year"].isin(_last5) & (raw["Salary_M"] > 0)].copy()
                if _league_sel != "Both Leagues":
                    _raw5 = _raw5[_raw5["League"] == _league_sel]
                if "All Teams" not in _team_sel and _team_sel:
                    _raw5 = _raw5[_raw5["Team"].isin(_team_sel)]
                if _stage_sel:
                    _raw5 = _raw5[_raw5["Stage_Clean"].isin(_stage_sel)]
                if _pos_sel:
                    _raw5 = _raw5[_raw5["Position"].isin(_pos_sel)]
                if _ptype_sel != "All":
                    _raw5 = _raw5[_raw5["Player_Type"] == _ptype_sel]

                _p5 = (
                    _raw5.groupby("Player")
                    .agg(
                        WAR5    =("WAR_Total",   "sum"),
                        Sal5    =("Salary_M",    "sum"),
                        Seasons =("Year",        "nunique"),
                        Team    =("Team",        "last"),
                        Stage   =("Stage_Clean", "last"),
                    )
                    .reset_index()
                )
                _p5 = _p5[(_p5["Sal5"] > 0) & (_p5["WAR5"] >= _pv_min_war)].copy()
                _p5["PVP5"] = _p5["WAR5"] / _p5["Sal5"].clip(lower=0.01)

                if len(_p5) < 5:
                    st.warning("Not enough players with 5-year data under current filters.")
                else:
                    _x5 = _p5["WAR5"].values.astype(float)
                    _y5 = _p5["Sal5"].values.astype(float)
                    if _reg_method == "OLS":
                        import statsmodels.api as _sm5
                        _m5  = _sm5.OLS(_y5, _sm5.add_constant(_x5)).fit()
                        _p5["_pred5"] = _m5.predict(_sm5.add_constant(_x5))
                        _x5seq = np.linspace(_x5.min(), _x5.max(), 200)
                        _y5seq = _m5.params[0] + _m5.params[1] * _x5seq
                        _cel5_lbl = f"PPEL5 (OLS \u00b7 R\u00b2={_m5.rsquared:.3f})"
                    elif _reg_method == "LOESS":
                        _lo5 = _sm_lowess(_y5, _x5, frac=0.75, return_sorted=True)
                        _p5["_pred5"] = np.interp(_x5, _lo5[:, 0], _lo5[:, 1])
                        _x5seq, _y5seq = _lo5[:, 0], _lo5[:, 1]
                        _cel5_lbl = "PPEL5 (LOESS)"
                    else:
                        _cf5 = np.polyfit(_x5, _y5, 2)
                        _pf5 = np.poly1d(_cf5)
                        _p5["_pred5"] = _pf5(_x5)
                        _x5seq = np.linspace(_x5.min(), _x5.max(), 200)
                        _y5seq = _pf5(_x5seq)
                        _cel5_lbl = "PPEL5 (Poly)"

                    _p5["_resid5"] = _p5["Sal5"] - _p5["_pred5"]
                    _p5["_above5"] = _p5["_resid5"] <= 0

                    _p5_hover = _p5.apply(lambda r: (
                        f"<b>{r['Player']}</b><br>"
                        + f"{r['Team']}  ({int(r['Seasons'])} seasons)<br>"
                        + f"5yr WAR: {r['WAR5']:.1f}  |  5yr Salary: ${r['Sal5']:.1f}M<br>"
                        + f"PPEL5: {r['PVP5']:.3f}  |  {r['Stage']}"
                    ), axis=1)

                    _clrs5 = ["#22c55e" if v else "#ef4444" for v in _p5["_above5"]]
                    fig_p5 = go.Figure()
                    fig_p5.add_trace(go.Scatter(
                        x=_p5["WAR5"], y=_p5["Sal5"],
                        mode="markers",
                        marker=dict(color=_clrs5, size=8, opacity=0.8,
                                    line=dict(color="rgba(0,0,0,0)", width=0)),
                        text=_p5_hover, hovertemplate="%{text}<extra></extra>",
                        name="Players", showlegend=False,
                    ))
                    fig_p5.add_trace(go.Scatter(
                        x=_x5seq, y=_y5seq, mode="lines",
                        line=dict(color="#f4a261", width=2.5),
                        name=_cel5_lbl,
                    ))
                    fig_p5.update_layout(**_pt(
                        title=f"PPEL5 \u2014 5-Year Cumulative WAR vs Salary  ({_last5[0]}\u2013{_last5[-1]})",
                        xaxis=dict(title="5-Year Total WAR"),
                        yaxis=dict(title="5-Year Total Salary ($M)"),
                        height=640, showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hoverlabel=dict(bgcolor=C.bg_card_surface, bordercolor=C.border_accent,
                                        font=dict(color=C.text_primary, size=12), namelength=-1),
                    ))
                    st.plotly_chart(fig_p5, width="stretch")

                    _p5_ranked = (
                        _p5.nlargest(_pv_n, "PVP5") if _pv_sort.startswith("Best")
                        else _p5.nsmallest(_pv_n, "PVP5")
                    ).sort_values("PVP5", ascending=_pv_sort.startswith("Worst"))
                    _p5_ranked["_lbl"] = (
                        _p5_ranked["Player"] + "  (" + _p5_ranked["Team"] + ")"
                    )
                    fig_bar5 = go.Figure(go.Bar(
                        y=_p5_ranked["_lbl"], x=_p5_ranked["PVP5"], orientation="h",
                        marker=dict(color=["#22c55e" if v else "#ef4444"
                                           for v in (_p5_ranked["_above5"])]),
                        text=_p5_ranked["PVP5"].round(3).astype(str),
                        textposition="outside", textfont=dict(color=C.text_primary, size=9),
                        hovertemplate=(
                            "%{y}<br>PPEL5: %{x:.3f}<br>"
                            "5yr WAR: " + _p5_ranked["WAR5"].round(1).astype(str)
                            + "  5yr Salary: $" + _p5_ranked["Sal5"].round(1).astype(str) + "M"
                            + "<extra></extra>"
                        ).tolist(),
                    ))
                    fig_bar5.update_layout(**_pt(
                        title="PPEL5 Ranking (5-Year WAR / 5-Year Salary $M)",
                        xaxis=dict(title="PPEL5"), yaxis=dict(autorange="reversed"),
                        height=max(350, _pv_n * 22), margin=dict(l=230, r=60, t=40, b=40),
                    ))
                    st.plotly_chart(fig_bar5, width="stretch")
                    _lg_pvp5 = _p5["WAR5"].sum() / _p5["Sal5"].sum()
                    st.caption(
                        f"Seasons used: **{', '.join(str(y) for y in _last5)}** "
                        f"({len(_p5):,} players with 5-year contract data)  |  "
                        f"League-average PPEL5: **{_lg_pvp5:.3f}** WAR per $M"
                    )

        # -- Tab 7 -- Pre-Arb Explorer -----------------------------------------
        with t7:
            _render_glossary([
                ("Pre-Arb", "Pre-Arbitration",
                 "Players in their first 1\u20133 years of service time earning near the league minimum "
                 "($740K\u2013$780K). They often deliver elite WAR per dollar."),
                ("WAR Trajectory", "Year-over-Year WAR Change",
                 "How a player's WAR evolves from season to season. Rising = development; "
                 "falling = regression or injury impact."),
                ("SD Threshold", "Classification Threshold",
                 "Players whose average annual WAR change exceeds \u00b1(threshold \u00d7 global SD) "
                 "are labelled Improving or Declining; the rest are Neutral."),
            ], title="\U0001f4d6 Pre-Arb Explorer", cols=3)

            _pa_c1, _pa_c2, _pa_c3 = st.columns([1, 1, 1])
            with _pa_c1:
                _pa_min_seasons = st.slider("Min seasons", 2, 5, 2, key="ef_pa_min_seasons")
            with _pa_c2:
                _pa_sd_thresh = st.slider("Trend SD threshold", 0.25, 1.5, 0.5, step=0.05,
                                          key="ef_pa_sd")
            with _pa_c3:
                _pa_ptype = st.selectbox("Player type", ["All", "Position Player", "Pitcher"],
                                         key="ef_pa_ptype")

            # Build pre-arb subset from the full raw dataset (all years, ignoring year filter)
            _pa_raw = raw[raw["Stage_Clean"] == "Pre-Arb"].copy()
            if _pa_ptype != "All":
                _pa_raw = _pa_raw[_pa_raw["Player_Type"] == _pa_ptype]
            if _league_sel != "Both Leagues":
                _pa_raw = _pa_raw[_pa_raw["League"] == _league_sel]
            if "All Teams" not in _team_sel and _team_sel:
                _pa_raw = _pa_raw[_pa_raw["Team"].isin(_team_sel)]

            _pa_counts  = _pa_raw.groupby("Player")["Year"].nunique()
            _pa_players = _pa_counts[_pa_counts >= _pa_min_seasons].index
            _pa_df      = (_pa_raw[_pa_raw["Player"].isin(_pa_players)]
                           .sort_values(["Player", "Year"]).copy())

            if len(_pa_df) < 3:
                st.info("No Pre-Arbitration players with enough seasons under current filters.")
            else:
                _pa_df["WAR_delta"] = _pa_df.groupby("Player")["WAR_Total"].diff()

                _pa_summary = (
                    _pa_df.groupby("Player")
                    .agg(
                        avg_delta=("WAR_delta",  "mean"),
                        seasons  =("Year",       "nunique"),
                        last_team=("Team",       "last"),
                        last_war =("WAR_Total",  "last"),
                        last_sal =("Salary_M",   "last"),
                        last_age =("Age",        "last"),
                        total_war=("WAR_Total",  "sum"),
                    )
                    .reset_index()
                    .dropna(subset=["avg_delta"])
                )

                _pa_sd_global = float(_pa_summary["avg_delta"].std()) or 1.0
                _pa_summary["Trend"] = _pa_summary["avg_delta"].apply(
                    lambda d: "Improving" if d > _pa_sd_thresh * _pa_sd_global
                    else ("Declining" if d < -_pa_sd_thresh * _pa_sd_global else "Neutral")
                )
                _TREND_COLORS = {
                    "Improving": "#22c55e",
                    "Neutral":   "#fbbf24",
                    "Declining": "#ef4444",
                }

                _pa_n_imp = int((_pa_summary["Trend"] == "Improving").sum())
                _pa_n_neu = int((_pa_summary["Trend"] == "Neutral").sum())
                _pa_n_dec = int((_pa_summary["Trend"] == "Declining").sum())
                _pa_n_tot = len(_pa_summary)

                st.markdown(f"""<div style="background:#090f1a;border:1px solid #1e3a5c;
border-radius:10px;padding:12px 16px;margin-bottom:12px;
display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
  <div style="text-align:center;">
    <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;">Pre-Arb Players</div>
    <div style="font-size:20px;font-weight:700;color:#e8f4ff;">{_pa_n_tot}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;">Improving</div>
    <div style="font-size:20px;font-weight:700;color:#22c55e;">{_pa_n_imp}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;">Neutral</div>
    <div style="font-size:20px;font-weight:700;color:#fbbf24;">{_pa_n_neu}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:9px;color:#4a7a9b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;">Declining</div>
    <div style="font-size:20px;font-weight:700;color:#ef4444;">{_pa_n_dec}</div>
  </div>
</div>""", unsafe_allow_html=True)

                # -- Player search filter --------------------------------------
                _pa_search_col, _pa_reset_col = st.columns([3, 1])
                with _pa_search_col:
                    _pa_all_players = sorted(_pa_summary["Player"].unique())
                    _pa_player_sel = st.multiselect(
                        "Filter by player(s)", _pa_all_players,
                        placeholder="Type or select player names\u2026",
                        key="ef_pa_player_sel",
                    )
                with _pa_reset_col:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                    if st.button("\u21ba  Reset", use_container_width=True, key="ef_pa_player_reset"):
                        if "ef_pa_player_sel" in st.session_state:
                            del st.session_state["ef_pa_player_sel"]
                        st.rerun()

                if _pa_player_sel:
                    _pa_df      = _pa_df[_pa_df["Player"].isin(_pa_player_sel)]
                    _pa_summary = _pa_summary[_pa_summary["Player"].isin(_pa_player_sel)]

                # WAR trajectory lines, one per player, coloured by trend
                fig_pa = go.Figure()
                _trend_map = dict(zip(_pa_summary["Player"], _pa_summary["Trend"]))
                _legend_shown = set()
                for _pname, _pgrp in _pa_df.groupby("Player"):
                    _trend = _trend_map.get(_pname, "Neutral")
                    _t_color = _TREND_COLORS.get(_trend, "#fbbf24")
                    _pg = _pgrp.sort_values("Year")
                    _show_leg = _trend not in _legend_shown
                    if _show_leg:
                        _legend_shown.add(_trend)
                    fig_pa.add_trace(go.Scatter(
                        x=_pg["Year"].astype(int), y=_pg["WAR_Total"],
                        mode="lines+markers",
                        name=_trend,
                        legendgroup=_trend,
                        showlegend=_show_leg,
                        line=dict(color=_t_color, width=1.5),
                        marker=dict(size=6, color=_t_color),
                        hovertemplate=(
                            f"<b>{_pname}</b><br>"
                            "Year: %{x}<br>WAR: %{y:.1f}"
                            f"<br>Team: {_pg['Team'].iloc[-1]}"
                            "<extra></extra>"
                        ),
                        opacity=0.65,
                    ))
                fig_pa.update_layout(**_pt(
                    title="Pre-Arbitration Player WAR Trajectories",
                    xaxis=dict(title="Season", tickformat="d"),
                    yaxis=dict(title="WAR"),
                    height=580, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                ))
                st.plotly_chart(fig_pa, width="stretch")
                if st.button("\u21ba  Reset Legend Filter", key="ef_pa_reset_legend"):
                    st.rerun()

                # Ranked summary table
                _pa_disp = (
                    _pa_summary
                    .sort_values("avg_delta", ascending=False)
                    .rename(columns={
                        "avg_delta": "Avg WAR \u0394/yr", "seasons":   "Seasons",
                        "last_team": "Last Team",    "last_war":  "Last WAR",
                        "last_sal":  "Last Sal ($M)", "last_age": "Age",
                        "total_war": "Total WAR",
                    })
                )

                def _clr_trend(val):
                    if val == "Improving": return "background-color:#14532d;color:white"
                    if val == "Declining": return "background-color:#7f1d1d;color:white"
                    return "background-color:#713f12;color:white"

                st.dataframe(
                    _pa_disp[["Player", "Last Team", "Trend", "Age", "Seasons",
                               "Avg WAR \u0394/yr", "Last WAR", "Total WAR", "Last Sal ($M)"]]
                    .style
                    .format({"Avg WAR \u0394/yr": "{:+.2f}", "Last WAR": "{:.1f}",
                             "Total WAR": "{:.1f}", "Last Sal ($M)": "${:.2f}M",
                             "Age": "{:.0f}"}, na_rep="\u2014")
                    .map(_clr_trend, subset=["Trend"]),
                    use_container_width=True, hide_index=True, height=410,
                )
                st.caption(
                    f"SD threshold = {_pa_sd_thresh:.2f} \u00d7 global SD ({_pa_sd_global:.2f} WAR/yr).  "
                    f"Improving: avg \u0394 > +{_pa_sd_thresh * _pa_sd_global:.2f}  |  "
                    f"Declining: avg \u0394 < \u2212{_pa_sd_thresh * _pa_sd_global:.2f}"
                )

        # -- Tab 8 -- WAR Stability (WSR) --------------------------------------
        with t8:
            st.markdown(
                "<div style='background:#090f1a;border:1px solid #1e3a5c;border-radius:10px;"
                "padding:14px 18px;margin-bottom:12px;'>"
                f"<div style='font-size:1rem;font-weight:700;color:{C.text_primary};margin-bottom:6px;'>fWAR Stability Rating (WSR)</div>"
                f"<div style='font-size:0.85rem;color:{C.text_secondary};line-height:1.7;'>"
                "Not all high-fWAR players are equally reliable. This chart plots each player's "
                "<b>average fWAR</b> (X-axis) against their <b>standard deviation</b> (Y-axis) \u2014 "
                "a measure of how much their production swings from year to year.<br><br>"
                "<b>How to read it:</b> Bottom-right = <span style='color:#22c55e;font-weight:600;'>Cornerstone</span> "
                "(high production, low variance \u2014 the most reliable stars). "
                "Top-right = <span style='color:#ef4444;font-weight:600;'>Star but Risky</span> "
                "(high ceiling but unpredictable). Bottom-left = consistent depth pieces. "
                "Top-left = volatile fringe players.<br><br>"
                "<b>Standard Deviation (SD)</b> measures how spread out a player's year-to-year fWAR values are. "
                "An SD of 0.5 means very consistent; SD of 2.0+ means huge swings between great and poor seasons. "
                "<b>WSR</b> = average fWAR \u00f7 (1 + SD) \u2014 higher WSR means more reliable, bankable production."
                "</div></div>",
                unsafe_allow_html=True,
            )

            _wsr_min_pa = st.slider("Min PA (hitters) / IP (pitchers)", 50, 300, 150,
                                     key="v2_war_stability_min_pa")

            # Recompute WSR with user threshold
            _wsr_raw = raw.copy()
            for _nc in ["WAR_Total", "PA", "IP"]:
                if _nc in _wsr_raw.columns:
                    _wsr_raw[_nc] = pd.to_numeric(_wsr_raw[_nc], errors="coerce")

            _is_pit = _wsr_raw["Position"].isin(["SP", "RP", "P", "TWP"])
            _wsr_qual = _wsr_raw[
                (_is_pit & (_wsr_raw["IP"].fillna(0) >= _wsr_min_pa)) |
                (~_is_pit & (_wsr_raw["PA"].fillna(0) >= _wsr_min_pa))
            ]
            _wsr_grp = _wsr_qual.groupby("Player").agg(
                WAR_Mean=("WAR_Total", "mean"),
                WAR_Std=("WAR_Total", "std"),
                Seasons=("Year", "nunique"),
                Team=("Team", "last"),
            ).reset_index()
            _wsr_grp = _wsr_grp[_wsr_grp["Seasons"] >= 2].copy()
            _wsr_grp["WAR_Std"] = _wsr_grp["WAR_Std"].fillna(0)
            _wsr_grp["WSR"] = (_wsr_grp["WAR_Mean"] / (1 + _wsr_grp["WAR_Std"])).round(3)

            def _wsr_tier(v):
                if v >= 3.5: return "Elite"
                if v >= 2.0: return "Reliable"
                if v >= 1.0: return "Volatile"
                return "Unstable"
            _wsr_grp["Tier"] = _wsr_grp["WSR"].apply(_wsr_tier)

            _WSR_COLORS = {"Elite": "#22c55e", "Reliable": "#14b8a6",
                           "Volatile": "#f59e0b", "Unstable": "#ef4444"}

            if not _wsr_grp.empty:
                _wsr_hover = _wsr_grp.apply(lambda r: (
                    f"<b>{r['Player']}</b><br>"
                    + f"{r['Team']} \u00b7 {int(r['Seasons'])} seasons<br>"
                    + f"Mean WAR: {r['WAR_Mean']:.2f} \u00b7 Std: {r['WAR_Std']:.2f}<br>"
                    + f"WSR: {r['WSR']:.3f} \u00b7 {r['Tier']}"
                ), axis=1)

                fig_wsr = go.Figure()
                for tier, color in _WSR_COLORS.items():
                    mask = _wsr_grp["Tier"] == tier
                    if mask.any():
                        sub = _wsr_grp[mask]
                        fig_wsr.add_trace(go.Scatter(
                            x=sub["WAR_Mean"], y=sub["WAR_Std"],
                            mode="markers", name=tier,
                            marker=dict(color=color, size=8, opacity=0.8),
                            text=_wsr_hover[mask],
                            hovertemplate="%{text}<extra></extra>",
                        ))

                _xm = float(_wsr_grp["WAR_Mean"].median())
                _ym = float(_wsr_grp["WAR_Std"].median())
                for _ax, _ay, _atxt in [
                    (_xm * 1.8, _ym * 1.8, "Star but Risky"),
                    (_xm * 0.3, _ym * 1.8, "Fringe Volatile"),
                    (_xm * 1.8, _ym * 0.3, "Cornerstone"),
                    (_xm * 0.3, _ym * 0.3, "Consistent Depth"),
                ]:
                    fig_wsr.add_annotation(x=_ax, y=_ay, text=_atxt,
                                           showarrow=False, font=dict(color=C.text_dim, size=10), opacity=0.6)

                fig_wsr.update_layout(**_pt(
                    title="WAR Stability \u2014 Mean WAR vs Standard Deviation",
                    xaxis=dict(title="Mean WAR (2021-2025)"),
                    yaxis=dict(title="WAR Std Dev"),
                    height=580, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hoverlabel=dict(bgcolor="#0d1f38", bordercolor="#1e3a5f",
                                    font=dict(color="#dbeafe", size=12)),
                ))
                st.plotly_chart(fig_wsr, width="stretch")
            else:
                st.info("No players meet the minimum qualifying threshold.")


# ---------------------------------------------------------------------------
# Helper: local razzball cache (mirrors the one in streamlit_app.py)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_razzball(razzball_path_arg: str) -> pd.DataFrame:
    """Load razzball MLBAM ID lookup table (local file or R2 URL)."""
    if not razzball_path_arg.startswith("http") and not os.path.exists(razzball_path_arg):
        return pd.DataFrame()
    try:
        df = read_csv(razzball_path_arg, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()
