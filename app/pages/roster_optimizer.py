"""MLB Toolbox -- Roster Optimizer page (extracted from streamlit_app.py).

Contains:
  - Team Planner (offseason planner with payroll charts, moves, contracts)
  - General Optimizer (archetype-based roster optimizer with 6 tabs)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- src imports ---
from src.projections import make_projections
from src.archetypes import build_archetype_definitions, assign_archetypes
from src.optimizer import run_optimizer
from src.simulation import monte_carlo
from src.backtest import run_backtest
from src.diagnostics import budget_frontier, marginal_analysis
from src.artifacts import write_run_artifacts
from src.team_mode import get_all_teams, get_team_payroll_history, build_offseason_scenario
from src.depth_chart import get_depth_chart_dir

# --- utils imports ---
from utils.constants import (
    ELIGIBLE_SLOTS_MAP as _ELIGIBLE_SLOTS_MAP,
    ROSTER_TEMPLATE as _ROSTER_TEMPLATE,
    OPTIONAL_SLOTS as _OPTIONAL_SLOTS,
    PG_CHART_COLORS as _PG_CHART_COLORS,
)
from utils.theme import plotly_theme as _pt
from utils.data_loading import (
    ROOT_DIR as _ROOT_DIR,
    DEFAULT_CONFIG as _DEFAULT_CONFIG,
    load_base_config as _load_base_config,
    resolve_data_path as _resolve_data_path,
    file_hash as _file_hash,
    dir_hash as _dir_hash,
    read_csv as _read_csv,
    cached_projections as _cached_projections,
    cached_archetypes as _cached_archetypes,
    cached_wins as _cached_wins,
    cached_payroll_history as _cached_payroll_history,
    cached_team_scenario as _cached_team_scenario,
    cached_2026_payroll as _cached_2026_payroll,
    cached_player_history as _cached_player_history,
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render(base_cfg: dict | None = None):
    """Roster Optimizer page: sub-tabs for Team Planner and General Optimizer."""
    st.markdown("### Roster Optimizer")

    t_planner, t_optimizer = st.tabs(["Team Planner", "General Optimizer"])

    with t_planner:
        _render_team_planner(base_cfg)

    with t_optimizer:
        _render_optimizer_page()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arch_label(arch_id: str) -> str:
    """Convert 'SP_FA_Elite' -> 'Elite Starter (Free Agent)'."""
    _pos = {
        "SP": "Starter",  "RP": "Reliever",  "C": "Catcher",
        "1B": "1st Base", "2B": "2nd Base",  "3B": "3rd Base", "SS": "Shortstop",
        "CI": "Corner IF","MI": "Middle IF",  "CF": "Center Field",
        "OF": "Corner OF","DH": "DH",
    }
    _stage = {"FA": "Free Agent", "Arb": "Arb-Eligible", "Pre-Arb": "Pre-Arb"}
    parts = arch_id.split("_", 2)
    if len(parts) != 3:
        return arch_id
    pos_str   = _pos.get(parts[0], parts[0])
    stage_str = _stage.get(parts[1], parts[1])
    tier_str  = parts[2]
    return f"{tier_str} {pos_str} ({stage_str})"


# ---------------------------------------------------------------------------
# Inline settings expander for the General Optimizer
# ---------------------------------------------------------------------------

def _build_inline_settings(base_cfg: dict) -> tuple[dict, bool]:
    """Render inline optimizer settings (expander) and return (cfg, run_clicked)."""
    cfg = copy.deepcopy(base_cfg)

    with st.expander("Settings", expanded=False):
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown("**Budget & Market**")
            cfg["budget_M"] = st.slider(
                "Budget ($M)", min_value=40, max_value=400,
                value=int(cfg.get("budget_M", 130)), step=5,
                key="opt_budget",
            )
            cfg["market_dpw_M"] = st.slider(
                "Market $/WAR ($M)", min_value=3.0, max_value=12.0,
                value=float(cfg.get("market_dpw_M", 5.5)), step=0.5,
                key="opt_mkt",
            )
            st.markdown("**Player Filters**")
            cfg["min_war_threshold"] = st.slider(
                "Min WAR threshold", -2.0, 4.0,
                float(cfg.get("min_war_threshold", -99)), 0.5,
                key="opt_minwar",
            )

        with s2:
            st.markdown("**Roster Structure**")
            cfg["roster_slots"]["SP"] = st.slider(
                "SP slots", min_value=3, max_value=6,
                value=int(cfg["roster_slots"].get("SP", 5)),
                key="opt_sp",
            )
            if st.checkbox(
                "Include DH slot",
                value=("DH" in cfg["roster_slots"] and cfg["roster_slots"]["DH"] > 0),
                key="opt_dh",
            ):
                cfg["roster_slots"]["DH"] = 1
            else:
                cfg["roster_slots"].pop("DH", None)

            st.markdown("**Optimizer Mode**")
            cfg["optimizer_mode"] = st.selectbox(
                "Mode", ["archetype", "player"],
                index=0 if cfg.get("optimizer_mode", "archetype") == "archetype" else 1,
                key="opt_mode",
            )

        with s3:
            st.markdown("**Objective Weights**")
            ow = cfg.get("objective_weights", {"wins": 1.0, "surplus": 0.0, "risk_penalty": 0.0})
            ow["wins"]         = st.slider("Wins weight",          0.0, 1.0, float(ow.get("wins",         1.0)), 0.05, key="opt_ow_wins")
            ow["surplus"]      = st.slider("Surplus value weight", 0.0, 1.0, float(ow.get("surplus",      0.0)), 0.05, key="opt_ow_surp")
            ow["risk_penalty"] = st.slider("Risk penalty",         0.0, 1.0, float(ow.get("risk_penalty", 0.0)), 0.05, key="opt_ow_risk")
            cfg["objective_weights"] = ow

            st.markdown("**Simulation**")
            cfg["mc_simulations"] = st.selectbox(
                "MC simulations", [100, 500, 1000, 5000, 10000], index=2,
                key="opt_mc",
            )
            cfg["mc_seed"] = st.number_input(
                "Random seed", value=int(cfg.get("mc_seed", 42)), step=1,
                key="opt_seed",
            )

    run_clicked = st.button("Run Optimizer", type="primary", key="opt_run_btn")
    return cfg, run_clicked


# ---------------------------------------------------------------------------
# General Optimizer page (6-tab layout)
# ---------------------------------------------------------------------------

def _render_optimizer_page():
    """Render the General Optimizer (original app logic, 6 tabs)."""
    import io as _io

    st.markdown("### General Optimizer")

    if not os.path.exists(_DEFAULT_CONFIG):
        st.error(f"Config file not found: {_DEFAULT_CONFIG}\nRun from mlb_optimizer/ root.")
        return

    base_cfg = _load_base_config(_DEFAULT_CONFIG)

    salary_path = _resolve_data_path(base_cfg["raw_salary_war_path"], _DEFAULT_CONFIG)
    wins_path   = _resolve_data_path(base_cfg["raw_wins_path"],       _DEFAULT_CONFIG)

    if not os.path.exists(salary_path):
        st.error(
            f"Salary/WAR data file not found:\n`{salary_path}`\n\n"
            "Update `raw_salary_war_path` in `configs/default_config.json`."
        )
        return

    cfg, run_btn = _build_inline_settings(base_cfg)

    salary_hash       = _file_hash(salary_path)
    proj_weights_json = json.dumps(base_cfg["projection_weights"], sort_keys=True)

    proj_df, raw_df = _cached_projections(
        salary_path, salary_hash, proj_weights_json,
        int(base_cfg["season"]),
        bool(base_cfg.get("clip_negative_war", True)),
        float(base_cfg.get("min_war_threshold", -99)),
        int(base_cfg.get("max_contract_years", 99)),
    )

    min_war_live = float(cfg.get("min_war_threshold", -99))
    proj_df_live = proj_df[proj_df["proj_WAR"] >= min_war_live].copy()

    proj_hash      = hashlib.md5(proj_df_live.to_csv(index=False).encode()).hexdigest()
    proj_json      = proj_df_live.to_json(orient="records")
    arch_df, proj_with_arch = _cached_archetypes(proj_hash, proj_json)

    wins_hash = _file_hash(wins_path)
    wins_df   = _cached_wins(wins_path, wins_hash)

    bt_cfg    = {"wins_intercept": float(cfg.get("wins_intercept", 48.0))}
    bt_result = run_backtest(raw_df, wins_df, bt_cfg)
    adj       = bt_result.adjustment_factor

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_cfg" not in st.session_state:
        st.session_state["last_cfg"] = None

    if run_btn or st.session_state["last_result"] is None:
        with st.spinner("Optimizing ..."):
            opt_result = run_optimizer(arch_df, cfg, projected_df=proj_with_arch)
        st.session_state["last_result"] = opt_result
        st.session_state["last_cfg"]    = copy.deepcopy(cfg)

    opt_result = st.session_state["last_result"]
    run_cfg    = st.session_state["last_cfg"] or cfg

    sim_result = monte_carlo(opt_result.roster_df, run_cfg, backtest_adjustment=adj)

    # --- Tabs ---
    tab_roster, tab_arch, tab_wins, tab_frontier, tab_diag, tab_export = st.tabs([
        "Roster", "Archetypes", "Win Dist", "Frontier", "Diagnostics", "Export",
    ])

    # ===========================================================
    # Tab 1 -- Roster
    # ===========================================================
    with tab_roster:
        st.subheader("Optimal Roster")
        if opt_result.roster_df.empty:
            st.warning(f"No feasible roster found. Solver status: {opt_result.status}")
        else:
            rdf        = opt_result.roster_df.copy()
            total_war  = rdf["war_mean"].sum()
            total_cost = rdf["cost_mean"].sum()
            budget_val = run_cfg["budget_M"]
            remaining  = budget_val - total_cost

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total proj. WAR",      f"{total_war:.1f}")
            col2.metric("Payroll used ($M)",     f"${total_cost:.1f}M")
            col3.metric("Budget remaining ($M)", f"${remaining:.1f}M")
            col4.metric("Solver",                opt_result.status)

            _rdf_disp = rdf.copy()
            _rdf_disp.insert(1, "Role", _rdf_disp["archetype_id"].apply(_arch_label))
            display_cols = [c for c in ["slot", "Role", "pos_group", "war_mean", "war_sd", "cost_mean", "n_players"]
                            if c in _rdf_disp.columns]
            st.dataframe(
                _rdf_disp[display_cols].rename(columns={
                    "slot": "Slot", "pos_group": "Pos Group",
                    "war_mean": "Proj WAR", "war_sd": "WAR SD",
                    "cost_mean": "Est Cost", "n_players": "# Players",
                }).style.format({
                    "Proj WAR": "{:.2f}",
                    "WAR SD":   "{:.2f}",
                    "Est Cost": "${:.1f}M",
                }),
                use_container_width=True,
            )

            try:
                slot_vals  = rdf["war_mean"].values.tolist()
                slot_labels = [
                    f"{r['slot']} \u00b7 {_arch_label(r['archetype_id'])}"
                    for _, r in rdf.iterrows()
                ]
                vmin, vmax = min(slot_vals), max(slot_vals)
                norm  = [(v - vmin) / max(vmax - vmin, 0.01) for v in slot_vals]
                def _ryg(t):
                    if t < 0.5:
                        r, g = 1.0, t * 2
                    else:
                        r, g = 1.0 - (t - 0.5) * 2, 1.0
                    return f"rgb({int(r*220)},{int(g*200)},60)"
                slot_colors = [_ryg(n) for n in norm]
                y_max = max(slot_vals) * 1.25 if slot_vals else 5
                fig = go.Figure(data=[go.Bar(
                    x=slot_labels, y=slot_vals,
                    marker_color=slot_colors, marker_line_width=0,
                    text=[f"{v:.2f}" for v in slot_vals],
                    textposition="outside", textfont=dict(color="#dbeafe", size=9),
                    hovertemplate="%{x}<br>WAR: %{y:.2f}<extra></extra>",
                )])
                fig.update_layout(**_pt(
                    title=f"Roster WAR by Slot  (${budget_val:.0f}M budget)",
                    yaxis=dict(title="Projected WAR", range=[0, y_max]),
                    xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
                    height=380,
                    margin=dict(l=50, r=20, t=45, b=110),
                ))
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})
            except Exception as e:
                st.info(f"Chart unavailable: {e}")

    # ===========================================================
    # Tab 2 -- Archetypes
    # ===========================================================
    with tab_arch:
        st.subheader("Archetype Mix")
        if opt_result.archetype_mix:
            mix_df = pd.DataFrame(
                [{"Archetype": k, "Slots": v} for k, v in opt_result.archetype_mix.items()]
            ).sort_values("Slots", ascending=False)
            st.dataframe(mix_df, width="stretch")

        st.markdown("---")
        st.subheader("Available Archetypes (gold inputs)")
        disp_arch = arch_df.copy()
        if "eligible_slots" in disp_arch.columns:
            disp_arch["eligible_slots"] = disp_arch["eligible_slots"].apply(
                lambda v: ", ".join(v) if isinstance(v, list) else str(v)
            )
        fmt = {"war_mean": "{:.2f}", "war_sd": "{:.2f}", "cost_mean": "${:.1f}M", "cost_sd": "${:.1f}M"}
        st.dataframe(
            disp_arch.style.format({k: v for k, v in fmt.items() if k in disp_arch.columns}),
            use_container_width=True,
        )

    # ===========================================================
    # Tab 3 -- Win Distribution
    # ===========================================================
    with tab_wins:
        st.subheader("Win Distribution (Monte Carlo)")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean wins",    f"{sim_result.mean_wins:.1f}")
        c2.metric("Median (P50)", f"{sim_result.median_wins:.1f}")
        c3.metric("P10",          f"{sim_result.p10:.1f}")
        c4.metric("P90",          f"{sim_result.p90:.1f}")
        c5.metric("Playoff odds", f"{sim_result.playoff_odds:.1%}")

        try:
            threshold = float(run_cfg.get("playoff_threshold_wins", 88))
            wins_arr  = sim_result.wins_array
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=wins_arr, nbinsx=50,
                marker_color="#2563eb", marker_line_width=0, opacity=0.85,
                name="Simulations",
                hovertemplate="Wins: %{x}<br>Count: %{y}<extra></extra>",
            ))
            for x_val, color, dash, lbl in [
                (sim_result.p10,         "#ef4444", "dash",  f"P10 = {sim_result.p10:.0f}"),
                (sim_result.median_wins, "#dbeafe", "solid", f"P50 = {sim_result.median_wins:.0f}"),
                (sim_result.p90,         "#22c55e", "dash",  f"P90 = {sim_result.p90:.0f}"),
                (threshold,              "#f59e0b", "dot",   f"Playoff >= {threshold:.0f}"),
            ]:
                fig.add_vline(x=x_val, line_color=color, line_dash=dash,
                              line_width=1.8, opacity=0.9,
                              annotation_text=lbl,
                              annotation_font_color=color,
                              annotation_position="top right")
            fig.add_vrect(x0=threshold, x1=float(np.max(wins_arr)) + 5,
                          fillcolor="#22c55e", opacity=0.06, line_width=0)
            fig.update_layout(**_pt(
                title=(f"Win Distribution -- {run_cfg['mc_simulations']:,} sims "
                       f"(playoff odds: {sim_result.playoff_odds:.1%})"),
                xaxis=dict(title="Season Wins"),
                yaxis=dict(title="Simulations"),
                height=380,
                showlegend=False,
            ))
            st.plotly_chart(fig, width="stretch",
                            config={"displayModeBar": False})
        except Exception as e:
            st.info(f"Chart unavailable: {e}")

        if bt_result.n_team_years > 0:
            st.caption(
                f"Backtest: RMSE = {bt_result.rmse:.1f} wins over {bt_result.n_team_years} team-seasons  "
                f"| Bias adjustment applied: {bt_result.adjustment_factor:+.1f} wins"
            )

    # ===========================================================
    # Tab 4 -- Budget Frontier
    # ===========================================================
    with tab_frontier:
        st.subheader("Budget Frontier")

        with st.spinner("Computing frontier (10 budget points) ..."):
            front_df = budget_frontier(arch_df, run_cfg)

        if not front_df.empty and "expected_wins" in front_df.columns:
            fdf = front_df.dropna(subset=["expected_wins"])

            try:
                curr_b    = run_cfg["budget_M"]
                threshold = float(run_cfg.get("playoff_threshold_wins", 88))
                bx = fdf["budget_M"].tolist()
                ew = fdf["expected_wins"].tolist()

                traces = []
                if "p10" in fdf.columns and "p90" in fdf.columns:
                    traces.append(go.Scatter(
                        x=bx + bx[::-1],
                        y=fdf["p90"].tolist() + fdf["p10"].tolist()[::-1],
                        fill="toself", fillcolor="rgba(59,130,246,0.12)",
                        line=dict(width=0), name="P10-P90 band",
                        hoverinfo="skip",
                    ))
                traces.append(go.Scatter(
                    x=bx, y=ew, mode="lines+markers",
                    line=dict(color="#3b82f6", width=2.5),
                    marker=dict(color="#3b82f6", size=7),
                    name="Expected wins",
                    hovertemplate="$%{x:.0f}M -> %{y:.1f} wins<extra></extra>",
                ))
                fig = go.Figure(data=traces)
                fig.add_vline(x=curr_b, line_color="#f59e0b", line_dash="dash",
                              line_width=1.5,
                              annotation_text=f"Current ${curr_b:.0f}M",
                              annotation_font_color="#f59e0b")
                fig.add_hline(y=threshold, line_color="#22c55e", line_dash="dot",
                              line_width=1.5,
                              annotation_text=f"Playoff >= {threshold:.0f}",
                              annotation_font_color="#22c55e")
                fig.update_layout(**_pt(
                    title="Wins vs Budget",
                    xaxis=dict(title="Budget ($M)"),
                    yaxis=dict(title="Expected wins"),
                    showlegend=True,
                    height=380,
                    margin=dict(l=50, r=20, t=45, b=50),
                ))
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})
            except Exception as e:
                st.info(f"Chart unavailable: {e}")

            st.dataframe(
                fdf[["budget_M","expected_wins","p10","p90","playoff_odds","total_cost_M","status"]]
                  .style.format({
                      "budget_M":      "${:.0f}M",
                      "expected_wins": "{:.1f}",
                      "p10":           "{:.1f}",
                      "p90":           "{:.1f}",
                      "playoff_odds":  "{:.1%}",
                      "total_cost_M":  "${:.1f}M",
                  }),
                use_container_width=True,
            )
        else:
            st.warning("Frontier computation returned no results.")

    # ===========================================================
    # Tab 5 -- Diagnostics
    # ===========================================================
    with tab_diag:
        st.subheader("Marginal Analysis")

        with st.spinner("Running marginal analysis ..."):
            upgrades_df, cuts_df = marginal_analysis(opt_result.roster_df, arch_df, run_cfg)

        col_u, col_c = st.columns(2)
        with col_u:
            st.markdown("**Top Upgrades** (best delta-wins / delta-cost)")
            if not upgrades_df.empty:
                st.dataframe(
                    upgrades_df.style.format({
                        "delta_war":        "{:+.2f}",
                        "delta_cost_M":     "${:+.1f}M",
                        "delta_wins":       "{:+.2f}",
                        "delta_wins_per_M": "{:.3f}",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("No upgrades available.")

        with col_c:
            st.markdown("**Top Cuts** (best savings / win cost)")
            if not cuts_df.empty:
                st.dataframe(
                    cuts_df.style.format({
                        "delta_war":    "{:+.2f}",
                        "delta_cost_M": "${:+.1f}M",
                        "savings_M":    "${:.1f}M",
                        "delta_wins":   "{:+.2f}",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("No cuts available.")

        st.markdown("---")
        st.subheader("Backtest Summary")
        if bt_result.n_team_years > 0:
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("RMSE (wins)",         f"{bt_result.rmse:.2f}")
            bc2.metric("Bias (mean residual)", f"{bt_result.bias:+.2f}")
            bc3.metric("Team-seasons",         str(bt_result.n_team_years))
        else:
            st.info("No wins data loaded -- backtest skipped. Check `raw_wins_path` in config.")

    # ===========================================================
    # Tab 6 -- Export
    # ===========================================================
    with tab_export:
        st.subheader("Export Run")

        cfg_str = json.dumps(run_cfg, indent=2)
        st.download_button(
            "Download config.json",
            data=cfg_str,
            file_name="config.json",
            mime="application/json",
        )

        if not opt_result.roster_df.empty:
            import io as _io2
            csv_buf = _io2.StringIO()
            opt_result.roster_df.to_csv(csv_buf, index=False)
            st.download_button(
                "Download roster.csv",
                data=csv_buf.getvalue(),
                file_name="roster.csv",
                mime="text/csv",
            )

        st.markdown("---")
        if st.button("Save full run to runs/ folder", key="export_save_btn"):
            with st.spinner("Writing artifacts ..."):
                from datetime import datetime
                ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = os.path.join(_ROOT_DIR, "runs", f"run_{ts}")
                os.makedirs(run_dir, exist_ok=True)
                diag_bundle = {
                    "frontier_df": front_df if "front_df" in dir() else pd.DataFrame(),
                    "upgrades_df": upgrades_df if "upgrades_df" in dir() else pd.DataFrame(),
                    "cuts_df":     cuts_df     if "cuts_df"     in dir() else pd.DataFrame(),
                    "backtest":    bt_result,
                    "sim_result":  sim_result,
                }
                write_run_artifacts(run_dir, run_cfg, arch_df, opt_result, diag_bundle, {})
            st.success(f"Run saved to: `{run_dir}`")

        st.markdown("---")
        st.subheader("Run Metadata")
        st.json({
            "solver_status":   opt_result.status,
            "objective_value": round(opt_result.objective_value, 4)
                               if opt_result.objective_value == opt_result.objective_value else None,
            "n_archetypes":    len(arch_df),
            "n_players_pool":  len(proj_df_live),
            "budget_M":        run_cfg["budget_M"],
            "optimizer_mode":  run_cfg["optimizer_mode"],
            "mc_simulations":  run_cfg["mc_simulations"],
            "mean_wins":       round(sim_result.mean_wins, 1),
            "playoff_odds":    round(sim_result.playoff_odds, 3),
            "backtest_adj":    round(adj, 2),
        })


# ---------------------------------------------------------------------------
# Team Moves recommendation panel
# ---------------------------------------------------------------------------

def _render_team_moves(
    scenario: dict,
    player_pool: pd.DataFrame,
    selected_team: str,
    available_M: float,
    mkt_rate: float,
) -> None:
    """Primary section: recommended FA signings, trade targets, and cost cuts."""
    st.markdown("---")
    st.markdown("### Recommended Offseason Moves")
    st.caption(
        f"Budget available after committed payroll: **${available_M:.1f}M**  "
        "Free agents can be signed directly. "
        "Arb & Pre-Arb players are under team control -- **acquisition requires a trade**."
    )

    if player_pool.empty:
        st.info("2026 payroll data not available -- cannot generate move recommendations.")
        return

    _on_team: set[str] = set()
    for _df in [
        scenario.get("locked_df",   pd.DataFrame()),
        scenario.get("arb_df",      pd.DataFrame()),
        scenario.get("expiring_df", pd.DataFrame()),
    ]:
        if not _df.empty and "Player" in _df.columns:
            _on_team.update(_df["Player"].tolist())

    pool = player_pool[~player_pool["Player"].isin(_on_team)].copy()

    remaining = scenario.get("remaining_slots", {})
    _open_pgs: set[str] = set()
    for _slot in remaining:
        for _pg, _slots in _ELIGIBLE_SLOTS_MAP.items():
            if _slot in _slots:
                _open_pgs.add(_pg)

    pool["_market_val_M"] = (pool["WAR_Total"].fillna(0) * mkt_rate).round(2)
    pool["_surplus_M"]    = (pool["_market_val_M"] - pool["Salary_M"].fillna(0)).round(2)
    pool["_fills_need"]   = pool["pos_group"].isin(_open_pgs)

    fa_pool    = pool[pool["Stage_Clean"] == "FA"].copy()
    trade_pool = pool[pool["Stage_Clean"].isin(["Arb", "Pre-Arb"])].copy()

    _acq_col   = "Acquisition"
    _show_base = ["Player", "Team", "Position", "Stage_Clean", "Age",
                  "WAR_Total", "Salary_M", "W_per_M", "_surplus_M", _acq_col]
    _fmt_mv = {
        "WAR_Total": "{:.1f}", "Salary_M": "${:.1f}M",
        "W_per_M": "{:.2f}", "_surplus_M": "${:+.1f}M", "Age": "{:.0f}",
    }
    _rename_mv = {
        "WAR_Total": "WAR (2025)", "Salary_M": "2026 Sal $M",
        "W_per_M": "W/$M", "_surplus_M": "Surplus vs Mkt", "Stage_Clean": "Stage",
    }

    tm_t1, tm_t2, tm_t3 = st.tabs([
        "Free Agent Signings",
        "Trade Targets",
        "Cost Cut Candidates",
    ])

    with tm_t1:
        fa_affordable = fa_pool[
            fa_pool["Salary_M"].notna() & (fa_pool["Salary_M"] <= available_M + 0.01)
        ].copy()
        fa_affordable[_acq_col] = "Sign (FA)"

        fa_priority = fa_affordable[fa_affordable["_fills_need"]].sort_values(
            ["W_per_M", "WAR_Total"], ascending=[False, False], na_position="last"
        )
        fa_other = fa_affordable[~fa_affordable["_fills_need"]].sort_values(
            ["W_per_M", "WAR_Total"], ascending=[False, False], na_position="last"
        )
        fa_sorted = pd.concat([fa_priority, fa_other]).drop_duplicates("Player")

        if fa_sorted.empty:
            st.info(f"No free agents found within the ${available_M:.0f}M available budget.")
        else:
            _fa_cols = [c for c in _show_base if c in fa_sorted.columns]
            st.caption(
                f"= fills an open roster position  |  {len(fa_sorted)} players affordable  "
                f"|  Sorted by efficiency (W/$M), needs first."
            )
            _fa_disp = fa_sorted[_fa_cols].rename(columns=_rename_mv).reset_index(drop=True)
            _fa_disp.insert(0, "Need", ["*" if v else "" for v in fa_sorted["_fills_need"].tolist()])
            st.dataframe(
                _fa_disp.style.format({k: v for k, v in {
                    **_fmt_mv,
                    "WAR (2025)": "{:.1f}", "2026 Sal $M": "${:.1f}M",
                    "W/$M": "{:.2f}", "Surplus vs Mkt": "${:+.1f}M", "Age": "{:.0f}",
                }.items() if k in _fa_disp.columns}),
                hide_index=True, use_container_width=True, height=500,
            )

    with tm_t2:
        st.caption(
            "Arb & Pre-Arb players are locked to their teams -- they **cannot** be signed as FAs. "
            "Any acquisition requires a trade. Sorted by projected WAR, needs first."
        )
        trade_priority = trade_pool[trade_pool["_fills_need"]].sort_values(
            "WAR_Total", ascending=False, na_position="last"
        )
        trade_other = trade_pool[~trade_pool["_fills_need"]].sort_values(
            "WAR_Total", ascending=False, na_position="last"
        )
        trade_sorted = pd.concat([trade_priority, trade_other]).drop_duplicates("Player")
        trade_sorted[_acq_col] = "Trade Required"

        if trade_sorted.empty:
            st.info("No Arb/Pre-Arb trade targets found in pool.")
        else:
            _tr_cols = [c for c in _show_base if c in trade_sorted.columns]
            _tr_disp = trade_sorted[_tr_cols].rename(columns=_rename_mv).reset_index(drop=True)
            _tr_disp.insert(0, "Need", ["*" if v else "" for v in trade_sorted["_fills_need"].tolist()])
            st.dataframe(
                _tr_disp.style.format({k: v for k, v in {
                    **_fmt_mv,
                    "WAR (2025)": "{:.1f}", "2026 Sal $M": "${:.1f}M",
                    "W/$M": "{:.2f}", "Surplus vs Mkt": "${:+.1f}M", "Age": "{:.0f}",
                }.items() if k in _tr_disp.columns}),
                hide_index=True, use_container_width=True, height=500,
            )

    with tm_t3:
        st.caption(
            "Current roster players delivering below-market value. "
            "Releasing or trading them frees budget for higher-impact additions."
        )
        _roster_rows: list[pd.DataFrame] = []
        for _df, _src in [
            (scenario.get("locked_df", pd.DataFrame()), "Under Contract"),
            (scenario.get("arb_df",    pd.DataFrame()), "Arb-Eligible"),
        ]:
            if _df.empty:
                continue
            _tmp = _df.copy()
            _tmp["_source"] = _src
            if "sal_2026_M" in _tmp.columns:
                _tmp["Salary_M"] = pd.to_numeric(_tmp["sal_2026_M"], errors="coerce")
            elif "sal_2025_M" in _tmp.columns:
                _tmp["Salary_M"] = pd.to_numeric(_tmp["sal_2025_M"], errors="coerce")
            if "war_2025" in _tmp.columns:
                _tmp["WAR_Total"] = pd.to_numeric(_tmp["war_2025"], errors="coerce")
            _roster_rows.append(_tmp)

        if not _roster_rows:
            st.info("No current roster data available.")
        else:
            _rc_df = pd.concat(_roster_rows, ignore_index=True)
            _rc_df["WAR_Total"] = pd.to_numeric(_rc_df.get("WAR_Total", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            _rc_df["Salary_M"]  = pd.to_numeric(_rc_df.get("Salary_M",  pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            _rc_df["_market_val_M"] = (_rc_df["WAR_Total"] * mkt_rate).round(2)
            _rc_df["_surplus_M"]    = (_rc_df["_market_val_M"] - _rc_df["Salary_M"]).round(2)
            _rc_df["W_per_M"]       = (_rc_df["WAR_Total"] / _rc_df["Salary_M"].clip(lower=0.1)).round(3)

            _cuts = _rc_df.sort_values("_surplus_M").reset_index(drop=True)
            _cut_cols = [c for c in ["Player", "pos_group", "age", "WAR_Total", "Salary_M",
                                     "W_per_M", "_surplus_M", "_source"]
                         if c in _cuts.columns]
            _cut_disp = _cuts[_cut_cols].rename(columns={
                "pos_group": "Position", "age": "Age",
                "WAR_Total": "WAR (2025)", "Salary_M": "2026 Sal $M",
                "W_per_M": "W/$M", "_surplus_M": "Surplus vs Mkt", "_source": "Status",
            }).reset_index(drop=True)

            def _clr_surplus(v):
                try:
                    return "background-color:#4a1a1a;color:white" if float(v) < 0 else ""
                except Exception:
                    return ""

            st.dataframe(
                _cut_disp.style
                    .format({k: v for k, v in {
                        "WAR (2025)": "{:.1f}", "2026 Sal $M": "${:.1f}M",
                        "W/$M": "{:.2f}", "Surplus vs Mkt": "${:+.1f}M", "Age": "{:.0f}",
                    }.items() if k in _cut_disp.columns})
                    .map(_clr_surplus, subset=["Surplus vs Mkt"] if "Surplus vs Mkt" in _cut_disp.columns else []),
                hide_index=True, use_container_width=True, height=500,
            )


# ---------------------------------------------------------------------------
# Hero panel + sticky bar
# ---------------------------------------------------------------------------

def _render_hero_panel(
    selected_team: str,
    budget_M: float,
    committed_M: float,
    available_M: float,
    total_war: float,
    open_slots: int,
    surplus_M: float,
) -> None:
    """Render styled hero metric cards at the top of the Team Planner."""
    est_wins    = 48 + total_war / 1.5
    budget_pct  = committed_M / max(budget_M, 1) * 100
    avail_bdr   = "#22c55e" if available_M >= 20 else "#f59e0b" if available_M >= 10 else "#ef4444"
    surp_bdr    = "#22c55e" if surplus_M >= 0 else "#ef4444"
    risk_lvl    = "Low" if open_slots <= 3 and available_M >= 20 else ("High" if open_slots >= 7 or available_M < 10 else "Mid")
    risk_color  = {"Low": "#22c55e", "Mid": "#f59e0b", "High": "#ef4444"}[risk_lvl]

    def _card(label, val, sub, border="#1e3250", vcolor="#d6e8f8"):
        return (
            f'<div style="background:#18243a;border:1px solid {border};'
            f'border-radius:10px;padding:14px 8px;text-align:center;">'
            f'<div style="font-size:10px;color:#4a687e;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">{label}</div>'
            f'<div style="font-size:26px;font-weight:800;color:{vcolor};line-height:1">{val}</div>'
            f'<div style="font-size:10px;color:#2e4a62;margin-top:4px">{sub}</div></div>'
        )

    html = (
        '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:16px 0 20px 0;">'
        + _card("Est. Wins",  f"{est_wins:.0f}",      "2026 Projection",                  "#253d58",  "#93c5fd")
        + _card("Total WAR",  f"{total_war:.1f}",     "Current Roster")
        + _card("Committed",  f"${committed_M:.0f}M", f"{budget_pct:.0f}% of ${budget_M:.0f}M")
        + _card("Available",  f"${available_M:.0f}M", "To Spend",                          avail_bdr, avail_bdr)
        + _card("Surplus",    f"${surplus_M:+.0f}M",  "vs Market Rate",                   surp_bdr,  surp_bdr)
        + _card("Risk",       risk_lvl,               f"{open_slots} open slot{'s' if open_slots != 1 else ''}", risk_color, risk_color)
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _inject_sticky_bar(
    selected_team: str,
    budget_M: float,
    committed_M: float,
    available_M: float,
    total_war: float,
    open_slots: int,
) -> None:
    """Inject a CSS fixed summary bar visible while the user scrolls."""
    est_wins    = 48 + total_war / 1.5
    avail_color = "#22c55e" if available_M >= 20 else "#f59e0b" if available_M >= 10 else "#ef4444"

    html = f"""<style>
.mlb-sbar{{position:fixed;bottom:0;top:auto;left:0;right:0;z-index:9998;
background:rgba(10,15,24,0.97);border-top:1px solid #1e3250;
padding:5px 24px;display:flex;align-items:center;gap:18px;
backdrop-filter:blur(6px);box-shadow:0 -2px 16px rgba(0,0,0,.4);}}
.mlb-sbar .sb-t{{font-size:14px;font-weight:800;color:#3b82f6;white-space:nowrap}}
.mlb-sbar .sb-i{{display:flex;flex-direction:column;align-items:center;gap:1px}}
.mlb-sbar .sb-l{{font-size:9px;color:#4a687e;text-transform:uppercase;letter-spacing:.7px}}
.mlb-sbar .sb-v{{font-size:13px;font-weight:700;color:#d6e8f8}}
</style>
<div class="mlb-sbar">
<span class="sb-t">{selected_team}</span>
<div class="sb-i"><span class="sb-l">Budget</span><span class="sb-v">${budget_M}M</span></div>
<div class="sb-i"><span class="sb-l">Committed</span><span class="sb-v">${committed_M:.0f}M</span></div>
<div class="sb-i"><span class="sb-l">Available</span><span class="sb-v" style="color:{avail_color}">${available_M:.0f}M</span></div>
<div class="sb-i"><span class="sb-l">WAR</span><span class="sb-v">{total_war:.1f}</span></div>
<div class="sb-i"><span class="sb-l">Est Wins</span><span class="sb-v">{est_wins:.0f}</span></div>
<div class="sb-i"><span class="sb-l">Open Slots</span><span class="sb-v">{open_slots}</span></div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Payroll charts
# ---------------------------------------------------------------------------

def _render_payroll_charts(
    depth_df: "pd.DataFrame | None",
    scenario: dict,
    budget_M: float,
    mkt_rate: float,
) -> None:
    """Payroll donut, WAR-by-position bar, and committed-vs-available budget bar."""
    col_l, col_r = st.columns(2)

    with col_l:
        if depth_df is not None and not depth_df.empty and "sal_2026_M" in depth_df.columns:
            def _broad(pg):
                if pg == "SP":                        return "Starting Pitching"
                if pg == "RP":                        return "Bullpen"
                if pg in ("C","1B","2B","3B","SS"):   return "Infield"
                if pg in ("OF","CF","LF","RF"):       return "Outfield"
                if pg == "DH":                        return "DH / Bench"
                return "Other"

            _dc = depth_df.copy()
            _dc["_cat"] = _dc["pos_group"].apply(_broad)
            _by_cat = (
                _dc.groupby("_cat")["sal_2026_M"].sum().reset_index()
                .rename(columns={"_cat": "Category"})
            )
            _by_cat = _by_cat[_by_cat["sal_2026_M"] > 0].sort_values("sal_2026_M", ascending=False)
            _pal = {
                "Starting Pitching": "#4a9eff", "Bullpen": "#7ee8a2",
                "Infield": "#f59e0b", "Outfield": "#a78bfa",
                "DH / Bench": "#fb923c", "Other": "#6b7280",
            }
            _total_sal = float(_dc["sal_2026_M"].sum())

            fig_d = go.Figure(go.Pie(
                labels=_by_cat["Category"],
                values=_by_cat["sal_2026_M"],
                hole=0.52,
                marker_colors=[_pal.get(c, "#6b7280") for c in _by_cat["Category"]],
                textinfo="label+percent",
                textfont=dict(size=9),
                hovertemplate="%{label}<br>$%{value:.1f}M (%{percent})<extra></extra>",
            ))
            fig_d.add_annotation(
                text=f"${_total_sal:.0f}M<br><span style='font-size:9px'>26-man</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="#e8f4ff"),
            )
            fig_d.update_layout(**_pt(
                title="Payroll Allocation (26-man)",
                height=300, showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
            ))
            st.plotly_chart(fig_d, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Depth chart not available -- payroll chart skipped.")

    with col_r:
        if depth_df is not None and not depth_df.empty and "proj_WAR" in depth_df.columns:
            _war_pg = (
                depth_df.groupby("pos_group")["proj_WAR"].sum().reset_index()
                .rename(columns={"proj_WAR": "WAR"})
            )
            _war_pg = _war_pg[_war_pg["WAR"].notna() & (_war_pg["WAR"].abs() > 0)]
            _war_pg = _war_pg.sort_values("WAR", ascending=True)

            fig_w = go.Figure(go.Bar(
                x=_war_pg["WAR"], y=_war_pg["pos_group"],
                orientation="h",
                marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in _war_pg["WAR"]],
                text=_war_pg["WAR"].apply(lambda v: f"{v:.1f}"),
                textposition="outside",
                textfont=dict(size=9, color="#a8c8e8"),
                hovertemplate="%{y}: %{x:.1f} WAR<extra></extra>",
            ))
            fig_w.update_layout(**_pt(
                title="Projected WAR by Position Group",
                xaxis=dict(title="Proj WAR"),
                height=300,
                margin=dict(l=50, r=60, t=40, b=30),
            ))
            st.plotly_chart(fig_w, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Depth chart not available -- WAR chart skipped.")

    # Budget bar
    _comm  = scenario.get("committed_payroll_M", 0.0)
    _avail = max(budget_M - _comm, 0.0)

    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(
        name="Committed", x=["2026 Budget"], y=[_comm],
        marker_color="#4a9eff",
        text=[f"${_comm:.0f}M committed"], textposition="inside",
        hovertemplate="Committed: $%{y:.1f}M<extra></extra>",
    ))
    fig_b.add_trace(go.Bar(
        name="Available", x=["2026 Budget"], y=[_avail],
        marker_color="#1a3a5c",
        text=[f"${_avail:.0f}M available"], textposition="inside",
        hovertemplate="Available: $%{y:.1f}M<extra></extra>",
    ))
    fig_b.add_shape(type="line", x0=-0.5, x1=0.5, y0=budget_M, y1=budget_M,
                    line=dict(color="#ef4444", width=2, dash="dash"))
    fig_b.add_annotation(x=0.45, y=budget_M, text=f"  Budget cap: ${budget_M}M",
                         showarrow=False, font=dict(color="#ef4444", size=10), xanchor="left")
    fig_b.update_layout(**_pt(
        title="Committed vs Available Budget",
        yaxis=dict(title="$M"), barmode="stack",
        height=200, showlegend=True,
        margin=dict(l=50, r=120, t=40, b=30),
    ))
    st.plotly_chart(fig_b, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Team Planner page
# ---------------------------------------------------------------------------

def _render_team_planner(base_cfg: dict | None = None):
    """Render the Team Offseason Planner."""
    st.markdown("### Team Offseason Planner")

    if base_cfg is None:
        if not os.path.exists(_DEFAULT_CONFIG):
            st.error(f"Config file not found: {_DEFAULT_CONFIG}")
            return
        base_cfg = _load_base_config(_DEFAULT_CONFIG)

    # -- Data paths --
    salary_abs    = _resolve_data_path(base_cfg["raw_salary_war_path"], _DEFAULT_CONFIG)
    data_dir      = os.path.dirname(salary_abs)
    combined_path = os.path.join(data_dir, "mlb_combined_2021_2025.csv")
    if not os.path.exists(combined_path):
        st.error(f"Combined data file not found:\n`{combined_path}`")
        return

    try:
        team_list = get_all_teams(data_dir)
    except Exception as e:
        st.error(f"Could not load team list: {e}")
        return

    # -- Controls row --
    col_sel, col_opts, col_dc = st.columns([2, 2, 2])
    with col_sel:
        selected_team = st.selectbox("Select a team", team_list, key="team_planner_team")
    with col_opts:
        include_arb = st.checkbox(
            "Include arb-eligible players as retained",
            value=True,
            help="Arbitration players are team-controlled for 2026.",
            key="team_planner_include_arb",
        )
    with col_dc:
        include_minors = st.checkbox(
            "Include AAA/AA depth players",
            value=False,
            help="Adds minor league players from 2026 depth charts. Unmatched players assumed at league min ($0.74M).",
            key="team_planner_include_minors",
        )

    _dc_dir = get_depth_chart_dir(data_dir)

    # -- Payroll history --
    pay_hist  = _cached_payroll_history(data_dir)
    team_hist = pay_hist[pay_hist["Team"] == selected_team].sort_values("Year")

    if not team_hist.empty:
        hist_cols = st.columns(len(team_hist))
        for i, (_, row) in enumerate(team_hist.iterrows()):
            hist_cols[i].metric(f"{int(row['Year'])} payroll", f"${row['payroll_M']:.0f}M")

    recent_3    = team_hist.sort_values("Year").tail(3)["payroll_M"]
    hist_avg    = float(recent_3.mean()) if not recent_3.empty else 130.0
    default_bgt = min(int(hist_avg * 1.05 / 5) * 5, 400)

    budget_M = st.slider(
        "2026 total budget ($M)", min_value=40, max_value=400,
        value=default_bgt, step=5,
        help="Default is ~5% above the team's recent average payroll.",
        key="team_planner_budget",
    )

    # -- Build scenario --
    roster_slots_json = json.dumps(base_cfg["roster_slots"], sort_keys=True)
    combined_hash     = _file_hash(combined_path)

    with st.spinner(f"Loading {selected_team} roster ..."):
        try:
            scenario = _cached_team_scenario(
                data_dir=data_dir, team=selected_team,
                combined_hash=combined_hash,
                roster_slots_json=roster_slots_json,
                market_dpw_M=float(base_cfg.get("market_dpw_M", 5.5)),
                include_arb=include_arb,
                budget_override_M=float(budget_M),
                depth_chart_dir=_dc_dir,
                include_minors=include_minors,
            )
        except Exception as e:
            st.error(f"Error building scenario: {e}")
            st.exception(e)
            return

    committed_M         = scenario["committed_payroll_M"]
    available_M         = scenario["available_budget_M"]
    remaining           = scenario["remaining_slots"]
    depth_df            = scenario.get("depth_chart_df", None)
    depth_total         = scenario.get("depth_total_M", 0.0)
    depth_avail         = scenario.get("depth_available_M", available_M)
    minors_df           = scenario.get("minors_df", None)
    minors_40man_df     = scenario.get("minors_40man_df", pd.DataFrame())
    minors_40man_cost_M = float(scenario.get("minors_40man_cost_M", 0.0))
    _mkt_rate           = float(base_cfg.get("market_dpw_M", 5.5))

    # -- Hero panel values --
    _has_dc = _dc_dir and depth_df is not None and not depth_df.empty
    if _has_dc:
        _hero_war    = float(depth_df["proj_WAR"].fillna(0).sum()) if "proj_WAR" in depth_df.columns else 0.0
        _hero_comm   = depth_total - minors_40man_cost_M
        _hero_avail  = depth_avail
    else:
        _hero_war    = (
            scenario["locked_df"]["war_2025"].clip(lower=0).sum()
            + (scenario["arb_df"]["war_2025"].clip(lower=0).sum() if include_arb else 0.0)
        )
        _hero_comm   = committed_M
        _hero_avail  = available_M

    _locked_rows: list[pd.DataFrame] = []
    for _df, _ in [
        (scenario.get("locked_df", pd.DataFrame()), "locked"),
        (scenario.get("arb_df",    pd.DataFrame()), "arb"),
    ]:
        if _df.empty:
            continue
        _tmp = _df.copy()
        _tmp["_sal_M"] = pd.to_numeric(
            _tmp.get("sal_2026_M", _tmp.get("sal_2025_M", pd.Series(dtype=float))), errors="coerce"
        ).fillna(0)
        _tmp["_war"]   = pd.to_numeric(_tmp.get("war_2025", pd.Series(dtype=float)), errors="coerce").fillna(0)
        _locked_rows.append(_tmp)

    _hero_surplus = 0.0
    if _locked_rows:
        _all_locked   = pd.concat(_locked_rows, ignore_index=True)
        _hero_surplus = float(
            (_all_locked["_war"].clip(lower=0) * _mkt_rate - _all_locked["_sal_M"]).sum()
        )

    _open_slots = sum(remaining.values())

    # -- Hero panel --
    _render_hero_panel(
        selected_team, budget_M, _hero_comm, _hero_avail,
        _hero_war, _open_slots, _hero_surplus,
    )

    # -- Sticky summary bar --
    _inject_sticky_bar(selected_team, budget_M, _hero_comm, _hero_avail, _hero_war, _open_slots)

    # -- Main workflow tabs --
    tab_overview, tab_moves, tab_contracts, tab_optimizer, tab_history = st.tabs([
        "Roster Overview",
        "Offseason Moves",
        "Contracts",
        "Optimizer",
        "Roster History",
    ])

    # ======================================================================
    # TAB 1 -- Roster Overview
    # ======================================================================
    with tab_overview:
        if _has_dc:
            _render_payroll_charts(depth_df, scenario, budget_M, _mkt_rate)
            st.markdown("---")

            with st.expander("2026 Projected 26-Man Roster (Depth Chart)", expanded=True):
                dc_show = depth_df[[
                    c for c in ["Player","pos_group","age","proj_WAR","sal_2026_M","contract_status","salary_source"]
                    if c in depth_df.columns
                ]].copy().rename(columns={
                    "pos_group":       "Position",
                    "proj_WAR":        "Proj WAR",
                    "sal_2026_M":      "2026 Salary ($M)",
                    "contract_status": "Status",
                    "salary_source":   "Source",
                })
                st.dataframe(
                    dc_show.style.format(
                        {"Proj WAR": "{:.1f}", "2026 Salary ($M)": "${:.2f}M", "age": "{:.0f}"},
                        na_rep="--",
                    ),
                    use_container_width=True, hide_index=True,
                )
                lg_min_count = (depth_df["salary_source"] == "League Min").sum() if "salary_source" in depth_df.columns else 0
                st.caption(
                    f"26-man payroll: **${_hero_comm:.1f}M**  "
                    f"({lg_min_count} player(s) at league minimum $0.74M)"
                )

            if not minors_40man_df.empty:
                with st.expander(
                    f"40-Man Optioned to Minors -- {len(minors_40man_df)} players  "
                    f"(${minors_40man_cost_M:.1f}M at league min, counted in total spend)",
                    expanded=False,
                ):
                    st.caption(
                        "On the 40-man but assigned to AAA/AA. Salary counted in total committed payroll, "
                        "**excluded from charts**."
                    )
                    _40m_cols = [c for c in ["Player","pos_group","age","proj_level","max_level"] if c in minors_40man_df.columns]
                    st.dataframe(
                        minors_40man_df[_40m_cols].rename(columns={
                            "pos_group": "Position", "proj_level": "Proj Level", "max_level": "Max Level",
                        }).style.format({"age": "{:.0f}"}, na_rep="--"),
                        use_container_width=True, hide_index=True,
                    )

            if include_minors and minors_df is not None and not minors_df.empty:
                with st.expander(f"Prospects (not on 40-man) -- {len(minors_df)} players", expanded=False):
                    m_cols = [c for c in ["Player","pos_group","age","proj_level","max_level"] if c in minors_df.columns]
                    st.dataframe(
                        minors_df[m_cols].rename(columns={
                            "pos_group": "Position", "proj_level": "Proj Level", "max_level": "Max Level",
                        }),
                        use_container_width=True, hide_index=True,
                    )
        else:
            _fc1, _fc2, _fc3, _fc4 = st.columns(4)
            _fc1.metric("2026 budget",        f"${budget_M}M")
            _fc2.metric("Committed payroll",  f"${committed_M:.1f}M")
            _fc3.metric("Available to spend", f"${available_M:.1f}M")
            _fc4.metric("Open roster slots",  str(_open_slots))
            if _dc_dir is None:
                st.caption("Depth chart directory not found -- showing payroll-only view.")

    # ======================================================================
    # TAB 2 -- Offseason Moves
    # ======================================================================
    with tab_moves:
        _payroll_dir_26 = os.path.join(_ROOT_DIR, "2026 Payroll")
        _player_pool_26 = pd.DataFrame()
        if os.path.exists(_payroll_dir_26):
            try:
                _player_pool_26 = _cached_2026_payroll(
                    _payroll_dir_26, combined_path, _dir_hash(_payroll_dir_26)
                )
            except Exception:
                pass
        _render_team_moves(scenario, _player_pool_26, selected_team, _hero_avail, _mkt_rate)

    # ======================================================================
    # TAB 3 -- Contracts
    # ======================================================================
    with tab_contracts:
        def _fmt_roster_df(df: pd.DataFrame, extra_cols=None) -> pd.DataFrame:
            base_cols = ["Player", "pos_group", "age", "war_2025", "sal_2025_M", "Contract"]
            if extra_cols:
                base_cols = base_cols + extra_cols
            show = [c for c in base_cols if c in df.columns]
            return df[show].rename(columns={
                "pos_group":  "Position",
                "war_2025":   "2025 WAR",
                "sal_2025_M": "2025 Salary ($M)",
            })

        with st.expander(f"Under Contract for 2026 -- {len(scenario['locked_df'])} players", expanded=True):
            if scenario["locked_df"].empty:
                st.info("No players with committed 2026 contracts found.")
            else:
                ldf     = scenario["locked_df"].copy()
                display = _fmt_roster_df(ldf, extra_cols=["sal_2026_M"])
                display = display.rename(columns={"sal_2026_M": "2026 Salary ($M)"})
                st.dataframe(
                    display.style.format({
                        "2025 WAR": "{:.1f}", "2025 Salary ($M)": "${:.2f}M",
                        "2026 Salary ($M)": "${:.2f}M", "age": "{:.0f}",
                    }),
                    use_container_width=True,
                )
                st.caption(f"Total locked 2026 payroll: **${ldf['sal_2026_M'].sum():.1f}M**")

        with st.expander(f"Arbitration-Eligible -- {len(scenario['arb_df'])} players", expanded=True):
            if scenario["arb_df"].empty:
                st.info("No arbitration-eligible players found.")
            else:
                adf     = scenario["arb_df"].copy()
                display = _fmt_roster_df(adf, extra_cols=["arb_tier", "proj_arb_cost_M", "recommendation"])
                display = display.rename(columns={
                    "proj_arb_cost_M": "Est. 2026 Arb Cost ($M)",
                    "arb_tier":        "Arb Year",
                    "recommendation":  "Recommendation",
                })

                def _color_rec(val):
                    if val == "Keep":       return "background-color: #1a4731; color: white"
                    if val == "Non-tender": return "background-color: #4a1a1a; color: white"
                    return ""

                styled = display.style.format({
                    "2025 WAR": "{:.1f}", "2025 Salary ($M)": "${:.2f}M",
                    "Est. 2026 Arb Cost ($M)": "${:.2f}M", "age": "{:.0f}",
                })
                if "Recommendation" in display.columns:
                    styled = styled.applymap(_color_rec, subset=["Recommendation"])
                st.dataframe(styled, width="stretch")
                keep_count = (adf.get("recommendation", pd.Series()) == "Keep").sum()
                st.caption(
                    f"Est. arb payroll if all retained: **${adf['proj_arb_cost_M'].sum():.1f}M** | "
                    f"Suggested keeps: **{keep_count}** of {len(adf)}"
                )
                if include_arb:
                    st.info("Arb players counted as retained. Non-tender candidates free up salary.")

        with st.expander(f"Expiring / Becoming Free Agent -- {len(scenario['expiring_df'])} players", expanded=False):
            if scenario["expiring_df"].empty:
                st.info("No expiring players found.")
            else:
                edf     = scenario["expiring_df"].copy()
                display = _fmt_roster_df(edf)
                st.dataframe(
                    display.style.format({
                        "2025 WAR": "{:.1f}", "2025 Salary ($M)": "${:.2f}M", "age": "{:.0f}",
                    }),
                    use_container_width=True,
                )
                st.caption("High-WAR expiring players may be worth re-signing.")

    # ======================================================================
    # TAB 4 -- Optimizer
    # ======================================================================
    with tab_optimizer:
        if not remaining:
            st.success("All roster slots are filled by locked/arb players -- nothing to optimize.")
        else:
            _oc1, _oc2 = st.columns([1, 2])
            with _oc1:
                st.markdown(f"**{_open_slots} Open Slots to Fill**")
                slot_df = pd.DataFrame([{"Slot": k, "Count": v} for k, v in sorted(remaining.items())])
                st.dataframe(slot_df, width="stretch", hide_index=True)
            with _oc2:
                st.markdown(
                    f"Budget remaining after committed payroll: **${_hero_avail:.1f}M**  \n"
                    "The optimizer will suggest the best available FA/Arb additions from the full "
                    "player pool to fill open slots within budget."
                )

            run_team_opt = st.button("Run Offseason Optimizer", type="primary", key="team_planner_run_opt")

            if "team_opt_result" not in st.session_state:
                st.session_state["team_opt_result"] = {}

            if run_team_opt:
                team_cfg = dict(base_cfg)
                team_cfg["budget_M"]       = float(_hero_avail)
                team_cfg["roster_slots"]   = dict(remaining)
                team_cfg["market_mode"]    = "all"
                team_cfg["optimizer_mode"] = "archetype"

                team_players     = set(scenario["roster_status_df"]["Player"].tolist())
                combined_df_full = _read_csv(combined_path, low_memory=False)
                filtered_raw     = combined_df_full[~combined_df_full["Player"].isin(team_players)].copy()

                with st.spinner("Building projections & running optimizer ..."):
                    try:
                        team_proj      = make_projections(filtered_raw, team_cfg)
                        team_proj_live = team_proj[
                            team_proj["proj_WAR"] >= float(team_cfg.get("min_war_threshold", 0.0))
                        ].copy()
                        team_arch      = build_archetype_definitions(team_proj_live)
                        team_proj_arch = assign_archetypes(team_proj_live)
                        opt            = run_optimizer(team_arch, team_cfg, projected_df=team_proj_arch)
                        st.session_state["team_opt_result"] = {
                            "opt": opt, "proj": team_proj_live, "arch": team_arch,
                            "cfg": team_cfg, "team": selected_team,
                        }
                    except Exception as e:
                        st.error(f"Optimizer error: {e}")
                        st.exception(e)

            team_opt_data = st.session_state.get("team_opt_result", {})
            opt           = team_opt_data.get("opt")

            if opt is not None and team_opt_data.get("team") == selected_team:
                rdf = opt.roster_df
                if rdf.empty:
                    st.warning(f"Optimizer returned no roster. Status: {opt.status}")
                else:
                    total_w  = rdf["war_mean"].sum()
                    total_c  = rdf["cost_mean"].sum()
                    full_war = total_w + (
                        scenario["locked_df"]["war_2025"].clip(lower=0).sum()
                        + (scenario["arb_df"]["war_2025"].clip(lower=0).sum() if include_arb else 0.0)
                    )

                    res1, res2, res3, res4 = st.columns(4)
                    res1.metric("Additions WAR",             f"{total_w:.1f}")
                    res2.metric("Additions cost",            f"${total_c:.1f}M")
                    res3.metric("Projected full-roster WAR", f"{full_war:.1f}")
                    res4.metric("Solver status",             opt.status)

                    st.markdown("#### Recommended Additions")
                    _tp_rdf  = rdf.copy()
                    _tp_rdf.insert(1, "Role", _tp_rdf["archetype_id"].apply(_arch_label))
                    _tp_cols = [c for c in ["slot","Role","pos_group","war_mean","war_sd","cost_mean","n_players"]
                                if c in _tp_rdf.columns]
                    st.dataframe(
                        _tp_rdf[_tp_cols].rename(columns={
                            "slot": "Slot", "pos_group": "Pos Group",
                            "war_mean": "Proj WAR", "war_sd": "WAR SD",
                            "cost_mean": "Est Cost", "n_players": "# Players",
                        }).style.format({"Proj WAR": "{:.2f}", "WAR SD": "{:.2f}", "Est Cost": "${:.1f}M"}),
                        use_container_width=True,
                    )

                    st.markdown("#### Full 2026 Projected Roster")
                    locked_summary = scenario["locked_df"][
                        ["Player", "pos_group", "war_2025", "sal_2026_M"]
                    ].copy().rename(columns={
                        "war_2025": "proj_WAR", "sal_2026_M": "cost_M", "pos_group": "Position",
                    })
                    locked_summary["source"]        = "Under Contract"
                    locked_summary["assigned_slot"] = locked_summary.apply(
                        lambda r: scenario["assignments_df"].set_index("Player")
                        .get("assigned_slot", pd.Series(dtype=str)).get(r["Player"], "-"),
                        axis=1,
                    )

                    if include_arb and not scenario["arb_df"].empty:
                        arb_keep = scenario["arb_df"][
                            scenario["arb_df"].get("recommendation", pd.Series()) != "Non-tender"
                        ][["Player", "pos_group", "war_2025", "proj_arb_cost_M"]].rename(columns={
                            "war_2025": "proj_WAR", "proj_arb_cost_M": "cost_M", "pos_group": "Position",
                        })
                        arb_keep["source"]        = "Arb (Retained)"
                        arb_keep["assigned_slot"] = "-"
                        locked_summary = pd.concat([locked_summary, arb_keep], ignore_index=True)

                    st.dataframe(
                        locked_summary[["Player","Position","assigned_slot","proj_WAR","cost_M","source"]]
                        .rename(columns={"assigned_slot": "Slot", "proj_WAR": "Proj WAR", "cost_M": "Cost ($M)"})
                        .style.format({"Proj WAR": "{:.1f}", "Cost ($M)": "${:.2f}M"}),
                        use_container_width=True,
                    )
                    st.caption(
                        f"Total projected 2026 payroll: **${committed_M + total_c:.1f}M** "
                        f"(committed ${committed_M:.1f}M + additions ${total_c:.1f}M)"
                    )

    # ======================================================================
    # TAB 5 -- Roster History (value analysis)
    # ======================================================================
    with tab_history:
        _hist_all = _cached_player_history(combined_path, _file_hash(combined_path))
        _th = _hist_all[_hist_all["Team"] == selected_team].copy()
        _th["WAR_Total"] = pd.to_numeric(_th["WAR_Total"], errors="coerce").fillna(0.0)
        _th["Year"]      = pd.to_numeric(_th["Year"],      errors="coerce")
        _th["Salary_M"]  = pd.to_numeric(_th.get("Salary_M", pd.Series(dtype=float)), errors="coerce").fillna(0.72)

        _psumm = (
            _th.groupby("Player").agg(
                Seasons    =("Year",      "count"),
                Total_WAR  =("WAR_Total", "sum"),
                Peak_WAR   =("WAR_Total", "max"),
                Total_Sal_M=("Salary_M",  "sum"),
            ).reset_index()
        )
        _psumm["Market_Val_M"] = (_psumm["Total_WAR"] * _mkt_rate).round(2)
        _psumm["Surplus_M"]    = (_psumm["Market_Val_M"] - _psumm["Total_Sal_M"]).round(2)

        _cur_players = set(scenario["locked_df"]["Player"].tolist())
        if include_arb:
            _cur_players |= set(scenario["arb_df"]["Player"].tolist())

        _cur_val = _psumm[_psumm["Player"].isin(_cur_players)].copy()
        _dep_val = _psumm[~_psumm["Player"].isin(_cur_players)].copy()

        with st.expander(
            f"Value Delivered by Current Roster (2021-2025) -- {len(_cur_val)} players tracked",
            expanded=True,
        ):
            if _cur_val.empty:
                st.info("No historical data found for current players on this team.")
            else:
                _c1, _c2, _c3, _c4 = st.columns(4)
                _c1.metric("Total WAR Delivered",  f"{_cur_val['Total_WAR'].sum():.1f}")
                _c2.metric("Market Value of WAR",  f"${_cur_val['Market_Val_M'].sum():.0f}M",
                           help=f"Total WAR x ${_mkt_rate}M/WAR market rate.")
                _c3.metric("Total Salary Paid",    f"${_cur_val['Total_Sal_M'].sum():.0f}M")
                _surplus = _cur_val["Surplus_M"].sum()
                _c4.metric("Total Surplus Value",  f"${_surplus:+.0f}M",
                           delta="above market" if _surplus >= 0 else "below market",
                           delta_color="normal" if _surplus >= 0 else "inverse")

                _cv     = _cur_val.sort_values("Total_WAR", ascending=True)
                _cv_clr = ["#22c55e" if s >= 0 else "#ef4444" for s in _cv["Surplus_M"]]
                _cv_txt = [
                    f"{r['Total_WAR']:.1f} WAR  |  paid ${r['Total_Sal_M']:.0f}M  |  surplus ${r['Surplus_M']:+.0f}M"
                    for _, r in _cv.iterrows()
                ]
                _fig_cur = go.Figure(go.Bar(
                    x=_cv["Total_WAR"], y=_cv["Player"], orientation="h",
                    marker_color=_cv_clr,
                    text=_cv_txt, textposition="outside",
                    textfont=dict(size=9, color="#7aa2c0"),
                    hovertemplate="%{y}<br>Total WAR: %{x:.1f}<extra></extra>",
                ))
                _fig_cur.update_layout(**_pt(
                    title=f"{selected_team} -- Current Players' Total WAR While on Team (2021-2025)",
                    xaxis=dict(title="Total WAR (cumulative)"),
                    height=max(320, len(_cv) * 28),
                    margin=dict(l=160, r=280, t=45, b=40),
                ))
                st.plotly_chart(_fig_cur, width="stretch", config={"displayModeBar": False})

                _cv_tbl = _cur_val.sort_values("Total_WAR", ascending=False)[
                    ["Player","Seasons","Total_WAR","Peak_WAR","Total_Sal_M","Market_Val_M","Surplus_M"]
                ].reset_index(drop=True)
                _cv_tbl.columns = ["Player","Seasons","Total WAR","Peak WAR","Salary Paid $M","Market Val $M","Surplus $M"]
                st.dataframe(
                    _cv_tbl.style.format({
                        "Total WAR": "{:.1f}", "Peak WAR": "{:.1f}",
                        "Salary Paid $M": "${:.1f}M", "Market Val $M": "${:.1f}M",
                        "Surplus $M": "${:+.1f}M",
                    }, na_rep="--"),
                    hide_index=True, use_container_width=True,
                )

        _dep_pos = _dep_val[_dep_val["Total_WAR"] > 0].sort_values("Total_WAR", ascending=False)
        with st.expander(
            f"Value From Departed Players (2021-2025) -- {len(_dep_pos)} tracked",
            expanded=False,
        ):
            if _dep_pos.empty:
                st.info("No departed players with positive WAR found.")
            else:
                _d1, _d2, _d3 = st.columns(3)
                _d1.metric("Total WAR From Departed", f"{_dep_pos['Total_WAR'].sum():.1f}")
                _d2.metric("Total Salary Paid",        f"${_dep_pos['Total_Sal_M'].sum():.0f}M")
                _d3.metric("Market Value Delivered",   f"${_dep_pos['Market_Val_M'].sum():.0f}M",
                           help="WAR x market rate -- how much those seasons were worth.")

                _dv     = _dep_pos.head(20).sort_values("Total_WAR", ascending=True)
                _dv_clr = ["#22c55e" if s >= 0 else "#ef4444" for s in _dv["Surplus_M"]]
                _dv_txt = [
                    f"{r['Total_WAR']:.1f} WAR  |  paid ${r['Total_Sal_M']:.0f}M  |  surplus ${r['Surplus_M']:+.0f}M"
                    for _, r in _dv.iterrows()
                ]
                _fig_dep = go.Figure(go.Bar(
                    x=_dv["Total_WAR"], y=_dv["Player"], orientation="h",
                    marker_color=_dv_clr,
                    text=_dv_txt, textposition="outside",
                    textfont=dict(size=9, color="#7aa2c0"),
                    hovertemplate="%{y}<br>Total WAR on team: %{x:.1f}<extra></extra>",
                ))
                _fig_dep.update_layout(**_pt(
                    title=f"{selected_team} -- Departed Players' Total WAR While on Team (Top 20)",
                    xaxis=dict(title="Total WAR (while on this team)"),
                    height=max(320, len(_dv) * 28),
                    margin=dict(l=160, r=280, t=45, b=40),
                ))
                st.plotly_chart(_fig_dep, width="stretch", config={"displayModeBar": False})

                _dv_tbl = _dep_pos.head(20)[
                    ["Player","Seasons","Total_WAR","Peak_WAR","Total_Sal_M","Market_Val_M","Surplus_M"]
                ].reset_index(drop=True)
                _dv_tbl.columns = ["Player","Seasons","Total WAR","Peak WAR","Salary Paid $M","Market Val $M","Surplus $M"]
                st.dataframe(
                    _dv_tbl.style.format({
                        "Total WAR": "{:.1f}", "Peak WAR": "{:.1f}",
                        "Salary Paid $M": "${:.1f}M", "Market Val $M": "${:.1f}M",
                        "Surplus $M": "${:+.1f}M",
                    }, na_rep="--"),
                    hide_index=True, use_container_width=True,
                )
