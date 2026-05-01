"""MLB Toolbox — Roster Simulator page (extracted from streamlit_app.py)."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.theme import plotly_theme as _pt
from utils.components import render_feedback_widget as _render_feedback_widget
from utils.components import render_glossary as _render_glossary
from utils.team_utils import cbt_info as _cbt_info
from utils.data_loading import (
    R2_BASE_URL as _DL_R2_BASE_URL,
    R2_MODE as _DL_R2_MODE,
    ROOT_DIR as _DL_ROOT_DIR,
    DEFAULT_CONFIG as _DL_DEFAULT_CONFIG,
    HEADSHOTS_DIR as _DL_HEADSHOTS_DIR,
    RAZZBALL_PATH as _DL_RAZZBALL_PATH,
    data_url as _DL_data_url,
    read_csv as _DL_read_csv,
    load_base_config as _DL_load_base_config,
    resolve_data_path as _DL_resolve_data_path,
    file_hash as _DL_file_hash,
    dir_hash as _DL_dir_hash,
    cached_simulator_data as _DL_cached_simulator_data,
    cached_2026_payroll as _DL_cached_2026_payroll,
    cached_40man_roster as _DL_cached_40man_roster,
    cached_war_reliability as _DL_cached_war_reliability,
    cached_player_history as _DL_cached_player_history,
    cached_razzball as _DL_cached_razzball,
    cached_mlbam_lookup as _DL_cached_mlbam_lookup,
)
from utils.constants import (
    C,
    ROSTER_TEMPLATE as _DL_ROSTER_TEMPLATE,
    ELIGIBLE_SLOTS_MAP as _DL_ELIGIBLE_SLOTS_MAP,
    OPTIONAL_SLOTS as _DL_OPTIONAL_SLOTS,
    PG_CHART_COLORS as _DL_PG_CHART_COLORS,
)


# ── Helpers (simulator-only) ────────────────────────────────────────────────

def _contract_decision(player_row: dict) -> str:
    """Return Extend / Hold / Trade / DFA recommendation for one player."""
    war   = float(player_row.get("WAR_Total") or 0)
    sal   = float(player_row.get("Salary_M")  or 0)
    age   = float(player_row.get("Age")        or 30)
    yrs   = float(player_row.get("Yrs_Left")   or 0)
    stage = str(player_row.get("Stage_Clean")  or "FA")

    # DFA: clearly negative or replacement-level with big contract
    if war < -0.5:
        return "DFA"
    if war < 0.5 and sal > 15 and age > 32:
        return "DFA"
    # Trade: declining value, expensive, locked in long-term
    if war < 1.0 and sal > 12 and yrs > 1:
        return "Trade"
    if war < 0.5 and sal > 8 and age > 33:
        return "Trade"
    # Extend: young, high-WAR, still cheap
    if age <= 27 and war >= 3.0 and stage in ("Pre-Arb", "Arb"):
        return "Extend"
    if age <= 26 and war >= 2.0 and stage == "Pre-Arb":
        return "Extend"
    return "Hold"


def _roster_grade(roster_df: pd.DataFrame) -> dict:
    """Return A-F grades for Production, Efficiency, Depth, Contract Health."""
    total_war  = float(roster_df["WAR_Total"].sum()) if "WAR_Total" in roster_df.columns else 0
    total_cost = float(roster_df["Salary_M"].sum())  if "Salary_M"  in roster_df.columns else 0

    prod_score  = min(100, int(total_war / 35 * 100))                            # 35 WAR = full marks
    eff_score   = min(100, int((total_war / max(total_cost, 0.1)) / 0.30 * 100)) # 0.30 WAR/$M = full marks
    pos_groups  = set(roster_df["pos_group"].tolist()) if "pos_group" in roster_df.columns else set()
    covered     = len(pos_groups & {"SP", "RP", "C", "1B", "2B", "3B", "SS", "CF", "OF"})
    depth_score = min(100, int(covered / 9 * 100))
    yrs_avg     = (float(roster_df["Yrs_Left"].mean())
                   if "Yrs_Left" in roster_df.columns and roster_df["Yrs_Left"].notna().any()
                   else 2.0)
    ctrc_score  = max(0, min(100, int(100 - abs(yrs_avg - 2.0) * 20)))

    def _g(s: int) -> str:
        return ("A+" if s >= 95 else "A" if s >= 85 else "B+" if s >= 75 else
                "B" if s >= 65 else "C" if s >= 50 else "D" if s >= 35 else "F")

    return {
        "Production":      (_g(prod_score),  prod_score),
        "Efficiency":      (_g(eff_score),   eff_score),
        "Depth":           (_g(depth_score), depth_score),
        "Contract Health": (_g(ctrc_score),  ctrc_score),
    }


# ── Sub-renderers ───────────────────────────────────────────────────────────

def _render_player_comparison(players_df: pd.DataFrame):
    """Side-by-side stats table for 2-4 selected players."""
    metrics = [
        ("Position",         "Position",        None),
        ("Age",              "Age",             "{:.0f}"),
        ("Stage",            "Stage_Clean",     None),
        ("2025 WAR",         "WAR_Total",       "{:.1f}"),
        ("2026 Salary $M",   "Salary_M",        "${:.1f}M"),
        ("Full Contract $M", "Total_Contract_M","${:.0f}M"),
        ("W/$M (Season)",    "W_per_M",         "{:.2f}"),
        ("W/$M (Contract)",  "PPR",             "{:.2f}"),
        ("Yrs Left",         "Yrs_Left",        "{:.0f}"),
        ("HR",               "HR",              "{:.0f}"),
        ("AVG",              "AVG",             "{:.3f}"),
        ("ERA",              "ERA",             "{:.2f}"),
        ("IP",               "IP",              "{:.0f}"),
        ("FIP",              "FIP",             "{:.2f}"),
    ]
    players = players_df.to_dict("records")
    header  = ["Metric"] + [p.get("Player", f"Player {i+1}") for i, p in enumerate(players)]
    rows    = []
    for label, col, fmt in metrics:
        vals = []
        for p in players:
            v = p.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append("\u2014")
            elif fmt:
                try:
                    vals.append(fmt.format(float(v)))
                except Exception:
                    vals.append(str(v))
            else:
                vals.append(str(v))
        if any(v != "\u2014" for v in vals):
            rows.append([label] + vals)
    comp_df = pd.DataFrame(rows, columns=header)
    st.dataframe(comp_df, hide_index=True, width="stretch",
                 height=min(60 + len(comp_df) * 35, 520))


def _render_trade_analyzer(roster_df: pd.DataFrame):
    """Interactive trade analyzer -- incoming vs outgoing player impact."""
    full_df = st.session_state.get("sim_df_full", pd.DataFrame())
    if full_df.empty:
        st.info("Full player pool not loaded yet \u2014 visit the Roster Simulator tab first.")
        return

    roster_names = roster_df["Player"].tolist() if "Player" in roster_df.columns else []
    all_names    = sorted(full_df["Player"].dropna().unique().tolist())
    available_in = [n for n in all_names if n not in roster_names]

    c_out, c_in = st.columns(2)
    with c_out:
        st.markdown("**Players Out** *(from your roster)*")
        out_players = st.multiselect("out", options=roster_names,
                                     key="trade_out", label_visibility="collapsed")
    with c_in:
        st.markdown("**Players In** *(acquire)*")
        in_players  = st.multiselect("in", options=available_in,
                                     key="trade_in", label_visibility="collapsed")

    if not out_players and not in_players:
        st.caption("Select players above to see trade impact.")
        return

    out_df = roster_df[roster_df["Player"].isin(out_players)] if out_players else pd.DataFrame(columns=roster_df.columns)
    in_df  = full_df[full_df["Player"].isin(in_players)]  if in_players  else pd.DataFrame(columns=full_df.columns)

    out_war  = float(out_df["WAR_Total"].sum()) if "WAR_Total" in out_df.columns else 0.0
    in_war   = float(in_df["WAR_Total"].sum())  if "WAR_Total" in in_df.columns  else 0.0
    out_sal  = float(out_df["Salary_M"].sum())  if "Salary_M"  in out_df.columns else 0.0
    in_sal   = float(in_df["Salary_M"].sum())   if "Salary_M"  in in_df.columns  else 0.0
    out_ppr  = float(out_df["PPR"].mean()) if "PPR" in out_df.columns and out_df["PPR"].notna().any() else 0.0
    in_ppr   = float(in_df["PPR"].mean())  if "PPR" in in_df.columns  and in_df["PPR"].notna().any()  else 0.0

    delta_war = in_war  - out_war
    delta_sal = in_sal  - out_sal
    delta_ppr = in_ppr  - out_ppr

    t1, t2, t3 = st.columns(3)
    t1.metric("Net WAR",      f"{delta_war:+.1f}",
              f"{in_war:.1f} in vs {out_war:.1f} out",
              delta_color="normal")
    t2.metric("Net Salary",   f"${delta_sal:+.1f}M",
              f"${in_sal:.1f}M in vs ${out_sal:.1f}M out",
              delta_color="inverse")
    t3.metric("Net Pay/Play", f"{delta_ppr:+.2f}",
              f"{in_ppr:.2f} in vs {out_ppr:.2f} out",
              delta_color="normal")

    if delta_war > 0 and delta_sal <= 0:
        verdict, vcolor = "\u2705 Strong Trade \u2014 more WAR, same or less money", "#2ecc71"
    elif delta_war > 0 and delta_sal > 0:
        verdict, vcolor = f"\u26a0\ufe0f Mixed \u2014 {delta_war:.1f} more WAR for ${delta_sal:.1f}M more", "#f39c12"
    elif delta_war <= 0 and delta_sal < 0:
        verdict, vcolor = f"\U0001f4b0 Salary dump \u2014 {-delta_war:.1f} WAR loss, saves ${-delta_sal:.1f}M", "#3498db"
    else:
        verdict, vcolor = "\u274c Net negative \u2014 WAR loss without salary relief", "#e74c3c"

    st.markdown(
        f"<p style='color:{vcolor};font-weight:600;margin:0.5rem 0;'>{verdict}</p>",
        unsafe_allow_html=True,
    )


def _render_position_coverage(roster_df: pd.DataFrame, deps=None):
    """Show position coverage vs standard roster template."""
    _ROSTER_TEMPLATE = _DL_ROSTER_TEMPLATE
    _ELIGIBLE_SLOTS_MAP = _DL_ELIGIBLE_SLOTS_MAP
    _OPTIONAL_SLOTS = _DL_OPTIONAL_SLOTS

    if "pos_group" not in roster_df.columns:
        st.info("No position data available.")
        return

    # Count eligible players per slot
    slot_have: dict[str, int] = {slot: 0 for slot in _ROSTER_TEMPLATE}
    for _, row in roster_df.iterrows():
        eligible = _ELIGIBLE_SLOTS_MAP.get(str(row.get("pos_group", "")), [])
        for slot in eligible:
            if slot in slot_have:
                slot_have[slot] += 1
        # Everyone can be bench
        slot_have["BENCH"] = slot_have.get("BENCH", 0) + 1

    rows = []
    for slot, needed in _ROSTER_TEMPLATE.items():
        have = slot_have.get(slot, 0)
        optional = slot in _OPTIONAL_SLOTS
        if have >= needed:
            indicator = "OK"
        elif optional and have == 0:
            indicator = "\u2014 optional"
        elif have > 0:
            indicator = f"{have}/{needed}"
        else:
            indicator = f"0/{needed} MISSING"
        rows.append({"Slot": slot, "Need": needed, "Have": have, "Status": indicator})

    cov_df = pd.DataFrame(rows)

    def _color_status(val):
        s = str(val)
        if s == "OK":
            return "background-color: #1a4731; color: #2ecc71; font-weight: bold"
        if "optional" in s:
            return "background-color: #1a2a3a; color: #7aa2c0; font-style: italic"
        if "MISSING" in s:
            return "background-color: #4a1a1a; color: #e74c3c; font-weight: bold"
        return "background-color: #3a2a00; color: #f39c12; font-weight: bold"

    styled = cov_df[["Slot", "Need", "Have", "Status"]].style.applymap(
        _color_status, subset=["Status"]
    )
    st.dataframe(styled, hide_index=True, width="stretch",
                 height=min(60 + len(cov_df) * 35, 490))


def _render_roster_summary(budget_M: float, deps=None):
    """Render the built roster summary panel."""
    import io

    _ROSTER_TEMPLATE = _DL_ROSTER_TEMPLATE
    _ELIGIBLE_SLOTS_MAP = _DL_ELIGIBLE_SLOTS_MAP
    _OPTIONAL_SLOTS = _DL_OPTIONAL_SLOTS
    _PG_CHART_COLORS = _DL_PG_CHART_COLORS

    roster_records = st.session_state.get("sim_roster", [])
    if not roster_records:
        return

    roster_df = pd.DataFrame(roster_records)

    st.markdown("---")
    st.markdown("### My Custom Roster")

    # -- Metrics row 1 --------------------------------------------------------
    total_war  = float(roster_df["WAR_Total"].sum())  if "WAR_Total"   in roster_df.columns else 0.0
    total_cost = float(roster_df["Salary_M"].sum())   if "Salary_M"    in roster_df.columns else 0.0
    n_fa    = int((roster_df["Stage_Clean"] == "FA").sum())      if "Stage_Clean" in roster_df.columns else 0
    n_arb   = int((roster_df["Stage_Clean"] == "Arb").sum())     if "Stage_Clean" in roster_df.columns else 0
    n_pre   = int((roster_df["Stage_Clean"] == "Pre-Arb").sum()) if "Stage_Clean" in roster_df.columns else 0
    avg_ppr = (float(roster_df["PPR"].mean())
               if "PPR" in roster_df.columns and roster_df["PPR"].notna().any()
               else None)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Players",            str(len(roster_df)))
    m2.metric("Total WAR",          f"{total_war:.1f}")
    m3.metric("Payroll ($M)",       f"${total_cost:.1f}M")
    m4.metric("$/WAR",              f"${total_cost / max(total_war, 0.1):.1f}M")
    m5.metric("FA / Arb / Pre-Arb", f"{n_fa} / {n_arb} / {n_pre}")
    m6.metric("Avg Pay/Play Ratio",
              f"{avg_ppr:.2f}" if avg_ppr is not None else "\u2014",
              help="Avg Pay vs Play Ratio: sum of actual WAR across all contract years \u00f7 total contract $M. Higher = better value.")

    # -- Metrics row 2: efficiency breakdown ----------------------------------
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("WAR / Player", f"{total_war / max(len(roster_df), 1):.1f}",
              help="Average WAR contributed per rostered player.")
    e2.metric("WAR / $M", f"{total_war / max(total_cost, 0.1):.2f}",
              help="Team-level production efficiency: total WAR divided by total payroll.")
    if "Stage_Clean" in roster_df.columns and "Salary_M" in roster_df.columns:
        pre_cost = float(roster_df[roster_df["Stage_Clean"] == "Pre-Arb"]["Salary_M"].sum())
        e3.metric("Pre-Arb % Pay", f"{pre_cost / max(total_cost, 0.1) * 100:.1f}%",
                  help="Fraction of payroll locked in pre-arb players \u2014 lower = more flexibility.")
    else:
        e3.metric("Pre-Arb % Pay", "\u2014")
    if "Yrs_Left" in roster_df.columns and roster_df["Yrs_Left"].notna().any():
        e4.metric("Avg Yrs Left", f"{float(roster_df['Yrs_Left'].mean()):.1f}",
                  help="Average contract years remaining across all rostered players.")
    else:
        e4.metric("Avg Yrs Left", "\u2014")
    e5.metric("Est. Win Total", f"~{47.7 + total_war:.0f} W",
              help="Rough estimate: 47.7 replacement-level wins + Total WAR.")

    # -- Roster Grade ----------------------------------------------------------
    grades = _roster_grade(roster_df)
    _GRADE_COLORS = {
        "A+": "#00e676", "A": "#4caf50", "B+": "#8bc34a", "B": "#cddc39",
        "C": "#ffc107",  "D": "#ff9800", "F": "#f44336",
    }
    st.markdown("##### Roster Grade")
    g_cols = st.columns(4)
    for i, (dim, (grade, score)) in enumerate(grades.items()):
        color = _GRADE_COLORS.get(grade, "#7aa2c0")
        g_cols[i].markdown(
            f"<div style='text-align:center;padding:0.6rem 0.4rem;background:#0d1f38;"
            f"border:1px solid #1e3a5f;border-radius:8px;'>"
            f"<div style='font-size:1.8rem;font-weight:700;color:{color};'>{grade}</div>"
            f"<div style='font-size:0.75rem;color:#7aa2c0;margin-top:2px;'>{dim}</div>"
            f"<div style='font-size:0.65rem;color:#4a6a8a;'>{score}/100</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("")

    # -- Roster table + position coverage -------------------------------------
    col_table, col_check = st.columns([3, 2])

    with col_table:
        st.markdown("##### Roster  <span style='font-size:0.78rem;color:#666;font-weight:normal;'>\u2014 check \u2715 to remove players</span>", unsafe_allow_html=True)
        show = [c for c in ["Player", "Team", "Position", "Stage_Clean",
                             "Age", "WAR_Total", "Salary_M", "W_per_M", "Yrs_Left", "PPR"]
                if c in roster_df.columns]
        edit_df = roster_df[show].copy()

        # Decision + Reliability columns
        reliability = st.session_state.get("sim_reliability", {})
        edit_df["Decision"]    = [_contract_decision(r) for r in roster_records]
        edit_df["Consistency"] = [reliability.get(r.get("Player", ""), {}).get("grade", "?")
                                   for r in roster_records]
        edit_df.insert(0, "Remove", False)

        # Row highlighting: amber for depth-chart-only players (no MLB stats)
        _dc_only_flags = [bool(r.get("_dc_only", False)) for r in roster_records]
        def _highlight_dc(row):
            return (
                ["background-color:#2d1f00;color:#fbbf24"] * len(row)
                if _dc_only_flags[row.name] else [""] * len(row)
            )
        _styled_edit = edit_df.style.apply(_highlight_dc, axis=1)

        if any(_dc_only_flags):
            st.caption("\U0001f7e1 Amber rows = depth chart only \u2014 no 2025 MLB stats in database (league min salary assumed)")

        edited = st.data_editor(
            _styled_edit,
            column_config={
                "Remove":      st.column_config.CheckboxColumn("\u2715",          width="small"),
                "WAR_Total":   st.column_config.NumberColumn("WAR",           format="%.1f",   width="small"),
                "Salary_M":    st.column_config.NumberColumn("Sal $M",        format="$%.1fM", width="small"),
                "W_per_M":     st.column_config.NumberColumn("W/$M (Ssn)",    format="%.2f",   width="small"),
                "Yrs_Left":    st.column_config.ProgressColumn("Yrs Left",    min_value=0, max_value=7, format="%d yr", width="small"),
                "PPR":         st.column_config.NumberColumn("W/$M (Ctrc)",   format="%.2f",   width="small",
                               help="Sum of actual WAR across all contract years \u00f7 total contract $M."),
                "Stage_Clean": st.column_config.TextColumn("Stage",           width="small"),
                "Position":    st.column_config.TextColumn("Pos",             width="small"),
                "Age":         st.column_config.NumberColumn("Age",           format="%d",     width="small"),
                "Decision":    st.column_config.TextColumn("Decision",        width="small",
                               help="Extend / Hold / Trade / DFA recommendation based on age, WAR, and contract."),
                "Consistency": st.column_config.TextColumn("Consistency",     width="small",
                               help="WAR year-to-year consistency grade (A = very consistent, F = 1 year of data only)."),
            },
            disabled=[c for c in edit_df.columns if c != "Remove"],
            hide_index=True,
            use_container_width=True,
            height=min(60 + (len(roster_df) + 1) * 35, 520),
            key="roster_editor",
        )

        if st.button("Remove Selected Players", key="roster_remove_btn", type="secondary"):
            keep = [rec for rec, rm in zip(roster_records, edited["Remove"].tolist()) if not rm]
            if len(keep) < len(roster_records):
                st.session_state["sim_roster"] = keep
                st.rerun()
            else:
                st.info("No players checked for removal \u2014 tick the \u2715 column first.")

    with col_check:
        st.markdown("##### Position Coverage")
        _render_position_coverage(roster_df, deps)

    # -- Charts: WAR by position group  |  Salary vs WAR scatter ---------------
    ch1, ch2 = st.columns(2)
    _PG_ORDER  = ["SP", "RP", "C", "1B", "2B", "3B", "SS", "CF", "OF", "DH"]
    _PG_COLORS = _PG_CHART_COLORS

    with ch1:
        st.markdown("##### WAR by Position Group")
        if "pos_group" in roster_df.columns and "WAR_Total" in roster_df.columns:
            pg_war = (
                roster_df.groupby("pos_group")["WAR_Total"]
                .sum()
                .reindex(_PG_ORDER)
                .dropna()
            )
            if not pg_war.empty:
                try:
                    cats    = pg_war.index.tolist()
                    vals    = pg_war.values.tolist()
                    bcolors = [_PG_COLORS[_PG_ORDER.index(p)] for p in cats if p in _PG_ORDER]
                    y_max   = max(vals) * 1.35 if vals else 10
                    fig = go.Figure(data=[go.Bar(
                        x=cats, y=vals,
                        marker_color=bcolors, marker_line_width=0,
                        text=[f"{v:.1f}" for v in vals],
                        textposition="outside",
                        textfont=dict(color="#dbeafe", size=11),
                        hovertemplate="%{x}: %{y:.1f} WAR<extra></extra>",
                    )])
                    fig.update_layout(**_pt(
                        title="WAR by Position Group",
                        yaxis=dict(title="WAR", range=[0, y_max]),
                        height=450,
                        margin=dict(l=50, r=20, t=45, b=50),
                    ))
                    st.plotly_chart(fig, width="stretch",
                                    config={"displayModeBar": False})
                except Exception:
                    pass

    with ch2:
        st.markdown("##### Salary vs WAR \u2014 Player Efficiency")
        if "Salary_M" in roster_df.columns and "WAR_Total" in roster_df.columns:
            try:
                scat_df = roster_df.dropna(subset=["Salary_M", "WAR_Total"])
                if not scat_df.empty:
                    dot_colors = (
                        [_PG_COLORS[_PG_ORDER.index(p)] if p in _PG_ORDER else "#60a5fa"
                         for p in scat_df["pos_group"]]
                        if "pos_group" in scat_df.columns
                        else C.accent_blue
                    )
                    labels = scat_df["Player"].tolist() if "Player" in scat_df.columns else [""] * len(scat_df)
                    x_max_ref = float(scat_df["Salary_M"].max()) * 1.15
                    x_ref = np.linspace(0, x_max_ref, 60)
                    y_ref = x_ref * (total_war / max(total_cost, 0.1))

                    fig2 = go.Figure()
                    # Team avg efficiency reference line
                    fig2.add_trace(go.Scatter(
                        x=x_ref, y=y_ref, mode="lines",
                        line=dict(color="#f59e0b", dash="dash", width=1.5),
                        opacity=0.65, name="Team avg WAR/$M", hoverinfo="skip",
                    ))
                    # Player dots
                    fig2.add_trace(go.Scatter(
                        x=scat_df["Salary_M"].tolist(),
                        y=scat_df["WAR_Total"].tolist(),
                        mode="markers+text",
                        text=labels,
                        textposition="top center",
                        textfont=dict(size=8, color="#7aa2c0"),
                        marker=dict(color=dot_colors, size=10, opacity=0.9,
                                    line=dict(color="#1e3a5f", width=1)),
                        name="Players",
                        hovertemplate="<b>%{text}</b><br>Salary: $%{x:.1f}M<br>WAR: %{y:.1f}<extra></extra>",
                        showlegend=False,
                    ))
                    fig2.update_layout(**_pt(
                        title="Salary vs WAR \u2014 Player Efficiency",
                        xaxis=dict(title="Salary ($M)"),
                        yaxis=dict(title="WAR"),
                        height=450,
                        showlegend=True,
                        margin=dict(l=50, r=20, t=45, b=50),
                    ))
                    st.plotly_chart(fig2, width="stretch",
                                    config={"displayModeBar": False})
            except Exception:
                pass

    # -- Trade Analyzer --------------------------------------------------------
    with st.expander("\U0001f504 Trade Analyzer", expanded=False):
        _render_trade_analyzer(roster_df)

    # -- Simulated Wins Explanation --------------------------------------------
    with st.expander("\u2139\ufe0f How is Est. Win Total calculated?", expanded=False):
        est_wins = 47.7 + total_war
        st.markdown(f"""
**Formula:** `Est. Wins \u2248 47.7 + Total WAR`

**Your roster:** 47.7 (baseline) + {total_war:.1f} (total WAR) = **~{est_wins:.0f} wins**

---

**Replacement Level Baseline (47.7):**
A team composed entirely of "replacement-level" players \u2014 freely available from waivers or AAA \u2014 is expected to win roughly 47\u201348 games. This is the universally accepted MLB baseline for WAR calculations.

**WAR Contribution:**
Each 1.0 WAR represents one additional win above that replacement level player. A 5-WAR player produces 5 more wins than his freely-available replacement.

**Reference points:**
| Total WAR | Est. Wins | Outcome |
|-----------|-----------|---------|
| 25 WAR    | ~73 W     | Rebuilding |
| 30 WAR    | ~78 W     | Fringe contender |
| 35 WAR    | ~83 W     | Wild card threat |
| 40 WAR    | ~88 W     | Playoff team |
| 45 WAR    | ~93 W     | Division favorite |
| 50 WAR    | ~98 W     | World Series contender |

**Full Monte Carlo Simulation:**
The Optimizer tab runs a more sophisticated model: `wins = intercept + slope \u00d7 WAR` with per-player WAR variance sampled from a normal distribution, plus group-level shocks (SP, RP, hitters). This produces a full win distribution (P10\u2013P90) and playoff probability rather than a single estimate.

*Tip: A balanced roster of ~40 total WAR with payroll under $130M is the sweet spot for sustainable contention.*
""")

    # -- Best Fits -------------------------------------------------------------
    _render_best_fits(roster_df, budget_M, deps)

    # -- Export + clear --------------------------------------------------------
    st.markdown("---")
    dl_col, clr_col, _ = st.columns([2, 2, 6])
    with dl_col:
        csv_buf = io.StringIO()
        roster_df.to_csv(csv_buf, index=False)
        st.download_button(
            "\u2b07 Export Roster (CSV)",
            data=csv_buf.getvalue(),
            file_name="my_custom_roster.csv",
            mime="text/csv",
            key="sim_export_btn",
        )
    with clr_col:
        if st.button("Clear Roster", key="sim_clear_btn"):
            for k in ("sim_roster", "sim_roster_war", "sim_roster_cost"):
                st.session_state.pop(k, None)
            st.rerun()


def _render_player_card(player_name: str, combined_path: str, file_hash: str, deps=None):
    """Render a player detail card: headshot + year-by-year stats (2021-2025)."""
    import io as _io

    _data_url = _DL_data_url
    _R2_MODE = _DL_R2_MODE
    _R2_BASE_URL = _DL_R2_BASE_URL
    _ROOT_DIR = _DL_ROOT_DIR
    _HEADSHOTS_DIR = _DL_HEADSHOTS_DIR
    _RAZZBALL_PATH = _DL_RAZZBALL_PATH
    _cached_player_history = _DL_cached_player_history
    _cached_simulator_data = _DL_cached_simulator_data
    _cached_2026_payroll = _DL_cached_2026_payroll
    _cached_razzball = _DL_cached_razzball
    _dir_hash = _DL_dir_hash

    try:
        import requests as _requests
        _requests_available = True
    except ImportError:
        _requests_available = False

    all_df = _cached_player_history(combined_path, file_hash)
    player_df = all_df[all_df["Player"] == player_name].sort_values("Year").reset_index(drop=True)

    if player_df.empty:
        st.warning(f"No historical data found for **{player_name}**.")
        return

    latest = player_df.iloc[-1]
    pos      = str(latest.get("Position", "?"))
    team     = str(latest.get("Team", "?"))
    stage    = str(latest.get("Stage_Clean", "?"))
    age_val  = latest.get("Age", None)
    age_str  = f"{int(age_val)}" if pd.notna(age_val) else "\u2014"
    is_pitcher = pos in ("SP", "RP", "P", "TWP")

    # Try to get 2026 salary from payroll data first; fall back to 2025 historical
    _payroll_dir_26 = os.path.join(_ROOT_DIR, "2026 Payroll")
    sal_26_val = None
    try:
        if os.path.exists(_payroll_dir_26):
            _df26_card = _cached_2026_payroll(
                _payroll_dir_26, combined_path, _dir_hash(_payroll_dir_26)
            )
            _row26 = _df26_card[_df26_card["Player"] == player_name]
            if not _row26.empty and pd.notna(_row26.iloc[0].get("Salary_M")):
                sal_26_val = float(_row26.iloc[0]["Salary_M"])
    except Exception:
        pass

    if sal_26_val is not None:
        sal_label = "Sal '26"
        sal_str   = f"${sal_26_val:.1f}M"
    else:
        sal_val = latest.get("Salary_M", None)
        sal_label = "Sal (2025)"
        sal_str   = f"${float(sal_val):.1f}M" if pd.notna(sal_val) and sal_val else "\u2014"

    # Look up PPR, total contract, years left from simulator data
    _ind_path = _data_url("data/2025mlbshared.csv")
    try:
        _sim_df   = _cached_simulator_data(combined_path, _ind_path, file_hash)
        _sim_row  = _sim_df[_sim_df["Player"] == player_name]
        ppr_val   = float(_sim_row.iloc[0]["PPR"]) if (
            not _sim_row.empty and "PPR" in _sim_row.columns
            and pd.notna(_sim_row.iloc[0]["PPR"])
        ) else None
        total_ctrc_val = float(_sim_row.iloc[0]["Total_Contract_M"]) if (
            not _sim_row.empty and "Total_Contract_M" in _sim_row.columns
            and pd.notna(_sim_row.iloc[0]["Total_Contract_M"])
        ) else None
        yrs_left_val = int(_sim_row.iloc[0]["Yrs_Left"]) if (
            not _sim_row.empty and "Yrs_Left" in _sim_row.columns
            and pd.notna(_sim_row.iloc[0]["Yrs_Left"])
        ) else None
    except Exception:
        ppr_val = None
        total_ctrc_val = None
        yrs_left_val = None
    ppr_str = f"{ppr_val:.2f} WAR/$M" if ppr_val is not None else "\u2014"
    if total_ctrc_val is not None:
        yrs_total = 1 + (yrs_left_val or 0)
        ctrc_str = f"${total_ctrc_val:.0f}M / {yrs_total} yr{'s' if yrs_total != 1 else ''}"
    else:
        ctrc_str = "\u2014"

    img_col, stats_col = st.columns([1, 3])

    # ---- Headshot ----
    with img_col:
        headshot_path = os.path.join(_HEADSHOTS_DIR, f"{player_name}.png")
        img_bytes = None

        # In R2 mode serve from bucket; otherwise use local cache
        if _R2_MODE and _requests_available:
            try:
                _r2_resp = _requests.get(
                    f"{_R2_BASE_URL}/data/headshots/{player_name}.png", timeout=5)
                if _r2_resp.status_code == 200:
                    img_bytes = _r2_resp.content
            except Exception:
                pass
        elif os.path.exists(headshot_path):
            with open(headshot_path, "rb") as fh:
                img_bytes = fh.read()
        if img_bytes is None and _requests_available:
            # Look up MLBAM ID from razzball.csv
            rz_df = _cached_razzball(_RAZZBALL_PATH)
            mlbam_id = None
            if not rz_df.empty and "MLBAMID" in rz_df.columns:
                parts = player_name.strip().split(" ", 1)
                first = parts[0] if parts else ""
                last  = parts[1] if len(parts) > 1 else ""
                rz_match = rz_df[
                    rz_df.get("First", pd.Series(dtype=str)).astype(str).str.strip().str.lower().eq(first.lower()) &
                    rz_df.get("Last",  pd.Series(dtype=str)).astype(str).str.strip().str.lower().eq(last.lower())
                ]
                if not rz_match.empty:
                    raw_id = str(rz_match.iloc[0]["MLBAMID"]).strip()
                    if raw_id.isdigit():
                        mlbam_id = raw_id

            if mlbam_id:
                img_url = (
                    "https://img.mlbstatic.com/mlb-photos/image/upload/"
                    "d_people:generic:headshot:67:current.png/w_426,q_auto:best"
                    f"/v1/people/{mlbam_id}/headshot/67/current"
                )
                try:
                    resp = _requests.get(img_url, timeout=8)
                    if resp.status_code == 200:
                        img_bytes = resp.content
                        # Cache locally for next time (dev mode only)
                        if not _R2_MODE:
                            try:
                                os.makedirs(_HEADSHOTS_DIR, exist_ok=True)
                                with open(headshot_path, "wb") as fh:
                                    fh.write(img_bytes)
                            except OSError:
                                pass
                except Exception:
                    pass

        if img_bytes:
            st.image(img_bytes, width=200)
        else:
            st.markdown(
                "<div style='width:200px;height:200px;background:#1a1a2e;"
                "display:flex;align-items:center;justify-content:center;"
                "border-radius:10px;font-size:4rem;border:1px solid #2a3a5a;'>\u26be</div>",
                unsafe_allow_html=True,
            )

    # ---- Stats ----
    with stats_col:
        st.markdown(
            f"<h4 style='margin:0 0 0.3rem;color:{C.text_primary};'>{player_name}</h4>"
            f"<div style='font-size:0.85rem;color:{C.text_secondary};margin-bottom:0.5rem;'>"
            f"<b>{pos}</b> &nbsp;|&nbsp; {team} &nbsp;|&nbsp; "
            f"Stage: <b>{stage}</b> &nbsp;|&nbsp; Age: <b>{age_str}</b> &nbsp;|&nbsp; "
            f"{sal_label}: <b>{sal_str}</b> &nbsp;|&nbsp; "
            f"Full Contract: <b>{ctrc_str}</b>"
            f"</div>"
            f"<div style='margin-bottom:0.3rem;'>"
            f"<span style='background:#1a3a1a;color:#2ecc71;padding:3px 10px;"
            f"border-radius:6px;font-size:0.85rem;font-weight:700;'>"
            f"&#x1F4C8; Pay vs Play Ratio: {ppr_str}</span>"
            f"<span style='color:#666;font-size:0.75rem;margin-left:8px;'>"
            f"projected WAR over contract &divide; total contract $M</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Year-by-year table
        year_cols = ["Year", "Team", "WAR_Total"]
        if is_pitcher:
            year_cols += [c for c in ["ERA", "FIP", "IP", "G", "GS", "K9", "BB9", "WHIP"] if c in player_df.columns]
        else:
            year_cols += [c for c in ["HR", "RBI", "AVG", "OBP", "SLG", "OPS", "G", "AB", "R"] if c in player_df.columns]
        year_cols = [c for c in year_cols if c in player_df.columns]

        yby_df = player_df[year_cols].copy()
        yby_df["Year"] = yby_df["Year"].astype(int)

        fmt: dict = {"WAR_Total": "{:.1f}"}
        for stat, f in [("ERA", "{:.2f}"), ("FIP", "{:.2f}"), ("IP",  "{:.1f}"),
                         ("K9",  "{:.2f}"), ("BB9", "{:.2f}"), ("WHIP","{:.2f}"),
                         ("AVG", "{:.3f}"), ("OBP", "{:.3f}"), ("SLG", "{:.3f}"), ("OPS", "{:.3f}")]:
            if stat in yby_df.columns:
                fmt[stat] = f

        st.dataframe(
            yby_df.rename(columns={"WAR_Total": "WAR"}).style.format(fmt, na_rep="\u2014"),
            hide_index=True,
            use_container_width=True,
            height=min(70 + len(yby_df) * 35, 310),
        )

    # ---- WAR trend + WAR-per-dollar charts (full width below headshot+stats) ----
    if "WAR_Total" in player_df.columns:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_df = player_df[["Year", "WAR_Total"]].copy()
        chart_df["Year"]      = chart_df["Year"].astype(int)
        chart_df["WAR_Total"] = pd.to_numeric(chart_df["WAR_Total"], errors="coerce")

        if "Salary_M" in player_df.columns:
            chart_df["Salary_M"] = pd.to_numeric(player_df["Salary_M"], errors="coerce")
            chart_df["WAR_per_M"] = chart_df.apply(
                lambda r: round(r["WAR_Total"] / r["Salary_M"], 3)
                if pd.notna(r["Salary_M"]) and r["Salary_M"] > 0 and pd.notna(r["WAR_Total"])
                else None,
                axis=1,
            )

        has_dollar = "WAR_per_M" in chart_df.columns and chart_df["WAR_per_M"].notna().any()
        years = chart_df["Year"].tolist()
        wars  = chart_df["WAR_Total"].tolist()

        try:
            bar_cols = ["#22c55e" if (pd.notna(w) and w >= 0) else "#ef4444" for w in wars]
            yr_strs  = [str(y) for y in years]
            war_vals = [w if pd.notna(w) else 0 for w in wars]

            fig_war = go.Figure(data=[go.Bar(
                x=yr_strs, y=war_vals,
                marker_color=bar_cols, marker_line_width=0,
                text=[f"{v:.1f}" if v != 0 else "" for v in war_vals],
                textposition="outside", textfont=dict(color="#dbeafe", size=9),
                hovertemplate="%{x}: %{y:.1f} WAR<extra></extra>",
            )])
            fig_war.update_layout(**_pt(
                title="WAR by Year",
                yaxis=dict(title="WAR"),
                xaxis=dict(tickvals=yr_strs),
                height=300,
                margin=dict(l=50, r=20, t=45, b=50),
            ))

            if has_dollar:
                wpm = chart_df["WAR_per_M"].tolist()
                wpm_vals = [w if pd.notna(w) else 0 for w in wpm]
                fig_wpm = go.Figure(data=[go.Bar(
                    x=yr_strs, y=wpm_vals,
                    marker_color=C.accent_blue, marker_line_width=0,
                    text=[f"{v:.2f}" if v != 0 else "" for v in wpm_vals],
                    textposition="outside", textfont=dict(color="#dbeafe", size=9),
                    hovertemplate="%{x}: %{y:.2f} WAR/$M<extra></extra>",
                )])
                fig_wpm.update_layout(**_pt(
                    title="WAR per $M by Year",
                    yaxis=dict(title="WAR / $M"),
                    xaxis=dict(tickvals=yr_strs),
                    height=300,
                    margin=dict(l=50, r=20, t=45, b=50),
                ))
                col_war, col_wpm = st.columns(2)
                with col_war:
                    st.plotly_chart(fig_war, width="stretch",
                                    config={"displayModeBar": False})
                with col_wpm:
                    st.plotly_chart(fig_wpm, width="stretch",
                                    config={"displayModeBar": False})
            else:
                st.plotly_chart(fig_war, width="stretch",
                                config={"displayModeBar": False})
        except Exception:
            pass


def _render_best_fits(roster_df: pd.DataFrame, budget_M: float, deps=None) -> None:
    """Recommend available players that best complement the current roster and budget."""
    _ROSTER_TEMPLATE = _DL_ROSTER_TEMPLATE
    _ELIGIBLE_SLOTS_MAP = _DL_ELIGIBLE_SLOTS_MAP
    _OPTIONAL_SLOTS = _DL_OPTIONAL_SLOTS

    all_players = st.session_state.get("sim_df_full", pd.DataFrame())
    if all_players.empty:
        return

    on_roster     = set(roster_df["Player"].tolist()) if "Player" in roster_df.columns else set()
    current_cost  = float(roster_df["Salary_M"].fillna(0).sum()) if "Salary_M" in roster_df.columns else 0.0
    remaining     = budget_M - current_cost

    # Open roster slots
    _filled: dict[str, int] = {}
    if "pos_group" in roster_df.columns:
        for _pg in roster_df["pos_group"].tolist():
            for _slot in _ELIGIBLE_SLOTS_MAP.get(_pg, []):
                if _slot in _ROSTER_TEMPLATE:
                    _filled[_slot] = _filled.get(_slot, 0) + 1
    _missing = [s for s, need in _ROSTER_TEMPLATE.items()
                if s not in _OPTIONAL_SLOTS and _filled.get(s, 0) < need]

    # Reverse map: slot -> list of pos_groups that fill it
    _pg_for_slot: dict[str, list[str]] = {}
    for _pg, _slots in _ELIGIBLE_SLOTS_MAP.items():
        for _s in _slots:
            _pg_for_slot.setdefault(_s, []).append(_pg)

    st.markdown("---")
    st.markdown("### Best Available Additions")

    # -- Status bar ------------------------------------------------------------
    _sb1, _sb2 = st.columns([1, 2])
    with _sb1:
        _rem_color = "#06d6a0" if remaining >= 0 else "#ef4444"
        st.markdown(
            f"<div style='padding:0.6rem 1rem;background:#162030;border:1px solid #243f5c;"
            f"border-radius:8px;'>"
            f"<div style='font-size:0.7rem;color:#a8c8e8;margin-bottom:2px;'>Remaining Budget</div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{_rem_color};'>${remaining:.1f}M</div>"
            f"<div style='font-size:0.7rem;color:#6a8aaa;'>${budget_M:.0f}M total \u2014 ${current_cost:.1f}M committed</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with _sb2:
        if _missing:
            st.warning(f"**{len(_missing)} open position(s):** {', '.join(_missing)}  "
                       f"\u2014 these slots are not covered by your current roster.")
        else:
            st.success("All roster slots filled \u2014 additions below would improve depth or replace weak spots.")

    if remaining <= 0.5:
        st.info("Under $0.5M remaining. Remove a player to free up budget space.")
        return

    # -- Available pool --------------------------------------------------------
    avail = all_players[
        ~all_players["Player"].isin(on_roster)
        & all_players["Salary_M"].notna()
        & (all_players["Salary_M"] <= remaining + 0.01)
    ].copy()

    if avail.empty:
        st.info("No players available within the remaining budget.")
        return

    avail["_score"] = avail["WAR_Total"].fillna(0) / avail["Salary_M"].clip(lower=0.1)

    # Acquisition type: check 40-man roster CSV for actual roster membership
    _roster_40_bf = st.session_state.get("_sim_roster_40", pd.DataFrame())
    _rostered_set = set()
    _rostered_teams: dict = {}
    if not _roster_40_bf.empty and "full_name" in _roster_40_bf.columns:
        _rostered_set = set(_roster_40_bf["full_name"].str.lower().str.strip())
        _rostered_teams = dict(zip(
            _roster_40_bf["full_name"].str.lower().str.strip(),
            _roster_40_bf["team"],
        ))
    avail["_is_rostered"] = avail["Player"].str.lower().str.strip().isin(_rostered_set)
    avail["Curr. Team"] = avail["Player"].str.lower().str.strip().map(_rostered_teams).fillna("")
    avail["Acquisition"] = avail["_is_rostered"].map({True: "Trade Required", False: "Free Agent"})
    # Free agents rank slightly higher at equal WAR (more accessible)
    avail["_adj_score"] = avail["_score"] * np.where(avail["_is_rostered"], 1.0, 1.2)

    _show_cols = ["Player", "Curr. Team", "Position", "Stage_Clean", "Age", "WAR_Total", "Salary_M", "_score", "Acquisition"]
    _show_cols = [c for c in _show_cols if c in avail.columns]

    _fmt = {"WAR_Total": "{:.1f}", "Salary_M": "${:.1f}M", "_score": "{:.2f}", "Age": "{:.0f}"}
    _rename = {"WAR_Total": "fWAR", "Salary_M": "Sal $M", "_score": "fWAR/$M", "Stage_Clean": "Stage"}

    fit_t1, fit_t2, fit_t3 = st.tabs(["Best Value Adds", "Fill Open Positions", "Best Group Fit"])

    # -- Tab 1: Best value adds (highest W/$M, any position) -------------------
    with fit_t1:
        st.caption(
            "Top affordable players by WAR/$ efficiency. Green = below-market value, "
            "red = below league avg. Excludes players already on your roster."
        )
        _top = (
            avail.sort_values("_adj_score", ascending=False)
                 .head(20)[_show_cols]
                 .rename(columns=_rename)
                 .reset_index(drop=True)
        )

        def _clr_score(v):
            try:
                f = float(v)
                if f >= 0.5: return "background-color:#14532d;color:white"
                if f >= 0.3: return "background-color:#166534;color:white"
                if f < 0.1:  return "background-color:#7f1d1d;color:white"
            except Exception:
                pass
            return ""

        st.dataframe(
            _top.style
                .format({k: v for k, v in {**_fmt, "WAR": "{:.1f}", "Sal $M": "${:.1f}M",
                          "W/$M": "{:.2f}", "Age": "{:.0f}"}.items() if k in _top.columns})
                .map(_clr_score, subset=["W/$M"] if "W/$M" in _top.columns else []),
            hide_index=True, use_container_width=True, height=440,
        )

    # -- Tab 2: Fill open positions --------------------------------------------
    with fit_t2:
        if not _missing:
            st.success("No open positions \u2014 all slots are filled by your current roster.")
            st.caption("Additions below would improve depth at occupied positions.")
            for _pos in list(_ROSTER_TEMPLATE.keys())[:4]:
                _pgs = _pg_for_slot.get(_pos, [_pos])
                _sub = avail[avail["pos_group"].isin(_pgs)].nlargest(3, "WAR_Total")
                if not _sub.empty:
                    with st.expander(f"{_pos} depth", expanded=False):
                        st.dataframe(
                            _sub[_show_cols].rename(columns=_rename)
                                .style.format({k: v for k, v in {**_fmt, "WAR": "{:.1f}",
                                    "Sal $M": "${:.1f}M", "W/$M": "{:.2f}", "Age": "{:.0f}"
                                }.items() if k in _sub.rename(columns=_rename).columns}),
                            hide_index=True, use_container_width=True,
                        )
        else:
            for _slot in _missing:
                _pgs = _pg_for_slot.get(_slot, [_slot])
                _slot_avail = avail[avail["pos_group"].isin(_pgs)].nlargest(5, "_score")
                _label = f"**{_slot}**  \u2014 {len(_slot_avail)} affordable option(s)"
                if _slot_avail.empty:
                    _label += "  \u26a0 none in budget"
                with st.expander(_label, expanded=True):
                    if _slot_avail.empty:
                        st.warning(f"No affordable {_slot} available within ${remaining:.0f}M.")
                    else:
                        _sd = _slot_avail[_show_cols].rename(columns=_rename)
                        st.dataframe(
                            _sd.style.format({k: v for k, v in {**_fmt, "WAR": "{:.1f}",
                                "Sal $M": "${:.1f}M", "W/$M": "{:.2f}", "Age": "{:.0f}"
                            }.items() if k in _sd.columns}),
                            hide_index=True, use_container_width=True,
                        )

    # -- Tab 3: Best group fit (greedy knapsack) -------------------------------
    with fit_t3:
        st.caption(
            "Players selected greedily by W/$M efficiency to maximize WAR within your "
            "remaining budget. Prioritises open positions first, then best overall value."
        )

        # Prioritise players that fill open slots
        _open_pgs = set()
        for _slot in _missing:
            for _pg in _pg_for_slot.get(_slot, [_slot]):
                _open_pgs.add(_pg)

        avail_sorted = pd.concat([
            avail[avail["pos_group"].isin(_open_pgs)].sort_values("_score", ascending=False),
            avail[~avail["pos_group"].isin(_open_pgs)].sort_values("_score", ascending=False),
        ]).drop_duplicates("Player")

        _group: list = []
        _g_cost = 0.0
        _g_war  = 0.0
        for _, _row in avail_sorted.iterrows():
            _s = float(_row.get("Salary_M") or 0)
            if _g_cost + _s <= remaining + 0.01:
                _group.append(_row)
                _g_cost += _s
                _g_war  += float(_row.get("WAR_Total") or 0)
            if len(_group) >= 10:
                break

        if _group:
            _gc1, _gc2, _gc3 = st.columns(3)
            _gc1.metric("Players in group", len(_group))
            _gc2.metric("Added WAR",        f"+{_g_war:.1f}")
            _gc3.metric("Group cost",       f"${_g_cost:.1f}M",
                        delta=f"${remaining - _g_cost:.1f}M still available", delta_color="normal")

            _gdf = pd.DataFrame(_group)[_show_cols].rename(columns=_rename).reset_index(drop=True)
            # Flag players that fill an open slot
            _gdf["Fills Gap"] = [
                "\u2713" if any(pg in _open_pgs for pg in [r.get("pos_group", "")])
                else ""
                for r in _group
            ]
            st.dataframe(
                _gdf.style.format({k: v for k, v in {**_fmt, "WAR": "{:.1f}",
                    "Sal $M": "${:.1f}M", "W/$M": "{:.2f}", "Age": "{:.0f}"
                }.items() if k in _gdf.columns}),
                hide_index=True, use_container_width=True, height=380,
            )

            # Projected roster totals after group
            _new_war  = (float(roster_df["WAR_Total"].sum()) if "WAR_Total" in roster_df.columns else 0) + _g_war
            _new_cost = current_cost + _g_cost
            st.info(
                f"**Projected after additions** \u2014 "
                f"WAR: **{_new_war:.1f}** (~{47.7 + _new_war:.0f} wins)  |  "
                f"Payroll: **${_new_cost:.1f}M** (${budget_M - _new_cost:.1f}M remaining)"
            )
        else:
            st.info("No players fit within the remaining budget.")


# ── Main render function ────────────────────────────────────────────────────

def render(deps: dict | None = None):
    """Render the Roster Simulator page -- full 2-column layout with sticky bar.

    All data functions are now imported directly from utils.data_loading.
    ``deps`` is accepted for backward compatibility but ignored.
    """
    import io

    # All dependencies now come from module-level imports
    _data_url = _DL_data_url
    _read_csv = _DL_read_csv
    _R2_MODE = _DL_R2_MODE
    _DEFAULT_CONFIG = _DL_DEFAULT_CONFIG
    _ROOT_DIR = _DL_ROOT_DIR
    _load_base_config = _DL_load_base_config
    _resolve_data_path = _DL_resolve_data_path
    _file_hash = _DL_file_hash
    _dir_hash = _DL_dir_hash
    _cached_simulator_data = _DL_cached_simulator_data
    _cached_2026_payroll = _DL_cached_2026_payroll
    _cached_40man_roster = _DL_cached_40man_roster
    _cached_war_reliability = _DL_cached_war_reliability
    _cached_player_history = _DL_cached_player_history
    _ROSTER_TEMPLATE = _DL_ROSTER_TEMPLATE
    _ELIGIBLE_SLOTS_MAP = _DL_ELIGIBLE_SLOTS_MAP
    _OPTIONAL_SLOTS = _DL_OPTIONAL_SLOTS
    _PG_CHART_COLORS = _DL_PG_CHART_COLORS

    # ---- CSS -----------------------------------------------------------------
    st.markdown("<style>"
        "/* == Simulator page ========================================================= */"
        ".sim-page-hdr{background:linear-gradient(135deg," + C.bg_card_surface + " 0%," + C.bg_primary + " 100%);"
        "  border:1px solid " + C.border_primary + ";border-radius:12px;padding:0.65rem 1.1rem;margin-bottom:0.6rem;}"
        ".sim-page-hdr h2{margin:0;font-size:1.15rem;color:" + C.text_primary + ";font-weight:800;letter-spacing:-0.01em;}"
        ".sim-page-hdr .sim-sub{font-size:0.68rem;color:" + C.text_dim + ";margin-top:0.1rem;line-height:1.4;}"
        "/* Chips */"
        ".sim-chips{display:flex;gap:0.3rem;flex-wrap:wrap;margin-top:0.4rem;}"
        ".sim-chip{padding:0.15rem 0.55rem;border-radius:999px;font-size:0.67rem;"
        "  font-weight:700;border:1px solid transparent;letter-spacing:0.01em;}"
        ".sim-chip.budget{background:#12213a;color:#93c5fd;border-color:#1e3a6a;}"
        ".sim-chip.remain-ok  {background:#0a1f14;color:#4ade80;border-color:#14532d;}"
        ".sim-chip.remain-warn{background:#1f1400;color:#fbbf24;border-color:#78450c;}"
        ".sim-chip.remain-over{background:#1f0a0a;color:#fca5a5;border-color:#7f1d1d;}"
        ".sim-chip.players{background:#141428;color:#a5b4fc;border-color:#2d2d5a;}"
        ".sim-chip.slots-ok  {background:#0a1f14;color:#86efac;border-color:#14532d;}"
        ".sim-chip.slots-open{background:#1f1400;color:#fcd34d;border-color:#78450c;}"
        "/* Section divider */"
        ".sim-divider{border:none;border-top:1px solid " + C.border_primary + ";margin:0.5rem 0;}"
        "/* Pool header -- bolder, more prominent */"
        ".sim-pool-hdr{display:flex;justify-content:space-between;align-items:center;"
        "  margin-bottom:0.3rem;padding-bottom:0.3rem;border-bottom:2px solid #1e3a5f;}"
        ".sim-pool-hdr h4{margin:0;color:" + C.text_primary + ";font-size:1.0rem;font-weight:800;letter-spacing:-0.01em;}"
        ".sim-pool-badge{background:#1e3a5f;color:#93c5fd;padding:0.1rem 0.5rem;"
        "  border-radius:999px;font-size:0.67rem;font-weight:700;}"
        "/* \"Already on roster\" tag in pool */"
        ".sim-added-tag{display:inline-block;background:#14532d;color:#86efac;"
        "  font-size:0.60rem;font-weight:700;padding:0.05rem 0.4rem;"
        "  border-radius:999px;margin-left:0.3rem;vertical-align:middle;}"
        "/* Add-to-roster action bar */"
        ".sim-action-bar{background:#0d1e35;border:1px solid #1e3a5f;border-radius:10px;"
        "  padding:0.5rem 0.7rem;margin-top:0.4rem;display:flex;align-items:center;gap:0.6rem;}"
        ".sim-sel-summary{font-size:0.70rem;color:" + C.text_muted + ";flex:1;}"
        ".sim-sel-summary strong{color:" + C.text_primary + ";}"
        "/* Roster right panel */"
        ".sim-roster-hdr{display:flex;justify-content:space-between;align-items:center;"
        "  border-bottom:2px solid #1e3a5f;padding-bottom:0.3rem;margin-bottom:0.45rem;}"
        ".sim-roster-hdr .sim-rh-title{font-weight:800;color:" + C.text_primary + ";font-size:1.0rem;letter-spacing:-0.01em;}"
        ".sim-roster-hdr .sim-rh-count{font-size:0.67rem;color:" + C.text_dim + ";font-weight:600;}"
        "/* KPI grid -- compact */"
        ".sim-kpi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.25rem;margin-bottom:0.4rem;}"
        ".sim-kpi-box{background:#0d1e35;border:1px solid " + C.border_primary + ";border-radius:7px;"
        "  padding:0.32rem 0.4rem;text-align:center;}"
        ".sim-kpi-box .kv{font-size:1.0rem;font-weight:800;color:" + C.text_primary + ";line-height:1.1;}"
        ".sim-kpi-box .kl{font-size:0.57rem;color:" + C.text_dim + ";margin-top:1px;text-transform:uppercase;letter-spacing:0.04em;}"
        "/* Grade strip */"
        ".sim-grade-strip{display:flex;gap:0.25rem;margin:0.3rem 0 0.5rem;}"
        ".sim-grade-box{flex:1;text-align:center;padding:0.28rem 0.2rem;"
        "  background:#0d1e35;border:1px solid " + C.border_primary + ";border-radius:7px;}"
        ".sim-grade-box .gv{font-size:1.15rem;font-weight:800;}"
        ".sim-grade-box .gl{font-size:0.55rem;color:" + C.text_dim + ";margin-top:1px;text-transform:uppercase;letter-spacing:0.03em;}"
        ".sim-grade-box .gs{font-size:0.52rem;color:#2e4a62;}"
        "/* CBT planning slider block */"
        ".sim-cbt-block{background:#0d1e35;border:1px solid #1e3a5f;border-radius:9px;"
        "  padding:0.55rem 0.7rem;margin:0.5rem 0;}"
        ".sim-cbt-block .cb-title{font-size:0.68rem;font-weight:700;color:" + C.text_muted + ";"
        "  text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.3rem;}"
        ".sim-cbt-row{display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;margin-top:0.2rem;}"
        ".sim-cbt-pill{padding:0.15rem 0.55rem;border-radius:999px;font-size:0.67rem;font-weight:700;border:1px solid transparent;}"
        ".sim-cbt-pill.ok  {background:#0a1f14;color:#4ade80;border-color:#14532d;}"
        ".sim-cbt-pill.warn{background:#1f1400;color:#fbbf24;border-color:#78450c;}"
        ".sim-cbt-pill.over{background:#1f0a0a;color:#fca5a5;border-color:#7f1d1d;}"
        "/* Sticky bar */"
        ".mlb-sbar{position:fixed;bottom:0;top:auto;left:0;right:0;z-index:9998;"
        "  background:rgba(8,13,22,0.98);border-top:1px solid " + C.border_primary + ";"
        "  padding:0.28rem 1.5rem;display:flex;align-items:center;gap:1.6rem;"
        "  font-size:0.72rem;color:" + C.text_muted + ";}"
        ".mlb-sbar .sb-team{font-weight:800;color:" + C.text_primary + ";font-size:0.80rem;}"
        ".mlb-sbar .sb-stat{display:flex;flex-direction:column;align-items:center;}"
        ".mlb-sbar .sb-val{font-weight:700;font-size:0.84rem;color:#93c5fd;}"
        ".mlb-sbar .sb-lbl{font-size:0.57rem;color:" + C.text_dim + ";}"
        ".mlb-sbar-pad{height:44px;}"
        "</style>", unsafe_allow_html=True)

    # ---- Player card sub-page routing ----------------------------------------
    if st.session_state.get("view_player"):
        player_name = st.session_state["view_player"]
        if st.button("\u2190 Back to Roster Simulator", key="card_back_btn", type="secondary"):
            st.session_state.pop("view_player", None)
            st.rerun()
        st.markdown(f"## {player_name}")
        st.markdown("---")
        if _R2_MODE:
            _comb = _data_url("data/mlb_combined_2021_2025.csv")
        else:
            if not os.path.exists(_DEFAULT_CONFIG):
                st.error(f"Config file not found: {_DEFAULT_CONFIG}")
                return
            _base = _load_base_config(_DEFAULT_CONFIG)
            _sal  = _resolve_data_path(_base["raw_salary_war_path"], _DEFAULT_CONFIG)
            _comb = os.path.join(os.path.dirname(_sal), "mlb_combined_2021_2025.csv")
        _hash = _file_hash(_comb)
        _render_player_card(player_name, _comb, _hash, deps)
        return

    # ---- Data loading --------------------------------------------------------
    if _R2_MODE:
        combined_path    = _data_url("data/mlb_combined_2021_2025.csv")
        ind_2025_path    = _data_url("data/2025mlbshared.csv")
        payroll_2026_dir = _data_url("2026 Payroll")
    else:
        if not os.path.exists(_DEFAULT_CONFIG):
            st.error(f"Config file not found: {_DEFAULT_CONFIG}")
            return
        base_cfg         = _load_base_config(_DEFAULT_CONFIG)
        salary_abs       = _resolve_data_path(base_cfg["raw_salary_war_path"], _DEFAULT_CONFIG)
        data_dir         = os.path.dirname(salary_abs)
        combined_path    = os.path.join(data_dir, "mlb_combined_2021_2025.csv")
        ind_2025_path    = os.path.join(data_dir, "2025mlbshared.csv")
        payroll_2026_dir = os.path.join(_ROOT_DIR, "2026 Payroll")

        if not os.path.exists(combined_path):
            st.error(f"Data file not found:\n`{combined_path}`")
            return

    if _R2_MODE or os.path.exists(payroll_2026_dir):
        try:
            df = _cached_2026_payroll(payroll_2026_dir, combined_path, _dir_hash(payroll_2026_dir))
        except Exception as e:
            st.error(f"Error loading 2026 payroll data: {e}")
            st.exception(e)
            return
    else:
        try:
            df = _cached_simulator_data(combined_path, ind_2025_path, _file_hash(combined_path))
        except Exception as e:
            st.error(f"Error loading player data: {e}")
            st.exception(e)
            return

    # ---- Load 40-man roster CSV ----------------------------------------------
    _roster_40_path = _data_url("40man_rosters_2025.csv")
    _roster_40 = _cached_40man_roster(_roster_40_path, _file_hash(_roster_40_path))
    st.session_state["_sim_roster_40"] = _roster_40

    df = df.copy()
    if "WAR_Total" in df.columns and "pos_group" in df.columns:
        df["WAR_Pct"] = (
            df.groupby("pos_group")["WAR_Total"]
            .rank(pct=True).mul(100).round(0).astype("Int64")
        )
    st.session_state["sim_df_full"] = df
    try:
        st.session_state["sim_reliability"] = _cached_war_reliability(
            combined_path, _file_hash(combined_path)
        )
    except Exception:
        pass

    all_teams = sorted(df["Team"].dropna().unique().tolist())

    # ---- Roster state --------------------------------------------------------
    roster_records = st.session_state.get("sim_roster", [])
    roster_df      = pd.DataFrame(roster_records) if roster_records else pd.DataFrame()
    n_rostered     = len(roster_records)
    _rc_cost = float(roster_df["Salary_M"].fillna(0).sum()) if (
        not roster_df.empty and "Salary_M" in roster_df.columns
    ) else 0.0

    # Pre-compute roster metrics (used in header, right panel, sticky bar)
    _r_total_war  = float(roster_df["WAR_Total"].sum())  if (not roster_df.empty and "WAR_Total"   in roster_df.columns) else 0.0
    _r_total_cost = _rc_cost
    _r_wpm        = _r_total_war / max(_r_total_cost, 0.1)
    _r_dpw        = _r_total_cost / max(_r_total_war, 0.1)
    _r_est_wins   = 47.7 + _r_total_war
    _r_n_fa  = int((roster_df["Stage_Clean"] == "FA").sum())      if (not roster_df.empty and "Stage_Clean" in roster_df.columns) else 0
    _r_n_arb = int((roster_df["Stage_Clean"] == "Arb").sum())     if (not roster_df.empty and "Stage_Clean" in roster_df.columns) else 0
    _r_n_pre = int((roster_df["Stage_Clean"] == "Pre-Arb").sum()) if (not roster_df.empty and "Stage_Clean" in roster_df.columns) else 0

    # ---- Page header card ----------------------------------------------------
    _hdr_budget = st.session_state.get("sim_budget_input", 130)
    _hdr_remain = _hdr_budget - _rc_cost
    _hdr_filled: dict[str, int] = {}
    if not roster_df.empty and "pos_group" in roster_df.columns:
        for _pg in roster_df["pos_group"].tolist():
            for _sl in _ELIGIBLE_SLOTS_MAP.get(_pg, []):
                if _sl in _ROSTER_TEMPLATE:
                    _hdr_filled[_sl] = _hdr_filled.get(_sl, 0) + 1
    _hdr_open = [s for s, need in _ROSTER_TEMPLATE.items()
                 if s not in _OPTIONAL_SLOTS and _hdr_filled.get(s, 0) < need]

    _rem_cls   = "remain-ok" if _hdr_remain >= 10 else ("remain-warn" if _hdr_remain >= 0 else "remain-over")
    _slots_cls = "slots-ok" if not _hdr_open else "slots-open"
    _slots_txt = "\u2705 Roster Full" if not _hdr_open else f"\u26a0 {len(_hdr_open)} open slot{'s' if len(_hdr_open)!=1 else ''}"

    _cbt_lbl, _cbt_bg, _cbt_fg, _cbt_nxt, _cbt_note = _cbt_info(_hdr_budget)
    _cbt_nxt_txt = (f" \u00b7 ${_cbt_nxt - _hdr_budget:.0f}M to next tier" if _cbt_nxt else "")
    st.markdown(
        f"<div class='sim-page-hdr'>"
        f"<h2>\U0001f3ae Roster Simulator</h2>"
        f"<div class='sim-sub'>Build and analyze a custom MLB roster \u2014 select players from the pool and click <strong>Add to Roster</strong>."
        f"  Salaries are 2026 contracts; stats are 2025 figures.</div>"
        f"<div class='sim-chips'>"
        f"  <span class='sim-chip budget'>\U0001f4b0 Budget: ${_hdr_budget:.0f}M</span>"
        f"  <span class='sim-chip {_rem_cls}'>${_hdr_remain:+.1f}M remaining</span>"
        f"  <span class='sim-chip players'>\U0001f465 {n_rostered} players</span>"
        f"  <span class='sim-chip {_slots_cls}'>{_slots_txt}</span>"
        f"  <span class='sim-chip' style='background:{_cbt_bg};color:{_cbt_fg};border-color:{_cbt_fg}33;'"
        f"    title='{_cbt_note}'>"
        f"    \U0001f3e6 {_cbt_lbl}{_cbt_nxt_txt}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ---- Optimizer Controls --------------------------------------------------
    # Apply any pending budget value BEFORE the number_input widget renders.
    # (Streamlit forbids setting a widget's session-state key after it has rendered.)
    if "_sim_pending_budget" in st.session_state:
        st.session_state["sim_budget_input"] = st.session_state.pop("_sim_pending_budget")

    with st.expander("\u2699\ufe0f Optimizer Controls", expanded=(n_rostered == 0)):
        oc1, oc2, oc3, oc4, oc5 = st.columns([3, 2, 3, 2, 2])
        with oc1:
            load_team = st.selectbox("Team", all_teams, key="sim_load_team")
        with oc2:
            if "sim_budget_input" not in st.session_state:
                st.session_state["sim_budget_input"] = 130
            budget_M = st.number_input(
                "Budget $M", min_value=50, max_value=350, step=5,
                key="sim_budget_input",
            )
            # CBT tier indicator
            _b_lbl, _b_bg, _b_fg, _b_nxt, _b_note = _cbt_info(float(budget_M))
            _b_nxt_s = f" \u00b7 ${_b_nxt - budget_M:.0f}M to next" if _b_nxt else ""
            st.markdown(
                f"<div style='background:{_b_bg};color:{_b_fg};border-radius:6px;"
                f"padding:0.18rem 0.5rem;font-size:0.65rem;font-weight:600;margin-top:2px;'>"
                f"\U0001f3e6 {_b_lbl}{_b_nxt_s}"
                + (f"<div style='font-size:0.58rem;opacity:0.8;margin-top:1px;'>{_b_note}</div>" if _b_note else "")
                + "</div>",
                unsafe_allow_html=True,
            )
        with oc3:
            _locked_opts = (roster_df["Player"].tolist()
                            if not roster_df.empty and "Player" in roster_df.columns else [])
            st.multiselect(
                "Lock Players", _locked_opts, default=[],
                placeholder="None locked", key="sim_locked_players",
            )
        with oc4:
            st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
            _load_btn = st.button("\U0001f4c2 Load Roster", key="sim_load_team_btn",
                                  type="primary", use_container_width=True)
        with oc5:
            st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
            _reset_btn = st.button("\U0001f504 Reset", key="sim_reset_btn",
                                   type="secondary", use_container_width=True)

    # Handle Load Roster -- use 40-man roster CSV as source of truth
    if _load_btn:
        _tp = pd.DataFrame()
        if not _roster_40.empty and "team" in _roster_40.columns:
            # Filter 40-man roster to selected team
            _r40_team = _roster_40[_roster_40["team"] == load_team].copy()
            if not _r40_team.empty:
                # Build a lookup from the payroll/combined DataFrame
                _lkp = df.set_index("Player") if not df.empty else pd.DataFrame()
                _rows = []
                _matched = 0
                _unmatched_names = []
                for _, _r in _r40_team.iterrows():
                    _nm = str(_r["full_name"])
                    if not _lkp.empty and _nm in _lkp.index:
                        _pr = _lkp.loc[_nm]
                        if isinstance(_pr, pd.DataFrame):
                            _pr = _pr.iloc[0]
                        _rec = _pr.to_dict()
                        _rec["Player"]   = _nm
                        _rec["_dc_only"] = False
                        _matched += 1
                    else:
                        # Player on 40-man but no stats -- league minimum
                        _pos = str(_r.get("position", "UNK"))
                        _pg = ("SP" if _pos == "P" else
                               "RP" if _pos in ("RP", "CL") else
                               _pos if _pos in ("C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH") else
                               "OF" if _pos in ("OF", "LF", "CF", "RF") else "UNK")
                        _rec = {
                            "Player":      _nm,
                            "Team":        load_team,
                            "Position":    _pos,
                            "pos_group":   _pg,
                            "Salary_M":    0.74,
                            "Stage_Clean": "Pre-Arb",
                            "WAR_Total":   float("nan"),
                            "W_per_M":     float("nan"),
                            "PPR":         float("nan"),
                            "_dc_only":    True,
                        }
                        _unmatched_names.append(_nm)
                    # Add status from 40-man CSV
                    _rec["_40man_status"] = str(_r.get("status", "Active"))
                    _rec["on_40man"] = True
                    _rows.append(_rec)
                _tp = pd.DataFrame(_rows)
                # Store debug info for the debug expander
                st.session_state["_sim_40man_debug"] = {
                    "team": load_team,
                    "total": len(_r40_team),
                    "matched": _matched,
                    "unmatched": len(_unmatched_names),
                    "unmatched_names": _unmatched_names,
                }
        # Fallback: if 40-man CSV unavailable, use payroll/combined data
        if _tp.empty:
            _tp = df[df["Team"] == load_team].copy()
            _tp["_dc_only"] = False
        if not _tp.empty:
            st.session_state["sim_roster"] = _tp.to_dict("records")
            _ab = float(_tp["Salary_M"].fillna(0.74).sum())
            if _ab > 0:
                st.session_state["_sim_pending_budget"] = max(50, round(_ab))
            st.rerun()
        else:
            st.warning(f"No players found for {load_team}.")

    # Handle Reset
    if _reset_btn:
        for _k in ("sim_roster", "sim_roster_war", "sim_roster_cost"):
            st.session_state.pop(_k, None)
        st.rerun()

    # ---- Debug expander (only when ?debug=1) ---------------------------------
    if st.query_params.get("debug") == "1" and "_sim_40man_debug" in st.session_state:
        _dbg = st.session_state["_sim_40man_debug"]
        with st.expander("\U0001f50d 40-Man Roster Debug", expanded=True):
            st.markdown(f"**Team:** {_dbg['team']}")
            st.markdown(f"**Player count:** {_dbg['total']}  (must be <= 40)")
            st.markdown(f"**Matched to stats CSV:** {_dbg['matched']}")
            st.markdown(f"**No stats (league min):** {_dbg['unmatched']}")
            if _dbg["unmatched_names"]:
                st.markdown("**Unmatched player names:**")
                for _un in _dbg["unmatched_names"]:
                    st.markdown(f"- {_un}")

    # ---- Fix 5 -- Roster Status Explainer ------------------------------------
    _render_glossary([
        ("40-Man Roster", "40-Man Roster",
         "The full group of players under MLB contract with a team. Includes the active 26-man roster, "
         "injured list players, and optioned minor leaguers. Teams have 40 slots \u2014 adding a player beyond "
         "40 requires removing someone first."),
        ("26-Man Roster", "Active Roster",
         "The 26 players eligible to play in any given game. Managers choose their lineup from this group. "
         "Expands to 28 in September."),
        ("60-Day IL", "60-Day Injured List",
         "Players sidelined for at least 60 days. They do NOT count against the active 26-man roster but "
         "still occupy a 40-man spot. Teams use this to free active roster space for healthy players."),
        ("15-Day IL", "15-Day Injured List",
         "Short-term injury designation. Player is out at least 15 days but retains their 40-man spot. "
         "Common for minor injuries. Frees one active roster space."),
        ("Option", "Optional Assignment",
         "A 40-man player sent to the minor leagues. Still occupies a 40-man spot but frees an active roster "
         "space. Players can be optioned up to 3 times before requiring waivers to move down."),
        ("DFA", "Designated for Assignment",
         "Team removes a player from the 40-man roster. Player has 10 days to be traded, claimed on waivers, "
         "or released. Used to clear 40-man space for new additions."),
        ("Service Time", "MLB Service Time",
         "Days accrued on the active roster. 172 days = 1 full service year. Determines arb eligibility "
         "(3 years), free agency (6 years), and Super Two status (~2.17 years)."),
        ("Trade Required", "Trade Required Acquisition",
         "This player is on another team's 40-man roster. Adding them to your custom roster would require "
         "a trade in real life. Players not on any 40-man roster can be signed directly as free agents."),
    ], title="\U0001f4cb Understanding Roster Rules & Status", cols=2)

    # ---- Filters -------------------------------------------------------------
    # Detect handedness columns (Bats for hitters, Throws for pitchers)
    _hand_col = next((c for c in ("Bats", "Throws", "Hand", "B", "T") if c in df.columns), None)
    _has_hand = _hand_col is not None
    _hand_opts = (["All"] + sorted(df[_hand_col].dropna().unique().tolist())
                  if _has_hand else ["All", "L", "R", "S"])

    if _has_hand:
        fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8 = st.columns([3, 2, 2, 1.5, 2, 2, 2, 1])
    else:
        fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8 = st.columns([3, 2, 2, 1.5, 2, 2, 2, 1])

    with fc1:
        teams_sel = st.multiselect("Team(s)", all_teams, default=[],
                                   placeholder="All teams", key="sim_teams")
    with fc2:
        player_type_sel = st.selectbox("Type", ["All", "Pitcher", "Position Player"],
                                       key="sim_ptype")
    _pitcher_pos  = ["SP", "RP"]
    _position_pos = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]
    if player_type_sel == "Pitcher":
        _pos_opts = ["All"] + _pitcher_pos
    elif player_type_sel == "Position Player":
        _pos_opts = ["All"] + _position_pos
    else:
        _pos_opts = ["All"] + _position_pos + _pitcher_pos
    with fc3:
        position_sel = st.selectbox("Position", _pos_opts, key="sim_pos")
    with fc4:
        hand_sel = st.selectbox(
            "Hand",
            _hand_opts,
            key="sim_hand",
            disabled=not _has_hand,
            help=("Filter by batting/throwing hand." if _has_hand
                  else "Add a 'Bats' or 'Throws' column to your data CSV to enable this filter."),
        )
    _stage_opts = (["All"] + sorted(df["Stage_Clean"].dropna().unique().tolist())
                   if "Stage_Clean" in df.columns else ["All"])
    with fc5:
        stage_sel = st.selectbox("Stage", _stage_opts, key="sim_stage")
    with fc6:
        min_war = st.slider("Min WAR", min_value=-5.0, max_value=8.0,
                            value=0.0, step=0.5, key="sim_min_war")
    with fc7:
        name_search = st.text_input("Search", placeholder="Player name\u2026", key="sim_search")
    with fc8:
        st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
        if st.button("\u2715", key="sim_filter_clear", help="Clear all filters"):
            for _fk in ("sim_teams", "sim_ptype", "sim_pos", "sim_hand", "sim_stage",
                        "sim_min_war", "sim_search"):
                st.session_state.pop(_fk, None)
            st.rerun()

    # ---- Apply filters -------------------------------------------------------
    filtered = df.copy()
    if teams_sel:
        filtered = filtered[filtered["Team"].isin(teams_sel)]
    if player_type_sel == "Pitcher":
        filtered = filtered[filtered["pos_group"].isin(["SP", "RP"])]
    elif player_type_sel == "Position Player":
        filtered = filtered[~filtered["pos_group"].isin(["SP", "RP"])]
    if position_sel != "All":
        filtered = filtered[filtered["Position"] == position_sel]
    if _has_hand and hand_sel != "All":
        filtered = filtered[filtered[_hand_col] == hand_sel]
    if stage_sel != "All" and "Stage_Clean" in filtered.columns:
        filtered = filtered[filtered["Stage_Clean"] == stage_sel]
    if min_war > -5.0 and "WAR_Total" in filtered.columns:
        filtered = filtered[filtered["WAR_Total"] >= min_war]
    if name_search and name_search.strip():
        filtered = filtered[
            filtered["Player"].str.contains(name_search.strip(), case=False, na=False)
        ]
    filtered = filtered.sort_values("WAR_Total", ascending=False,
                                    na_position="last").reset_index(drop=True)

    # ---- Terms glossary ------------------------------------------------------
    _render_glossary([
        ("WAR",        "Wins Above Replacement",
         "How many wins a player adds vs a replacement-level player (e.g. a minor leaguer or bench fill-in). "
         "League average is ~2 WAR; All-Star level is 5+; MVP-caliber is 8+."),
        ("W/$M",       "WAR per $M (Season)",
         "Single-season WAR divided by 2026 salary. Higher = more efficient. "
         "League avg is roughly 0.3\u20130.5 WAR/$M for free agents; Pre-Arb players often exceed 2.0."),
        ("PPR",        "Pay-to-Performance Ratio (Contract)",
         "Total career WAR earned divided by total contract value ($M). Accounts for multi-year deals \u2014 "
         "a player on a 5-year $100M contract needs ~50 WAR to break even at market rate."),
        ("PPEL",       "Pay-Performance Efficiency Line",
         "The regression line of WAR vs Salary across all players. Sits at the league-average cost of a win. "
         "Players below the line are underpaid; above are overpaid relative to their output."),
        ("Stage",      "Contract Stage",
         "FA = Free Agent (negotiated market contract). "
         "Arb = Arbitration-eligible (salary set by panel or hearing). "
         "Pre-Arb = Club-controlled, first 3 years of service time (near league minimum ~$740K)."),
        ("CBT",        "Competitive Balance Tax",
         "MLB luxury tax on payrolls above $244M (2026). Tiers are $20M bands: "
         "$244M \u00b7 $264M (1st Apron) \u00b7 $284M \u00b7 $304M (2nd Apron). "
         "The two aprons also carry roster-building restrictions \u2014 "
         "1st Apron limits trades/signings; 2nd Apron adds draft pick penalties."),
    ], title="\U0001f4d6 Terms & Definitions")

    # ---- Vertical layout: Player Pool -> My Custom Roster --------------------

    # ---- TOP: Player Pool ----------------------------------------------------
    with st.container():
        # Players already on the custom roster
        _on_roster_names = {r.get("Player", "") for r in roster_records}
        _n_available = len([p for p in filtered["Player"] if p not in _on_roster_names])

        st.markdown(
            f"<div class='sim-pool-hdr'>"
            f"<h4>Player Pool</h4>"
            f"<span class='sim-pool-badge'>{_n_available:,} available"
            f" \u00b7 {len(filtered):,} shown \u00b7 {len(df):,} total</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        base_show  = ["Player", "Team", "Position", "Stage_Clean",
                      "Age", "WAR_Total", "WSR", "Salary_M", "W_per_M", "PPR"]
        hitter_ext = [c for c in ["HR", "AVG", "OBP"] if c in filtered.columns]
        pitch_ext  = [c for c in ["ERA", "IP"]         if c in filtered.columns]
        show_cols  = [c for c in base_show + hitter_ext[:2] + pitch_ext[:1]
                      if c in filtered.columns]
        display_df = filtered[show_cols].copy()

        # Mark players already on roster
        display_df["Added"] = display_df["Player"].apply(
            lambda p: "\u2713" if p in _on_roster_names else ""
        )
        # Show 40-man roster membership (current team or blank for true FAs)
        _r40_team_map = {}
        if not _roster_40.empty and "full_name" in _roster_40.columns:
            _r40_team_map = dict(zip(
                _roster_40["full_name"].str.lower().str.strip(),
                _roster_40["team"],
            ))
        display_df["40-Man"] = display_df["Player"].str.lower().str.strip().map(_r40_team_map).fillna("")
        show_cols_with_added = ["Added"] + show_cols + ["40-Man"]

        col_cfg: dict = {
            "Added":      st.column_config.TextColumn("\u2713",       width=30),
            "Player":     st.column_config.TextColumn("Player",   width="medium"),
            "Team":       st.column_config.TextColumn("Team",     width="small"),
            "Position":   st.column_config.TextColumn("Pos",      width="small"),
            "Stage_Clean":st.column_config.TextColumn("Stage",    width="small"),
            "Age":        st.column_config.NumberColumn("Age",    format="%d",     width="small"),
            "WAR_Total":  st.column_config.NumberColumn("WAR \u2195",  format="%.1f",   width="small",
                          help="Wins Above Replacement \u2014 how many wins this player adds vs a "
                               "replacement-level fill-in. Scale: 0\u20131 = bench, 2 = solid starter, "
                               "4\u20135 = All-Star, 7+ = MVP candidate."),
            "Salary_M":   st.column_config.NumberColumn("Sal $M \u2195", format="$%.1fM", width="small"),
            "W_per_M":    st.column_config.NumberColumn("W/$M \u2195", format="%.2f",   width="small",
                          help="Season WAR \u00f7 2026 salary. Higher = more efficient."),
            "PPR":        st.column_config.NumberColumn("Ctrc W/$M", format="%.2f", width="small",
                          help="Contract WAR per $M over the full contract length."),
            "WSR":        st.column_config.NumberColumn("WSR", format="%.2f", width="small",
                          help="WAR Stability Rating: mean WAR / (1 + std WAR). Higher = more consistent production. "
                               "Elite \u2265 3.5, Reliable \u2265 2.0, Volatile \u2265 1.0, Unstable < 1.0."),
            "40-Man":     st.column_config.TextColumn("40-Man", width="small",
                          help="Current 40-man roster team. Blank = free agent (can be signed directly)."),
        }
        for _st in hitter_ext[:2] + pitch_ext[:1]:
            if _st in ("AVG", "OBP"):
                col_cfg[_st] = st.column_config.NumberColumn(_st, format="%.3f", width="small")
            elif _st == "HR":
                col_cfg[_st] = st.column_config.NumberColumn("HR", format="%d",   width="small")
            elif _st == "ERA":
                col_cfg[_st] = st.column_config.NumberColumn("ERA", format="%.2f", width="small")
            else:
                col_cfg[_st] = st.column_config.NumberColumn(_st, format="%.1f", width="small")

        _fsig = f"{sorted(teams_sel)}{player_type_sel}{position_sel}{hand_sel}{stage_sel}{min_war}{name_search}"
        _fkey = hashlib.md5(_fsig.encode()).hexdigest()[:10]

        event = st.dataframe(
            display_df[show_cols_with_added],
            column_config=col_cfg,
            hide_index=True,
            use_container_width=True,
            height=480,
            on_select="rerun",
            selection_mode="multi-row",
            key=f"sim_table_{_fkey}",
        )

        sel_indices   = event.selection.rows if hasattr(event, "selection") else []
        selected_rows = (filtered.iloc[sel_indices].copy() if sel_indices
                         else pd.DataFrame(columns=filtered.columns))
        # Exclude players already on roster from the add action
        selected_new  = selected_rows[~selected_rows["Player"].isin(_on_roster_names)]
        n_sel         = len(selected_rows)
        n_new         = len(selected_new)

        # ---- Action bar ----
        _sel_war  = float(selected_new["WAR_Total"].sum()) if (n_new > 0 and "WAR_Total" in selected_new.columns) else 0.0
        _sel_cost = float(selected_new["Salary_M"].sum())  if (n_new > 0 and "Salary_M"  in selected_new.columns) else 0.0
        _budget_after = float(budget_M) - _rc_cost - _sel_cost

        _ba1, _ba2, _ba3 = st.columns([4, 3, 5])
        with _ba1:
            _add_clicked = st.button(
                f"\u2795 Add {n_new} to Roster" if n_new > 0 else (
                    f"\u2713 Already Added" if n_sel > 0 else "\u2191 Select players above"
                ),
                type="primary" if n_new > 0 else "secondary",
                use_container_width=True, key="sim_add_btn", disabled=(n_new == 0),
            )
        with _ba2:
            _card_clicked = st.button(
                "\U0001f0cf Player Card", type="secondary",
                use_container_width=True, key="sim_card_btn", disabled=(n_sel == 0),
                help="Open detailed player card for the first selected player.",
            )
        with _ba3:
            if n_new > 0:
                _after_color = "#4ade80" if _budget_after >= 10 else ("#fbbf24" if _budget_after >= 0 else "#fca5a5")
                st.markdown(
                    f"<div style='padding:0.3rem 0.5rem;background:#0d1e35;border:1px solid #1e3a5f;"
                    f"border-radius:8px;font-size:0.75rem;color:{C.text_secondary};line-height:1.5;'>"
                    f"<strong style='color:{C.text_primary};'>{n_new}</strong> new \u00b7 "
                    f"WAR <strong style='color:{C.text_primary};'>+{_sel_war:.1f}</strong> \u00b7 "
                    f"Cost <strong style='color:{C.text_primary};'>${_sel_cost:.1f}M</strong> \u00b7 "
                    f"Budget after <strong style='color:{_after_color};'>${_budget_after:+.1f}M</strong>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif n_sel > 0 and n_new == 0:
                st.caption("All selected players are already on your roster.")

        # Player comparison (2-4 selected)
        if 2 <= n_sel <= 4:
            with st.expander(f"\U0001f504 Compare {n_sel} Players", expanded=True):
                _render_player_comparison(selected_rows)

        # Handle Add to Roster
        if _add_clicked and n_sel > 0:
            _existing       = st.session_state.get("sim_roster", [])
            _existing_names = {r.get("Player") for r in _existing}
            _new_recs       = [r for r in selected_rows.to_dict("records")
                               if r.get("Player") not in _existing_names]
            st.session_state["sim_roster"] = _existing + _new_recs
            st.rerun()

        if _card_clicked and n_sel > 0:
            st.session_state["view_player"] = selected_rows.iloc[0]["Player"]
            st.rerun()

    # ---- BOTTOM: My Custom Roster --------------------------------------------
    with st.container():
        if not roster_records:
            st.markdown(
                "<div style='background:#0d1f38;border:1px dashed #1e3a5c;border-radius:12px;"
                "padding:2.5rem 1rem;text-align:center;color:#4a6a8a;margin-top:2rem;'>"
                "<div style='font-size:2rem;'>\U0001f3df</div>"
                "<div style='font-size:0.88rem;margin-top:0.5rem;color:#7aa2c0;'>Roster is empty</div>"
                "<div style='font-size:0.72rem;margin-top:0.3rem;'>Load a team or select players and click <strong>Add to Roster</strong></div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            _r_remaining = float(budget_M) - _r_total_cost
            _rem_color   = "#4ade80" if _r_remaining >= 0 else "#f87171"

            # Roster title + KPI grid
            st.markdown(
                f"<div class='sim-roster-hdr'>"
                f"  <span class='sim-rh-title'>My Custom Roster</span>"
                f"  <span class='sim-rh-count'>{n_rostered} players</span>"
                f"</div>"
                f"<div class='sim-kpi-grid'>"
                f"  <div class='sim-kpi-box'><div class='kv'>{_r_total_war:.1f}</div><div class='kl'>Total WAR</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv'>${_r_total_cost:.0f}M</div><div class='kl'>Payroll</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv' style='color:{_rem_color}'>${_r_remaining:+.0f}M</div><div class='kl'>Remaining</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv'>~{_r_est_wins:.0f}W</div><div class='kl'>Est. Wins</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv'>${_r_dpw:.1f}M</div><div class='kl'>$/WAR</div>"
                f"  <div style='font-size:0.58rem;color:{C.text_dim};margin-top:2px;'>${_r_total_cost / max(_r_total_war + 47.7, 1):.2f}M $/Win</div></div>"
                f"  <div class='sim-kpi-box'><div class='kv'>{_r_n_fa}/{_r_n_arb}/{_r_n_pre}</div><div class='kl'>FA/Arb/Pre</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Export / Clear
            _ec1, _ec2 = st.columns(2)
            with _ec1:
                _csv_buf = io.StringIO()
                roster_df.to_csv(_csv_buf, index=False)
                st.download_button(
                    "\u2b07 Export CSV", data=_csv_buf.getvalue(),
                    file_name="my_custom_roster.csv", mime="text/csv",
                    key="sim_export_btn", use_container_width=True,
                )
            with _ec2:
                if st.button("\U0001f5d1 Clear Roster", key="sim_clear_btn",
                             use_container_width=True, type="secondary"):
                    for _k in ("sim_roster", "sim_roster_war", "sim_roster_cost"):
                        st.session_state.pop(_k, None)
                    st.rerun()

            # Roster Grades strip
            _grades = _roster_grade(roster_df)
            _GRADE_COLORS = {
                "A+": "#00e676", "A": "#4caf50", "B+": "#8bc34a", "B": "#cddc39",
                "C":  "#ffc107", "D": "#ff9800", "F": "#f44336",
            }
            _gh = "<div class='sim-grade-strip'>"
            for _dim, (_gr, _sc) in _grades.items():
                _gc = _GRADE_COLORS.get(_gr, "#7aa2c0")
                _gh += (
                    f"<div class='sim-grade-box'>"
                    f"<div class='gv' style='color:{_gc};'>{_gr}</div>"
                    f"<div class='gl'>{_dim}</div>"
                    f"<div class='gs'>{_sc}/100</div>"
                    f"</div>"
                )
            _gh += "</div>"
            st.markdown(_gh, unsafe_allow_html=True)

            # Tabs: Roster | Analysis | Best Additions
            rt1, rt2, rt3 = st.tabs(["\U0001f4cb Roster", "\U0001f4ca Analysis", "\u2728 Best Additions"])

            with rt1:
                _reliability = st.session_state.get("sim_reliability", {})
                _show = [c for c in ["Player", "Position", "Stage_Clean", "Age",
                                     "WAR_Total", "Salary_M", "W_per_M", "PPR"]
                         if c in roster_df.columns]
                _edf = roster_df[_show].copy()
                _edf["Decision"]    = [_contract_decision(r) for r in roster_records]
                _edf["Consistency"] = [_reliability.get(r.get("Player", ""), {}).get("grade", "?")
                                       for r in roster_records]
                _edf.insert(0, "\u2715", False)
                _dc_flags = [bool(r.get("_dc_only", False)) for r in roster_records]
                def _hl_dc(row):
                    return (["background-color:#2d1f00;color:#fbbf24"] * len(row)
                            if _dc_flags[row.name] else [""] * len(row))
                if any(_dc_flags):
                    st.caption("\U0001f7e1 Amber = depth chart only (no MLB stats, league min salary assumed)")
                _edited = st.data_editor(
                    _edf.style.apply(_hl_dc, axis=1),
                    column_config={
                        "\u2715":           st.column_config.CheckboxColumn("\u2715",          width="small"),
                        "WAR_Total":   st.column_config.NumberColumn("WAR",           format="%.1f",   width="small"),
                        "Salary_M":    st.column_config.NumberColumn("Sal $M",        format="$%.1fM", width="small"),
                        "W_per_M":     st.column_config.NumberColumn("W/$M",          format="%.2f",   width="small"),
                        "PPR":         st.column_config.NumberColumn("Ctrc W/$M",     format="%.2f",   width="small"),
                        "Stage_Clean": st.column_config.TextColumn("Stage",           width="small"),
                        "Position":    st.column_config.TextColumn("Pos",             width="small"),
                        "Age":         st.column_config.NumberColumn("Age",           format="%d",     width="small"),
                        "Decision":    st.column_config.TextColumn("Action",          width="small"),
                        "Consistency": st.column_config.TextColumn("Consistency",     width="small"),
                    },
                    disabled=[c for c in _edf.columns if c != "\u2715"],
                    hide_index=True, use_container_width=True,
                    height=min(60 + (n_rostered + 1) * 35, 460),
                    key="roster_editor",
                )
                if st.button("Remove Selected", key="roster_remove_btn", type="secondary"):
                    _keep = [rec for rec, rm in zip(roster_records, _edited["\u2715"].tolist()) if not rm]
                    if len(_keep) < len(roster_records):
                        st.session_state["sim_roster"] = _keep
                        st.rerun()
                    else:
                        st.info("No players checked \u2014 tick the \u2715 column first.")
                st.markdown("##### Position Coverage")
                _render_position_coverage(roster_df, deps)

                # ---- CBT Threshold Planner ----
                st.markdown("<hr class='sim-divider'>", unsafe_allow_html=True)
                if "sim_cbt_adj" not in st.session_state:
                    st.session_state["sim_cbt_adj"] = 0
                _cbt_adj = st.slider(
                    "Luxury Tax Threshold Adjustment ($M)",
                    min_value=-20, max_value=20, step=1,
                    key="sim_cbt_adj",
                    help=(
                        "Shift the CBT base threshold ($244M for 2026) to model different "
                        "planning targets. Negative = more conservative, Positive = more aggressive."
                    ),
                )
                _eff_thresh = 244.0 + _cbt_adj
                _cbt_lbl2, _, _cbt_fg2, _, _cbt_note2 = _cbt_info(_r_total_cost)
                _rem_vs_thresh = _eff_thresh - _r_total_cost
                _pill_cls2 = "ok" if _rem_vs_thresh >= 0 else "over"
                _rem_vs_txt = (
                    f"${_rem_vs_thresh:.0f}M under target"
                    if _rem_vs_thresh >= 0
                    else f"${abs(_rem_vs_thresh):.0f}M over target"
                )
                _note_txt = _cbt_note2 if _cbt_note2 else "\u2713 Under CBT \u2014 no luxury tax"
                st.markdown(
                    f"<div class='sim-cbt-block'>"
                    f"  <div class='cb-title'>\U0001f9fe CBT Threshold Planner</div>"
                    f"  <div class='sim-cbt-row'>"
                    f"    <span style='font-size:0.75rem;color:{C.text_secondary};'>"
                    f"      Target: <strong style='color:{C.text_primary};'>${_eff_thresh:.0f}M</strong>"
                    f"    </span>"
                    f"    <span class='sim-cbt-pill {_pill_cls2}'>{_rem_vs_txt}</span>"
                    f"  </div>"
                    f"  <div style='font-size:0.63rem;color:{C.text_dim};margin-top:0.3rem;'>"
                    f"    <span style='color:{_cbt_fg2};font-weight:700;'>{_cbt_lbl2}</span>"
                    f"    {(' — ' + _note_txt) if _note_txt else ''}"
                    f"  </div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with rt2:
                _PG_ORDER  = ["SP", "RP", "C", "1B", "2B", "3B", "SS", "CF", "OF", "DH"]
                _PG_COLORS = _PG_CHART_COLORS
                if "pos_group" in roster_df.columns and "WAR_Total" in roster_df.columns:
                    _pg_war = (roster_df.groupby("pos_group")["WAR_Total"]
                               .sum().reindex(_PG_ORDER).dropna())
                    if not _pg_war.empty:
                        _cats  = _pg_war.index.tolist()
                        _vals  = _pg_war.values.tolist()
                        _bcols = [_PG_COLORS[_PG_ORDER.index(p)] for p in _cats if p in _PG_ORDER]
                        _ymax  = max(_vals) * 1.35 if _vals else 10
                        _fig1  = go.Figure(data=[go.Bar(
                            x=_cats, y=_vals, marker_color=_bcols, marker_line_width=0,
                            text=[f"{v:.1f}" for v in _vals], textposition="outside",
                            textfont=dict(color="#dbeafe", size=10),
                            hovertemplate="%{x}: %{y:.1f} WAR<extra></extra>",
                        )])
                        _fig1.update_layout(**_pt(
                            title="WAR by Position Group",
                            yaxis=dict(title="WAR", range=[0, _ymax]),
                            height=260, margin=dict(l=40, r=10, t=36, b=36),
                        ))
                        st.plotly_chart(_fig1, width="stretch",
                                        config={"displayModeBar": False})
                if "Salary_M" in roster_df.columns and "WAR_Total" in roster_df.columns:
                    _sc_df = roster_df.dropna(subset=["Salary_M", "WAR_Total"])
                    if not _sc_df.empty:
                        _dc2 = ([_PG_COLORS[_PG_ORDER.index(p)] if p in _PG_ORDER else "#60a5fa"
                                 for p in _sc_df["pos_group"]]
                                if "pos_group" in _sc_df.columns else C.accent_blue)
                        _xlim = float(_sc_df["Salary_M"].max()) * 1.15
                        _xr   = np.linspace(0, _xlim, 60)
                        _yr   = _xr * _r_wpm
                        _fig2 = go.Figure()
                        _fig2.add_trace(go.Scatter(
                            x=_xr, y=_yr, mode="lines",
                            line=dict(color="#f59e0b", dash="dash", width=1.5),
                            opacity=0.6, name="Avg WAR/$M", hoverinfo="skip",
                        ))
                        _fig2.add_trace(go.Scatter(
                            x=_sc_df["Salary_M"].tolist(), y=_sc_df["WAR_Total"].tolist(),
                            mode="markers+text",
                            text=(_sc_df["Player"].tolist() if "Player" in _sc_df.columns else []),
                            textposition="top center",
                            textfont=dict(size=7, color="#7aa2c0"),
                            marker=dict(color=_dc2, size=8, opacity=0.9,
                                        line=dict(color="#1e3a5f", width=1)),
                            hovertemplate="<b>%{text}</b><br>$%{x:.1f}M \u00b7 WAR %{y:.1f}<extra></extra>",
                            showlegend=False,
                        ))
                        _fig2.update_layout(**_pt(
                            title="Salary vs WAR",
                            xaxis=dict(title="Salary ($M)"),
                            yaxis=dict(title="WAR"),
                            height=260, showlegend=True,
                            margin=dict(l=40, r=10, t=36, b=36),
                        ))
                        st.plotly_chart(_fig2, width="stretch",
                                        config={"displayModeBar": False})
                # ---- Fix 6 -- Future Payroll Commitments ----
                if n_rostered >= 5:
                    st.markdown("<hr class='sim-divider'>", unsafe_allow_html=True)
                    st.markdown("##### \U0001f4c5 Future Payroll Commitments")

                    _fut_rows = []
                    for _, _fp in roster_df.iterrows():
                        _pn  = _fp.get("Player", "?")
                        _sal = float(_fp.get("Salary_M") or 0.74)
                        _stg = str(_fp.get("Stage_Clean", ""))
                        _age = int(_fp["Age"]) if pd.notna(_fp.get("Age")) else 28
                        # 2027/2028 from payroll data if available
                        _s27 = float(_fp["2027"]) if pd.notna(_fp.get("2027")) and float(_fp.get("2027") or 0) > 0 else None
                        _s28 = float(_fp["2028"]) if pd.notna(_fp.get("2028")) and float(_fp.get("2028") or 0) > 0 else None
                        # Estimate if not available
                        if _s27 is None:
                            if _stg == "Pre-Arb":
                                _s27 = 0.74
                            elif _stg == "Arb":
                                _s27 = round(_sal * 1.25, 1)
                            else:
                                _s27 = None  # FA -- unknown
                        if _s28 is None:
                            if _stg == "Pre-Arb":
                                _s28 = round(max(2.0, _sal * 3), 1)  # enters arb
                            elif _stg == "Arb":
                                _s28 = round(_sal * 1.5, 1)
                            else:
                                _s28 = None
                        _fut_rows.append({"Player": _pn, "Stage": _stg,
                                          "2026": _sal, "2027": _s27, "2028": _s28})

                    _fut_df = pd.DataFrame(_fut_rows)

                    # Stacked bar chart by stage
                    _stg_colors = {"Pre-Arb": "#22c55e", "Arb": "#f59e0b", "FA": C.accent_blue}
                    _years_lbl = ["2026", "2027", "2028"]
                    _fig_fut = go.Figure()
                    for stg, clr in _stg_colors.items():
                        _stg_sub = _fut_df[_fut_df["Stage"] == stg]
                        vals = [float(_stg_sub[yr].dropna().sum()) for yr in _years_lbl]
                        _fig_fut.add_trace(go.Bar(
                            x=_years_lbl, y=vals, name=stg,
                            marker_color=clr, opacity=0.85,
                            hovertemplate=f"{stg}<br>%{{x}}: $%{{y:.1f}}M<extra></extra>",
                        ))
                    # Unknown/FA years
                    _unk_vals = [0, 0, 0]
                    for yr_i, yr in enumerate(_years_lbl):
                        _unk_vals[yr_i] = float(_fut_df[_fut_df[yr].isna()].shape[0] * 0)  # placeholder
                    _fig_fut.add_hline(y=244, line_dash="dash", line_color="#f59e0b", opacity=0.5,
                                       annotation_text="CBT $244M", annotation_position="top right",
                                       annotation_font_color="#f59e0b")
                    _fig_fut.add_hline(y=float(budget_M), line_dash="dot", line_color=C.accent_blue, opacity=0.4,
                                       annotation_text=f"Budget ${budget_M}M", annotation_position="bottom right",
                                       annotation_font_color=C.accent_blue)
                    _fig_fut.update_layout(**_pt(
                        title="Committed Payroll by Stage (2026\u20132028)",
                        yaxis=dict(title="Total $M"), height=340,
                        barmode="stack", showlegend=True,
                        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
                    ))
                    st.plotly_chart(_fig_fut, width="stretch", config={"displayModeBar": False})

                    # Summary table
                    _fut_show = _fut_df[["Player", "Stage", "2026", "2027", "2028"]].copy()
                    _fut_show["2026"] = _fut_show["2026"].apply(lambda v: f"${v:.1f}M" if pd.notna(v) else "\u2014")
                    _fut_show["2027"] = _fut_show["2027"].apply(lambda v: f"~${v:.1f}M" if pd.notna(v) else "Free Agent")
                    _fut_show["2028"] = _fut_show["2028"].apply(lambda v: f"~${v:.1f}M" if pd.notna(v) else "Free Agent")
                    st.dataframe(_fut_show, hide_index=True, width="stretch",
                                 height=min(60 + n_rostered * 35, 400))
                    st.caption(
                        "2026 salaries reflect actual contracts. 2027\u20132028 figures for arbitration-eligible "
                        "players are estimates based on typical raise rates. Free agent years (shown as "
                        "'Free Agent') assume the player's contract expires."
                    )

                with st.expander("\U0001f504 Trade Analyzer", expanded=False):
                    _render_trade_analyzer(roster_df)

            with rt3:
                _render_best_fits(roster_df, float(budget_M), deps)

    # ---- Sticky roster summary strip -----------------------------------------
    if roster_records:
        _sb_remain = float(budget_M) - _r_total_cost
        _sb_rc = "#4ade80" if _sb_remain >= 0 else "#f87171"
        st.markdown(
            f"<div class='mlb-sbar'>"
            f"  <span class='sb-team'>\U0001f3df Custom Roster</span>"
            f"  <span class='sb-stat'><span class='sb-val'>{_r_total_war:.1f}</span>"
            f"    <span class='sb-lbl'>WAR</span></span>"
            f"  <span class='sb-stat'><span class='sb-val'>${_r_total_cost:.0f}M</span>"
            f"    <span class='sb-lbl'>Payroll</span></span>"
            f"  <span class='sb-stat'><span class='sb-val' style='color:{_sb_rc}'>${_sb_remain:+.0f}M</span>"
            f"    <span class='sb-lbl'>Remaining</span></span>"
            f"  <span class='sb-stat'><span class='sb-val'>~{_r_est_wins:.0f}W</span>"
            f"    <span class='sb-lbl'>Est. Wins</span></span>"
            f"  <span class='sb-stat'><span class='sb-val'>{n_rostered}</span>"
            f"    <span class='sb-lbl'>Players</span></span>"
            f"</div>"
            f"<div class='mlb-sbar-pad'></div>",
            unsafe_allow_html=True,
        )

    _render_feedback_widget("simulator")
