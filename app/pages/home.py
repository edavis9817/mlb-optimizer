"""MLB Toolbox — Home / Landing page (Stadium Tunnel + Starting Lineup)."""

import streamlit as st

from utils.player_utils import (
    fix_player_name as _fix_player_name,
    headshot_url as _headshot_url,
)


def render(
    data_url,
    read_csv,
    cached_mlbam_lookup,
    razzball_path: str,
):
    """Landing page: Stadium Tunnel + Starting Lineup concept."""

    # Player headshots for tunnel walls (top fWAR leaders)
    _mlbam_lk = cached_mlbam_lookup(razzball_path)
    def _hs(name):
        mid = _mlbam_lk.get(_fix_player_name(name), "")
        return _headshot_url(mid, width=96) if mid else ""

    _left_players = [("Aaron Judge", _hs("Aaron Judge")),
                     ("Shohei Ohtani", _hs("Shohei Ohtani")),
                     ("Cal Raleigh", _hs("Cal Raleigh")),
                     ("Bobby Witt Jr.", _hs("Bobby Witt Jr."))]
    _right_players = [("Tarik Skubal", _hs("Tarik Skubal")),
                      ("Corbin Carroll", _hs("Corbin Carroll")),
                      ("Paul Skenes", _hs("Paul Skenes")),
                      ("Trea Turner", _hs("Trea Turner"))]

    # Dynamic stats for lineup badges
    try:
        _det = read_csv(data_url("efficiency_detail.csv"))
        _det25 = _det[_det["Year"] == 2025]
        _best_team = _det25.loc[_det25["dollar_gap_M"].idxmin()]
        _badge_rank = f"{_best_team['Team']} ${_best_team['dollar_gap_M']:.0f}M"
    except Exception:
        _badge_rank = "30 teams"

    _lineup = [
        ("rankings", "Rankings", "Efficiency rankings across all 30 MLB teams", "SS", _badge_rank, "#5dc9a5"),
        ("team", "Team Analysis", "Deep-dive into any team's spending and roster", "CF", "30 teams", "#7a9ebc"),
        ("league", "Player Analysis", "fWAR vs salary for every player (2021–2025)", "RF", "4,000+ players", "#7a9ebc"),
        ("simulator", "Roster Simulator", "Build custom rosters and optimize efficiency", "DH", "Full pool", "#7a9ebc"),
        ("glossary", "Methodology", "How every metric is calculated and sourced", "P", "Transparent", "#7a9ebc"),
    ]

    # ------------------------------------------------------------------
    # Build tunnel player images HTML
    # ------------------------------------------------------------------
    def _wall_imgs(players, side):
        html = ""
        for i, (name, url) in enumerate(players):
            if not url:
                continue
            op = max(0.15, 0.5 - i * 0.1)
            pos = f"{'left' if side == 'L' else 'right'}:12px;top:{20 + i * 72}px;"
            html += (f"<img src='{url}' style='position:absolute;{pos}width:48px;height:60px;"
                     f"object-fit:cover;border-radius:6px;opacity:{op};' "
                     f"onerror=\"this.style.display='none'\">")
        return html

    _left_wall = _wall_imgs(_left_players, "L")
    _right_wall = _wall_imgs(_right_players, "R")

    # Build lineup rows HTML
    _lineup_html = ""
    for i, (page, title, desc, pos, badge, badge_clr) in enumerate(_lineup):
        _num_bg = "#1a2a44"
        _num_clr = "#c9a94e" if i == 0 else "#4a90d9"
        _num_bdr = "#c9a94e" if i == 0 else "#253d58"
        _border = "border-bottom:1px solid #0e1828;" if i < len(_lineup) - 1 else ""
        _lineup_html += (
            f"<a href='?page={page}' target='_self' style='text-decoration:none;color:inherit;"
            f"display:flex;align-items:center;padding:12px 16px;{_border}"
            f"transition:background 0.15s;cursor:pointer;' "
            f"onmouseover=\"this.style.background='rgba(255,255,255,0.03)'\" "
            f"onmouseout=\"this.style.background='transparent'\">"
            f"<div style='width:28px;height:28px;border-radius:50%;background:{_num_bg};"
            f"border:1px solid {_num_bdr};display:flex;align-items:center;justify-content:center;"
            f"font-size:12px;font-weight:700;color:{_num_clr};flex-shrink:0;'>{i+1}</div>"
            f"<div style='flex:1;margin-left:12px;'>"
            f"<div style='font-size:13px;font-weight:500;color:#e8f4ff;'>{title}</div>"
            f"<div style='font-size:11px;color:#4a687e;margin-top:1px;'>{desc}</div></div>"
            f"<div style='font-size:11px;color:#4a687e;margin-right:12px;'>{pos}</div>"
            f"<div style='font-size:11px;color:{badge_clr};white-space:nowrap;'>{badge}</div>"
            f"</a>"
        )

    # ------------------------------------------------------------------
    # Full page HTML
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Render stadium tunnel + lineup card
    # ------------------------------------------------------------------
    st.markdown(f"""
    <style>
    @keyframes gold-pulse {{ 0%,100%{{opacity:0.5;}} 50%{{opacity:1;}} }}
    @keyframes row-in {{ from{{opacity:0;transform:translateX(-8px);}} to{{opacity:1;transform:translateX(0);}} }}
    </style>

    <!-- Tunnel Section -->
    <div style="position:relative;min-height:320px;background:#080e1a;overflow:hidden;
                border-radius:0 0 12px 12px;margin:-1rem -1rem 0 -1rem;padding:0 1rem;">

      <!-- Gold tunnel lines -->
      <div style="position:absolute;left:60px;top:0;bottom:0;width:1px;background:linear-gradient(180deg,transparent,#c9a94e22,transparent);"></div>
      <div style="position:absolute;right:60px;top:0;bottom:0;width:1px;background:linear-gradient(180deg,transparent,#c9a94e22,transparent);"></div>
      <div style="position:absolute;left:61px;top:30%;width:40px;height:1px;background:#c9a94e18;"></div>
      <div style="position:absolute;left:61px;top:55%;width:30px;height:1px;background:#c9a94e10;"></div>
      <div style="position:absolute;right:61px;top:35%;width:40px;height:1px;background:#c9a94e18;"></div>
      <div style="position:absolute;right:61px;top:60%;width:30px;height:1px;background:#c9a94e10;"></div>

      <!-- Left wall headshots -->
      {_left_wall}
      <!-- Right wall headshots -->
      {_right_wall}

      <!-- Dark gradient overlays on edges -->
      <div style="position:absolute;left:0;top:0;bottom:0;width:100px;
                  background:linear-gradient(90deg,#080e1a,transparent);z-index:1;"></div>
      <div style="position:absolute;right:0;top:0;bottom:0;width:100px;
                  background:linear-gradient(270deg,#080e1a,transparent);z-index:1;"></div>

      <!-- Center content -->
      <div style="position:relative;z-index:2;display:flex;flex-direction:column;align-items:center;
                  justify-content:center;min-height:320px;text-align:center;">
        <div style="font-size:11px;color:#c9a94e;letter-spacing:3px;text-transform:uppercase;
                    animation:gold-pulse 3s ease-in-out infinite;margin-bottom:8px;">ENTERING THE FIELD</div>
        <div style="font-size:28px;font-weight:500;color:#e8f4ff;margin-bottom:4px;">MLB Toolbox</div>
        <div style="font-size:13px;color:#4a687e;">Data-driven baseball analysis</div>
      </div>
    </div>

    <!-- Tunnel floor line -->
    <div style="height:3px;background:linear-gradient(90deg,transparent 5%,#1e3050 50%,transparent 95%);margin:0 -1rem;"></div>
    <div style="height:1px;background:linear-gradient(90deg,transparent 10%,#2a5a3a44 50%,transparent 90%);margin:0 -1rem 1.5rem;"></div>

    <!-- Lineup Card -->
    <div style="max-width:480px;margin:0 auto;background:#0b1422;border:1px solid #1e3050;border-radius:8px;overflow:hidden;">
      <!-- Card header -->
      <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 16px;
                  border-bottom:1px solid #0e1828;">
        <span style="font-size:13px;font-weight:500;color:#e8f4ff;">Today's lineup</span>
        <span style="font-size:10px;color:#4a687e;letter-spacing:1px;">2025 SEASON</span>
      </div>

      <!-- Lineup rows -->
      {_lineup_html}
    </div>

    <!-- Footer stats -->
    <div style="text-align:center;margin-top:1.5rem;margin-bottom:1rem;">
      <span style="font-size:11px;color:#2a4a3a;">150 team-seasons analyzed across 5 years</span>
    </div>
    """, unsafe_allow_html=True)
