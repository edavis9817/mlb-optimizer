"""MLB Toolbox — Home / Landing page (Stadium Tunnel + Starting Lineup)."""

import streamlit as st

from utils.constants import C
from utils.player_utils import (
    fix_player_name as _fix_player_name,
    headshot_url as _headshot_url,
)
from utils.data_loading import (
    data_url,
    read_csv,
    cached_mlbam_lookup,
    RAZZBALL_PATH as razzball_path,
)


def render(*_args, **_kwargs):
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
        _badge_rank = f"{_best_team['Team']} &#36;{_best_team['dollar_gap_M']:.0f}M"
    except Exception:
        _badge_rank = "30 teams"

    _lineup = [
        ("rankings", "Rankings", "Efficiency rankings across all 30 teams", "SS", _badge_rank, "#5dc9a5"),
        ("team", "Team Analysis", "Deep-dive into any team's spending and roster", "CF", "30 teams", C.text_muted),
        ("league", "Player Analysis", "fWAR vs salary for every player (2021\u20132025)", "RF", "4,000+ players", C.text_muted),
        ("simulator", "Roster Simulator", "Build custom rosters and optimize efficiency", "DH", "Full pool", C.text_muted),
        ("glossary", "Methodology", "How every metric is calculated and sourced", "P", "Transparent", C.text_muted),
    ]

    # ------------------------------------------------------------------
    # Build tunnel player images (absolutely positioned, larger, spaced)
    # ------------------------------------------------------------------
    _left_wall = ""
    _right_wall = ""
    _img_h, _img_w = 80, 64
    _spacing = 90
    _start_top = 30
    for i, (name, url) in enumerate(_left_players):
        if not url:
            continue
        op = max(0.15, 0.45 - i * 0.08)
        _left_wall += (
            f"<img src='{url}' style='position:absolute;left:10px;top:{_start_top + i * _spacing}px;"
            f"width:{_img_w}px;height:{_img_h}px;object-fit:cover;border-radius:6px;"
            f"opacity:{op};z-index:0;' onerror=\"this.style.display='none'\">")
    for i, (name, url) in enumerate(_right_players):
        if not url:
            continue
        op = max(0.15, 0.45 - i * 0.08)
        _right_wall += (
            f"<img src='{url}' style='position:absolute;right:10px;top:{_start_top + i * _spacing}px;"
            f"width:{_img_w}px;height:{_img_h}px;object-fit:cover;border-radius:6px;"
            f"opacity:{op};z-index:0;' onerror=\"this.style.display='none'\">")

    # ------------------------------------------------------------------
    # Build lineup rows
    # ------------------------------------------------------------------
    _lineup_html = ""
    for i, (page, title, desc, pos, badge, badge_clr) in enumerate(_lineup):
        _num_clr = "#c9a94e" if i == 0 else "#4a90d9"
        _num_bdr = "#c9a94e" if i == 0 else C.border_accent
        _border = "border-bottom:1px solid #0e1828;" if i < len(_lineup) - 1 else ""
        _lineup_html += (
            f"<a href='?page={page}' target='_self' style='text-decoration:none;color:inherit;"
            f"display:flex;align-items:center;padding:10px 16px;{_border}"
            f"transition:background 0.15s;cursor:pointer;' "
            f"onmouseover=\"this.style.background='rgba(255,255,255,0.03)'\" "
            f"onmouseout=\"this.style.background='transparent'\">"
            f"<div style='width:28px;height:28px;border-radius:50%;background:#1a2a44;"
            f"border:1px solid {_num_bdr};display:flex;align-items:center;justify-content:center;"
            f"font-size:12px;font-weight:700;color:{_num_clr};flex-shrink:0;'>{i+1}</div>"
            f"<div style='flex:1;margin-left:12px;min-width:0;'>"
            f"<div style='font-size:13px;font-weight:500;color:#e8f4ff;'>{title}</div>"
            f"<div style='font-size:11px;color:{C.text_dim};margin-top:1px;'>{desc}</div></div>"
            f"<div style='font-size:11px;color:{C.text_dim};width:28px;text-align:center;"
            f"flex-shrink:0;margin:0 8px;'>{pos}</div>"
            f"<div style='font-size:11px;color:{badge_clr};white-space:nowrap;"
            f"text-align:right;min-width:60px;flex-shrink:0;'>{badge}</div>"
            f"</a>")

    # ------------------------------------------------------------------
    # Full page: ONE seamless container
    # ------------------------------------------------------------------
    st.markdown("""
    <style>
    @keyframes gold-pulse { 0%,100%{opacity:0.5;} 50%{opacity:1;} }
    </style>
    """, unsafe_allow_html=True)

    # Build as a single flat string — no indentation that Streamlit could
    # misinterpret as Markdown code blocks.
    _page = (
        "<div style='background:" + C.bg_dark + ";border-radius:12px;overflow:hidden;position:relative;"
        "margin:-0.5rem -1rem 0;padding:0;'>"
        "<div style='position:relative;min-height:380px;'>"
        # Gold vertical tunnel lines
        "<div style='position:absolute;left:78px;top:0;bottom:0;width:1px;"
        "background:linear-gradient(180deg,transparent,#c9a94e22,transparent);'></div>"
        "<div style='position:absolute;right:78px;top:0;bottom:0;width:1px;"
        "background:linear-gradient(180deg,transparent,#c9a94e22,transparent);'></div>"
        # Gold horizontal stripes
        "<div style='position:absolute;left:79px;top:25%;width:45px;height:1px;background:#c9a94e18;'></div>"
        "<div style='position:absolute;left:79px;top:45%;width:35px;height:1px;background:#c9a94e10;'></div>"
        "<div style='position:absolute;left:79px;top:65%;width:25px;height:1px;background:#c9a94e0c;'></div>"
        "<div style='position:absolute;right:79px;top:30%;width:45px;height:1px;background:#c9a94e18;'></div>"
        "<div style='position:absolute;right:79px;top:50%;width:35px;height:1px;background:#c9a94e10;'></div>"
        "<div style='position:absolute;right:79px;top:70%;width:25px;height:1px;background:#c9a94e0c;'></div>"
        # Player headshots
        + _left_wall + _right_wall +
        # Dark gradient overlays
        "<div style='position:absolute;left:0;top:0;bottom:0;width:90px;"
        "background:linear-gradient(90deg," + C.bg_dark + " 0%,transparent 100%);z-index:1;'></div>"
        "<div style='position:absolute;right:0;top:0;bottom:0;width:90px;"
        "background:linear-gradient(270deg," + C.bg_dark + " 0%,transparent 100%);z-index:1;'></div>"
        # Center title
        "<div style='position:relative;z-index:2;text-align:center;padding:50px 80px 24px;'>"
        "<div style='font-size:11px;color:#c9a94e;letter-spacing:3px;text-transform:uppercase;"
        "animation:gold-pulse 3s ease-in-out infinite;margin-bottom:10px;'>ENTERING THE FIELD</div>"
        "<div style='font-size:28px;font-weight:500;color:#e8f4ff;margin-bottom:6px;'>MLB Toolbox</div>"
        "<div style='font-size:13px;color:" + C.text_dim + ";'>Data-driven baseball analysis</div>"
        "</div>"
        # Lineup card
        "<div style='position:relative;z-index:2;padding:0 60px 0;'>"
        "<div style='max-width:500px;margin:0 auto;background:" + C.bg_dark + ";"
        "border:1px solid " + C.border_primary + ";border-radius:8px;overflow:hidden;'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;"
        "padding:10px 16px;border-bottom:1px solid #0e1828;'>"
        "<span style='font-size:13px;font-weight:500;color:#e8f4ff;'>Today's lineup</span>"
        "<span style='font-size:10px;color:" + C.text_dim + ";letter-spacing:1px;'>2025 SEASON</span>"
        "</div>"
        + _lineup_html +
        "</div></div>"
        "<div style='height:30px;'></div>"
        "</div>"
        # Tunnel floor
        "<div style='height:3px;background:linear-gradient(90deg,transparent 5%," + C.border_primary + " 50%,transparent 95%);"
        "margin:0 40px;'></div>"
        "<div style='height:1px;background:linear-gradient(90deg,transparent 15%,#2a5a3a44 50%,transparent 85%);"
        "margin:4px 60px 0;opacity:0.6;'></div>"
        # Footer
        "<div style='text-align:center;padding:14px 0 18px;'>"
        "<span style='font-size:11px;color:#2a4a3a;'>150 team-seasons analyzed across 5 years</span>"
        "</div>"
        "</div>"
    )
    st.markdown(_page, unsafe_allow_html=True)
