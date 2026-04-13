"""
streamlit_app.py
----------------
MLB Toolbox -- Interactive Streamlit UI.

Run with:
  streamlit run app/streamlit_app.py
  (from the mlb_optimizer/ root directory)

This file is a thin router.  All page logic lives under app/pages/.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_APP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from utils.theme import inject_meta_tags, get_current_page, render_nav_bar
from utils.data_loading import (
    DEFAULT_CONFIG as _DEFAULT_CONFIG,
    load_base_config as _load_base_config,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLB Toolbox",
    page_icon="\u26be",
    layout="wide",
    initial_sidebar_state="auto",
)

# Override Streamlit's default meta tags for link previews (iOS, iMessage, etc.)
inject_meta_tags()


# ---------------------------------------------------------------------------
# Page renderers (lazy imports keep startup fast)
# ---------------------------------------------------------------------------

def _page_home():
    from pages.home import render
    render()


def _page_rankings():
    from pages.rankings import render
    render()


def _page_league():
    from pages.player_analysis import render
    render()


def _page_simulator():
    from pages.roster_simulator import render
    render({})


def _page_roster_optimizer():
    from pages.roster_optimizer import render
    base_cfg = _load_base_config(_DEFAULT_CONFIG) if os.path.exists(_DEFAULT_CONFIG) else {}
    render(base_cfg)


def _page_team():
    from pages.team_analysis import render
    from utils.data_loading import (
        data_url, read_csv, load_enriched_roster, team_logo_url,
        fetch_2026_standings, fetch_2026_standings_full, fetch_2026_team_stats,
        R2_MODE,
    )
    render(data_url, read_csv, load_enriched_roster, team_logo_url,
           fetch_2026_standings, fetch_2026_standings_full, fetch_2026_team_stats,
           R2_MODE)


def _page_glossary():
    from pages.methodology import render
    render()


def _page_feedback():
    from pages.feedback import render
    render()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
_PAGES = {
    "home":             _page_home,
    "rankings":         _page_rankings,
    "league":           _page_league,
    "simulator":        _page_simulator,
    "roster_optimizer": _page_roster_optimizer,
    "team":             _page_team,
    "glossary":         _page_glossary,
    "feedback":         _page_feedback,
}


def main():
    page = get_current_page()

    # Skip nav bar on the home page -- it has its own navigation cards
    if page != "home":
        render_nav_bar()

    _PAGES.get(page, _page_home)()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
