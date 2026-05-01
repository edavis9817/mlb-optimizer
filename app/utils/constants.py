"""MLB Toolbox — shared constants used across all pages and modules."""

# ══════════════════════════════════════════════════════════════════════════════
# Seasons & Years
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SEASON = 2025
SALARY_SEASON = 2026
YEAR_RANGE = list(range(2021, 2026))
FUTURE_YEARS = list(range(2026, 2033))

# ══════════════════════════════════════════════════════════════════════════════
# Financial Constants
# ══════════════════════════════════════════════════════════════════════════════

REPLACEMENT_LEVEL_WINS = 47.7
LEAGUE_MIN_SALARY_M = 0.780
LEAGUE_MIN_ANNUAL_INCREASE_M = 0.020

CBT_BASE = 244.0
CBT_FIRST_APRON = 264.0
CBT_TIER_3 = 284.0
CBT_SECOND_APRON = 304.0

CBT_THRESHOLDS_BY_YEAR = {
    2026: 244, 2027: 251, 2028: 258, 2029: 265,
    2030: 272, 2031: 279, 2032: 286,
}

CBT_TIERS: list[tuple[float, str, str, str, str]] = [
    (244.0, "Under CBT",       "#0c2218", "#22c55e", ""),
    (264.0, "Tier 1 \u2265$244M",   "#281a08", "#fbbf24",
     "Above CBT \u2014 paying luxury tax on overages"),
    (284.0, "1st Apron \u2265$264M", "#2d1408", "#f97316",
     "1st Apron \u2014 trade & signing restrictions apply"),
    (304.0, "Tier 3 \u2265$284M",   "#2d0c0c", "#ef4444",
     "Above 1st Apron \u2014 severe roster-building limits"),
    (9999., "2nd Apron \u2265$304M", "#1f0808", "#fca5a5",
     "2nd Apron \u2014 hardest restrictions, draft pick penalty"),
]

ARB_SALARY_ESTIMATES = {1: 2.0, 2: 4.0, 3: 7.0, 4: 10.0}

# ══════════════════════════════════════════════════════════════════════════════
# Service Time & Career Stages
# ══════════════════════════════════════════════════════════════════════════════

PRE_ARB_YEARS = 3
FREE_AGENT_YEARS = 6

STAGE_DISPLAY_MAP = {
    "Guaranteed": "Free Agent", "Arb-Eligible": "Arb", "Arb": "Arb",
    "Pre-Arb": "Pre-Arb", "FA": "Free Agent", "Off 40-Man": "Off 40-Man",
}

STAGE_COLORS = {
    "Pre-Arb": "#4ade80", "Arb": "#14b8a6",
    "Free Agent": "#60a5fa", "Off 40-Man": "#94a3b8",
}

STAGE_BG_COLORS = {
    "Pre-Arb": "#1a6b3a", "Arb": "#0c2a2a",
    "Guaranteed": "#0c1a2d", "Free Agent": "#0c1a2d",
}

# ══════════════════════════════════════════════════════════════════════════════
# Team Data
# ══════════════════════════════════════════════════════════════════════════════

PAYROLL_TEAM_MAP: dict[str, str] = {
    "Angels": "LAA", "Astros": "HOU", "Athletics": "ATH",
    "Blue Jays": "TOR", "Braves": "ATL", "Brewers": "MIL",
    "Cardinals": "STL", "Cubs": "CHC", "Diamondbacks": "ARI",
    "Dodgers": "LAD", "Giants": "SFG", "Guardians": "CLE",
    "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM",
    "Nationals": "WSN", "Orioles": "BAL", "Padres": "SDP",
    "Phillies": "PHI", "Pirates": "PIT", "Rangers": "TEX",
    "Rays": "TBR", "Red Sox": "BOS", "Reds": "CIN",
    "Rockies": "COL", "Royals": "KCR", "Tigers": "DET",
    "Twins": "MIN", "White Sox": "CHW", "Yankees": "NYY",
}

ABBR_TO_FULL: dict[str, str] = {v: k for k, v in PAYROLL_TEAM_MAP.items()}

TEAM_CITIES: dict[str, str] = {
    "ARI": "Arizona", "ATH": "Las Vegas", "ATL": "Atlanta", "BAL": "Baltimore",
    "BOS": "Boston", "CHC": "Chicago", "CHW": "Chicago", "CIN": "Cincinnati",
    "CLE": "Cleveland", "COL": "Colorado", "DET": "Detroit", "HOU": "Houston",
    "KCR": "Kansas City", "LAA": "Los Angeles", "LAD": "Los Angeles", "MIA": "Miami",
    "MIL": "Milwaukee", "MIN": "Minnesota", "NYM": "New York", "NYY": "New York",
    "PHI": "Philadelphia", "PIT": "Pittsburgh", "SDP": "San Diego", "SEA": "Seattle",
    "SFG": "San Francisco", "STL": "St. Louis", "TBR": "Tampa Bay", "TEX": "Texas",
    "TOR": "Toronto", "WSN": "Washington",
}

TEAM_COLORS: dict[str, tuple[str, str, str]] = {
    "ARI": ("#a71930", "#e3d4ad", "#1a0810"), "ATH": ("#003831", "#efb21e", "#0a1a14"),
    "ATL": ("#13274f", "#e85a6f", "#081420"), "BAL": ("#27251f", "#ff6b2b", "#0e0e0d"),
    "BOS": ("#0c2340", "#bd3039", "#081420"), "CHC": ("#0e3386", "#cc3433", "#081a2d"),
    "CHW": ("#27251f", "#c4ced4", "#0e0e0d"), "CIN": ("#000000", "#e8344e", "#0e0e0e"),
    "CLE": ("#00385d", "#e31937", "#081420"), "COL": ("#33006f", "#c4ced4", "#10082a"),
    "DET": ("#0c2340", "#fa4616", "#081420"), "HOU": ("#002d62", "#eb6e1f", "#081420"),
    "KCR": ("#004687", "#bd9b60", "#081420"), "LAA": ("#ba0021", "#4a7fbf", "#1a0810"),
    "LAD": ("#005a9c", "#ef3e42", "#081828"), "MIA": ("#00a3e0", "#ef3340", "#081a20"),
    "MIL": ("#12284b", "#ffc52f", "#081420"), "MIN": ("#002b5c", "#d31145", "#081420"),
    "NYM": ("#002d72", "#ff5910", "#081420"), "NYY": ("#003087", "#c4ced4", "#081420"),
    "PHI": ("#002d72", "#ef4444", "#081420"), "PIT": ("#27251f", "#fdb827", "#0e0e0d"),
    "SDP": ("#2f241d", "#ffc425", "#0e0c0a"), "SEA": ("#0c2c56", "#14b8a6", "#081420"),
    "SFG": ("#27251f", "#ff7a3d", "#0e0e0d"), "STL": ("#0c2340", "#ef4444", "#081420"),
    "TBR": ("#092c5c", "#8fbce6", "#081420"), "TEX": ("#003278", "#c0111f", "#081420"),
    "TOR": ("#134a8e", "#e63946", "#081828"), "WSN": ("#14225a", "#ef4444", "#081420"),
}

LOGO_FILE_NAMES: dict[str, str] = {
    "ARI": "Arizona Diamondbacks", "ATH": "Oakland Athletics", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox", "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox", "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers", "HOU": "Houston Astros",
    "KCR": "Kansas City Royals", "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins",
    "NYM": "New York Mets", "NYY": "New York Yankees", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
}

MLB_TEAM_ID_MAP: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KCR", 119: "LAD",
    120: "WSN", 121: "NYM", 133: "ATH", 134: "PIT", 135: "SDP", 136: "SEA",
    137: "SFG", 138: "STL", 139: "TBR", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CHW", 146: "MIA", 147: "NYY", 158: "MIL",
}

AL_DIVISIONS = [
    ("East",    ["BAL", "BOS", "NYY", "TBR", "TOR"]),
    ("Central", ["CHW", "CLE", "DET", "KCR", "MIN"]),
    ("West",    ["ATH", "HOU", "LAA", "SEA", "TEX"]),
]
NL_DIVISIONS = [
    ("East",    ["ATL", "MIA", "NYM", "PHI", "WSN"]),
    ("Central", ["CHC", "CIN", "MIL", "PIT", "STL"]),
    ("West",    ["ARI", "COL", "LAD", "SDP", "SFG"]),
]

ALL_TEAMS = sorted(ABBR_TO_FULL.keys())

# ══════════════════════════════════════════════════════════════════════════════
# Position & Roster Data
# ══════════════════════════════════════════════════════════════════════════════

POS_GROUP_MAP = {
    "C": "C", "1B": "1B", "3B": "3B", "IF": "1B",
    "2B": "2B", "SS": "SS",
    "LF": "OF", "RF": "OF", "OF": "OF", "CF": "CF",
    "SP": "SP", "RP": "RP", "TWP": "SP", "DH": "DH", "P": "SP",
}

ELIGIBLE_SLOTS_MAP = {
    "C": ["C"], "1B": ["1B"], "2B": ["2B"], "3B": ["3B"], "SS": ["SS"],
    "CF": ["CF"], "OF": ["LF", "RF"],
    "SP": ["SP"], "RP": ["RP"], "DH": ["DH"],
}

OPTIONAL_SLOTS = {"DH"}

ROSTER_TEMPLATE = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
    "LF": 1, "CF": 1, "RF": 1, "DH": 1, "BENCH": 3,
    "SP": 5, "RP": 8,
}

PITCHER_POSITIONS = {"SP", "RP", "P", "CL", "TWP"}

POS_GROUP_ORDER = ["SP", "RP", "C", "1B", "2B", "3B", "SS", "CF", "OF", "DH"]

PG_CHART_COLORS = [
    "#4873b8", "#7865b8", "#2e9080", "#5889c8", "#3898a8",
    "#458068", "#887898", "#b88840", "#5e9860", "#6a7890",
]

# ══════════════════════════════════════════════════════════════════════════════
# UI / Theme Constants
# ══════════════════════════════════════════════════════════════════════════════


def _is_light() -> bool:
    """Return True when the user has light mode active (default)."""
    try:
        import streamlit as st
        return st.session_state.get("light_mode", True)
    except Exception:
        return True


class _ThemeColors:
    """Dynamic color palette that adapts to light/dark mode.

    Access like module-level constants: ``C.bg_primary``, ``C.text_primary``, etc.
    Each access checks the current session state so the colors are always in sync.
    """

    # ── dark palette ────────────────────────────────────────────────────
    _DARK = dict(
        bg_primary="#141d2e", bg_card="#1c2a42", bg_card_surface="#18243a",
        bg_sidebar="#111a24", bg_dark="#0d1b2a", bg_input="#12202e",
        bg_hover="#1d2f47",
        border_primary="#1e3250", border_accent="#253d58",
        text_primary="#d6e8f8", text_secondary="#93b8d8",
        text_muted="#7a9ebc", text_dim="#4a687e", text_heading="#d6e8f8",
        accent_blue="#3b82f6", accent_green="#22c55e", accent_red="#ef4444",
        accent_orange="#f59e0b", accent_teal="#14b8a6", accent_gold="#c9a94e",
        eff_top="#14532d", eff_above="#1a3a20", eff_avg="",
        eff_below="#2d1f0c", eff_bottom="#2d0c0c",
        stage_bg_pre_arb="#1a6b3a", stage_bg_arb="#0c2a2a",
        stage_bg_guaranteed="#0c1a2d", stage_bg_fa="#0c1a2d",
    )

    # ── light palette ───────────────────────────────────────────────────
    _LIGHT = dict(
        bg_primary="#f0f2f6", bg_card="#ffffff", bg_card_surface="#f8fafc",
        bg_sidebar="#e8ebf0", bg_dark="#e2e8f0", bg_input="#ffffff",
        bg_hover="#f1f5f9",
        border_primary="#e2e8f0", border_accent="#cbd5e1",
        text_primary="#1a1a2e", text_secondary="#2d3748",
        text_muted="#4a5568", text_dim="#718096", text_heading="#1a1a2e",
        accent_blue="#2b5cc8", accent_green="#16a34a", accent_red="#dc2626",
        accent_orange="#d97706", accent_teal="#0d9488", accent_gold="#b8860b",
        eff_top="#dcfce7", eff_above="#ecfdf5", eff_avg="",
        eff_below="#fef3c7", eff_bottom="#fee2e2",
        stage_bg_pre_arb="#dcfce7", stage_bg_arb="#ccfbf1",
        stage_bg_guaranteed="#dbeafe", stage_bg_fa="#dbeafe",
    )

    def __getattr__(self, name: str) -> str:
        pal = self._LIGHT if _is_light() else self._DARK
        try:
            return pal[name]
        except KeyError:
            raise AttributeError(f"No theme color named {name!r}")


C = _ThemeColors()

# Legacy aliases — kept for any direct imports elsewhere
BG_PRIMARY = "#141d2e"
BG_CARD = "#1c2a42"
BG_SIDEBAR = "#111a24"
BG_DARK = "#0d1b2a"

BORDER_PRIMARY = "#1e3250"
BORDER_ACCENT = "#253d58"

TEXT_PRIMARY = "#d6e8f8"
TEXT_SECONDARY = "#93b8d8"
TEXT_MUTED = "#7a9ebc"
TEXT_DIM = "#4a687e"

ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#22c55e"
ACCENT_RED = "#ef4444"
ACCENT_ORANGE = "#f59e0b"
ACCENT_TEAL = "#14b8a6"
ACCENT_GOLD = "#c9a94e"

EFFICIENCY_TIER_COLORS = {
    "Top Tier": "#14532d", "Above Average": "#1a3a20",
    "Average": "", "Below Average": "#2d1f0c", "Bottom": "#2d0c0c",
}


def efficiency_tier_colors() -> dict:
    """Return efficiency tier bg colors for the active theme."""
    return {
        "Top Tier": C.eff_top, "Above Average": C.eff_above,
        "Average": C.eff_avg, "Below Average": C.eff_below, "Bottom": C.eff_bottom,
    }

# ══════════════════════════════════════════════════════════════════════════════
# fWAR Benchmarks (from 2021-2025 data)
# ══════════════════════════════════════════════════════════════════════════════

FWAR_AVG_PLAYER = 0.7
FWAR_ABOVE_AVG_PLAYER = 1.8
FWAR_ALL_STAR = 3.2
FWAR_MVP_CANDIDATE = 4.0
FWAR_AVG_TEAM = 33.8
FWAR_TOP5_TEAM_AVG = 51.8
FWAR_BOTTOM5_TEAM_AVG = 14.7
FWAR_PLAYOFF_THRESHOLD = 27.0
FWAR_WS_CHAMP_AVG = 42.4
FWAR_CONTENDER_FLOOR = 30

# ══════════════════════════════════════════════════════════════════════════════
# Feedback
# ══════════════════════════════════════════════════════════════════════════════

GOOGLE_SHEET_WEBHOOK = (
    "https://script.google.com/macros/s/"
    "AKfycbxfujsC1uRLp1bD9Bk4JyK6L8Z7ZT4fBgy6vaFRgwGOJc9NYfyX76-9cJ_64cvV6e-NMQ/exec"
)
