"""
mlb_rosters.py
--------------
Fetch 40-man rosters from the official MLB Stats API via the MLB-StatsAPI package.

Usage:
    from src.mlb_rosters import fetch_40man_roster, fetch_all_40man_rosters, TEAM_IDS
"""

import logging
import time
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# ── Full MLB team-ID map ────────────────────────────────────────────────────
TEAM_IDS: dict[str, int] = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CHW": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KCR": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "ATH": 133,
    "PHI": 143, "PIT": 134, "SDP": 135, "SEA": 136, "SFG": 137,
    "STL": 138, "TBR": 139, "TEX": 140, "TOR": 141, "WSN": 120,
}

# Reverse lookup: team_id → abbreviation
_ID_TO_ABBR: dict[int, str] = {v: k for k, v in TEAM_IDS.items()}

# Statuses that represent players currently on the 40-man roster
_ACTIVE_STATUSES = {"Active", "Injured 60-Day", "Paternity", "Bereavement",
                    "Restricted", "Suspended"}


def fetch_40man_roster(team_id: int, season: int = 2025) -> pd.DataFrame:
    """Fetch a single team's 40-man roster from the MLB Stats API (JSON endpoint).

    Returns a DataFrame with columns:
        player_id, full_name, position, jersey_number, status, team
    """
    import statsapi

    try:
        data = statsapi.get("team_roster", {
            "teamId": team_id,
            "rosterType": "40Man",
            "season": season,
        })
    except Exception as exc:
        logger.warning("Failed to fetch roster for team_id=%d season=%d: %s", team_id, season, exc)
        return pd.DataFrame(columns=["player_id", "full_name", "position", "jersey_number", "status", "team"])

    team_abbr = _ID_TO_ABBR.get(team_id, str(team_id))
    rows: list[dict] = []

    for entry in data.get("roster", []):
        status_desc = entry.get("status", {}).get("description", "")
        # Only keep players currently on the 40-man (Active, IL, etc.)
        if status_desc not in _ACTIVE_STATUSES:
            continue

        person = entry.get("person", {})
        pos = entry.get("position", {})

        rows.append({
            "player_id": person.get("id", 0),
            "full_name": person.get("fullName", ""),
            "position": pos.get("abbreviation", ""),
            "jersey_number": entry.get("jerseyNumber", ""),
            "status": status_desc,
            "team": team_abbr,
        })

    return pd.DataFrame(rows)


# ── TTL-aware LRU cache (1-hour expiry) ────────────────────────────────────
_CACHE_TTL = 3600  # seconds


def _ttl_hash() -> int:
    """Return an int that changes every _CACHE_TTL seconds."""
    return int(time.time() // _CACHE_TTL)


@lru_cache(maxsize=4)
def fetch_all_40man_rosters(season: int = 2025, _ttl: int = 0) -> pd.DataFrame:
    """Fetch 40-man rosters for all 30 MLB teams.

    The ``_ttl`` parameter is auto-set for cache-busting every hour.
    Call as: ``fetch_all_40man_rosters(season=2025, _ttl=_ttl_hash())``
    """
    frames: list[pd.DataFrame] = []
    for abbr, tid in sorted(TEAM_IDS.items()):
        logger.info("Fetching 40-man roster: %s (id=%d, season=%d)", abbr, tid, season)
        df = fetch_40man_roster(tid, season=season)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["player_id", "full_name", "position", "jersey_number", "status", "team"])

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total 40-man roster players fetched: %d across %d teams", len(combined), len(frames))
    return combined
