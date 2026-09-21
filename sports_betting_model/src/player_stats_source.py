"""Shared fetch/cache for nflverse's weekly per-player stats.

Used by both `team_epa.py` (aggregated to team level) and `player_props.py`
(kept at player level for prop projections), so the fetch/cache logic lives
in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats_{year}.parquet"
)


def fetch_player_stats_year(year: int, force_refresh: bool = False) -> pd.DataFrame | None:
    """Returns None if this year's data hasn't been published yet (e.g. the
    current in-progress season may lag behind the schedule/odds data source).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"player_stats_{year}.parquet"
    if not force_refresh and cache_path.exists():
        return pd.read_parquet(cache_path)

    url = PLAYER_STATS_URL.format(year=year)
    head = requests.head(url, allow_redirects=True, timeout=15)
    if head.status_code == 404:
        return None

    df = pd.read_parquet(url)
    df.to_parquet(cache_path)
    return df


def fetch_player_stats_years(years: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """Concat multiple years, silently skipping any not yet published."""
    frames = []
    for y in years:
        yearly = fetch_player_stats_year(y, force_refresh=force_refresh)
        if yearly is None:
            print(f"  (no player_stats data published yet for {y}, skipping that season)")
            continue
        frames.append(yearly)
    return pd.concat(frames, ignore_index=True)
