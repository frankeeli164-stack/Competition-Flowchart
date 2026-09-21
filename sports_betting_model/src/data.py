"""Fetch and cache NFL game results + historical closing lines.

Source: nflverse/nfldata `games.csv`, which has one row per NFL game since 1999
with final scores plus closing moneyline/spread/total lines. This is the same
data `nfl_data_py.import_schedules()` uses, but we fetch it straight from
raw.githubusercontent.com because the package's own default host
(habitatring.com) is blocked by outbound network policy in some environments.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GAMES_CACHE = DATA_DIR / "games.csv"


def fetch_games(force_refresh: bool = False) -> pd.DataFrame:
    """Load the full games table, using a local cache unless `force_refresh`."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force_refresh or not GAMES_CACHE.exists():
        resp = requests.get(GAMES_URL, timeout=30)
        resp.raise_for_status()
        GAMES_CACHE.write_bytes(resp.content)
    return pd.read_csv(GAMES_CACHE, low_memory=False)


def load_completed_games(min_season: int = 1999, force_refresh: bool = False) -> pd.DataFrame:
    """Games with a known final score, regular season + playoffs, sorted chronologically.

    Drops games missing a result (future/unplayed games still in the source file)
    and games before `min_season`.
    """
    df = fetch_games(force_refresh=force_refresh)
    df = df[df["season"] >= min_season].copy()
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()

    df["gameday"] = pd.to_datetime(df["gameday"])
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    # result == 0 is a tie; keep it out of the binary win/loss label used for
    # moneyline classification but leave the row in for Elo margin-of-victory updates.
    df["is_tie"] = (df["home_score"] == df["away_score"]).astype(int)

    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return df
