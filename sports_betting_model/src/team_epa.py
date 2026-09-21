"""Team-level offensive/defensive EPA-per-play, built from weekly player stats.

This is the feature the original model was missing: `elo.py` and `features.py`
only ever saw final scores. EPA (expected points added) per play is a much
richer measure of how well a team is actually playing, and is the standard
building block of every serious NFL prediction model (538, nflfastR-based
models, etc).

Source: nflverse's `player_stats_{year}.parquet` release files (one row per
player per week), which already have `passing_epa`/`rushing_epa`/
`receiving_epa` computed from play-by-play. We aggregate to team-week level
ourselves rather than pulling full play-by-play, which is much smaller and
faster to fetch (~5-6k rows/season vs. millions of play-level rows) while
still capturing the core signal.

Scope note: this only covers REGULAR SEASON games. Postseason week numbers
don't line up cleanly with regular-season week numbers across data sources,
and postseason is <5% of games, so it's deliberately left out rather than
risking a subtle join bug that leaks data across weeks.
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

ROLLING_WINDOW = 8
MIN_PERIODS = 3


def _fetch_player_stats_year(year: int, force_refresh: bool = False) -> pd.DataFrame | None:
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


def _team_week_off_epa(years: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """One row per (season, week, team): offensive EPA per play, regular season only.

    Sums passing_epa + rushing_epa (team-level passing/rushing production).
    receiving_epa is deliberately excluded: it re-attributes the same pass
    play's EPA to the targeted receiver, so summing it alongside passing_epa
    would double-count every passing play's value.
    """
    frames = []
    for y in years:
        yearly = _fetch_player_stats_year(y, force_refresh=force_refresh)
        if yearly is None:
            print(f"  (no player_stats data published yet for {y}, skipping EPA for that season)")
            continue
        frames.append(yearly)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"] == "REG"]

    grouped = (
        df.groupby(["season", "week", "recent_team"])
        .agg(
            off_epa=("passing_epa", "sum"),
            rushing_epa_sum=("rushing_epa", "sum"),
            attempts=("attempts", "sum"),
            carries=("carries", "sum"),
        )
        .reset_index()
    )
    grouped["off_epa"] = grouped["off_epa"] + grouped["rushing_epa_sum"]
    grouped["plays"] = grouped["attempts"] + grouped["carries"]
    grouped["off_epa_per_play"] = grouped["off_epa"] / grouped["plays"].replace(0, pd.NA)
    grouped = grouped.rename(columns={"recent_team": "team"})
    return grouped[["season", "week", "team", "off_epa_per_play"]]


def _team_game_log(reg: pd.DataFrame, team_week: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per team-appearance-in-a-game, with that team's own
    off_epa_per_play for the week and their opponent's (= what the team's
    defense faced that week), unsorted.
    """
    home_side = reg[["game_id", "season", "week", "gameday", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away_side = reg[["game_id", "season", "week", "gameday", "home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    long_log = pd.concat([home_side, away_side], ignore_index=True)

    long_log = long_log.merge(team_week, on=["season", "week", "team"], how="left")
    long_log = long_log.merge(
        team_week.rename(columns={"team": "opponent", "off_epa_per_play": "opp_off_epa_per_play"}),
        on=["season", "week", "opponent"],
        how="left",
    )
    return long_log


def build_trailing_epa_features(
    games: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    min_periods: int = MIN_PERIODS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Add trailing (pre-game, no-leakage) offensive and defensive EPA/play
    features to a REGULAR-SEASON-only subset of `games`.

    Returns a filtered copy of `games` (game_type == 'REG' only) with four new
    columns: home_off_epa, home_def_epa_allowed, away_off_epa, away_def_epa_allowed.
    Each is a rolling mean of the team's own (or opponents', for defense) EPA/play
    over their prior `window` games, computed with `.shift(1)` so the current
    game is never included in its own feature.
    """
    reg = games[games["game_type"] == "REG"].copy()
    years = sorted(reg["season"].unique().tolist())

    team_week = _team_week_off_epa(years, force_refresh=force_refresh)
    long_log = _team_game_log(reg, team_week)
    long_log = long_log.sort_values(["team", "gameday"])

    long_log["trailing_off_epa"] = (
        long_log.groupby("team")["off_epa_per_play"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    )
    # A team's defense "allowed" EPA is the opponent's own offensive EPA that week.
    long_log["trailing_def_epa_allowed"] = (
        long_log.groupby("team")["opp_off_epa_per_play"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    )

    league_avg_off = team_week["off_epa_per_play"].mean()
    long_log["trailing_off_epa"] = long_log["trailing_off_epa"].fillna(league_avg_off)
    long_log["trailing_def_epa_allowed"] = long_log["trailing_def_epa_allowed"].fillna(league_avg_off)

    feat_cols = long_log[["game_id", "team", "trailing_off_epa", "trailing_def_epa_allowed"]]

    reg = reg.merge(
        feat_cols.rename(
            columns={
                "team": "home_team",
                "trailing_off_epa": "home_off_epa",
                "trailing_def_epa_allowed": "home_def_epa_allowed",
            }
        ),
        on=["game_id", "home_team"],
        how="left",
    )
    reg = reg.merge(
        feat_cols.rename(
            columns={
                "team": "away_team",
                "trailing_off_epa": "away_off_epa",
                "trailing_def_epa_allowed": "away_def_epa_allowed",
            }
        ),
        on=["game_id", "away_team"],
        how="left",
    )

    return reg


def current_team_strength(
    completed_games: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    min_periods: int = MIN_PERIODS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Each team's offense/defense EPA-per-play *as of right now* -- i.e.
    including their most recent completed game, for use predicting a game
    that hasn't been played yet. `completed_games` must only contain games
    with a known final score (REG season rows are used; others ignored).

    This is deliberately a different rolling computation from
    `build_trailing_epa_features`: that one excludes each game from its own
    feature (correct for training/backtesting), this one wants the fullest
    up-to-date picture of a team's current form (correct for a live pick).

    Returns one row per team: team, off_epa, def_epa_allowed.
    """
    reg = completed_games[completed_games["game_type"] == "REG"].copy()
    years = sorted(reg["season"].unique().tolist())

    team_week = _team_week_off_epa(years, force_refresh=force_refresh)
    long_log = _team_game_log(reg, team_week)
    long_log = long_log.sort_values(["team", "gameday"])

    long_log["off_epa_now"] = (
        long_log.groupby("team")["off_epa_per_play"]
        .transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    )
    long_log["def_epa_allowed_now"] = (
        long_log.groupby("team")["opp_off_epa_per_play"]
        .transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    )

    league_avg_off = team_week["off_epa_per_play"].mean()
    latest = long_log.sort_values("gameday").groupby("team").tail(1)
    latest = latest[["team", "off_epa_now", "def_epa_allowed_now"]].rename(
        columns={"off_epa_now": "off_epa", "def_epa_allowed_now": "def_epa_allowed"}
    )
    latest["off_epa"] = latest["off_epa"].fillna(league_avg_off)
    latest["def_epa_allowed"] = latest["def_epa_allowed"].fillna(league_avg_off)
    return latest.reset_index(drop=True)
