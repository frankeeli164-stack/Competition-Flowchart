"""Player prop projections: trailing player rate x opponent-allowed adjustment.

Same "keep it simple and honest" approach as `team_epa.py`: rather than
fitting a per-player regression (most players don't have enough games for
that to be anything but overfit noise), a player's projection for a stat is:

    projected = player's own trailing per-game rate
                * (opponent's trailing allowed rate to that position / league average)

This is the standard, defensible technique public fantasy-football and prop
projection tools use (a "strength of schedule" adjustment on a raw per-game
average), not a novel model. See README for what this does and doesn't
validate -- there is no free historical player-prop-odds data to backtest
against, only projection accuracy.

All "trailing" functions take a `shift` flag: `shift=True` (default) excludes
each row's own game from its own rolling window, which is required for
leakage-free training/backtesting. `shift=False` includes it, giving the
truest "as of right now" estimate for projecting a game that hasn't been
played yet -- there's no future game to exclude a value from.
"""

from __future__ import annotations

import pandas as pd

from .player_stats_source import fetch_player_stats_years

STAT_COLUMNS = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
ROLLING_WINDOW = 8
MIN_PERIODS = 3
ADJUSTMENT_CLIP = (0.6, 1.6)  # damp noisy small-sample defense-allowed swings

# Which position groups a stat is actually a meaningful prop market for.
# Mixing in irrelevant positions (e.g. a lineman's always-zero passing yards)
# would swamp any real signal in accuracy/calibration metrics.
STAT_RELEVANT_POSITIONS = {
    "passing_yards": ["QB"],
    "rushing_yards": ["RB", "QB"],
    "receiving_yards": ["WR", "TE", "RB"],
    "receptions": ["WR", "TE", "RB"],
    "td_rate": ["RB", "WR", "TE", "QB"],
}


def player_week_stats(years: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """Regular-season weekly stats, one row per player per week, with just the
    columns this module needs.
    """
    df = fetch_player_stats_years(years, force_refresh=force_refresh)
    df = df[df["season_type"] == "REG"].copy()
    df["td_rate"] = df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0)
    cols = [
        "player_id", "player_display_name", "position_group", "recent_team",
        "opponent_team", "season", "week",
        *STAT_COLUMNS, "td_rate",
    ]
    return df[cols].sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def defense_allowed_by_position(player_week: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, opponent_team, position_group): total stats
    that team's defense allowed to that position group in that game.
    """
    grouped = (
        player_week.groupby(["season", "week", "opponent_team", "position_group"])[
            STAT_COLUMNS + ["td_rate"]
        ]
        .sum()
        .reset_index()
        .rename(columns={"opponent_team": "defense_team"})
    )
    return grouped


def _trailing(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    window: int = ROLLING_WINDOW,
    min_periods: int = MIN_PERIODS,
    shift: bool = True,
) -> pd.DataFrame:
    """Add `trailing_<col>` for each of `value_cols`, rolling within groups,
    sorted by (season, week). `shift=True` excludes the row's own game.
    """
    out = df.sort_values(group_cols + ["season", "week"]).copy()
    grouped = out.groupby(group_cols)
    for col in value_cols:
        series = grouped[col]
        if shift:
            out[f"trailing_{col}"] = series.transform(
                lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
            )
        else:
            out[f"trailing_{col}"] = series.transform(
                lambda s: s.rolling(window, min_periods=min_periods).mean()
            )
    return out


def trailing_player_features(
    player_week: pd.DataFrame,
    shift: bool = True,
    window: int = ROLLING_WINDOW,
    min_periods: int = MIN_PERIODS,
) -> pd.DataFrame:
    return _trailing(
        player_week, ["player_id"], STAT_COLUMNS + ["td_rate"], window=window, min_periods=min_periods, shift=shift
    )


def trailing_defense_features(
    defense_allowed: pd.DataFrame,
    shift: bool = True,
    window: int = ROLLING_WINDOW,
    min_periods: int = MIN_PERIODS,
) -> pd.DataFrame:
    return _trailing(
        defense_allowed,
        ["defense_team", "position_group"],
        STAT_COLUMNS + ["td_rate"],
        window=window,
        min_periods=min_periods,
        shift=shift,
    )


def position_std_dev(player_week: pd.DataFrame) -> pd.DataFrame:
    """Empirical per-game standard deviation of each stat, by position group.
    Used as the spread for the Normal-approximation over/under probability --
    individual players don't have enough games to estimate their own std
    reliably, so this borrows strength across the position.
    """
    return player_week.groupby("position_group")[STAT_COLUMNS].std().reset_index()


def league_average_allowed(defense_allowed: pd.DataFrame) -> pd.DataFrame:
    """Per (position_group, stat): league-average amount allowed per team-game.
    The denominator of the opponent adjustment factor.
    """
    return defense_allowed.groupby("position_group")[STAT_COLUMNS + ["td_rate"]].mean().reset_index()


DEFAULT_ADJUSTMENT_SHRINK = 0.15


def project_stat(
    player_trailing_value: float,
    opponent_allowed: float,
    league_avg_allowed: float,
    shrink: float = DEFAULT_ADJUSTMENT_SHRINK,
) -> float:
    """Combine a player's own trailing rate with an opponent strength-of-
    schedule adjustment, shrunk toward "no adjustment" (factor=1.0) by
    `shrink`.

    The shrinkage is empirically load-bearing, not cosmetic: a defense's
    trailing allowed-stats-per-position sample is much noisier than a team's
    overall offensive EPA (see team_epa.py), because it's aggregating far
    fewer plays per game. Backtesting found the full, unshrunk adjustment
    (shrink=1.0) makes projections ~2-4% *worse* by MAE than the player's own
    trailing average alone; a small shrink (~0.1-0.15) is roughly neutral to
    marginally better. The takeaway: for individual player props, a player's
    own recent form dominates far more than opponent matchup does -- unlike
    team-level game outcomes, where the opponent-strength adjustment measurably
    helped. See scripts/player_props_report.py to reproduce this.

    The result is additionally clipped to guard against a small-sample
    defense-allowed value (e.g. early season) producing an absurd projection.
    """
    if league_avg_allowed <= 0 or pd.isna(league_avg_allowed):
        return player_trailing_value
    raw_factor = opponent_allowed / league_avg_allowed
    factor = 1.0 + shrink * (raw_factor - 1.0)
    factor = min(max(factor, ADJUSTMENT_CLIP[0]), ADJUSTMENT_CLIP[1])
    return player_trailing_value * factor
