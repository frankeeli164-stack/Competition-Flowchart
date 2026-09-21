"""Sequential Elo rating system for NFL teams.

Follows the general shape of FiveThirtyEight's NFL Elo methodology: a K-factor
scaled by margin of victory, a home-field advantage bonus, and partial
regression to the mean between seasons. Ratings are updated strictly in
chronological order and every prediction uses only *pre-game* ratings, so
there is no lookahead leakage when this is used as a model feature.
"""

from __future__ import annotations

import math

import pandas as pd

INITIAL_RATING = 1500.0
DEFAULT_K = 20.0
DEFAULT_HOME_ADVANTAGE = 55.0
DEFAULT_SEASON_REGRESSION = 1.0 / 3.0  # fraction each team reverts toward 1500 pre-season


def win_probability(elo_diff: float) -> float:
    """Probability the higher-rated side (by `elo_diff` points) wins, logistic curve."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def _mov_multiplier(point_margin: float, elo_diff_winner: float) -> float:
    """Margin-of-victory multiplier so blowouts move ratings more than close games,
    while damping the effect when the win was already expected (elo_diff_winner large).
    """
    return math.log(abs(point_margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def run_elo(
    games: pd.DataFrame,
    k: float = DEFAULT_K,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
    season_regression: float = DEFAULT_SEASON_REGRESSION,
    return_final_ratings: bool = False,
):
    """Compute pre-game Elo ratings and implied home win probability for every game.

    `games` must be sorted chronologically and have columns: season, home_team,
    away_team, home_score, away_score.

    Returns a copy of `games` with added columns:
      home_elo_pre, away_elo_pre  - ratings entering the game (no leakage)
      elo_home_win_prob           - Elo's pre-game home win probability
    """
    games = games.reset_index(drop=True).copy()
    ratings: dict[str, float] = {}
    current_season: int | None = None

    home_elo_pre = []
    away_elo_pre = []
    home_win_prob = []

    for row in games.itertuples(index=False):
        season = row.season
        if current_season is None:
            current_season = season
        elif season != current_season:
            # New season: regress every known team's rating partway back to the mean.
            for team in list(ratings.keys()):
                ratings[team] += (INITIAL_RATING - ratings[team]) * season_regression
            current_season = season

        home_team, away_team = row.home_team, row.away_team
        home_rating = ratings.get(home_team, INITIAL_RATING)
        away_rating = ratings.get(away_team, INITIAL_RATING)

        home_elo_pre.append(home_rating)
        away_elo_pre.append(away_rating)

        elo_diff = (home_rating + home_advantage) - away_rating
        p_home = win_probability(elo_diff)
        home_win_prob.append(p_home)

        home_score, away_score = row.home_score, row.away_score
        margin = home_score - away_score

        if margin > 0:
            home_result, away_result = 1.0, 0.0
        elif margin < 0:
            home_result, away_result = 0.0, 1.0
        else:
            home_result, away_result = 0.5, 0.5

        winner_elo_diff = elo_diff if margin >= 0 else -elo_diff
        mult = _mov_multiplier(margin, winner_elo_diff)

        shift = k * mult * (home_result - p_home)
        ratings[home_team] = home_rating + shift
        ratings[away_team] = away_rating - shift

    games["home_elo_pre"] = home_elo_pre
    games["away_elo_pre"] = away_elo_pre
    games["elo_home_win_prob"] = home_win_prob

    if return_final_ratings:
        return games, ratings
    return games
