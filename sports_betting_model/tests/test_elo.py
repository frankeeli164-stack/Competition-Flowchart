import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.elo import INITIAL_RATING, run_elo, win_probability


def test_win_probability_equal_ratings_is_50_50():
    assert win_probability(0.0) == pytest.approx(0.5)


def test_win_probability_increases_with_elo_diff():
    assert win_probability(200) > win_probability(0) > win_probability(-200)


def _fake_games(results):
    """results: list of (season, home_team, away_team, home_score, away_score)"""
    return pd.DataFrame(
        results, columns=["season", "home_team", "away_team", "home_score", "away_score"]
    )


def test_run_elo_no_lookahead_first_game_uses_initial_ratings():
    games = _fake_games([(2020, "A", "B", 24, 17)])
    out = run_elo(games, home_advantage=0.0)
    assert out.loc[0, "home_elo_pre"] == INITIAL_RATING
    assert out.loc[0, "away_elo_pre"] == INITIAL_RATING


def test_run_elo_winner_rating_increases():
    games = _fake_games(
        [
            (2020, "A", "B", 24, 17),
            (2020, "A", "B", 24, 17),
        ]
    )
    out = run_elo(games, home_advantage=0.0)
    # Team A won game 1, so its pre-game rating for game 2 should exceed initial.
    assert out.loc[1, "home_elo_pre"] > INITIAL_RATING
    assert out.loc[1, "away_elo_pre"] < INITIAL_RATING


def test_run_elo_home_advantage_shifts_win_prob():
    games = _fake_games([(2020, "A", "B", 24, 17)])
    no_hfa = run_elo(games, home_advantage=0.0)
    with_hfa = run_elo(games, home_advantage=65.0)
    assert with_hfa.loc[0, "elo_home_win_prob"] > no_hfa.loc[0, "elo_home_win_prob"]


def test_run_elo_season_regression_pulls_toward_mean():
    games = _fake_games(
        [
            (2020, "A", "B", 40, 0),  # A blows out B, big rating swing
            (2021, "A", "C", 24, 17),  # new season game for A
        ]
    )
    out = run_elo(games, home_advantage=0.0, season_regression=1.0)
    # season_regression=1.0 means full reset to mean between seasons
    assert out.loc[1, "home_elo_pre"] == pytest.approx(INITIAL_RATING)
