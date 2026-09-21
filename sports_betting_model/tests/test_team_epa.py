import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

import src.team_epa as team_epa


def _fake_team_week():
    # Team A plays great in weeks 1-2, terribly in week 3. Team B is average
    # throughout. Team C only shows up in week 3 (for the opponent join).
    return pd.DataFrame(
        [
            {"season": 2020, "week": 1, "team": "A", "off_epa_per_play": 0.30},
            {"season": 2020, "week": 2, "team": "A", "off_epa_per_play": 0.30},
            {"season": 2020, "week": 3, "team": "A", "off_epa_per_play": -0.50},
            {"season": 2020, "week": 1, "team": "B", "off_epa_per_play": 0.00},
            {"season": 2020, "week": 2, "team": "B", "off_epa_per_play": 0.00},
            {"season": 2020, "week": 3, "team": "B", "off_epa_per_play": 0.00},
            {"season": 2020, "week": 3, "team": "C", "off_epa_per_play": 0.10},
        ]
    )


def _fake_games():
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "season": 2020,
                "week": 1,
                "game_type": "REG",
                "gameday": pd.Timestamp("2020-09-06"),
                "home_team": "A",
                "away_team": "B",
            },
            {
                "game_id": "g2",
                "season": 2020,
                "week": 2,
                "game_type": "REG",
                "gameday": pd.Timestamp("2020-09-13"),
                "home_team": "A",
                "away_team": "B",
            },
            {
                "game_id": "g3",
                "season": 2020,
                "week": 3,
                "game_type": "REG",
                "gameday": pd.Timestamp("2020-09-20"),
                "home_team": "A",
                "away_team": "C",
            },
        ]
    )


def test_trailing_epa_has_no_lookahead(monkeypatch):
    monkeypatch.setattr(team_epa, "_team_week_off_epa", lambda years, force_refresh=False: _fake_team_week())

    # min_periods=2 isolates the rolling/shift logic itself from the
    # (separately tested) league-average fallback for insufficient history.
    out = team_epa.build_trailing_epa_features(_fake_games(), min_periods=2)
    out = out.set_index("game_id")

    # Week 1: team A has zero prior history -> falls back to league average,
    # NOT team A's own week-1 performance (that would be leakage).
    league_avg = _fake_team_week()["off_epa_per_play"].mean()
    assert out.loc["g1", "home_off_epa"] == pytest.approx(league_avg)

    # Week 3: team A's trailing off_epa should reflect ONLY weeks 1-2 (0.30, 0.30),
    # not week 3's own -0.50 value.
    assert out.loc["g3", "home_off_epa"] == pytest.approx(0.30)
    assert out.loc["g3", "home_off_epa"] != pytest.approx(-0.50)


def test_trailing_def_epa_allowed_uses_opponent_offense(monkeypatch):
    monkeypatch.setattr(team_epa, "_team_week_off_epa", lambda years, force_refresh=False: _fake_team_week())

    out = team_epa.build_trailing_epa_features(_fake_games(), min_periods=2)
    out = out.set_index("game_id")

    # Going into week 3, team A's defense has faced team B (0.00 epa/play) twice.
    assert out.loc["g3", "home_def_epa_allowed"] == pytest.approx(0.00)


def test_insufficient_history_falls_back_to_league_average(monkeypatch):
    monkeypatch.setattr(team_epa, "_team_week_off_epa", lambda years, force_refresh=False: _fake_team_week())

    # Default min_periods=3: team A only has 2 prior games by week 3, so it
    # must fall back to the league average rather than use an under-sampled mean.
    out = team_epa.build_trailing_epa_features(_fake_games())
    out = out.set_index("game_id")

    league_avg = _fake_team_week()["off_epa_per_play"].mean()
    assert out.loc["g3", "home_off_epa"] == pytest.approx(league_avg)


def test_postseason_games_excluded():
    games = _fake_games()
    games.loc[0, "game_type"] = "WC"
    import src.team_epa as te

    def fake_fetch(years, force_refresh=False):
        return _fake_team_week()

    orig = te._team_week_off_epa
    te._team_week_off_epa = fake_fetch
    try:
        out = te.build_trailing_epa_features(games)
    finally:
        te._team_week_off_epa = orig

    assert "g1" not in out["game_id"].values
    assert len(out) == 2
