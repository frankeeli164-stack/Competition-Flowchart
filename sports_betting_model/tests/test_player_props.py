import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.player_props import (
    defense_allowed_by_position,
    league_average_allowed,
    position_std_dev,
    project_stat,
    trailing_defense_features,
    trailing_player_features,
)


def _fake_player_week():
    return pd.DataFrame(
        [
            {"player_id": "P1", "player_display_name": "A", "position_group": "WR",
             "recent_team": "X", "opponent_team": "Y", "season": 2020, "week": 1,
             "passing_yards": 0, "rushing_yards": 0, "receiving_yards": 40, "receptions": 4, "td_rate": 0},
            {"player_id": "P1", "player_display_name": "A", "position_group": "WR",
             "recent_team": "X", "opponent_team": "Z", "season": 2020, "week": 2,
             "passing_yards": 0, "rushing_yards": 0, "receiving_yards": 60, "receptions": 5, "td_rate": 1},
            {"player_id": "P1", "player_display_name": "A", "position_group": "WR",
             "recent_team": "X", "opponent_team": "Y", "season": 2020, "week": 3,
             "passing_yards": 0, "rushing_yards": 0, "receiving_yards": 200, "receptions": 10, "td_rate": 2},
        ]
    )


def test_trailing_player_features_shift_excludes_own_game():
    # min_periods=2 isolates the rolling/shift logic from the (separately
    # tested) insufficient-history fallback.
    out = trailing_player_features(_fake_player_week(), shift=True, min_periods=2).set_index("week")
    # Week 3's trailing value should reflect only weeks 1-2 (40, 60 -> mean 50),
    # never week 3's own 200.
    assert out.loc[3, "trailing_receiving_yards"] == pytest.approx(50.0)
    assert out.loc[3, "trailing_receiving_yards"] != pytest.approx(200.0)


def test_trailing_player_features_no_shift_includes_own_game():
    out = trailing_player_features(_fake_player_week(), shift=False, min_periods=2).set_index("week")
    # "As of now" (no shift): week 3's value includes week 3 itself -> mean(40,60,200)=100
    assert out.loc[3, "trailing_receiving_yards"] == pytest.approx(100.0)


def test_trailing_player_features_first_game_is_nan_when_shifted():
    out = trailing_player_features(_fake_player_week(), shift=True).set_index("week")
    assert pd.isna(out.loc[1, "trailing_receiving_yards"])


def test_defense_allowed_by_position_aggregates_correctly():
    pw = _fake_player_week()
    # Add a second WR facing the same defense (Y) in week 1 to check summation.
    pw = pd.concat(
        [
            pw,
            pd.DataFrame(
                [{"player_id": "P2", "player_display_name": "B", "position_group": "WR",
                  "recent_team": "X", "opponent_team": "Y", "season": 2020, "week": 1,
                  "passing_yards": 0, "rushing_yards": 0, "receiving_yards": 30, "receptions": 3, "td_rate": 0}]
            ),
        ],
        ignore_index=True,
    )
    allowed = defense_allowed_by_position(pw)
    row = allowed[(allowed["defense_team"] == "Y") & (allowed["season"] == 2020) & (allowed["week"] == 1)]
    assert row["receiving_yards"].iloc[0] == pytest.approx(40 + 30)


def test_trailing_defense_features_no_lookahead():
    pw = _fake_player_week()
    allowed = defense_allowed_by_position(pw)
    # Defense "Y" only has 2 appearances in this fixture; min_periods=1 isolates
    # the shift/rolling logic itself from the insufficient-history fallback.
    out = trailing_defense_features(allowed, shift=True, min_periods=1)
    # Defense "Y" faced this WR position group in weeks 1 and 3 (40, then 200).
    # Going into week 3, trailing allowed should reflect ONLY week 1 (40), not week 3's own 200.
    y_week3 = out[(out["defense_team"] == "Y") & (out["week"] == 3)]
    assert y_week3["trailing_receiving_yards"].iloc[0] == pytest.approx(40.0)


def test_position_std_dev_positive():
    std = position_std_dev(_fake_player_week())
    wr_std = std[std["position_group"] == "WR"]["receiving_yards"].iloc[0]
    assert wr_std > 0


def test_league_average_allowed_matches_manual_mean():
    pw = _fake_player_week()
    allowed = defense_allowed_by_position(pw)
    avg = league_average_allowed(allowed)
    wr_avg = avg[avg["position_group"] == "WR"]["receiving_yards"].iloc[0]
    assert wr_avg == pytest.approx(allowed["receiving_yards"].mean())


def test_project_stat_no_adjustment_when_opponent_equals_average():
    assert project_stat(75.0, opponent_allowed=100.0, league_avg_allowed=100.0) == pytest.approx(75.0)


def test_project_stat_scales_up_against_weak_defense():
    projected = project_stat(75.0, opponent_allowed=150.0, league_avg_allowed=100.0)
    assert projected > 75.0


def test_project_stat_clips_extreme_adjustment():
    # opponent allowed 10x league average -> factor would be 10.0, must clip to 1.6
    projected = project_stat(75.0, opponent_allowed=1000.0, league_avg_allowed=100.0)
    assert projected == pytest.approx(75.0 * 1.6)


def test_project_stat_handles_zero_league_average():
    # Guard against division by zero -- falls back to the player's own rate unadjusted.
    assert project_stat(75.0, opponent_allowed=0.0, league_avg_allowed=0.0) == pytest.approx(75.0)
