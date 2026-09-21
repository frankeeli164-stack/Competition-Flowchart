import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.odds_math import (
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    devig_two_way,
    expected_value,
    ev_percent,
    has_edge,
)


def test_american_to_decimal_favorite():
    assert american_to_decimal(-110) == pytest.approx(1.9091, abs=1e-3)


def test_american_to_decimal_underdog():
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_decimal_american_roundtrip():
    for odds in [-250, -110, 100, 120, 300]:
        d = american_to_decimal(odds)
        assert decimal_to_american(d) == pytest.approx(odds, abs=1e-6)


def test_american_to_decimal_zero_raises():
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_implied_prob_even_money():
    assert american_to_implied_prob(100) == pytest.approx(0.5)


def test_implied_prob_favorite_higher_than_underdog():
    assert american_to_implied_prob(-200) > american_to_implied_prob(150)


def test_devig_two_way_removes_vig_to_100_percent():
    # Typical -110/-110 market: each side implies ~52.4%, summing to ~104.8%
    p_home = american_to_implied_prob(-110)
    p_away = american_to_implied_prob(-110)
    fair_home, fair_away = devig_two_way(p_home, p_away)
    assert fair_home + fair_away == pytest.approx(1.0)
    assert fair_home == pytest.approx(0.5)


def test_devig_two_way_uneven_market():
    p_home = american_to_implied_prob(-200)  # ~0.667
    p_away = american_to_implied_prob(170)  # ~0.370
    fair_home, fair_away = devig_two_way(p_home, p_away)
    assert fair_home + fair_away == pytest.approx(1.0)
    assert fair_home > fair_away


def test_expected_value_zero_at_exact_breakeven_price():
    # Plugging the vig-included implied probability back into EV at that same
    # price is EV's break-even point by construction: exactly 0.
    odds = -110
    p = american_to_implied_prob(odds)
    assert expected_value(p, odds, stake=100) == pytest.approx(0.0, abs=1e-9)


def test_expected_value_negative_at_true_fair_price_on_vig_market():
    # The realistic case: true (de-vigged) prob is *lower* than the book's
    # vig-included implied prob -> betting at market price is -EV, which is
    # exactly what the vig is for.
    odds = -110
    fair_p, _ = devig_two_way(
        american_to_implied_prob(odds), american_to_implied_prob(odds)
    )
    assert expected_value(fair_p, odds, stake=100) < 0


def test_expected_value_positive_with_real_edge():
    # Book prices team at 50% (+100) but true prob is 60% -> positive EV.
    ev = expected_value(0.60, 100, stake=100)
    assert ev == pytest.approx(20.0)


def test_expected_value_matches_hand_calc_favorite():
    # -200 odds: bet $200 to win $100. True prob 0.75.
    ev = expected_value(0.75, -200, stake=200)
    # win: 0.75 * 100 ; lose: 0.25 * -200
    assert ev == pytest.approx(0.75 * 100 - 0.25 * 200)


def test_ev_percent_and_has_edge_consistent():
    assert has_edge(0.60, 100, min_edge_pct=5.0) == (ev_percent(0.60, 100) > 5.0)


def test_true_prob_out_of_range_raises():
    with pytest.raises(ValueError):
        expected_value(1.5, 100)
    with pytest.raises(ValueError):
        expected_value(-0.1, 100)
