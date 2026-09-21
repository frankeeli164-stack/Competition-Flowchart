import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.props_math import prob_anytime_td, prob_over, prob_under


def test_prob_over_at_the_mean_is_50_percent():
    assert prob_over(75.0, projected_mean=75.0, std_dev=20.0) == pytest.approx(0.5)


def test_prob_over_and_under_sum_to_one():
    p_over = prob_over(80.0, projected_mean=75.0, std_dev=20.0)
    p_under = prob_under(80.0, projected_mean=75.0, std_dev=20.0)
    assert p_over + p_under == pytest.approx(1.0)


def test_prob_over_decreases_as_line_increases():
    low = prob_over(50.0, projected_mean=75.0, std_dev=20.0)
    high = prob_over(100.0, projected_mean=75.0, std_dev=20.0)
    assert low > high


def test_prob_over_one_std_above_mean():
    # For a Normal distribution, P(X > mean + 1 SD) ~= 0.1587
    p = prob_over(95.0, projected_mean=75.0, std_dev=20.0)
    assert p == pytest.approx(0.1587, abs=1e-3)


def test_prob_over_invalid_std_raises():
    with pytest.raises(ValueError):
        prob_over(75.0, projected_mean=75.0, std_dev=0.0)
    with pytest.raises(ValueError):
        prob_over(75.0, projected_mean=75.0, std_dev=-5.0)


def test_prob_anytime_td_zero_rate_is_zero():
    assert prob_anytime_td(0.0) == pytest.approx(0.0)


def test_prob_anytime_td_increases_with_rate():
    low = prob_anytime_td(0.3)
    high = prob_anytime_td(0.9)
    assert 0.0 < low < high < 1.0


def test_prob_anytime_td_known_value():
    # lambda=1 -> 1 - e^-1 ~= 0.6321
    assert prob_anytime_td(1.0) == pytest.approx(0.6321, abs=1e-3)


def test_prob_anytime_td_negative_rate_raises():
    with pytest.raises(ValueError):
        prob_anytime_td(-0.1)
