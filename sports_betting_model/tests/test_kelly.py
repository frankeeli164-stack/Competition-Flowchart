import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.kelly import fractional_kelly_stake, kelly_fraction


def test_kelly_fraction_no_edge_is_zero():
    # True prob matches vig-included market implied prob (-110 ~ 0.5238) -> no edge
    assert kelly_fraction(0.5, 100) == 0.0


def test_kelly_fraction_classic_example():
    # Even money (+100, b=1), true prob 0.6 -> f* = (1*0.6 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 100) == pytest.approx(0.2)


def test_kelly_fraction_never_negative():
    # Bad bet: true prob 0.4 at even money should give 0, not negative stake
    assert kelly_fraction(0.4, 100) == 0.0


def test_kelly_fraction_out_of_range_prob_raises():
    with pytest.raises(ValueError):
        kelly_fraction(1.2, 100)


def test_fractional_kelly_stake_scales_down():
    full = kelly_fraction(0.6, 100)
    half_stake = fractional_kelly_stake(0.6, 100, bankroll=1000, fraction=0.5, max_stake_pct=1.0)
    assert half_stake == pytest.approx(1000 * full * 0.5, abs=0.01)


def test_fractional_kelly_stake_respects_cap():
    # Huge apparent edge should still be capped by max_stake_pct
    stake = fractional_kelly_stake(0.95, 100, bankroll=10_000, fraction=1.0, max_stake_pct=0.05)
    assert stake == pytest.approx(500.0)


def test_fractional_kelly_stake_zero_when_no_edge():
    assert fractional_kelly_stake(0.5, 100, bankroll=1000) == 0.0


def test_fractional_kelly_invalid_fraction_raises():
    with pytest.raises(ValueError):
        fractional_kelly_stake(0.6, 100, bankroll=1000, fraction=0)


def test_fractional_kelly_invalid_bankroll_raises():
    with pytest.raises(ValueError):
        fractional_kelly_stake(0.6, 100, bankroll=0)
