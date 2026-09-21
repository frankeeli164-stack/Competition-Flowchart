"""Kelly criterion stake sizing."""

from __future__ import annotations

from .odds_math import american_to_decimal


def kelly_fraction(true_prob: float, american_odds: float) -> float:
    """Full Kelly fraction of bankroll to stake.

    f* = (b*p - q) / b
    where b = net decimal odds (profit per unit staked), p = true win prob, q = 1-p.

    Returns 0.0 (never negative) when the bet has no edge — Kelly never tells you
    to bet the *other* side of a moneyline based on this formula alone.
    """
    if not 0.0 <= true_prob <= 1.0:
        raise ValueError("true_prob must be in [0, 1]")
    b = american_to_decimal(american_odds) - 1.0
    q = 1.0 - true_prob
    f = (b * true_prob - q) / b
    return max(f, 0.0)


def fractional_kelly_stake(
    true_prob: float,
    american_odds: float,
    bankroll: float,
    fraction: float = 0.5,
    max_stake_pct: float = 0.05,
) -> float:
    """Recommended stake in currency units.

    `fraction` scales down full Kelly (e.g. 0.5 = "half Kelly") to reduce variance,
    which is standard practice since real edges are uncertain and full Kelly is
    brutal on drawdowns when your probability estimate is even slightly off.

    `max_stake_pct` is a hard cap on bankroll % per bet regardless of what Kelly
    says, as a sanity backstop against a bad probability estimate blowing up the
    stake size.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")
    f = kelly_fraction(true_prob, american_odds) * fraction
    f = min(f, max_stake_pct)
    return round(f * bankroll, 2)
