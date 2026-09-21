"""Odds conversion and expected-value math.

All "odds" inputs are American odds (e.g. -110, +150) unless stated otherwise.
"""

from __future__ import annotations


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds (total payout multiple per unit stake)."""
    if american_odds == 0:
        raise ValueError("American odds cannot be 0")
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    return 1.0 + 100.0 / abs(american_odds)


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds back to American odds."""
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    if decimal_odds >= 2.0:
        return (decimal_odds - 1.0) * 100.0
    return -100.0 / (decimal_odds - 1.0)


def american_to_implied_prob(american_odds: float) -> float:
    """Implied win probability from American odds, *including* the sportsbook's vig."""
    return 1.0 / american_to_decimal(american_odds)


def devig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove vig from a two-way market by normalizing implied probabilities to sum to 1.

    This is the standard proportional (multiplicative) devig method. It is an
    approximation — it does not correctly separate favorite/longshot bias — but it
    is simple, well understood, and adequate as a "fair line" baseline to compare
    a model against.
    """
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError("Probabilities must be positive")
    return prob_a / total, prob_b / total


def expected_value(true_prob: float, american_odds: float, stake: float = 1.0) -> float:
    """Expected profit (not return) of betting `stake` at `american_odds`, given a true
    (de-vigged, model) win probability `true_prob`.

    EV = p * profit_if_win - (1-p) * stake
    """
    if not 0.0 <= true_prob <= 1.0:
        raise ValueError("true_prob must be in [0, 1]")
    decimal_odds = american_to_decimal(american_odds)
    profit_if_win = stake * (decimal_odds - 1.0)
    return true_prob * profit_if_win - (1.0 - true_prob) * stake


def ev_percent(true_prob: float, american_odds: float) -> float:
    """Expected value as a percentage of stake (i.e. expected ROI of this single bet)."""
    return expected_value(true_prob, american_odds, stake=1.0) * 100.0


def has_edge(true_prob: float, american_odds: float, min_edge_pct: float = 0.0) -> bool:
    """True if the bet's expected ROI exceeds `min_edge_pct` (e.g. 2.0 for a 2% minimum edge)."""
    return ev_percent(true_prob, american_odds) > min_edge_pct
