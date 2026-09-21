"""Probability math for player props.

Unlike a moneyline (a clean binary outcome with a market-quoted probability
built in), a yardage/reception prop's "true probability" has to come from a
projected distribution, because we have no market odds to lean on for most
of this (see README -- no free historical player-prop odds exist). These are
explicit, documented approximations, not a black box:

- Yardage/reception counts are modeled as Normal(projected_mean, position_std).
  Real game-to-game yardage is closer to right-skewed/heavy-tailed than
  Normal, especially near zero, but Normal is the standard, defensible
  starting approximation used by most public projection tools, and it's easy
  to reason about and unit test.
- "Anytime touchdown" is modeled as Poisson: P(at least one) = 1 - exp(-lambda),
  where lambda is the player's projected TDs-per-game rate. This is standard
  for rare-event counting stats.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def prob_over(line: float, projected_mean: float, std_dev: float) -> float:
    """P(actual stat > line), assuming Normal(projected_mean, std_dev).

    `std_dev` should come from historical game-to-game variance for that
    stat/position (see `player_props.position_std_dev`), not a per-player
    guess -- individual players don't have enough games to estimate their
    own variance reliably.
    """
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")
    z = (line - projected_mean) / std_dev
    return 1.0 - norm.cdf(z)


def prob_under(line: float, projected_mean: float, std_dev: float) -> float:
    return 1.0 - prob_over(line, projected_mean, std_dev)


def prob_anytime_td(projected_td_rate: float) -> float:
    """P(at least one TD), given a projected touchdowns-per-game rate, via a
    Poisson(lambda) assumption. `projected_td_rate` must be >= 0.
    """
    if projected_td_rate < 0:
        raise ValueError("projected_td_rate must be non-negative")
    return 1.0 - math.exp(-projected_td_rate)
