"""Turn model predictions + live odds into a ranked list of recommended bets.

This is the "for bets not yet played" counterpart to `backtest.py` (which
scores the model against history). You supply current odds yourself — this
project does not include a live odds feed (see README for why).
"""

from __future__ import annotations

import pandas as pd

from .kelly import fractional_kelly_stake
from .odds_math import ev_percent


def find_value_bets(
    upcoming: pd.DataFrame,
    bankroll: float,
    min_edge_pct: float = 2.0,
    kelly_fraction: float = 0.5,
) -> pd.DataFrame:
    """`upcoming` needs columns: event, home_team, away_team, model_home_prob,
    home_moneyline, away_moneyline.

    Returns one row per game/side that clears `min_edge_pct` expected ROI, with
    a suggested fractional-Kelly stake, sorted by edge descending. A game with
    no edge on either side is simply absent from the result — this is not a
    "pick every game" tool.
    """
    rows = []
    for r in upcoming.itertuples(index=False):
        p_home = r.model_home_prob
        home_edge = ev_percent(p_home, r.home_moneyline)
        away_edge = ev_percent(1.0 - p_home, r.away_moneyline)

        if home_edge > min_edge_pct and home_edge >= away_edge:
            side, team, odds, p_win, edge = "home", r.home_team, r.home_moneyline, p_home, home_edge
        elif away_edge > min_edge_pct:
            side, team, odds, p_win, edge = "away", r.away_team, r.away_moneyline, 1.0 - p_home, away_edge
        else:
            continue

        stake = fractional_kelly_stake(p_win, odds, bankroll, fraction=kelly_fraction)
        rows.append(
            {
                "event": r.event,
                "bet_on": team,
                "side": side,
                "american_odds": odds,
                "model_prob": round(p_win, 4),
                "edge_pct": round(edge, 2),
                "suggested_stake": stake,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("edge_pct", ascending=False).reset_index(drop=True)
    return result
