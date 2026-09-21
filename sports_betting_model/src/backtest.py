"""Evaluate model predictions against outcomes and against the market itself.

The real test of a betting model isn't "is it more accurate than a coin flip" —
it's "does it beat the closing line," because a sportsbook's closing odds are
one of the most efficient prices in all of gambling. This module scores
calibration (Brier score, log loss) for the model, for raw Elo, and for the
de-vigged market itself as the benchmark, then simulates flat-stake and
Kelly-stake betting strategies using only games where the odds columns exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .kelly import fractional_kelly_stake
from .odds_math import american_to_implied_prob, devig_two_way, expected_value


def add_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Add a de-vigged 'fair' market home win probability from closing moneylines.

    Drops rows with no moneyline data (older seasons / occasional missing games).
    """
    df = df[df["home_moneyline"].notna() & df["away_moneyline"].notna()].copy()

    home_vig_prob = df["home_moneyline"].apply(american_to_implied_prob)
    away_vig_prob = df["away_moneyline"].apply(american_to_implied_prob)

    fair = [devig_two_way(h, a) for h, a in zip(home_vig_prob, away_vig_prob)]
    df["market_home_prob"] = [f[0] for f in fair]
    return df


def score_calibration(df: pd.DataFrame, prob_col: str, label: str) -> dict:
    """Brier score and log loss of `prob_col` predicting `home_win`. Lower is better
    for both. A well-calibrated model should score close to (and a genuine edge
    would score *below*) the market's own Brier/log-loss.
    """
    valid = df[df[prob_col].notna() & (df["is_tie"] == 0)]
    y = valid["home_win"]
    p = valid[prob_col].clip(1e-6, 1 - 1e-6)
    return {
        "label": label,
        "n_games": len(valid),
        "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def simulate_flat_stake(
    df: pd.DataFrame,
    prob_col: str,
    stake: float = 100.0,
    min_edge_pct: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    """Bet a flat `stake` on any game where `prob_col` implies edge over the
    actual moneyline price (after accounting for the book's vig on that side),
    exceeding `min_edge_pct` percent expected ROI. Bets the side with edge; skips
    games with no edge on either side.
    """
    rows = []
    for r in df.itertuples(index=False):
        p_model = getattr(r, prob_col)
        if pd.isna(p_model):
            continue

        home_ev_pct = expected_value(p_model, r.home_moneyline, stake=1.0) * 100.0
        away_ev_pct = expected_value(1.0 - p_model, r.away_moneyline, stake=1.0) * 100.0

        if home_ev_pct >= away_ev_pct and home_ev_pct > min_edge_pct:
            side, odds = "home", r.home_moneyline
        elif away_ev_pct > min_edge_pct:
            side, odds = "away", r.away_moneyline
        else:
            continue

        if r.is_tie:
            # A tied game is a push on a moneyline bet: stake is returned, no win/loss.
            won, profit = None, 0.0
        else:
            won = r.home_win == 1 if side == "home" else r.home_win == 0
            profit = expected_value(1.0, odds, stake=stake) if won else -stake

        rows.append(
            {
                "game_id": r.game_id,
                "season": r.season,
                "side": side,
                "odds": odds,
                "stake": stake,
                "won": won,
                "profit": profit,
            }
        )

    bets = pd.DataFrame(rows)
    if bets.empty:
        return bets, {"n_bets": 0, "total_staked": 0.0, "total_profit": 0.0, "roi_pct": None}

    decided = bets[bets["won"].notna()]
    summary = {
        "n_bets": len(bets),
        "win_rate": decided["won"].mean() if len(decided) else None,
        "total_staked": bets["stake"].sum(),
        "total_profit": bets["profit"].sum(),
        "roi_pct": 100.0 * bets["profit"].sum() / bets["stake"].sum(),
    }
    return bets, summary


def simulate_kelly_stake(
    df: pd.DataFrame,
    prob_col: str,
    starting_bankroll: float = 10_000.0,
    fraction: float = 0.5,
    min_edge_pct: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    """Same edge-detection as `simulate_flat_stake`, but sizes each bet as a
    fraction of the *current* bankroll via Kelly, applied sequentially in
    chronological order. `df` must already be sorted by game date.
    """
    bankroll = starting_bankroll
    rows = []

    for r in df.itertuples(index=False):
        p_model = getattr(r, prob_col)
        if pd.isna(p_model):
            continue

        home_ev_pct = expected_value(p_model, r.home_moneyline, stake=1.0) * 100.0
        away_ev_pct = expected_value(1.0 - p_model, r.away_moneyline, stake=1.0) * 100.0

        if home_ev_pct >= away_ev_pct and home_ev_pct > min_edge_pct:
            side, odds, p_win = "home", r.home_moneyline, p_model
        elif away_ev_pct > min_edge_pct:
            side, odds, p_win = "away", r.away_moneyline, 1.0 - p_model
        else:
            continue

        stake = fractional_kelly_stake(p_win, odds, bankroll, fraction=fraction)
        if stake <= 0:
            continue

        if r.is_tie:
            won, profit = None, 0.0
        else:
            won = r.home_win == 1 if side == "home" else r.home_win == 0
            profit = expected_value(1.0, odds, stake=stake) if won else -stake
        bankroll += profit

        rows.append(
            {
                "game_id": r.game_id,
                "season": r.season,
                "side": side,
                "odds": odds,
                "stake": stake,
                "won": won,
                "profit": profit,
                "bankroll_after": bankroll,
            }
        )

    bets = pd.DataFrame(rows)
    summary = {
        "n_bets": len(bets),
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": bankroll,
        "roi_pct": 100.0 * (bankroll - starting_bankroll) / starting_bankroll,
    }
    return bets, summary
