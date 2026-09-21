#!/usr/bin/env python3
"""End-to-end: fetch data -> Elo -> walk-forward model -> backtest report.

Usage:
    python3 scripts/run_pipeline.py [--refresh] [--min-season 1999] [--min-edge 2.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import (
    add_market_probs,
    score_calibration,
    simulate_flat_stake,
    simulate_kelly_stake,
)
from src.data import load_completed_games
from src.elo import run_elo
from src.features import build_features
from src.model import walk_forward_predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-download source data")
    parser.add_argument("--min-season", type=int, default=1999)
    parser.add_argument("--min-train-seasons", type=int, default=3)
    parser.add_argument("--min-edge", type=float, default=2.0, help="Min EV%% to place a bet")
    args = parser.parse_args()

    print(f"Loading NFL games since {args.min_season}...")
    games = load_completed_games(min_season=args.min_season, force_refresh=args.refresh)
    print(f"  {len(games)} completed games loaded ({games['season'].min()}-{games['season'].max()})")

    print("Computing Elo ratings...")
    games = run_elo(games)

    print("Building features and training walk-forward model...")
    games = build_features(games)
    games = walk_forward_predict(games, min_train_seasons=args.min_train_seasons)

    print("Attaching de-vigged market probabilities...")
    games = add_market_probs(games)

    eval_games = games[games["season"] >= sorted(games["season"].unique())[args.min_train_seasons]]
    eval_games = eval_games[eval_games["model_home_win_prob"].notna()]
    print(f"  {len(eval_games)} games available for out-of-sample evaluation\n")

    print("=" * 60)
    print("CALIBRATION (lower is better; market is the benchmark to beat)")
    print("=" * 60)
    for prob_col, label in [
        ("elo_home_win_prob", "Elo (raw)"),
        ("model_home_win_prob", "Logistic model (walk-forward)"),
        ("market_home_prob", "Market (de-vigged, closing line)"),
    ]:
        scored = eval_games[eval_games["market_home_prob"].notna()]
        m = score_calibration(scored, prob_col, label)
        print(f"  {m['label']:<32} n={m['n_games']:<5} brier={m['brier_score']:.4f}  log_loss={m['log_loss']:.4f}")

    print()
    print("=" * 60)
    print(f"FLAT-STAKE BACKTEST (bet $100 when edge > {args.min_edge}%, vs. closing lines)")
    print("=" * 60)
    for prob_col, label in [
        ("model_home_win_prob", "Logistic model"),
        ("elo_home_win_prob", "Elo (raw)"),
    ]:
        odds_games = eval_games[eval_games["home_moneyline"].notna()]
        _, summ = simulate_flat_stake(odds_games, prob_col, stake=100.0, min_edge_pct=args.min_edge)
        if summ["n_bets"] == 0:
            print(f"  {label}: no qualifying bets")
            continue
        print(
            f"  {label:<20} n_bets={summ['n_bets']:<5} win_rate={summ['win_rate']:.1%} "
            f"staked=${summ['total_staked']:,.0f} profit=${summ['total_profit']:,.0f} "
            f"ROI={summ['roi_pct']:.2f}%"
        )

    print()
    print("=" * 60)
    print(f"HALF-KELLY BACKTEST (model probs, $10,000 starting bankroll)")
    print("=" * 60)
    odds_games = eval_games[eval_games["home_moneyline"].notna()]
    _, ksumm = simulate_kelly_stake(odds_games, "model_home_win_prob", min_edge_pct=args.min_edge)
    print(
        f"  n_bets={ksumm['n_bets']}  start=${ksumm['starting_bankroll']:,.0f}  "
        f"end=${ksumm['ending_bankroll']:,.0f}  ROI={ksumm['roi_pct']:.2f}%"
    )

    print()
    print("Reminder: this backtest bets against historical CLOSING lines, one of the")
    print("hardest prices in sports betting to beat. A positive ROI here over a large")
    print("sample is a meaningfully good sign; a negative one means the model has no")
    print("real edge yet and should not be bet with real money. See README.md.")


if __name__ == "__main__":
    main()
