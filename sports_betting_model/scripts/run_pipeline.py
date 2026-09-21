#!/usr/bin/env python3
"""End-to-end: fetch data -> Elo -> walk-forward model -> backtest report.

Runs two model comparisons:
  1. Baseline (Elo + rest + division game) vs. market, across ALL completed
     games (regular season + playoffs) -- the original model.
  2. EPA-enhanced (baseline + trailing offensive/defensive EPA-per-play) vs.
     the same baseline vs. market, on REGULAR SEASON games only (EPA features
     aren't computed for the postseason -- see src/team_epa.py).

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
from src.features import FEATURE_COLUMNS, FEATURE_COLUMNS_V2, build_features
from src.model import walk_forward_predict
from src.team_epa import build_trailing_epa_features


def print_calibration(eval_games, prob_cols_labels):
    print("=" * 60)
    print("CALIBRATION (lower is better; market is the benchmark to beat)")
    print("=" * 60)
    scored = eval_games[eval_games["market_home_prob"].notna()]
    for prob_col, label in prob_cols_labels:
        m = score_calibration(scored, prob_col, label)
        print(f"  {m['label']:<38} n={m['n_games']:<5} brier={m['brier_score']:.4f}  log_loss={m['log_loss']:.4f}")


def print_flat_backtest(eval_games, prob_cols_labels, min_edge):
    print()
    print("=" * 60)
    print(f"FLAT-STAKE BACKTEST (bet $100 when edge > {min_edge}%, vs. closing lines)")
    print("=" * 60)
    odds_games = eval_games[eval_games["home_moneyline"].notna()]
    for prob_col, label in prob_cols_labels:
        _, summ = simulate_flat_stake(odds_games, prob_col, stake=100.0, min_edge_pct=min_edge)
        if summ["n_bets"] == 0:
            print(f"  {label}: no qualifying bets")
            continue
        print(
            f"  {label:<38} n_bets={summ['n_bets']:<5} win_rate={summ['win_rate']:.1%} "
            f"staked=${summ['total_staked']:,.0f} profit=${summ['total_profit']:,.0f} "
            f"ROI={summ['roi_pct']:.2f}%"
        )


def print_kelly_backtest(eval_games, prob_col, label, min_edge):
    print()
    print("=" * 60)
    print(f"HALF-KELLY BACKTEST: {label} ($10,000 starting bankroll)")
    print("=" * 60)
    odds_games = eval_games[eval_games["home_moneyline"].notna()]
    _, ksumm = simulate_kelly_stake(odds_games, prob_col, min_edge_pct=min_edge)
    print(
        f"  n_bets={ksumm['n_bets']}  start=${ksumm['starting_bankroll']:,.0f}  "
        f"end=${ksumm['ending_bankroll']:,.0f}  ROI={ksumm['roi_pct']:.2f}%"
    )


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
    games = build_features(games)

    # ---- Section 1: baseline model, all completed games (reg + playoffs) ----
    baseline_all = walk_forward_predict(
        games.copy(), feature_columns=FEATURE_COLUMNS, min_train_seasons=args.min_train_seasons
    )
    baseline_all = add_market_probs(baseline_all)
    warmup_season = sorted(baseline_all["season"].unique())[args.min_train_seasons]
    eval_all = baseline_all[baseline_all["season"] >= warmup_season]
    eval_all = eval_all[eval_all["model_home_win_prob"].notna()]

    print(f"\n{len(eval_all)} games available for out-of-sample evaluation (all game types)\n")
    print_calibration(
        eval_all,
        [
            ("elo_home_win_prob", "Elo (raw)"),
            ("model_home_win_prob", "Baseline logistic (elo+rest+div)"),
            ("market_home_prob", "Market (de-vigged, closing line)"),
        ],
    )
    print_flat_backtest(
        eval_all,
        [
            ("model_home_win_prob", "Baseline logistic"),
            ("elo_home_win_prob", "Elo (raw)"),
        ],
        args.min_edge,
    )
    print_kelly_backtest(eval_all, "model_home_win_prob", "Baseline logistic", args.min_edge)

    # ---- Section 2: EPA-enhanced model, regular season only, same warmup ----
    print("\nBuilding trailing offense/defense EPA features (regular season only)...")
    reg = build_trailing_epa_features(games, force_refresh=args.refresh)

    reg_baseline = walk_forward_predict(
        reg.copy(), feature_columns=FEATURE_COLUMNS, min_train_seasons=args.min_train_seasons
    ).rename(columns={"model_home_win_prob": "model_v1_prob"})
    reg_epa = walk_forward_predict(
        reg.copy(), feature_columns=FEATURE_COLUMNS_V2, min_train_seasons=args.min_train_seasons
    ).rename(columns={"model_home_win_prob": "model_v2_prob"})

    reg = reg.merge(reg_baseline[["game_id", "model_v1_prob"]], on="game_id")
    reg = reg.merge(reg_epa[["game_id", "model_v2_prob"]], on="game_id")
    reg = add_market_probs(reg)

    warmup_season_reg = sorted(reg["season"].unique())[args.min_train_seasons]
    eval_reg = reg[reg["season"] >= warmup_season_reg]
    eval_reg = eval_reg[eval_reg["model_v2_prob"].notna()]

    print(f"\n{len(eval_reg)} regular-season games available for EPA-model evaluation\n")
    print_calibration(
        eval_reg,
        [
            ("elo_home_win_prob", "Elo (raw)"),
            ("model_v1_prob", "Baseline logistic (elo+rest+div)"),
            ("model_v2_prob", "EPA-enhanced logistic (+ off/def EPA)"),
            ("market_home_prob", "Market (de-vigged, closing line)"),
        ],
    )
    print_flat_backtest(
        eval_reg,
        [
            ("model_v2_prob", "EPA-enhanced logistic"),
            ("model_v1_prob", "Baseline logistic"),
        ],
        args.min_edge,
    )
    print_kelly_backtest(eval_reg, "model_v2_prob", "EPA-enhanced logistic", args.min_edge)

    print()
    print("Reminder: this backtest bets against historical CLOSING lines, one of the")
    print("hardest prices in sports betting to beat. A positive ROI here over a large")
    print("sample is a meaningfully good sign; a negative one means the model has no")
    print("real edge yet and should not be bet with real money. See README.md.")


if __name__ == "__main__":
    main()
