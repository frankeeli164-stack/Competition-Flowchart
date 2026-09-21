#!/usr/bin/env python3
"""Backtest player-prop projection accuracy and calibration.

No historical player-prop odds exist freely, so this can't report ROI --
only whether the projections themselves are accurate and unbiased. See
src/props_backtest.py's module docstring for exactly what is and isn't
being validated here.

Usage:
    python3 scripts/player_props_report.py [--refresh] [--min-season 2010]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.player_props import STAT_COLUMNS, player_week_stats, position_std_dev
from src.props_backtest import accuracy_report, build_projections, over_calibration, td_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--min-season", type=int, default=2010)
    parser.add_argument("--max-season", type=int, default=2024)
    args = parser.parse_args()

    years = list(range(args.min_season, args.max_season + 1))
    print(f"Loading player weekly stats, {years[0]}-{years[-1]}...")
    player_week = player_week_stats(years, force_refresh=args.refresh)
    print(f"  {len(player_week)} player-game rows loaded\n")

    print("Building leakage-free projections (trailing player rate x opponent adjustment)...")
    projections = build_projections(player_week)

    print()
    print("=" * 72)
    print("PROJECTION ACCURACY: naive (player's own trailing avg) vs. opponent-adjusted")
    print("Lower MAE is better. If 'adjusted' isn't beating 'naive', the opponent")
    print("adjustment isn't adding real signal for that stat.")
    print("=" * 72)
    acc = accuracy_report(projections)
    for _, row in acc.iterrows():
        improvement = 100 * (row["naive_mae"] - row["adjusted_mae"]) / row["naive_mae"]
        print(
            f"  {row['stat']:<18} n={row['n']:<6.0f} naive_mae={row['naive_mae']:.2f}  "
            f"adjusted_mae={row['adjusted_mae']:.2f}  ({improvement:+.1f}% {'better' if improvement > 0 else 'worse'})"
        )

    print()
    print("=" * 72)
    print("OVER/UNDER CALIBRATION (actual results vs. each game's own projected mean)")
    print("Should be close to 50% if the projection is unbiased.")
    print("=" * 72)
    std_devs = position_std_dev(player_week)
    for stat in STAT_COLUMNS:
        avg_std = std_devs[stat].mean()
        cal = over_calibration(projections, stat, avg_std)
        print(f"  {stat:<18} n={cal['n']:<6}  actual-over-own-projection rate: {cal['actual_over_own_projection_rate']:.1%}")

    print()
    print("=" * 72)
    print("ANYTIME-TD CALIBRATION (Poisson approximation)")
    print("=" * 72)
    td_cal = td_calibration(projections)
    print(
        f"  n={td_cal['n']}  brier_score={td_cal['brier_score']:.4f}  "
        f"actual_td_rate={td_cal['actual_td_rate']:.1%}  avg_predicted_prob={td_cal['avg_predicted_prob']:.1%}"
    )

    print()
    print("Reminder: none of this is a backtested ROI claim -- there's no free")
    print("historical player-prop odds to backtest against. This only validates")
    print("that the projections themselves are accurate and roughly unbiased.")
    print("Real-money use requires comparing these probabilities to actual")
    print("sportsbook prop odds (see scripts/prop_bet_check.py) and paper-trading")
    print("results over time before trusting this with real stakes.")


if __name__ == "__main__":
    main()
