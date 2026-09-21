#!/usr/bin/env python3
"""Rank next week's games by the model's win confidence and build a parlay.

This is a LIVE prediction, not a backtest: it fits the model on ALL
available completed history (no held-out season) and applies it to games
that haven't been played yet, using each team's current Elo rating and
current trailing EPA (not the leakage-safe, shifted training-time version --
there's no "current game" to exclude when the game hasn't happened).

Usage:
    python3 scripts/next_week_picks.py [--refresh] [--n-legs 5] [--stake 100]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import fetch_games
from src.elo import DEFAULT_HOME_ADVANTAGE, run_elo, win_probability
from src.features import FEATURE_COLUMNS_V2, build_features
from src.odds_math import american_to_decimal, ev_percent
from src.team_epa import build_trailing_epa_features, current_team_strength


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--n-legs", type=int, default=5)
    parser.add_argument("--stake", type=float, default=100.0)
    args = parser.parse_args()

    print("Loading full schedule (completed + upcoming)...")
    all_games = fetch_games(force_refresh=args.refresh)
    all_games["gameday"] = pd.to_datetime(all_games["gameday"])
    all_games = all_games.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    completed = all_games[all_games["home_score"].notna()].copy()
    completed["home_win"] = (completed["home_score"] > completed["away_score"]).astype(int)
    completed["is_tie"] = (completed["home_score"] == completed["away_score"]).astype(int)

    upcoming = all_games[all_games["home_score"].isna()].copy()
    upcoming_reg = upcoming[upcoming["game_type"] == "REG"].sort_values(["season", "week", "gameday"])
    if upcoming_reg.empty:
        print("No upcoming games found in the schedule.")
        return

    # "Next week" = the week after the current one, not just the earliest
    # unplayed game -- a week with games still in progress (e.g. tonight's
    # Monday-night leftover) is THIS week, not next week.
    latest_season = int(completed[completed["game_type"] == "REG"]["season"].max())
    current_week = int(
        completed[(completed["season"] == latest_season) & (completed["game_type"] == "REG")]["week"].max()
    )
    next_season, next_week = latest_season, current_week + 1

    next_games = upcoming_reg[(upcoming_reg["season"] == next_season) & (upcoming_reg["week"] == next_week)].copy()
    if next_games.empty:
        # Fallback: whole next season hasn't started, or numbering doesn't line up.
        next_season, next_week = upcoming_reg.iloc[0][["season", "week"]]
        next_games = upcoming_reg[(upcoming_reg["season"] == next_season) & (upcoming_reg["week"] == next_week)].copy()
    print(f"Current week appears to be {current_week}. Next week: season {next_season}, week {next_week} "
          f"-- {len(next_games)} games\n")

    if next_games["home_moneyline"].isna().any():
        print("(Some games in this week have no posted moneyline yet -- excluding those.)")
        next_games = next_games[next_games["home_moneyline"].notna()]

    print("Computing current Elo ratings from completed games...")
    _, final_ratings = run_elo(completed, return_final_ratings=True)

    print("Computing current trailing offense/defense EPA...")
    strength = current_team_strength(completed, force_refresh=args.refresh).set_index("team")
    league_avg_epa = strength["off_epa"].mean()

    print("Training model on ALL completed history (this is a live pick, not a backtest)...")
    completed_elo = run_elo(completed)
    completed_elo = build_features(completed_elo)
    completed_epa = build_trailing_epa_features(completed_elo, force_refresh=args.refresh)
    trainable = completed_epa[completed_epa["is_tie"] == 0]

    clf = LogisticRegression()
    clf.fit(trainable[FEATURE_COLUMNS_V2], trainable["home_win"])

    rows = []
    for r in next_games.itertuples(index=False):
        home_elo = final_ratings.get(r.home_team, 1500.0)
        away_elo = final_ratings.get(r.away_team, 1500.0)
        elo_diff = (home_elo + DEFAULT_HOME_ADVANTAGE) - away_elo
        elo_home_prob = win_probability(elo_diff)

        home_off = strength["off_epa"].get(r.home_team, league_avg_epa)
        home_def = strength["def_epa_allowed"].get(r.home_team, league_avg_epa)
        away_off = strength["off_epa"].get(r.away_team, league_avg_epa)
        away_def = strength["def_epa_allowed"].get(r.away_team, league_avg_epa)

        home_rest = r.home_rest if pd.notna(r.home_rest) else 7
        away_rest = r.away_rest if pd.notna(r.away_rest) else 7
        div_game = int(r.div_game) if pd.notna(r.div_game) else 0

        feat = pd.DataFrame(
            [{
                "elo_logit": _logit(elo_home_prob),
                "rest_diff": home_rest - away_rest,
                "div_game": div_game,
                "home_off_epa": home_off,
                "home_def_epa_allowed": home_def,
                "away_off_epa": away_off,
                "away_def_epa_allowed": away_def,
            }]
        )[FEATURE_COLUMNS_V2]

        p_home = clf.predict_proba(feat)[0, 1]

        if p_home >= 0.5:
            side, team, odds, p_win = "home", r.home_team, r.home_moneyline, p_home
        else:
            side, team, odds, p_win = "away", r.away_team, r.away_moneyline, 1.0 - p_home

        rows.append(
            {
                "matchup": f"{r.away_team} @ {r.home_team}",
                "pick": team,
                "side": side,
                "american_odds": odds,
                "model_prob": p_win,
                "edge_pct": ev_percent(p_win, odds),
            }
        )

    picks = pd.DataFrame(rows).sort_values("model_prob", ascending=False).reset_index(drop=True)

    print("=" * 78)
    print(f"ALL {len(picks)} GAMES, RANKED BY MODEL CONFIDENCE")
    print("=" * 78)
    print(picks.to_string(index=False, formatters={
        "model_prob": "{:.1%}".format,
        "edge_pct": "{:+.2f}%".format,
    }))

    top = picks.head(args.n_legs).copy()
    print()
    print("=" * 78)
    print(f"TOP {args.n_legs} PICKS -> PARLAY")
    print("=" * 78)
    print(top.to_string(index=False, formatters={
        "model_prob": "{:.1%}".format,
        "edge_pct": "{:+.2f}%".format,
    }))

    decimals = [american_to_decimal(o) for o in top["american_odds"]]
    combined_decimal = 1.0
    for d in decimals:
        combined_decimal *= d
    combined_model_prob = top["model_prob"].prod()
    combined_market_implied = 1.0 / combined_decimal

    stake = args.stake
    payout = stake * combined_decimal
    profit_if_win = payout - stake
    parlay_ev = combined_model_prob * profit_if_win - (1 - combined_model_prob) * stake
    parlay_ev_pct = 100 * parlay_ev / stake

    print()
    print(f"Parlay ({args.n_legs} legs), ${stake:.0f} stake:")
    print(f"  Combined odds:              {combined_decimal:.2f}x decimal")
    print(f"  Payout if ALL {args.n_legs} hit:        ${payout:,.2f}  (profit ${profit_if_win:,.2f})")
    print(f"  Model prob ALL {args.n_legs} legs hit:   {combined_model_prob:.1%}")
    print(f"  Market-implied prob (w/ vig): {combined_market_implied:.1%}")
    print(f"  Parlay expected value:      ${parlay_ev:,.2f}  ({parlay_ev_pct:+.2f}% of stake)")

    print()
    print("Same 5 picks as SEPARATE straight bets instead (for comparison):")
    per_leg_stake = stake / args.n_legs
    straight_ev_total = 0.0
    for _, row in top.iterrows():
        d = american_to_decimal(row["american_odds"])
        leg_profit_if_win = per_leg_stake * (d - 1.0)
        leg_ev = row["model_prob"] * leg_profit_if_win - (1 - row["model_prob"]) * per_leg_stake
        straight_ev_total += leg_ev
    print(f"  ${per_leg_stake:.2f} on each of {args.n_legs} legs, ${stake:.0f} total staked")
    print(f"  Combined expected value:    ${straight_ev_total:,.2f}  ({100*straight_ev_total/stake:+.2f}% of stake)")
    print(f"  (vs. parlay's ${parlay_ev:,.2f} on the same ${stake:.0f} -- and if any ONE leg loses,")
    print(f"   the parlay pays $0; straight bets still pay out on every leg that hits.)")


if __name__ == "__main__":
    main()
