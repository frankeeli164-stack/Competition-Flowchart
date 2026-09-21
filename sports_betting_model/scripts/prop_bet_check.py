#!/usr/bin/env python3
"""Evaluate ONE specific player prop against a line/odds you supply from your
own sportsbook. There is no live odds feed in this project (see README) --
you provide the real number, this checks it against the projection.

Usage:
    python3 scripts/prop_bet_check.py --player "Puka Nacua" --stat receiving_yards \
        --opponent SF --side over --line 74.5 --odds -110

    python3 scripts/prop_bet_check.py --player "Christian McCaffrey" --stat anytime_td \
        --opponent SEA --odds 145
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.odds_math import american_to_implied_prob, ev_percent
from src.player_props import (
    STAT_COLUMNS,
    defense_allowed_by_position,
    league_average_allowed,
    player_week_stats,
    position_std_dev,
    project_stat,
    trailing_defense_features,
    trailing_player_features,
)
from src.props_math import prob_anytime_td, prob_over, prob_under

ALL_STATS = STAT_COLUMNS + ["anytime_td"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True, help="Player name (substring match)")
    parser.add_argument("--stat", required=True, choices=ALL_STATS)
    parser.add_argument("--opponent", required=True, help="Opponent team abbreviation, e.g. SF")
    parser.add_argument("--line", type=float, help="Prop line (not needed for --stat anytime_td)")
    parser.add_argument("--side", choices=["over", "under"], default="over")
    parser.add_argument("--odds", type=float, required=True, help="American odds for this side")
    parser.add_argument("--min-season", type=int, default=2010)
    parser.add_argument("--max-season", type=int, default=2026)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.stat != "anytime_td" and args.line is None:
        parser.error("--line is required unless --stat anytime_td")

    years = list(range(args.min_season, args.max_season + 1))
    print(f"Loading player stats {years[0]}-{years[-1]}...")
    player_week = player_week_stats(years, force_refresh=args.refresh)
    latest_season = int(player_week["season"].max())

    seasons_stale = args.max_season - latest_season
    if seasons_stale >= 1:
        print()
        print("!" * 78)
        print(f"! WARNING: data only published through {latest_season}. If it's currently the")
        print(f"! {args.max_season} season (or later), every projection below is based on")
        print(f"! player/defense performance from {seasons_stale}+ season(s) ago -- it does NOT know")
        print(f"! about trades, injuries, role changes, or current form since then.")
        print(f"! This is a data-availability gap (nflverse hasn't published newer data),")
        print(f"! not something this script can fix. Treat this output accordingly.")
        print("!" * 78)
    print()

    matches = player_week[
        player_week["player_display_name"].str.contains(args.player, case=False, na=False)
    ]["player_id"].unique()
    if len(matches) == 0:
        print(f"No player found matching '{args.player}'.")
        return
    if len(matches) > 1:
        names = player_week[player_week["player_id"].isin(matches)]["player_display_name"].unique()
        print(f"Multiple players match '{args.player}': {list(names)}")
        print("Be more specific.")
        return

    player_id = matches[0]
    player_rows = player_week[player_week["player_id"] == player_id]
    player_name = player_rows["player_display_name"].iloc[0]
    position_group = player_rows["position_group"].iloc[-1]
    n_games = len(player_rows)
    print(f"Found: {player_name} ({position_group}), {n_games} games in history\n")

    if n_games < 3:
        print("WARNING: fewer than 3 games of history -- projection will fall back")
        print("to a league/position default and is not meaningful for this player.\n")

    # "Current" (as-of-now) trailing stats: shift=False, since there's no future
    # game to exclude for a genuinely upcoming game.
    player_current = trailing_player_features(player_rows, shift=False)
    player_latest = player_current.sort_values(["season", "week"]).iloc[-1]

    defense_allowed = defense_allowed_by_position(player_week)
    defense_current = trailing_defense_features(defense_allowed, shift=False)
    opp_row = defense_current[
        (defense_current["defense_team"] == args.opponent)
        & (defense_current["position_group"] == position_group)
    ].sort_values(["season", "week"])
    if opp_row.empty:
        print(f"No defensive data found for {args.opponent} vs {position_group}. Using league average.")
        opp_row = None
    else:
        opp_row = opp_row.iloc[-1]

    league_avg = league_average_allowed(defense_allowed)
    league_avg_row = league_avg[league_avg["position_group"] == position_group]
    if league_avg_row.empty:
        print(f"No league-average data for position group {position_group}.")
        return
    league_avg_row = league_avg_row.iloc[0]

    if args.stat == "anytime_td":
        player_rate = player_latest["trailing_td_rate"]
        opp_rate = opp_row["trailing_td_rate"] if opp_row is not None else league_avg_row["td_rate"]
        projected_rate = project_stat(player_rate, opp_rate, league_avg_row["td_rate"])
        model_prob = prob_anytime_td(projected_rate)
        print(f"Projected TDs/game: {projected_rate:.3f}")
        print(f"Model P(anytime TD): {model_prob:.1%}")
        side_label = "anytime TD"
    else:
        stat = args.stat
        player_rate = player_latest[f"trailing_{stat}"]
        opp_rate = opp_row[f"trailing_{stat}"] if opp_row is not None else league_avg_row[stat]
        projected_mean = project_stat(player_rate, opp_rate, league_avg_row[stat])

        std_devs = position_std_dev(player_week)
        std_dev = std_devs[std_devs["position_group"] == position_group][stat].iloc[0]

        prob_fn = prob_over if args.side == "over" else prob_under
        model_prob = prob_fn(args.line, projected_mean, std_dev)

        print(f"Projected {stat}: {projected_mean:.1f} (position std dev: {std_dev:.1f})")
        print(f"Line: {args.side} {args.line}")
        print(f"Model P({args.side}): {model_prob:.1%}")
        side_label = f"{args.side} {args.line} {stat}"

    market_implied = american_to_implied_prob(args.odds)
    edge = ev_percent(model_prob, args.odds)

    print()
    print(f"Odds offered: {args.odds:+.0f}  (market-implied prob, with vig: {market_implied:.1%})")
    print(f"Model edge: {edge:+.2f}%  {'(model likes this bet)' if edge > 0 else '(model does NOT like this bet)'}")
    print()
    print(f"Reminder: this is one Normal/Poisson-approximation model with no historical")
    print(f"prop-odds validation behind it (see README). Treat this as one input, not a")
    print(f"verdict -- especially for {player_name} with only {n_games} games of history.")


if __name__ == "__main__":
    main()
