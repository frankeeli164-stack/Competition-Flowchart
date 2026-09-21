"""Validate player-prop projection quality.

Important honesty note: there is no free historical player-prop-odds dataset
(unlike game moneylines, which nflverse conveniently publishes). So unlike
`backtest.py`, this CANNOT measure ROI against a real market -- there's no
market to compare against. What it CAN measure, honestly:

1. Projection accuracy (MAE) vs. actual results, compared against a naive
   baseline (player's own trailing average, no opponent adjustment) -- does
   the opponent-strength adjustment actually help, or is it just noise?
2. Calibration of the Normal-approximation over/under probability: using
   each projection's own mean as a synthetic "line", roughly half of actual
   outcomes should land above it if the model is unbiased.
3. Calibration of the anytime-TD probability via Brier score against actual
   TD-or-not outcomes.

All of this is leakage-free: every projection uses only `shift=True`
trailing features and a league average computed from strictly prior seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .player_props import (
    STAT_COLUMNS,
    STAT_RELEVANT_POSITIONS,
    defense_allowed_by_position,
    project_stat,
    trailing_defense_features,
    trailing_player_features,
)
from .props_math import prob_anytime_td, prob_over


def _season_expanding_league_average(defense_allowed: pd.DataFrame) -> pd.DataFrame:
    """For each season, the league-average allowed per (position_group, stat),
    computed from STRICTLY PRIOR seasons only (no leakage). The first season
    present has no prior data and is dropped by the caller.
    """
    seasons = sorted(defense_allowed["season"].unique())
    rows = []
    for s in seasons:
        prior = defense_allowed[defense_allowed["season"] < s]
        if prior.empty:
            continue
        avg = prior.groupby("position_group")[STAT_COLUMNS + ["td_rate"]].mean().reset_index()
        avg["season"] = s
        rows.append(avg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_projections(player_week: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game with actual values, the naive projection
    (player's own trailing rate), and the opponent-adjusted projection --
    all leakage-free (shift=True throughout, prior-seasons-only league avg).
    """
    player_trail = trailing_player_features(player_week, shift=True)
    defense_allowed = defense_allowed_by_position(player_week)
    defense_trail = trailing_defense_features(defense_allowed, shift=True)
    league_avg = _season_expanding_league_average(defense_allowed)

    merged = player_trail.merge(
        defense_trail.rename(columns={"defense_team": "opponent_team"}),
        on=["opponent_team", "position_group", "season", "week"],
        how="left",
        suffixes=("", "_defense"),
    )
    merged = merged.merge(
        league_avg.rename(columns={c: f"leagueavg_{c}" for c in STAT_COLUMNS + ["td_rate"]}),
        on=["position_group", "season"],
        how="left",
    )

    for stat in STAT_COLUMNS + ["td_rate"]:
        merged[f"naive_proj_{stat}"] = merged[f"trailing_{stat}"]
        merged[f"adj_proj_{stat}"] = merged.apply(
            lambda row, s=stat: project_stat(
                row[f"trailing_{s}"], row[f"trailing_{s}_defense"], row[f"leagueavg_{s}"]
            )
            if pd.notna(row[f"trailing_{s}"]) and pd.notna(row[f"trailing_{s}_defense"])
            else np.nan,
            axis=1,
        )

    return merged


def _relevant(projections: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Restrict to the position groups this stat is actually a prop market
    for -- otherwise e.g. linemen's always-zero passing yards swamp any real
    signal in a QB-only stat like passing yards.
    """
    return projections[projections["position_group"].isin(STAT_RELEVANT_POSITIONS[stat])]


def accuracy_report(projections: pd.DataFrame) -> pd.DataFrame:
    """Per-stat MAE of the naive vs. opponent-adjusted projection against
    actual results (rows with insufficient trailing history, or an
    irrelevant position for that stat, excluded).
    """
    rows = []
    for stat in STAT_COLUMNS:
        valid = _relevant(projections, stat)
        valid = valid[valid[f"adj_proj_{stat}"].notna()]
        naive_mae = (valid[stat] - valid[f"naive_proj_{stat}"]).abs().mean()
        adj_mae = (valid[stat] - valid[f"adj_proj_{stat}"]).abs().mean()
        rows.append({"stat": stat, "n": len(valid), "naive_mae": naive_mae, "adjusted_mae": adj_mae})
    return pd.DataFrame(rows)


def over_calibration(projections: pd.DataFrame, stat: str, std_dev: float) -> dict:
    """Using each row's OWN projected mean as a synthetic 'line', what fraction
    of actual results landed above it? Should be close to 50% for an unbiased
    projection (this is not evidence of market-beating value -- see module docstring).
    """
    valid = _relevant(projections, stat)
    valid = valid[valid[f"adj_proj_{stat}"].notna()].copy()
    valid["p_over"] = valid[f"adj_proj_{stat}"].apply(lambda m: prob_over(m, m, std_dev))
    actual_over_rate = (valid[stat] > valid[f"adj_proj_{stat}"]).mean()
    return {"stat": stat, "n": len(valid), "actual_over_own_projection_rate": actual_over_rate}


def td_calibration(projections: pd.DataFrame) -> dict:
    valid = _relevant(projections, "td_rate")
    valid = valid[valid["adj_proj_td_rate"].notna()].copy()
    valid["p_td"] = valid["adj_proj_td_rate"].apply(prob_anytime_td)
    actual = (valid["td_rate"] > 0).astype(int)
    return {
        "n": len(valid),
        "brier_score": brier_score_loss(actual, valid["p_td"].clip(1e-6, 1 - 1e-6)),
        "actual_td_rate": actual.mean(),
        "avg_predicted_prob": valid["p_td"].mean(),
    }
