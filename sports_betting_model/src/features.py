"""Build the feature matrix used by the win-probability model.

All features are computed from information available *before* kickoff, so this
is safe to use for both training and live prediction without leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["elo_logit", "rest_diff", "div_game"]


def _logit(p: pd.Series, eps: float = 1e-6) -> pd.Series:
    p = p.clip(eps, 1 - eps)
    return np.log(p / (1 - p))


def build_features(games_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Add model feature columns to a games dataframe that already has Elo columns
    (i.e. has been through `elo.run_elo`).
    """
    df = games_with_elo.copy()

    df["elo_logit"] = _logit(df["elo_home_win_prob"])

    # away_rest/home_rest are days of rest before this game; missing for very
    # early-season games in some years, default to a normal week (7 days).
    df["home_rest"] = df["home_rest"].fillna(7)
    df["away_rest"] = df["away_rest"].fillna(7)
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    df["div_game"] = df["div_game"].fillna(0).astype(int)

    return df
