"""Walk-forward logistic regression that recalibrates Elo into a win probability.

"Walk-forward" means: to predict season S, we only ever train on seasons
strictly before S. This is the minimum bar for an honest backtest — training
and testing on the same season, or shuffling games randomly across seasons
into train/test, would leak future information into predictions of the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import FEATURE_COLUMNS


def walk_forward_predict(
    df: pd.DataFrame,
    feature_columns: list[str] = FEATURE_COLUMNS,
    target_column: str = "home_win",
    min_train_seasons: int = 3,
) -> pd.DataFrame:
    """Return a copy of `df` with a `model_home_win_prob` column of out-of-sample
    predictions. Rows in the first `min_train_seasons` seasons (used only to warm
    up training data) get NaN and should be dropped before evaluating.
    """
    df = df.copy()
    df["model_home_win_prob"] = np.nan

    seasons = sorted(df["season"].unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"Need more than {min_train_seasons} seasons of data to walk-forward train."
        )

    # Ties (is_tie==1) are dropped from training/eval of the binary win model;
    # they're rare (a small handful in modern NFL history) and undefined as a
    # "win" label. They keep their NaN prediction here.
    trainable = df[df["is_tie"] == 0]

    for test_season in seasons[min_train_seasons:]:
        train_mask = trainable["season"] < test_season
        test_mask = df["season"] == test_season

        X_train = trainable.loc[train_mask, feature_columns]
        y_train = trainable.loc[train_mask, target_column]
        if y_train.nunique() < 2:
            continue

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        X_test = df.loc[test_mask, feature_columns]
        probs = clf.predict_proba(X_test)[:, 1]
        df.loc[test_mask, "model_home_win_prob"] = probs

    return df
