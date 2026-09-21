"""A plain-CSV bet log for real (non-backtest) bets you actually place.

This is intentionally simple: one row per bet, append-only until settled. It
does not talk to any sportsbook — you log bets yourself (manually or from
`ev.find_value_bets`) and settle them once the game finishes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "bet_id",
    "date_placed",
    "event",
    "side",
    "american_odds",
    "model_prob",
    "stake",
    "status",  # pending | win | loss | push
    "profit",
    "notes",
]

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "bet_log.csv"


def _load(log_path: Path) -> pd.DataFrame:
    if log_path.exists():
        return pd.read_csv(log_path)
    return pd.DataFrame(columns=COLUMNS)


def add_bet(
    event: str,
    side: str,
    american_odds: float,
    stake: float,
    model_prob: float | None = None,
    date_placed: str | None = None,
    notes: str = "",
    log_path: Path = DEFAULT_LOG_PATH,
) -> int:
    """Append a new pending bet to the log. Returns the new bet_id."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df = _load(log_path)

    bet_id = int(df["bet_id"].max()) + 1 if len(df) else 1
    row = {
        "bet_id": bet_id,
        "date_placed": date_placed or pd.Timestamp.now().strftime("%Y-%m-%d"),
        "event": event,
        "side": side,
        "american_odds": american_odds,
        "model_prob": model_prob,
        "stake": stake,
        "status": "pending",
        "profit": None,
        "notes": notes,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(log_path, index=False)
    return bet_id


def settle_bet(bet_id: int, status: str, log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Mark a bet win/loss/push and compute its profit."""
    if status not in {"win", "loss", "push"}:
        raise ValueError("status must be one of: win, loss, push")

    df = _load(log_path)
    if bet_id not in df["bet_id"].values:
        raise ValueError(f"No bet with id {bet_id}")

    from .odds_math import american_to_decimal

    idx = df.index[df["bet_id"] == bet_id][0]
    stake = float(df.loc[idx, "stake"])
    odds = float(df.loc[idx, "american_odds"])

    if status == "win":
        profit = stake * (american_to_decimal(odds) - 1.0)
    elif status == "loss":
        profit = -stake
    else:
        profit = 0.0

    df.loc[idx, "status"] = status
    df.loc[idx, "profit"] = profit
    df.to_csv(log_path, index=False)


def summary(log_path: Path = DEFAULT_LOG_PATH) -> dict:
    df = _load(log_path)
    settled = df[df["status"] != "pending"]
    return {
        "n_bets_total": len(df),
        "n_pending": (df["status"] == "pending").sum(),
        "n_settled": len(settled),
        "total_staked": settled["stake"].sum(),
        "total_profit": settled["profit"].sum(),
        "roi_pct": (
            100.0 * settled["profit"].sum() / settled["stake"].sum()
            if len(settled) and settled["stake"].sum() > 0
            else None
        ),
    }
