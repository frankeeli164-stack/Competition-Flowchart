# NFL Sports Betting Model

A full pipeline: predict game outcomes → compare to market odds → size the
stake → track results. Built as an honest research/backtesting tool, not a
turnkey money-printer — see **Current results** and **Limitations** before
you decide whether to bet a cent of real money on this.

## What's actually in here

- **Elo ratings** (`src/elo.py`) — sequential team ratings with home-field
  advantage, margin-of-victory scaling, and season-to-season regression to
  the mean. Every rating used to predict a game is computed only from games
  *before* it — no lookahead.
- **Walk-forward logistic model** (`src/model.py`) — recalibrates Elo's win
  probability using logistic regression, retrained each season using only
  prior seasons' data. This is the minimum standard for an honest sports
  betting backtest: train/test split by time, never by random shuffle.
- **Odds math** (`src/odds_math.py`) — American ↔ decimal conversion,
  implied probability, two-way de-vig, expected value. Unit tested.
- **Kelly staking** (`src/kelly.py`) — full and fractional Kelly bet sizing
  with a hard bankroll-percentage cap. Unit tested.
- **Backtest** (`src/backtest.py`) — scores calibration (Brier score, log
  loss) of Elo, the logistic model, and the de-vigged market itself as the
  benchmark; simulates flat-stake and Kelly-stake betting against historical
  closing lines.
- **Value-bet finder** (`src/ev.py`) — given upcoming games with your model's
  probabilities and current odds you supply, ranks +EV bets with a
  suggested Kelly stake. No live odds feed is included (see below).
- **Bet tracker** (`src/tracker.py`) — plain CSV log to record real bets you
  place and settle them later.

## Data source

`src/data.py` pulls `games.csv` from the `nflverse/nfldata` GitHub repo —
7,300+ NFL games since 1999 with final scores *and* historical closing
moneylines/spreads/totals, free, no API key. (`nfl_data_py`'s own default
source, `habitatring.com`, is blocked by network policy in some sandboxed
environments, so this fetches the same file straight from
`raw.githubusercontent.com` instead.)

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/run_pipeline.py
```

This fetches data (cached in `data/games.csv` after the first run), builds
Elo ratings, trains the walk-forward model, and prints a calibration +
backtest report. Re-run with `--refresh` to re-download the source data, or
`--min-edge 5.0` to require a bigger edge before a backtested bet is placed.

Run the test suite:

```bash
python3 -m pytest tests/ -v
```

## Current results (as of this build, 1999–2026 data)

```
CALIBRATION (lower is better; market is the benchmark to beat)
  Elo (raw)                        n=4633  brier=0.2202  log_loss=0.6302
  Logistic model (walk-forward)    n=4633  brier=0.2201  log_loss=0.6298
  Market (de-vigged, closing line) n=4633  brier=0.2105  log_loss=0.6084

FLAT-STAKE BACKTEST (bet $100 when edge > 2.0%, vs. closing lines)
  Logistic model   n_bets=3615  win_rate=39.3%  ROI=-5.03%
  Elo (raw)        n_bets=3621  win_rate=40.6%  ROI=-4.99%

HALF-KELLY BACKTEST (model probs, $10,000 starting bankroll)
  n_bets=3413  end=$0.10  ROI=-100.00%
```

**Read this honestly, not optimistically:**

- The market's de-vigged closing line is *better calibrated* than both Elo
  and the logistic model (lower Brier score, lower log loss). That's
  expected — a closing line prices in injuries, weather, public and sharp
  money, and everything else this project doesn't model. Beating it is a
  professional-level result, not a weekend-project baseline.
- Because the model disagrees with the market on ~78% of games at only a 2%
  edge threshold, and is *worse* calibrated than the market, betting on
  those disagreements loses money — about -5% ROI on flat stakes across
  3,600+ bets, which is a large, statistically real sample, not noise.
- The half-Kelly simulation is the sharpest illustration of why this matters:
  sizing bets as a fraction of bankroll based on an overconfident,
  miscalibrated probability doesn't just lose slowly — it compounds losses
  and rides the bankroll down to essentially zero. Kelly staking amplifies a
  bad probability estimate; it does not protect you from having one.
- **In short: this model does not currently have a real betting edge.** It's
  an honest, working, leakage-free pipeline you can extend, not a validated
  strategy. Do not bet real money against a sportsbook's closing line using
  the probabilities this produces as-is.

## What would make this better (not yet built)

- More features: recent-form/EPA-based metrics (nflfastR play-by-play),
  QB status, weather, rest/travel beyond simple day counts.
- Spread and totals models, not just moneyline win probability.
- Backtesting against *opening* lines rather than closing lines, since
  closing lines are the hardest number in the sport to beat and a more
  realistic target for a bettor is to find value before the line moves.
- A live odds feed (e.g. The Odds API) to actually run `src/ev.py` on
  upcoming games instead of only backtesting on history.
- Cross-validated hyperparameters for Elo's K-factor and home-field
  advantage instead of the current fixed defaults.

## Project layout

```
sports_betting_model/
  src/
    odds_math.py   american/decimal odds, implied prob, de-vig, EV
    kelly.py        Kelly criterion stake sizing
    data.py         fetch + cache NFL games/odds
    elo.py          sequential Elo ratings
    features.py     pre-game feature engineering (no leakage)
    model.py        walk-forward logistic regression
    backtest.py      calibration scoring + betting simulation
    ev.py           value-bet finder for upcoming games (you supply odds)
    tracker.py       CSV bet log for bets you actually place
  scripts/
    run_pipeline.py end-to-end CLI: fetch -> train -> backtest -> report
  tests/            unit tests for odds/Kelly math and Elo
```
