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
- **Trailing offense/defense EPA** (`src/team_epa.py`) — rolling 8-game
  offensive and defensive EPA-per-play for each team, built from weekly
  player stats (regular season only). Every value is a `.shift(1).rolling()`
  computed strictly before the game it's attached to — verified by a
  dedicated no-lookahead unit test, the same way Elo is.
- **Walk-forward logistic model** (`src/model.py`) — recalibrates Elo's win
  probability (optionally plus EPA features) using logistic regression,
  retrained each season using only prior seasons' data. This is the minimum
  standard for an honest sports betting backtest: train/test split by time,
  never by random shuffle.
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
- **Live weekly picks** (`scripts/next_week_picks.py`) — fits the model on
  all completed history and ranks the upcoming week's games by win
  confidence, with an example parlay built from the top picks (and a
  side-by-side comparison to betting the same picks straight — see
  **Player props** below for why a parlay of good picks isn't itself a good
  bet).
- **Player prop projections** (`src/player_props.py`, `src/props_math.py`) —
  per-player passing/rushing/receiving yardage, receptions, and anytime-TD
  probability, projected from each player's own trailing form plus a
  (deliberately small — see below) opponent-strength adjustment. See
  **Player props: what this can and can't validate**.

## Player props: what this can and can't validate

Player props are a fundamentally weaker validation story than the moneyline
model above, for one structural reason: **there is no free historical
player-prop-odds dataset**, unlike game moneylines (which `nflverse`
conveniently publishes going back to 1999). Historical prop lines exist only
behind paid data vendors. That means:

- `scripts/player_props_report.py` can validate that projections are
  *accurate* (MAE against actual results, compared to a naive "just use the
  player's own trailing average" baseline) and *roughly unbiased*
  (calibration), using the same leakage-free `shift(1).rolling()` discipline
  as everything else in this project.
- It **cannot** validate ROI against a real market, because there's no
  historical market to test against. Don't mistake "the projection is
  accurate" for "this beats sportsbook prop pricing" — those are different
  claims and only the first one is tested here.
- `scripts/prop_bet_check.py` is the practical tool: give it a player, stat,
  opponent, and the actual line/odds from your sportsbook, and it computes
  the model's probability and edge against that real number. This is
  intentionally manual — no live odds feed exists in this project.

**A real finding from backtesting, not a design choice made in advance:**
the opponent-strength adjustment (a defense's trailing allowed-yards to a
position group) was originally implemented the same way as team-level EPA,
and it made projections measurably *worse* (~2-4% higher MAE) at full
strength. Position-level defense-allowed stats are much noisier than
team-level EPA (far fewer relevant plays per game to average over). Heavily
shrinking the adjustment toward "ignore it" (`DEFAULT_ADJUSTMENT_SHRINK =
0.15` in `player_props.py`) got it to roughly neutral. The honest takeaway:
**for individual player props, a player's own recent form dominates far more
than opponent matchup does** — the opposite of what was found for team-level
game outcomes, where the opponent adjustment measurably helped. Run
`scripts/player_props_report.py` yourself to see this.

**Data staleness is worse here than for team games.** `nflverse`'s
`player_stats` release currently only goes through 2024 — no 2025 or 2026
data is published yet. `scripts/prop_bet_check.py` prints an unmissable
warning when this applies, but the practical effect is real: right now,
every live player projection reflects a player's performance as of the end
of the 2024 season, not their current form. A player who changed teams, got
injured, lost/gained a role, or simply improved or declined since then is
invisible to this model until `nflverse` publishes newer data. Check the
warning banner before trusting any live prop output.

## Data source

`src/data.py` pulls `games.csv` from the `nflverse/nfldata` GitHub repo —
7,300+ NFL games since 1999 with final scores *and* historical closing
moneylines/spreads/totals, free, no API key. (`nfl_data_py`'s own default
source, `habitatring.com`, is blocked by network policy in some sandboxed
environments, so this fetches the same file straight from
`raw.githubusercontent.com` instead.)

`src/team_epa.py` pulls `player_stats_{year}.parquet` release files from
`nflverse/nflverse-data` — weekly per-player stats that already include
passing/rushing/receiving EPA, aggregated here to team-week offense/defense
strength. Note: this release can lag behind `games.csv` for the current
in-progress season; missing years are skipped with a printed warning rather
than crashing the pipeline.

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
ALL GAME TYPES (reg + playoffs), baseline model (Elo + rest + division game):
  Elo (raw)                         n=4633  brier=0.2202  log_loss=0.6302
  Baseline logistic                 n=4633  brier=0.2201  log_loss=0.6298
  Market (de-vigged, closing line)  n=4633  brier=0.2105  log_loss=0.6084
  Baseline flat-stake backtest: n_bets=3615  win_rate=39.3%  ROI=-5.03%
  Baseline half-Kelly backtest: n_bets=3413  end=$0.10  ROI=-100.00%

REGULAR SEASON ONLY, same games, baseline vs. EPA-enhanced model:
  Elo (raw)                         n=4434  brier=0.2201  log_loss=0.6299
  Baseline logistic                 n=4434  brier=0.2200  log_loss=0.6296
  EPA-enhanced logistic             n=4434  brier=0.2189  log_loss=0.6272
  Market (de-vigged, closing line)  n=4434  brier=0.2103  log_loss=0.6079
  EPA-enhanced flat-stake backtest: n_bets=3464  win_rate=40.4%  ROI=-3.73%
  Baseline flat-stake backtest (same games): n_bets=3457  win_rate=39.4%  ROI=-5.10%
  EPA-enhanced half-Kelly backtest: n_bets=3464  end=$2  ROI=-99.98%
```

**Read this honestly, not optimistically:**

- Adding trailing offense/defense EPA-per-play is a *real, measured*
  improvement: Brier score moved from 0.2200 to 0.2189, and flat-stake ROI
  improved from -5.10% to -3.73% on the identical set of games. That's not
  nothing — it closes part of the gap to the market. It is still nowhere
  close to closing it, and it's still a losing strategy.
- The market's de-vigged closing line remains *better calibrated* than every
  model version here (lowest Brier score, lowest log loss). That's expected
  — a closing line prices in injuries, weather, public and sharp money, and
  everything else this project doesn't model. Beating it is a
  professional-level result, not a weekend-project baseline.
- The model still disagrees with the market on the large majority of games
  at only a 2% edge threshold, which is a sign the model is still
  overconfident relative to the market, not that it's found 3,000+ real
  edges. A higher `--min-edge` would cut bet volume but wouldn't turn the
  sign of the ROI positive on its own — the underlying probabilities need to
  keep improving.
- The half-Kelly simulation is the sharpest illustration of why calibration
  matters more than raw accuracy: sizing bets as a fraction of bankroll
  based on an overconfident probability doesn't just lose slowly — it
  compounds losses and rides the bankroll toward zero. Kelly amplifies a bad
  probability estimate; it does not protect you from having one.
- **In short: this model does not currently have a real betting edge**, but
  the EPA work shows the right direction to keep pushing in. Do not bet real
  money against a sportsbook's closing line using these probabilities as-is.

## What would make this better (not yet built)

- Full play-by-play features (not just weekly aggregates): success rate,
  explosive-play rate, EPA splits by down/distance, QB-specific EPA separate
  from team EPA (a backup QB shouldn't inherit the starter's trailing stats).
- Non-performance signals: injury reports, weather, Vegas total/spread
  movement itself as an input, coaching/scheme changes.
- Spread and totals models, not just moneyline win probability.
- Backtesting against *opening* lines rather than closing lines, since
  closing lines are the hardest number in the sport to beat and a more
  realistic target for a bettor is to find value before the line moves.
- A live odds feed (e.g. The Odds API) to actually run `src/ev.py` on
  upcoming games instead of only backtesting on history.
- Cross-validated hyperparameters (Elo K-factor/home-field advantage, EPA
  rolling window length) instead of the current fixed defaults.
- Extend EPA features to the postseason (needs careful week-number alignment
  between data sources — deliberately skipped for now, see `team_epa.py`).
- Real per-player modeling for props instead of a population-level Normal
  approximation (e.g. per-position variance models, usage/target-share
  trends, snap counts) — the current version borrows strength across a whole
  position group because most individual players don't have enough games to
  fit their own distribution.
- A paid historical player-prop-odds dataset, to actually validate ROI for
  props the way `backtest.py` does for moneylines (currently impossible with
  free data — see **Player props** above).

## Project layout

```
sports_betting_model/
  src/
    odds_math.py         american/decimal odds, implied prob, de-vig, EV
    kelly.py              Kelly criterion stake sizing
    data.py               fetch + cache NFL games/odds
    elo.py                sequential Elo ratings
    player_stats_source.py  shared nflverse player-stats fetch/cache
    team_epa.py           trailing offense/defense EPA-per-play (no leakage)
    features.py           pre-game feature engineering (no leakage)
    model.py              walk-forward logistic regression
    backtest.py            calibration scoring + betting simulation
    ev.py                 value-bet finder for upcoming games (you supply odds)
    tracker.py             CSV bet log for bets you actually place
    player_props.py       trailing player stats + opponent-adjusted projections
    props_math.py         over/under and anytime-TD probability math
    props_backtest.py     player-prop projection accuracy/calibration (no ROI -- see above)
  scripts/
    run_pipeline.py       end-to-end CLI: fetch -> train -> backtest -> report
    next_week_picks.py    live model picks for the upcoming week + example parlay
    player_props_report.py  player-prop projection accuracy/calibration backtest
    prop_bet_check.py     evaluate ONE prop against a line/odds you supply
  tests/                  unit tests: odds/Kelly math, Elo, EPA, player props (all leak-checked)
```
