# Daily Trades (UK and US)

A walk-forward predictive model of every share's daily OHLCV, in C++ with fused kernels (WGSL,
translated to C++ by `tools/wgsl2cpp.py` and run on every CPU thread), and a page with the next
open's trades and the model's backtest, for the London Stock Exchange and for NYSE and NASDAQ.

## The model (`apps/ofm_research`, `kernels/v2`, `kernels/net`)

- **Features** (`features.wgsl`): 27 per share and day from its own daily open, high, low, close
  and volume: momentum over 2 to 64 days, intraday and overnight momentum (past open-to-close moves
  continue, past overnight gaps reverse), volume and range shocks, the day's open-to-close,
  close-to-close and gap, the close's place in its range, wicks, signed volume, liquidity (traded
  value) and the volatility level. Only the past enters each day's features.
- **Interactions** (`interact.wgsl`): the key signals times the share's liquidity and volatility,
  so the model can weight each signal by where it works.
- **Walk-forward fit** (`zscore.wgsl`, `gram.wgsl`, `predict.wgsl`): each close, a linear model of
  the next day's open-to-close, refitted on every earlier day only. A neural network
  (`kernels/net`: forward pass, gradient, Adam) is in the research output for comparison.
- **Bounce-free**: the trades use the prediction made one close earlier, so the bid-ask bounce
  between the last close and the next open is not learnt.
- **US shares**: ordinary shares only (no preferred shares, notes, rights, units, warrants or
  funds), told apart by name and, on NASDAQ, the fifth ticker letter.
- **Outliers**: impossible bars (a low above the open or close, a high below them) are dropped as
  missing; each day a share more than 3 robust standard deviations from the day's median on its
  last 60 days' traded value (low side), unmoved closes (stale prices) or daily range (wide) is
  left out of the trades.
- **Trades**: the next open's 10 highest predictions, each a buy limit d standard deviations below
  the last close: filled at the open when the share opens at or below it, during the day when its
  low trades half a bid-ask spread through it, otherwise not traded (cash). UK: sold at the
  take-profit, the stop-loss or the close, the levels relative to each share's own moves from the
  open over its last 100 days (the fall and the rise it reached on a learnt fraction, 1 in 2 to 1
  in 10, of those days; the stop blended with its parabolic SAR); after a fill during the day the
  take-profit is not credited (the high may have come first). US: sold at the close (`--exit
  close`). Learnt continuously each day from the records so far: the pool (how many of the most
  traded shares), the ranking (from the prediction in volatilities to the expected return), the
  limit's depth d (pool, ranking and limit for return), the levels (for return per unit of
  downside, Sortino). A stop sells half a spread below its level, and a take-profit fills only
  when the high clears it by half a spread. One unit in each trade;
  no fees or costs. US: the 2,000 most traded shares per exchange; UK: the 500 most traded.

`ofm_research --csv stocks.csv [--series futures.csv] [--lag 1] [--top N] [--json scan.json]`
prints each feature's day-by-day predictive power and the backtests, and writes the page's scan.

## Data

EODData's daily bars (key in the `EODDATA_API_KEY` secret) are kept in the private repository
`svr-net/eod` (`EOD_TOKEN`), checked out by every run, which downloads only what is missing
(`tools/eoddata`). EODData's licence does not allow redistributing prices: only results are
published.

## Schedule

`.github/workflows/uk-portfolio.yml` runs on weekday evenings (UK 19:00 UTC, US 22:30 UTC):
updates the data, runs the model, publishes the scan to the page and sends the daily email.

Research only, not investment advice.
