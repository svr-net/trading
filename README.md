# Daily Trades (UK and US)

A walk-forward predictive model of every share's daily OHLCV, in C++ with fused kernels (WGSL,
translated to C++ by `tools/wgsl2cpp.py` and run on every CPU thread), and a page with the next
open's trades and the model's backtest, for the London Stock Exchange and for NYSE and NASDAQ.

## The UK model (`apps/ofm_uk`): the London orders

The UK's live strategy (`config/markets.json`) is a separate, deliberately simple model whose every
order is one a person would place, and whose backtest trades exactly those orders:

- **Shares**: ordinary shares in the order book's auctions only (SET1, SET3, AMSM, SSMM in
  `tools/eoddata/lse_segments.csv`): no closed-end funds (STMM), ETFs/ETPs or quote-driven SETSqx
  shares. At least 20p, 250 days of history, a bar on 18 of the last 20 sessions, a median of GBP
  250,000 traded a day, no 30% move on the day (a bid, a placing, bad data); the 300 most traded.
  Delisted shares stay in the backtest (no survivorship bias from today's list).
- **Forecast**: one walk-forward ridge regression across the shares of the next open-to-close (in
  each share's own volatilities) on the signals of `apps/swing.hpp`, the share's volatility, liquidity
  and stamp duty, each ranked across the day's shares; refitted every 21 sessions on the 750 before.
- **Expected return**: the forecast calibrated on the model's own out-of-sample record (the last 500
  sessions' top fifth of forecasts against what those shares then did, the slope shrunk by its t
  statistic). No record, or no skill in it, means no orders.
- **Bought at the open, never on a dip**: a limit at the last close plus 0.5 of the share's daily
  volatility (it does not chase a big gap up), good only until 2 minutes after the open (IBKR's GTD;
  if refused, a day order the watch cancels then), so it buys in the opening auction or not at all.
  The amount is the slot's capital, **at most 1% of the share's median day's value** (`maxCapital`,
  which the budget and `tools/ibkr/submit.py` respect), at least GBP 500.
- **Stop-loss and take-profit from the fill**: the order file carries them as fractions of the price
  bought at (`stopRel`, `takeRel`); the bridge's watch (`submit.py --rebuy`) places them once the
  auction has filled, from the fill's own price, one cancelling the other (reduced together on a part
  fill, placed again when more is bought, never twice), so a gap down never leaves the stop above the
  entry. Whatever is still held is sold in the closing auction. The pair is chosen each day from a
  grid (stop 1 to 4 daily volatilities, take-profit 1 to 3 or none) by what it added to the model's
  top forecasts over the last 250 sessions against the 4-volatility stop alone, shrunk by its t
  statistic: levels are tightened only where they have paid. The daily bar cannot tell which came
  first, so a day that touched both counts as stopped; a stop sells half a spread below its level, a
  take-profit fills only when the high clears it by half a spread.
- **Only orders that pay**: the expected return must clear IBKR's commission on the order's own
  amount, the 0.5% stamp duty (not AIM shares), a quarter-spread at each auction, plus 0.05%. Many
  days have fewer orders than slots, some none.
- **Checks**: `tools/check/uk_leak.sh` (CI and the daily run): on a generated market it must find a
  planted effect, learn a take-profit where the day's highs spike and fade and none on a random walk,
  every order must be sensible (bought at the open only, levels around the entry, an amount within the
  share's liquidity) and unchanged by later data. `tools/ibkr/test/levels.py` (CI) runs the orders
  through `submit.py` against a stand-in IBKR: the auction-only buy, the levels from the fill, a part
  fill, a refused good-till time, the stop cancelling the take-profit.

`ofm_uk --csv stocks.csv --costs UK [--capital 5000] [--slots 5] [--stamp-free aim.txt] [--segments
lse_segments.csv] [--override ibkr.csv] [--json scan.json] [--orders orders.json]` prints the
backtest by year (with the forecast's daily rank correlation), the fill rate, how often trades were
stopped or took profit, every stop / take-profit pair's record, the average trade before and after
costs and the next session's orders. The previous UK live strategy (`regime-trend50`, the
research model below) runs on as a shadow for comparison; EU, US and APAC keep the research model.

## The model (`apps/ofm_research`, `kernels/v2`, `kernels/net`)

- **Features** (`features.wgsl`): 27 per share and day from its own daily open, high, low, close
  and volume: momentum over 2 to 64 days, intraday and overnight momentum (past open-to-close moves
  continue, past overnight gaps reverse), volume and range shocks, the day's open-to-close,
  close-to-close and gap, the close's place in its range, wicks, signed volume, liquidity (traded
  value) and the volatility level. Only the past enters each day's features.
- **Interactions** (`interact.wgsl`): the key signals times the share's liquidity and volatility,
  so the model can weight each signal by where it works.
- **Residual and literature signals** (`resid.wgsl`): the residual reversal over 5 and 21 days and
  the tug of war (overnight up, daytime down), the largest daily return of the last month (Bali,
  Cakici and Whitelaw 2011) and the idiosyncratic volatility (Ang, Hodrick, Xing and Zhang 2006);
  the 52-week high (George and Hwang 2004) is computed but left out, as it did not help.
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
- **Costs**: the model, its gate and the backtest run gross of the broker's costs (commission, fees, stamp duty; the spread stays). The final order list is net: each order's cost comes off its gain if it buys and (times its chance to buy) off its expected return, and the orders are sorted by that expected return, highest first. Each order in `--orders` carries its round-trip `cost` (a share of its value) and `costAmount`; `--net` charges them in the model and backtest as before.
- **Trades**: the next open's 10 highest predictions, each a buy limit d standard deviations below
  the last close: filled at the open when the share opens at or below it, during the day when its
  low trades half a bid-ask spread through it, otherwise not traded (cash). Sold at the
  take-profit, the stop-loss or the close, the levels relative to each share's own moves from the
  open over its last 100 days (the fall and the rise it reached on a learnt fraction, 1 in 2 to 1
  in 10, of those days; the stop blended with its parabolic SAR); after a fill during the day the
  take-profit is not credited (the high may have come first); `--exit close` sells every trade
  at the close instead; `--exit safety` (every market) keeps only a safety stop, at the fall the share
  reached on a learnt 1 in 20 to 1 in 100 of its days, at most a learnt 1.5 to 6 of its own daily sigmas below the buy price, with a take-profit where the share's own rises faded (from its last 40 to 750 days, a learnt window or none: the level that would have paid the most, each day sold at it once the high cleared it, else at the close), and sells at the close otherwise (US shares
  stopped during the day mostly recover by the close: +1% a year with learnt stops and
  take-profits against +10% with the safety stop, after costs; the UK +55% a year, Sharpe 4.94, against +79%, 4.98: the stop no longer sells into the opening volatility). Every lattice point's daily trades are one fused kernel
  (kernels/v2/levels.wgsl and records.wgsl). Learnt continuously each day from the records so far: the pool (how many of the most
  traded shares), the ranking (the expected return net of each share's own costs, from in volatilities
  to as a return), the limit's depth d (pool, ranking and limit for return, each by its record
  averaged over the level settings), the UK's daily cap on AIM shares (0 to 10, for return per
  unit of downside), the levels (for return per unit of
  downside, Sortino). Only trades expected to gain after costs are taken or published (the
  expectation calibrated on the past days' top 10 before this filter, so it keeps learning). A stop sells half a spread below its level, and a take-profit fills only
  when the high clears it by half a spread. Net of costs: every filled trade pays Interactive Brokers' commissions (UK 0.05% an
  order, at least GBP 1; US USD 0.0035 a share, at least 0.35; plus exchange, clearing and US
  regulatory fees) on GBP/USD 5,000 a trade, and UK buys 0.5% stamp duty, except AIM shares, exempt since 2014 (`--costs UK|US
  --capital 5000 --stamp 0.005 --stamp-free tools/eoddata/aim.txt`, the list refreshed daily by
  tools/eoddata/aim.mjs); everything above is learnt net of them. One unit in each trade. US: the 2,000 most traded shares per exchange; UK: the 500 most traded.

Orders for Interactive Brokers: `--orders orders.json` writes the next open's orders in prices (to
the private store, by the daily job), and `tools/ibkr/submit.py` places them through IB Gateway; see
[docs/ibkr.md](docs/ibkr.md) for opening the account, connecting and the daily routine.

`ofm_research --csv stocks.csv [--series futures.csv] [--lag 1] [--top N] [--json scan.json]`
prints each feature's day-by-day predictive power and the backtests, and writes the page's scan.

`ofm_paths paths/UK/*.csv` (or `--bars` 5-minute bars): a buy-back after a stop replayed on intraday
paths, against the bounds daily bars allow (research; `X_REBUY=f,mode[,w]` puts it in `ofm_research`'s
backtest: mode 0 the worst case, 2 only days the close settles, 1 an unclear day at weight w; off by default).

## Data

EODData's daily bars (key in the `EODDATA_API_KEY` secret) are kept in the private repository
`svr-net/eod` (`EOD_TOKEN`), checked out by every run, which downloads only what is missing
(`tools/eoddata`). EODData's licence does not allow redistributing prices: only results are
published.

## Schedule

`.github/workflows/uk-portfolio.yml` runs on weekday evenings (UK 19:00 UTC, US 22:30 UTC):
updates the data, runs the model, publishes the scan to the page and sends the daily email.

Research only, not investment advice.
