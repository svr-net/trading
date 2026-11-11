# Daily Trades (UK and US)

A walk-forward predictive model of every share's daily OHLCV, in C++ with fused kernels (WGSL,
translated to C++ by `tools/wgsl2cpp.py` and run on every CPU thread), and a page with the next
open's trades and the model's backtest, for the London Stock Exchange and for NYSE and NASDAQ.

## The UK model (`apps/ofm_uk`): the London orders

**Europe too**: the same model is Europe's live strategy (`--costs EU --venues tools/venues.txt --hold 5`:
Paris, Amsterdam, Brussels, Lisbon, Madrid and Frankfurt; tickers `EXCHANGE:CODE`, prices in euros). Its
orders pay each exchange's commission and taxes on a share held past the day (`tools/venues.txt`: the
French FTT 0.4% and the Spanish 0.2% on buying, the Portuguese 0.3%, the Belgian TOB 0.35% each side;
none in Amsterdam or Frankfurt), applied to the final orders only. The research model (`base`) runs on
as Europe's shadow.

**And the US**: the US's live strategy too (`--costs US --hold 5`: NYSE and NASDAQ, dollars, shares of USD 5
or more). Its orders pay IBKR's USD 0.005 a share (at least USD 1, at most 1%) each way and the SEC's and
FINRA's fees on the sale; there is no tax on buying. The research model (`base`) runs on as its shadow.
**And Asia-Pacific**: the ASX and the SGX (`--costs APAC --venues tools/venues.txt --hold 5`, tickers
`EXCHANGE:CODE`): each exchange's commission and fees, no tax on buying. Each order is good only through its
own exchange's opening auction: the ASX opens in alphabetical groups from 10:00 to about 10:09 Sydney time
(good until 10:12), the SGX at 09:00 Singapore time (good until 09:02). An order's amount is in the ASX's AUD
(an SGX order's taken as SGD, about the same). The research model (`base`) runs on as its shadow.
**INDEX and FOREX**: INDEX trades index ETFs listed in New York (`tools/eoddata/index.txt`: the S&P 500,
Nasdaq 100, Dow, Russell 2000 and world indices, about 25 MSCI country indices and the 11 S&P sectors), with
the US's costs, on New York's clock (placed at 09:21, the bridge's `AUTO_MARKETS` naming INDEX). FOREX models
currency pairs (`tools/eoddata/forex.txt`, each with its inverse, `CODE-`, so the model can take either side;
IBKR's FX commission; no volume, so no liquidity cap) as research only: it has no budget, and the bridge does
not trade it. Both are fetched as named series (`tools/eoddata/series_list.py`: each from the first of its
EODData exchanges with bars) and scanned with the US's evening run.
What follows is said of the UK; the other markets are the same but for prices, segments and costs.

The UK's live strategy (`config/markets.json`) is a separate, deliberately simple model whose every
order is one a person would place, and whose backtest trades exactly those orders:

- **Shares**: ordinary shares in the order book's auctions only (SET1, SET3, AMSM, SSMM in
  `tools/eoddata/lse_segments.csv`): no closed-end funds (STMM), ETFs/ETPs or quote-driven SETSqx
  shares. At least 20p, 250 days of history, a bar on 18 of the last 20 sessions, a median of GBP
  250,000 traded a day, no 30% move on the day (a bid, a placing, bad data), a daily volatility of at
  most 4% (`--max-vol`: the most volatile shares' falls took most of a small book); the 300 most traded.
  Delisted shares stay in the backtest (no survivorship bias from today's list).
- **Forecast**: one walk-forward ridge regression across the shares of the next open-to-close (in
  each share's own volatilities) on the signals of `apps/swing.hpp`, the share's volatility, liquidity
  and stamp duty, each ranked across the day's shares; refitted every 21 sessions on the 750 before.
- **Expected return**: the forecast calibrated on the model's own out-of-sample record of the trade as
  placed (the last 500 sessions' top fifth of forecasts, bought at the open and sold at the day's stop,
  take-profit or close: a triple barrier, López de Prado 2018), the slope shrunk by its t statistic.
  No record, or no skill in it, means no orders.
- **Held up to 5 days, as few as it needs (the live setting, `--hold 5`)**: each share is bought in the
  opening auction and sold in the closing auction of the first day it closes at or above its exit level
  (a learnt 0 to 1.5 daily volatilities above the price paid: the level with the best return per day
  held over the last 250 sessions, so short holds win unless a longer one clearly pays), else at the
  close of day 5. The 0.5% stamp duty is paid once a trade, not once a day. The expected return is
  calibrated on the whole trade (before costs); each final order's costs come off it. The order file is a
  multi-day book (`"book": "hold"`): the bridge places it with its hold book (`submit.py --holds`: the buy
  good only through the opening auction, each close a LOC sale at the exit level, a MOC sale on the last
  day), never within the day, and the evening's close and leftover checks leave its positions alone.
  **Updated every day**: each morning every position held is looked at again with the last close's
  forecast: one the model would no longer buy (its expected return as a new trade, before costs; the
  costs paid are spent) is sold in the opening auction; the others keep, their exit level set again from
  the day's. The order file carries the update for every share on the day's list (`"updates"`); the
  bridge's hold book applies it before the open; the backtest makes the same decisions.
  Without `--hold`, the same-day trades below.
- **Bought at the open, never on a dip**: a limit at the last close plus 0.5 of the share's daily
  volatility (it does not chase a big gap up), good only until 2 minutes after the open (IBKR's GTD;
  if refused, a day order the watch cancels then), so it buys in the opening auction or not at all.
  The amount is the slot's capital, **at most 1% of the share's median day's value** (`maxCapital`,
  which the budget and `tools/ibkr/submit.py` respect), at least GBP 500.
- **Stop-loss and take-profit from the fill, moving during the day**: the order file carries them as
  fractions of the price bought at (`stopRel`, `takeRel`) and how the stop moves (`stopMode`): fixed;
  `trail`, IBKR's own trailing stop `trailRel` under the highest price since the fill (moved at IBKR,
  whether or not the bridge is up; a fixed stop where IBKR refuses it); or `breakeven`, moved by the
  bridge's watch to the price paid once the share has traded `breakevenRel` (one volatility) above it
  (from the day's high in IBKR's quote: with delayed data the move comes late, never early). The watch
  (`submit.py --rebuy`) places them once the auction has filled, from the fill's own price, one
  cancelling the other (reduced together on a part fill, placed again when more is bought, never
  twice), so a gap down never leaves the stop above the entry. Whatever is still held is sold in the
  closing auction. The levels are chosen each day from 75 (stop 1 to 4 daily volatilities, fixed,
  trailing or break-even; take-profit 1 to 3 or none) by what each added to the model's top forecasts
  over the last 250 sessions against the fixed 4-volatility stop alone, shrunk by its t statistic:
  levels are tightened, or made to move, only where they have paid.
- **Replayed on 5-minute paths**: the levels are tested on each share-day's 5-minute bars where there
  are any (`--bars`: the store's LSE bars and the bridge's 30-second recordings of each day's
  candidates, `tools/eoddata/intraday_export.mjs --paths UK`; each bar's low taken before its high).
  Without them, on the daily bar both ways (the low first, the high first), the worse taken: a moving
  stop cannot be credited on a daily bar, so on daily bars alone it is seldom chosen. The report says
  how many trades were replayed on paths and each kind of stop's best record.
- **Costs only on the final orders**: the model learns and its backtest trades before the broker's
  costs (the report shows what they would have been, never charges them). The next session's orders
  then have them applied, each on its own amount: IBKR's commission and the 0.5% stamp duty (not AIM
  shares); no spread, as they trade in the opening and closing auctions (one price for all); the expected return after them is what the order file
  carries (`ifFilled`, and `expected` times the chance to buy), and an order whose costs take all of it
  is not placed. Many days have fewer orders than slots, some none.
- **Ideas from the literature**: [docs/uk-research.md](docs/uk-research.md) (what the 2025-2026 reading lists' books suggest, what
  is applied, what is next).
- **Checks**: `tools/check/uk_leak.sh` (CI and the daily run): on a generated market it must find a
  planted effect, learn a take-profit where the day's highs spike and fade and none on a random walk,
  on 5-minute bars a trailing or break-even stop where the day's runs fade and a fixed one on a random walk,
  every order must be sensible (bought at the open only, levels around the entry, an amount within the
  share's liquidity) and unchanged by later data. `tools/ibkr/test/levels.py` (CI) runs the orders
  through `submit.py` against a stand-in IBKR: the auction-only buy, the levels from the fill, a part
  fill, a refused good-till time, the stop cancelling the take-profit, IBKR's trailing stop (and a fixed
  one where it is refused), the break-even move.

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
