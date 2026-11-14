# Ideas from the algorithmic trading literature (for the UK model, `apps/ofm_uk`)

A survey of the books most recommended in 2025-2026 reading lists ([QuantVPS](https://www.quantvps.com/blog/best-algorithmic-trading-books),
[Quant Matter](https://quantmatter.com/top-algo-trading-books/), [Nurp](https://nurp.com/algorithmic-trading-blog/5-best-algorithmic-trading-books-to-read/),
[Pocketful](https://www.pocketful.in/blog/trading/best-algorithmic-trading-books/)) and of the papers behind them, and what each means for
this model. (These lists are largely marketing pages; the books themselves are the established ones.)

| Source | Idea | In the model |
| --- | --- | --- |
| López de Prado, *Advances in Financial Machine Learning* (ch. 3, [triple-barrier labelling](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy/)) | Label a trade by how it actually ends: the take-profit, the stop-loss or the time limit, whichever comes first, not by the plain return to the horizon | **Applied**: the expected return is calibrated on the trade as placed (bought at the open, out at its stop, its take-profit or the close), per stop / take-profit pair |
| López de Prado, meta-labelling | A second model learns only *whether* to act on the first model's signal (bet or no bet), not its direction | Partly: the calibration's shrunk slope and the gate "no skill, no orders" play that role; a full second model is a candidate (below) |
| López de Prado, purged and embargoed cross-validation | Training samples whose outcomes overlap the test period leak it | Already so: every fit uses only targets settled before the day it predicts (one-day horizon, nothing overlaps); `tools/check/uk_leak.sh` checks it |
| Bailey and López de Prado (2012, 2014), probabilistic / deflated Sharpe ratio | A backtest's Sharpe is uncertain, more so with skew, fat tails and many trials | **Applied**: the report prints the probabilistic Sharpe ratio (chance the true Sharpe is above 0) and how many level pairs were tried |
| Chan, *Algorithmic Trading* and *Quantitative Trading* (2nd ed.) | Mean reversion of gaps; stops fit momentum strategies and hurt mean-reverting ones | **Applied as a test**, not an assumption: each stop / take-profit pair is used only where its own record over the last 250 sessions shows it paid; on a random walk the model keeps the wide stop alone |
| Davey, *Building Winning Algorithmic Trading Systems* | Few rules, fixed in advance; walk-forward; paper-trade (incubate) before money | The model's settings are fixed and documented; the previous live strategy runs on as a shadow beside it |
| Narang, *Inside the Black Box* | Separate the alpha, risk, transaction-cost and execution models | **Applied**: the model learns and is backtested before costs; the cost model (commission at the order's own amount, stamp duty, auction spread) is applied to the final orders only; execution in `tools/ibkr/submit.py` |
| Kissell, *The Science of Algorithmic Trading*; Cartea, Jaimungal and Penalva, *Algorithmic and High-Frequency Trading* | Market impact grows with the order's share of the day's volume | **Applied**: each order is at most 1% of the share's median day's value |
| Gao, Han, Li and Zhou (JFE 2018), [market intraday momentum](https://econpapers.repec.org/RePEc:eee:jfinec:v:129:y:2018:i:2:p:394-414) | The market's first half-hour return predicts its last half-hour, more so on volatile, high-volume days (12 of 16 markets, Li, Sakkas and Urquhart) | Candidate: needs the 5-minute bars the store keeps (`tools/eoddata/intraday_export.mjs`): hold the trades to the close only on days the first half-hour was up |
| Lou, Polk and Skouras (JFE 2019), the tug of war | Intraday returns continue, overnight returns reverse | Already among the inputs (`apps/swing.hpp`) |

## Candidates for the next step

1. **Meta-labelling**: a second ridge on the top forecasts that predicts whether the triple-barrier trade ends above 0, from the
   inputs plus the day's market state; act only where it says so.
2. **Intraday momentum exit**: from the 5-minute bars, sell at 08:30 instead of holding to the close when the first half-hour fell.
3. ~~Resolve the stop / take-profit order from 5-minute bars~~: done, with trailing and break-even stops (`--bars`); the more
   LSE paths the store and the bridge's recordings hold, the more of the record is exact rather than the daily bar's worse case.
4. **Deflated Sharpe ratio**: deflate for the 25 level pairs (and the budget's 4 trade counts) with the variance of their Sharpe ratios.

## Research switches from the literature (13 Nov 2026)

The model already used three findings from this literature:
- the overnight versus intraday split of past returns (Lou, Polk and Skouras 2019, "A Tug of War", JFE);
- the opening gap's reversal (Berkman, Koch, Tuttle and Zhang 2012, JFQA);
- short-term losers (Jegadeesh 1990; Lehmann 1990).

Three more are now switches in `apps/ofm_uk`, measured as shadows (`edge`: all three; `resid-target`: the first
and the third) in every market before any goes live:

- **`X_RESID=1`, residual reversal.** Blitz, Huij, Lansdorp and Verbeek (2013, JFM, "Short-term residual
  reversal") and Da, Liu and Schaumburg (2014, Management Science) find that the reversal of returns left after
  the market's (and the industry's) move earns two to four times the plain reversal's risk-adjusted return. It
  stays profitable after costs in large caps, because it no longer carries time-varying factor bets. Here: the
  1, 5 and 20-day losses less beta times the market's. Beta is fitted over 60 sessions; the market is the mean
  return of that day's shares.
- **`X_VOLREGIME=1`, reversal scaled by market volatility.** Nagel (2012, RFS, "Evaporating Liquidity") shows
  that reversal returns are the price of supplying liquidity, and the VIX forecasts them: they are highest when
  volatility is high. Here: the 1 and 5-day loss ranks times the market's 20-session volatility over its median
  of the year before, less 1. The fit learns how much more the losers earn in turbulent markets.
- **`X_TARGET=5`, the fit's target matching the 5-day hold.** Gârleanu and Pedersen (2013, JF, "Dynamic Trading
  with Predictable Returns and Transaction Costs"): with costs, a signal is worth what it earns over the
  horizon it is held, and persistent signals deserve more weight than fast-decaying ones. The target becomes the
  open after the signal to the 5th close, over sigma times the root of 5, instead of the next session's open to
  close. The fit uses only targets known by the day it is made (`tools/check/uk_leak.sh` checks it).

Other findings that shaped the design:
- **Costs:** Frazzini, Israel and Moskowitz (2018) find reversal is the anomaly most constrained by costs; De
  Groot, Huij and Zhou (2012) find most of the cost comes from trading small caps. Hence the costs on the final
  orders only, few large orders, the liquidity floor and the volatility cap.
- **Not adopted:** volatility-managed sizing (Moreira and Muir 2017, JF) would cut exposure exactly when Nagel's
  reversal pays most, so it is left out for a reversal book.

### First results on the store's data (13 Nov 2026; before costs, 5 slots of 5,000, the same model without the switches as `hold5-base`)

| market | hold5-base | edge (all three) | resid-target |
|---|---|---|---|
| UK | +30.7% a year, Sharpe 1.20, worst fall 38% | **+46.8%, 1.50, 30%** | +43.8%, 1.43, 30% |
| EU | **+119.0%, 3.19, 34%** | +54.6%, 1.73, 44% | +55.3%, 1.74, 43% |
| US | **+6.6%, 0.36, 56%** | −1.4%, 0.12, 65% | −1.5%, 0.12, 63% |
| APAC | **+34.3%, 1.24, 45%** | +29.6%, 1.05, 44% | +27.2%, 0.99, 48% |

Year by year, `edge` beats `hold5-base` in the UK in 5 of 7 years (2022: +26% against −21%) and in 2026. The UK's
live strategy now runs with all three switches; `hold5-base` keeps the model without them in view. In Europe,
`hold5-base` leads over the whole period, but only +3% in 2026 against +18% (`edge`) and +29% (`resid-target`):
watched, not switched.

### Every market (13 Nov 2026)

At the owner's request, every market's live strategy now runs with all three switches, as the UK's does. On the
store's data this lowered the whole-period backtests in Europe, the US and Asia-Pacific (see the table above);
they beat the model without the switches in 2026. Each market's `hold5-base` shadow keeps that model in view;
going back is the live strategy's `env` set to `{}` in `config/markets.json`.

## Missing bars rolled forward (14 Nov 2026)

A session with no bar for a share used to break its inputs, and with them the requirement of 18 bars in the last
20 sessions. INDEX's Frankfurt-floor ETFs lost 27,141 share-days that way. Now `apps/ofm_uk --roll 5` (the
default) rolls the share's last close forward over up to 5 missing sessions in a row: open, high, low and close
all equal to it, and no volume. This feeds only the inputs: signals, volatility, the history and gap test, and
the traded value. The trades, the fit's targets and the backtest still use the real bars only, and a share is
listed only on a session with a real bar of its own. `tools/check/uk_leak.sh`, with one bar in seven missing:
2,360 orders with the roll against 145 without, none on a rolled bar, none changed by later data. The shadow
`noroll` (`--roll 0`) keeps the model without it in view.

## Learning that does not collapse (14 Nov 2026)

The US backtest lost 49% in 2022. The model then stood aside for about 290 sessions, until its 500-session
calibration record showed an edge again. One window of 750 sessions learns a break slowly, and a book sized the
same in calm and turbulent markets rides a bad run to its end. Two switches address this, measured as the shadows
`adapt`, `risk` and `adapt-risk` beside each market's live model:

- **`X_ADAPT=1`, forecast combination.** Following Timmermann (2006, Handbook of Economic Forecasting) and
  Rapach, Strauss and Zhou (2010, RFS), three fits over the last 250, 750 and 1,500 sessions are combined each
  day by their recent skill. Skill is each fit's daily rank correlation with the outcome, exponentially weighted
  with a half-life of 60 sessions, from outcomes known by then; only positive weights count, else they're equal.
  After a break, the short window's fit takes over within weeks. The calibration record that decides whether to
  trade is 250 sessions instead of 500, so the book is back in sooner.
- **`X_RISK=1`, the book's own volatility target.** Following Moreira and Muir (2017, JF) and Barroso and
  Santa-Clara (2015, JFE, on momentum's crashes), each day's new orders are scaled by 1% over the book's daily
  volatility over its last 60 sessions, between a quarter and the whole amount. A run of large losses shrinks
  the bets until the book calms down. The costs of the final orders are worked out before the scaling, so a
  scaled order's minimum commission is slightly understated.

`tools/check/uk_leak.sh` checks both: no order changed by later data.

### Results (14 Nov 2026, the store's data, before costs, 5 slots of 5,000, on top of the live switches)

| market | live | `adapt` | `risk` | `adapt-risk` |
|---|---|---|---|---|
| UK | +43.6% a year, Sharpe 1.44, worst fall 33.5% | +42.6%, 1.44, 37.2% | +32.4%, 1.41, **25.7%** | +32.1%, 1.43, 29.0% |
| EU | +49.8%, 1.65, 39.6% | +35.3%, 1.31, 48.1% | +32.1%, 1.43, **34.5%** | +25.1%, 1.20, 39.0% |
| US | −1.6%, 0.12, 65.4% | −4.7%, 0.00, 68.9% | −0.3%, 0.10, **49.0%** | −1.7%, 0.03, 53.5% |
| APAC | +26.1%, 0.96, 43.9% | +25.0%, 0.92, 43.3% | +27.8%, **1.25**, **29.3%** | +18.6%, 0.91, 32.5% |
| INDEX | +45.5%, 2.39, 15.5% | +44.1%, 2.32, 15.8% | +42.5%, 2.37, 15.4% | +40.9%, 2.30, 15.7% |

The live column is the budget's sizing; the others are 5 slots of 5,000.

- **`X_ADAPT` doesn't help.** The short window's fit is noisier than the break it was meant to catch: Europe's 2022
  went from +45% to −18%. It stays a shadow.
- **`X_RISK` cuts every market's worst fall,** by up to 16 points (the US 65% → 49%, APAC 44% → 29%), and halves
  APAC's 2026 loss (−37% → −17%), for a lower return in the calm years. It is now live in every market. The
  shadow `norisk` keeps the model without it in view.

## Costs: on the next day's final orders only (14 Nov 2026)

At the owner's instruction: no cost enters the learning, the backtests or the split of the account's money over
the markets. The budget's record of each market is now its backtested daily results before costs. Before, it
subtracted the trade log's broker costs, `costs_of`, now removed. The costs come off the next day's final order
list only (`apps/ofm_uk`, `ordersAt(T - 1, true)`): each order's expected return after its own commission, stamp
duty and taxes, and an order expected to lose after them is not made.

## Buying more surely (14 Nov 2026)

An order that does not buy is a lost opportunity. The live buy is good only in the opening auction, at most the
last close plus 0.5 daily volatilities (`--cap-sd 0.5`). Two experiments, as shadows in every market:

- **Fixed caps.** `cap1`, `cap2` and `moo` set the cap at 1, 2 and 8 volatilities. At 8 the order in effect buys
  at the auction's price.
- **`X_CAPLEARN=1`, each order's own cap, learnt.** Two things are learnt walk-forward:
  - What a gap costs or adds: for each size of opening gap over the close (bands from below −1 to above 3
    daily volatilities), the market's shares' outcome against the day's mean, over the last 750 sessions whose
    outcome is known.
  - The share's chance to open in each band, from its own gaps over its last 250 sessions.

  The cap is the band edge that makes the most of the order: its chance to buy in each band, times its expected
  return given that gap, added up. A strong pick takes a higher cap and buys more surely; a weak one keeps a tight
  one. The order's chance to fill and its expected return if filled follow from the cap, and so do its costs and
  its rank. The calibration stays on `--cap-sd`. `uk_leak.sh`: no order changed by later data.

### Results (14 Nov 2026, the store's data, before costs)

| market | live (bought, Sharpe, worst fall) | `cap1` | `cap2` | `moo` | `caplearn` |
|---|---|---|---|---|---|
| UK | 86%, 1.52, 26.8% | 96%, 0.84, 32.7% | 99%, 0.42, 40.7% | 100%, 0.46, 34.9% | 67%, **3.54**, **16.5%** |
| EU | 89%, 1.51, 34.1% | 95%, 1.51, 37.8% | 97%, 1.45, 39.2% | 98%, 1.37, 40.7% | 77%, **1.89**, **25.6%** |
| US | 87%, 0.08, 45.1% | 97%, 0.22, 39.5% | 99%, 0.41, 40.6% | 100%, 0.29, 41.0% | 96%, **0.74**, 44.1% |
| APAC | 85%, 0.78, 36.5% | 95%, 0.99, 39.1% | 98%, 1.10, 33.1% | 99%, 1.15, 28.6% | 79%, 1.08, **22.9%** |
| INDEX | 70%, **2.37**, 15.4% | 83%, 2.03, 14.1% | 89%, 1.88, 16.2% | 89%, 1.85, 19.2% | 78%, 2.23, 14.8% |

The live column is the budget's sizing; the rest are 5 slots of 5,000.

- **Buying more often is not better on its own.** In the UK and Europe a share that opens well above its close
  tends to give it back over the hold, so a higher fixed cap adds losing trades.
- **The learnt cap separates the two cases per order.** It tightens where gap-ups lose (UK 67% bought, Sharpe
  1.52 → 3.54) and opens up where they don't (the US 96% bought, Sharpe 0.08 → 0.74).
- **Now live** in the UK, Europe, the US and Asia-Pacific; INDEX keeps the close plus 0.5. Shadow `nocaplearn`
  keeps the model without it in view.

**A ceiling on the learnt cap (14 Nov 2026).** The first live scan let US orders pay up to 31% over the close
(8 volatilities of a volatile share). In the US, gap-ups had not hurt on average, but a gap that large is news (an
earnings shock, a bid), and the record of such gaps is too thin to trust. The learnt cap is now at most 2
volatilities and at most 3% over the close (`--cap-max 0.03`). Each gap band's estimate is shrunk towards 0 by
its number of trades (n / (n + 500)), so a band with few trades cannot pull the cap up. `uk_leak.sh` checks the
ceiling.
