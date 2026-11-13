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
