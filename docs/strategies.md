# Strategies and back ends

The strategy engine (`apps/ofm_strategy`) runs a market's strategy on its daily bars. Strategies and compute back
ends are injected: each is one source file that registers itself, and CMake builds every file in `strategies/`
and `backends/` into the engine. The strategy engine replaced every earlier model (the auction model `ofm_uk`,
the research model `ofm_research` and the single-share models) on 15 Nov 2026.

## Adding a strategy

Create `strategies/<name>.cpp`:

```cpp
#include "strategy/strategy.hpp"

namespace {
class MyStrategy : public strat::Strategy {
 public:
  std::string about() const override { return "what it does (the evidence)"; }
  strat::Params defaults() const override {
    // read by the engine: rebalance, warmup, capsd, capmax, maxdays; add your own
    return {{"lookback", 126}, {"rebalance", 21}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 260}};
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    out.assign(v.N(), strat::kNaN);  // NaN: not wanted
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {  // the chosen back end's loop
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) out[i] = v.change(v.t() - std::size_t(p.at("lookback")), v.t(), i);
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override { return "why, in a sentence"; }
};
}  // namespace

REGISTER_STRATEGY("mystrategy", MyStrategy);
```

Rebuild, then check it is listed (`ofm_strategy --list`) and passes `sh tools/check/strategy_check.sh
build/ofm_strategy`. Choose it for a market in `config/markets.json`:
`"live": {"name": "mine", "strategy": "mystrategy", "params": {"lookback": 63}}`.

What a strategy can read is the `View`: the closes (carried over a few missing bars), opens, log returns,
volatilities, traded values, the market's daily return and the day's tradable shares. It covers every day up to
the close it decides on, and a read of a later day throws, so a strategy cannot look ahead.

## Adding a back end

Create `backends/<name>.cpp` with a class derived from `strat::Backend` (`include/strategy/backend.hpp`). It
implements `parallelFor(n, f)`, which runs `f(begin, end)` over chunks covering `[0, n)`, and registers with
`REGISTER_BACKEND("name", Class)`. Every kernel chunk writes only its own indices, so any split gives the same
numbers. The checks run every strategy on every back end and require identical output. Choose a back end per
market with `"backend": "name"` in `config/markets.json`, or `--backend name`.

## Costs

Costs never enter a strategy, its backtest or the split of the money. The final order list is sorted by
expected return before costs, highest first. Then each order's own IBKR costs come off:
- commission both ways;
- the US regulators' fees on the sale;
- UK stamp duty (none on AIM shares), or the exchange's tax (`tools/venues.txt`).

An order its costs would make a loss is dropped. No other order takes its place, and the order of the list
does not change.

## The daily strategies (26 Nov 2026)

The monthly and weekly strategies (momentum, trend, low volatility, weekly reversal, residual momentum, 52-week
high, no lottery shares and long-term reversal) were removed and replaced by seven daily ones from the published
research on short-horizon returns. Each strategy decides after every close (`rebalance: 1`). A share stops being
wanted when its signal ends, and the engine then sells it at the next open. `maxdays` is the time stop.

| Strategy | Rule | Evidence |
|---|---|---|
| `rsi2` | close over its 200-day average and RSI(2) under 10; held until a close over the 5-day average | Connors and Alvarez 2009 |
| `ibs` | close in the bottom 20% of the day's range (internal bar strength); held a day | Pagonidis 2013; Pandey and Joshi 2023 (arXiv 2306.12434) |
| `revday` | the day's largest falls net of beta times the market, at least 1.5 volatilities; held a day | Lehmann 1990; Lo and MacKinlay 1990; Nagel 2012 |
| `volrev` | 3-day falls against the market on at least 1.5 times the usual volume | Conrad, Hameed and Niden 1994; Avramov, Chordia and Goyal 2006 |
| `hvp` | a day of at least 2.5 times the median volume of the 49 sessions before, that did not close lower; held up to 5 sessions | Gervais, Kaniel and Mingelgrin 2001; Kaniel, Ozoguz and Starks 2012 |
| `tom` | from the month's last session to its third, the highest betas over their 200-day average; read from the calendar, never a later day | Lakonishok and Smidt 1988; McConnell and Xu 2008 |
| `breakout` | a new 20-day closing high within 5 sessions, over the 200-day average and the 10-day low | Donchian; Faith 2007; George and Hwang 2004 |

The View gained `high`, `low` and `volume`, and the shared helpers live in `include/strategy/daily.hpp`.

**Backtests on the store's bars** (`markets/<MARKET>/`, Nov 2017 to 25 Nov 2026, a year of warm-up), 10 equal
positions, no Monte Carlo and no stop, **before any cost**. IBKR's tiered costs come off only the final order list
(the last column). "Opposite picks" holds the lowest scores of the same list. For a filter strategy, that means the
weakest signals among the shares that qualify, so "own edge" there measures the ranking, not the filter.

| Market | Strategy | Annual | Sharpe | Max DD | Opposite picks | Own edge | Trades/yr | Avg trade | Win | Held | Next orders: list -> after costs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UK | breakout | +10.7% | 0.69 | 32% | +7.8% | +2.9% | 662 | +0.18% | 45% | 4.0 | 10 -> 0 |
| UK | hvp | +21.9% | 1.18 | 42% | +11.6% | +10.3% | 746 | +0.31% | 51% | 3.6 | 10 -> 0 |
| UK | ibs | -3.9% | -0.13 | 55% | +26.3% | -30.2% | 2415 | -0.01% | 48% | 1.1 | 10 -> 0 |
| UK | revday | +11.0% | 0.65 | 42% | +12.8% | -1.8% | 1948 | +0.07% | 50% | 1.1 | 9 -> 0 |
| UK | rsi2 | +8.3% | 0.57 | 34% | +6.3% | +2.0% | 916 | +0.11% | 54% | 2.6 | 10 -> 0 |
| UK | tom | +6.0% | 0.57 | 14% | +2.6% | +3.4% | 176 | +0.39% | 51% | 2.9 | 0 -> 0 |
| UK | volrev | +9.7% | 0.59 | 44% | +9.1% | +0.6% | 1197 | +0.10% | 52% | 1.7 | 10 -> 1 |
| EU | breakout | +25.5% | 1.54 | 18% | +9.2% | +16.3% | 870 | +0.34% | 42% | 3.6 | 8 -> 1 |
| EU | hvp | +11.9% | 0.69 | 37% | +12.8% | -0.9% | 960 | +0.16% | 44% | 3.4 | 8 -> 0 |
| EU | ibs | -6.0% | -0.27 | 49% | -12.8% | +6.8% | 2775 | -0.03% | 40% | 1.1 | 10 -> 0 |
| EU | revday | -6.0% | -0.31 | 52% | -7.4% | +1.4% | 2251 | -0.04% | 41% | 1.1 | 4 -> 0 |
| EU | rsi2 | +0.7% | 0.12 | 34% | +0.8% | -0.1% | 1111 | +0.01% | 50% | 2.5 | 10 -> 0 |
| EU | tom | +7.8% | 0.76 | 23% | +0.1% | +7.7% | 177 | +0.57% | 53% | 3.1 | 0 -> 0 |
| EU | volrev | -1.5% | -0.01 | 39% | +2.6% | -4.1% | 1218 | -0.01% | 46% | 1.9 | 5 -> 0 |
| US | breakout | -1.8% | 0.05 | 44% | -3.5% | +1.7% | 753 | +0.01% | 43% | 3.5 | 10 -> 2 |
| US | hvp | +0.1% | 0.16 | 57% | +9.7% | -9.6% | 747 | +0.07% | 51% | 3.6 | 10 -> 6 |
| US | ibs | +6.5% | 0.36 | 48% | +2.3% | +4.2% | 2452 | +0.04% | 51% | 1.0 | 10 -> 6 |
| US | revday | +7.9% | 0.41 | 40% | +20.9% | -13.0% | 2350 | +0.05% | 51% | 1.0 | 10 -> 8 |
| US | rsi2 | +12.2% | 0.62 | 33% | +10.0% | +2.2% | 1062 | +0.14% | 57% | 2.4 | 10 -> 8 |
| US | tom | +4.2% | 0.30 | 34% | +0.8% | +3.4% | 182 | +0.41% | 50% | 2.9 | 0 -> 0 |
| US | volrev | +14.1% | 0.58 | 54% | +15.0% | -0.9% | 1308 | +0.14% | 53% | 1.8 | 10 -> 8 |
| APAC | breakout | +0.9% | 0.14 | 49% | +10.6% | -9.7% | 719 | +0.02% | 39% | 3.7 | 10 -> 2 |
| APAC | hvp | +18.0% | 0.87 | 37% | -3.1% | +21.1% | 750 | +0.27% | 48% | 3.6 | 10 -> 6 |
| APAC | ibs | +21.8% | 1.18 | 36% | +8.8% | +13.0% | 2495 | +0.09% | 46% | 1.0 | 10 -> 0 |
| APAC | revday | +5.3% | 0.36 | 40% | +12.5% | -7.2% | 2209 | +0.03% | 47% | 1.1 | 10 -> 0 |
| APAC | rsi2 | +14.8% | 0.80 | 39% | +11.9% | +2.9% | 980 | +0.17% | 54% | 2.5 | 10 -> 2 |
| APAC | tom | +5.8% | 0.41 | 25% | +0.3% | +5.5% | 178 | +0.43% | 47% | 3.0 | 0 -> 0 |
| APAC | volrev | +5.1% | 0.34 | 52% | +4.3% | +0.8% | 1280 | +0.06% | 49% | 1.7 | 10 -> 0 |

**What it shows:**
- **No strategy works everywhere.** The best per market: UK `hvp` (Sharpe 1.18), EU `breakout` (1.54), US `rsi2`
  (0.62) and APAC `ibs` (1.18). Each one is weak or negative somewhere else. For example, `ibs` returns −4% in the UK
  and −6% in the EU, and `breakout` returns −2% in the US.
- **Small trades.** A trade earns +0.01% to +0.57% before costs. That is less than a UK round trip (stamp duty: about
  0.6%) and less than most EU ones (France's tax). So the final cost step drops nearly every UK and EU order. In the
  US and APAC, where costs are a few basis points, most orders survive.
- **Turn of the month.** It is the only strategy whose average trade beats every market's costs (+0.4% to +0.6%).
  It is invested only about 4 sessions a month, which explains its low drawdowns.
- **Weaker than the old strategies before costs.** The removed monthly momentum made +34% a year in the UK and US
  (5 positions, store data to 13 Nov 2026, below), against +22% and +12% here for the best daily strategy.

**The live choice:** one rule for every market, the highest Sharpe among the strategies whose picks beat their own
opposite picks: UK `hvp`, EU `breakout`, US `rsi2`, APAC `ibs`. INDEX takes `tom` (the effect is a market-index
one, McConnell and Xu's 31 of 35 countries). FOREX takes `rsi2` (research only; currency bars carry no volume, so
`hvp` and `volrev` cannot run). Picking the best of seven on the same data overstates what to expect: the
research figures and the opposite-picks gap are the better guide.

## The removed monthly and weekly strategies

The sections below record the strategies replaced on 26 Nov 2026.

### The first backtests (store data to 13 Nov 2026)

Each strategy on each market, with 5 equal positions, before costs: annual return, Sharpe ratio and the worst
drawdown, from Nov 2018 (a year of warm-up) to 13 Nov 2026. The US figures are for NYSE only, because NASDAQ
couldn't be exported from the store's cache.

| Market | momentum | trend | lowvol | reversal |
|---|---|---|---|---|
| UK | +34%, 1.08, 44% | +13%, 0.60, 41% | +2%, 0.20, 34% | +21%, 0.89, 37% |
| EU | +13%, 0.57, 47% | +6%, 0.37, 45% | +1%, 0.13, 27% | +6%, 0.41, 38% |
| US | +34%, 0.85, 58% | +26%, 0.80, 49% | −6%, −0.38, 46% | +16%, 0.63, 44% |
| APAC | +31%, 0.82, 61% | +52%, 1.25, 42% | −1%, −0.23, 17% | +18%, 0.71, 50% |
| INDEX | +11%, 0.66, 30% | +11%, 0.70, 30% | +10%, 0.72, 30% | +15%, 0.87, 35% |
| FOREX | −1%, −0.08, 23% | −1%, −0.18, 20% | +0%, 0.42, 3% | +2%, 0.48, 14% |

The live choice follows the research, not this table: momentum for shares, trend following for INDEX and FOREX.
Picking each market's best strategy from the same data it is judged on is the overfitting the research warns
against. These are 5-position books, so the drawdowns are large; the budget's larger books (10 positions)
spread the risk.

## Stops and buy-backs (16 Nov 2026)

The engine adds a stop to any strategy, along with a rule for buying back a share after it has been stopped out:

- **`stop` with `rebuy`:** a fixed stop-loss, this fraction under the price paid. After a stop, the share is bought
  back once its close is `rebuy` above the stop level; `rebuy: -1` means never.
- **`psar: 1` with `rebuy`:** a Parabolic SAR trailing stop (Wilder 1978).
  - It starts at the lowest low of the 5 sessions before the buy.
  - Each day it rises by the step (0.02, growing by 0.02 at each new high, to at most 0.2) times the gap to the
    highest high since the buy.
  - It never rises above the last two lows.
  - After a stop, the share's SAR on the way down is followed. The share is bought back once a close is above it,
    while the strategy still wants the share (`rebuy` ≥ 0).

A stop is hit when the day's low reaches it. The position is sold at the stop, or at the open if the share
opened under it. At IBKR it is a good-till-cancelled stop sell (`tools/ibkr/submit.py --holds`):
- placed once the buy fills;
- moved each morning to the engine's level for the day (`stopPrice` in the orders file);
- cancelled before the book sells the position any other way.

Tested on the store's data (Nov 2018 to 13 Nov 2026), before costs. The cost estimate is the extra buys a year
times each order's round trip at the 15 Nov order sizes:

| Market | No stop | PSAR | PSAR with buy-back | PSAR with buy-back, about after costs |
|---|---|---|---|---|
| UK | +34%, 1.08, 44% | +6%, 0.75, 16% | +19%, 1.02, 28% | +7% (no stop: +30%) |
| EU | +15%, 0.75, 36% | −1%, −0.07, 14% | +11%, 0.93, 20% | −13% (no stop: +7%) |
| US (NYSE) | +28%, 0.79, 56% | −1%, −0.04, 25% | −5%, −0.11, 45% | −10% (no stop: +26%) |
| APAC | +39%, 1.06, 56% | +6%, 0.71, 15% | +25%, 1.23, 25% | +7% (no stop: +34%) |
| INDEX | +11%, 0.70, 30% | +4%, 0.53, 15% | +6%, 0.65, 14% | 0% (no stop: +6%) |

Fixed stops (10%, 15%, 20%) did worse than PSAR with buy-back, with or without a buy-back.

The PSAR stop with buy-back went live in every market on 16 Nov 2026, chosen to roughly halve the worst falls.
It trades about three times as often, so the costs take most of its return before costs. The paper account's
fills will show the real figure.

## The nested Monte Carlo (16 Nov 2026)

`mc: 1` (`include/strategy/montecarlo.hpp`) models with a nested Monte Carlo:

- **Outer loop, resets on the real market:** at every rebalance in the history, the simulation is reset to the
  real daily OHLCV up to that close. That means the real prices, the real tradable shares and the strategy's real
  list.
- **Inner loop:** `mcpaths` futures (500) of `mchorizon` sessions (default: the rebalance period). Each future is
  built from blocks of `mcblock` (5) whole real days, drawn from the `mcwindow` (750) sessions before the reset,
  never later.
  - Every share takes its own bar of each drawn day (open, high, low and close relative to its close before), so
    a market-wide fall hits every share on the same day, and runs of volatile days are kept.
  - With `mcdrift: 1`, the default, each share's own mean daily move over the window is replaced by the market's.
    The futures then carry the share's risk without assuming its past trend continues.
  - The book runs on every future: the buy under its cap, the stop (fixed or PSAR) on the day's low, and the
    buy-back.
- **Real check:** the one real future after each reset goes through the same simulator, which shows whether the
  simulation is honest.
- **Orders:** each order's expected return before costs is its mean over the futures from the last close; costs
  still come off at the end. The page also shows each order's 5% and 95% outcomes and its chances of a loss and
  of a stop, plus the book's.

**Fused kernel:** each future's draw and every share's trade on it are computed in one pass, run in parallel over
the futures by the back end. The random draws are counter-based (a hash of the seed, the reset and the path), so
every back end gives the same numbers, and the check enforces it.

**First run on the store's data** (live settings, PSAR with buy-back). "Inside the band" is the share of resets
where the real period fell inside the simulated 5–95% band; an honest simulation scores 90%. The average
percentile should be 50.

| Market | Inside the band | Average percentile | Simulated mean against real, a period |
|---|---|---|---|
| UK | 82% of 98 | 56 | +0.3% against +1.3% |
| EU | 86% of 116 | 51 | +0.5% against +0.7% |
| US | 89% of 97 | 48 | +1.0% against +0.1% |
| APAC | 84% of 99 | 55 | +0.7% against +2.5% |
| INDEX | 92% of 401 | 49 | +0.2% against +0.1% |

**What the first run shows:**
- **Ranges:** close to honest. The bands are slightly narrow in the UK, EU and APAC, so real crashes are a little
  under-represented.
- **No forecasting skill:** across resets, the simulated mean doesn't predict the real one (correlation about 0).
  The simulation measures risk; it doesn't forecast.
- **Own-trend version:** keeping each share's own trend (`mcdrift: 0`) made the means too optimistic, for example
  US +2.5% a period against the real +0.05%.
- **Stops:** the PSAR stop is hit in 83–98% of the futures of the momentum books' shares within a month, against
  about 9% for the INDEX ETFs.

## The PSAR stop switched off (16 Nov 2026)

With the PSAR stop and buy-back live, the first scans published almost no orders: UK 0 of 5, US 0 of 10, EU, APAC
and INDEX one each. The stop sells shares on ordinary dips and buys them back higher. Before any cost, that cut
the backtests (UK +34% to +19% a year, US +28% to −5%, APAC +39% to +25%). It also left each trade expecting so
little that the final cost step dropped most orders. Costs were applied only at the end, as always. The stop is
now off in every market (`params: {mc: 1}`), and the Monte Carlo still gives each order's expected return and
range. The stop remains in the engine and the bridge, to switch on per market if wanted.

## Review: what was wrong, and what was fixed (16 Nov 2026)

1. **The PSAR stop:** a wrong tool, not a calculation error. I logged all 444 stops in the UK backtest.
   - Every stop was a real touch of the day's low.
   - But the PSAR sat a median 0.78 daily volatilities under the close (a quarter within 0.43). A share's ordinary
     daily low reaches that far, so it sold on normal noise, after a median of 8 days.
   - **Fixed:** switched off. A stop meant to survive normal moves needs about 2–3 daily volatilities.
2. **Bad prices:** 0.01–0.08% of bars had a low or high that is impossible for the day's open and close (for
   example AstraZeneca's low at 109p on a day it traded at 7,372p).
   - **Fixed:** such a bar is dropped, and the last real close is rolled forward over it, for the signals, the
     marks, the stops and the simulation's days. The rule looks at the day's own bar only, never a later one.
   - The exporter no longer drops a share for a one-day move of over 60% that comes back within 5 days. Its bad
     bar is rolled over instead, so real crashes stay in the data.
3. **Share lists chosen with hindsight:** each market's list was today's most traded shares, applied back to 2018.
   - **Fixed as far as the data allows:** the scan now takes every share the store keeps, and the engine picks
     each day's most traded as of that day (UK 300, US 1,000, EU 500, APAC 400).
   - **What remains:** the store was built from shares listed today, so most that collapsed or were delisted are
     missing. The backtests therefore still overstate.
   - The scan now measures this every run. It also runs the strategy's opposite picks (`invert: 1`) on the same
     list and book. The gap is the strategy's own edge; what the opposite picks also make comes from the list.
     On the store's data:

| Market | Momentum | Opposite picks | Own edge, a year |
|---|---|---|---|
| UK | +30% | +26% | about +4% |
| EU | +10% | +4% | about +6% |
| US | +33% | +16% | about +17% |
| APAC | +40% | −3% | about +43% |

   These are 5–10 position books, before costs. In the UK the opposite picks earn nearly as much as momentum, so
   the edge, not the backtest, is the figure to expect. A full fix needs share lists that include delisted shares.
4. **Expected returns:** the Monte Carlo's ranges are close to honest, but its means don't predict (correlation
   about 0 across resets). The order list is still sorted by expected return, as instructed, so the order within
   the list carries little information.
5. **Costs:** they come off only the final order list. Nothing else in the engine uses them.
