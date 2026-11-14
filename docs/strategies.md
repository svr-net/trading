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

## The first backtests (store data to 13 Nov 2026)

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
