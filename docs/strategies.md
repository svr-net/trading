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
