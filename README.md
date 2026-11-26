# Daily Trades

Research-backed trading strategies for the UK, Europe, the US, Asia-Pacific, Xetra index ETFs (INDEX) and
currencies (FOREX, research only), run every evening by a C++ strategy engine. The strategies and the compute
back ends are injectable. A page shows the next session's orders and each strategy's backtest, and a bridge on
the NAS places the orders at Interactive Brokers.

## The strategy engine (`apps/ofm_strategy`)

- **Strategies (`strategies/*.cpp`):** each one is its own file. It defines a class derived from
  `strat::Strategy` (`include/strategy/strategy.hpp`) and registers it with `REGISTER_STRATEGY("name", Class)`.
  CMake builds every file in the folder into the engine, and `config/markets.json` chooses a market's strategy
  by name, with optional parameter overrides. See [docs/strategies.md](docs/strategies.md).
- **Back ends (`backends/*.cpp`):** they decide how the fused kernels and the strategies' per-share loops run.
  `serial` uses one thread and is the WebAssembly build's back end; `threads` uses every CPU thread and is the
  default. A new one registers with `REGISTER_BACKEND("name", Class)`. All back ends give identical results,
  and the checks enforce that.
- **Fused kernels (`include/strategy/kernels.hpp`):**
  - one sweep per share computes the carried-forward close, the log return, the 60-session volatility, the
    20-bar median traded value and the first bar;
  - one pass per day computes the tradable shares and the market's return.
- **No look-ahead by construction:** a strategy only sees the market through a `View` that ends at the close
  it decides on. Reading a later day throws.

The engine does the same for every strategy:

1. **Universe:** a real bar, a minimum price and traded value, a year of history, and a volatility.
2. **Book:** a few equal positions. Every rebalance it holds the strategy's highest scores.
3. **Selling:** a share the strategy drops is sold at the next open.
4. **Buying:** a target share is bought in the opening auction under a price cap, and tried again the next day
   if it doesn't fill.
5. **Backtest:** walk-forward and **before any cost**.
6. **Expected return:** each order's comes from the strategy's own past trades at the same rank.
7. **Costs at the end:** the broker's costs are applied only to the final order list, which is sorted by
   expected return, highest first. An order its costs would make a loss is dropped.

| Strategy | What it does (decided after every close, bought at the next open) | Evidence |
|---|---|---|
| `rsi2` | RSI(2) under 10 while over the 200-day average; held until a close over the 5-day average | Connors and Alvarez 2009 |
| `ibs` | a close in the bottom 20% of the day's range; held a day | Pagonidis 2013; Pandey and Joshi 2023 |
| `revday` | the day's largest falls against the market, in volatilities; held a day | Lehmann 1990; Lo and MacKinlay 1990; Nagel 2012 |
| `volrev` | 3-day falls against the market on at least 1.5 times the usual volume; held days | Conrad, Hameed and Niden 1994; Avramov, Chordia and Goyal 2006 |
| `hvp` | a day of at least 2.5 times the usual volume that did not close lower; held up to a week | Gervais, Kaniel and Mingelgrin 2001; Kaniel, Ozoguz and Starks 2012 |
| `tom` | held from the month's last session to its third, the highest betas over their 200-day average | Lakonishok and Smidt 1988; McConnell and Xu 2008 |
| `breakout` | a new 20-day closing high over the 200-day average; out under the 10-day low | Donchian; Faith 2007; George and Hwang 2004 |

The monthly and weekly strategies (momentum, trend, low volatility, weekly reversal and the rest) were replaced by
these daily ones on 26 Nov 2026 (docs/strategies.md).

The engine's command:

```
ofm_strategy --csv stocks.csv --costs UK --strategy rsi2 [--param name=value] [--backend threads]
             [--capital 5000] [--slots 5] [--stamp-free aim.txt] [--venues tools/venues.txt] [--override ibkr.csv]
             [--json page.json] [--orders orders.json] [--trace orders.trace]
ofm_strategy --list
```

Checks (CI, and every scan before it publishes): `tools/check/strategy_check.sh` runs every strategy on a
generated market. It confirms that:
- a run without the last 120 days makes the same orders on every day both runs have;
- every back end gives the same results;
- the final orders are sorted by expected return, with their costs.

`tools/check/wasm_same.mjs` checks that the WebAssembly build gives the native results.

## The nested Monte Carlo and its neural overlay

Every market's book is chosen by the nested Monte Carlo (`include/strategy/montecarlo.hpp`, `neural.hpp`):

- **Resets on the real market:** at every rebalance the simulation restarts from the real daily bars up to that
  close. 500 futures of 21 sessions are built from blocks of 5 whole real days drawn from the 750 sessions before,
  so the days the whole market moved together are kept.
- **Differential neural network:** for every share at every reset, it shapes those futures with a daily drift and a
  volatility scale, learnt from point-in-time features. It is trained walk-forward, only on resets whose outcome was
  known by then, by the CRPS of the simulated futures against the real return, with exact gradients through the
  paths. It learns every week (`mcnnstep=5`) from the bars realised since, its training cases still taken at the
  resets: learning only at the monthly resets left it up to a month behind; taking a day's cases every day crowded
  its window with the last months (both tested walk-forward, 17 Nov).
- **Every strategy injected:** the seven daily strategies (RSI(2), internal bar strength, daily reversal, volume
  reversal, the high-volume premium, the turn of the month and the breakout) each score every share at every reset. Their
  ranks are inputs to the network, so it learns which strategy to trust for which share, in which market.
- **The choice:** each strategy proposes its best 2 a slot, and the simulation keeps the k with the highest expected
  return. The same futures give each order's expected return, its 5-95% band and its chance of a loss. The real
  future after every reset is checked against the simulated ones (the calibration).
- **Costs:** none while modelling. The broker's costs come off the final order list only.

## Data

EODData's daily bars (key in the `EODDATA_API_KEY` secret) are kept in the private repository `svr-net/eod`
(`EOD_TOKEN`). Every run brings its clone up to date and downloads only what is missing, and pushes what it added,
through libgit2 and libcurl (`apps/ofm_eod`: `sync`, `shares`, `series`, `market`, `profiles`, `aim`, `symbols`,
`snapshot`, `push`). IBKR's own daily bars replace the store's where it has them. EODData's licence does not allow redistributing prices, so only
results are published.

## Schedule

`.github/workflows/uk-portfolio.yml` runs on weekday evenings. It updates the data, runs each market's strategy
(`apps/ofm_scan run`, which also sizes the book and splits the account's money over the markets),
publishes the results to the page (`ofm_scan track`, `index`), writes the orders to the private store for the
bridge and emails the report (`ofm_scan mail`).

## Native tools (Docker, Alpine)

The engine, the loaders and the scan are C++ only. `docker/native/Dockerfile` builds them on Alpine in cached
layers (the toolchain once; a code change rebuilds only the sources' layer) and makes the runtime image
`ofm-tools` only when every strategy's checks and the scan's (`ofm_scan test`) pass. The workflows build it with
GitHub's layer cache (`.github/actions/ofm-tools`) and run each tool in it with `tools/ofm.sh`.

Locally: `docker buildx build -f docker/native/Dockerfile --target tools -t ofm-tools --load .`, then
`sh tools/ofm.sh ofm_scan test`.

## The bridge (`tools/ibkr`)

The bridge runs on the NAS. It places each market's book at IBKR: buys in the opening auction, and each
morning sells at the open what the strategy dropped (the orders file's `updates`). A position with no exit
level (`exitAbove: null`) is held until the strategy drops it. See [docs/ibkr.md](docs/ibkr.md).

## The Lab (WebAssembly)

`docker/wasm/Dockerfile` builds the engine with every strategy and the serial back end to WebAssembly, in
cached layers. The pinned Emscripten toolchain is built once; the engine's sources are the only layer a
strategy change rebuilds. It is built with `-O3 -flto -msimd128`. The page's **Lab** runs it in a Web Worker on
the bars from the bridge (the prices stay on the device), with a choice of strategy.

Locally: `docker buildx build -f docker/wasm/Dockerfile --target out --output type=local,dest=web/lab .`

Research only, not investment advice.
