# trading — self-adaptive machine-learning trading library

[![CI](https://github.com/svr-net/trading/actions/workflows/ci.yml/badge.svg)](https://github.com/svr-net/trading/actions/workflows/ci.yml)
[![Pages](https://github.com/svr-net/trading/actions/workflows/pages.yml/badge.svg)](https://github.com/svr-net/trading/actions/workflows/pages.yml)

**Live demo: [svr-net.github.io/trading](https://svr-net.github.io/trading/)**: the whole method running in your browser (WebAssembly, and WebGPU where available).

A C++17 library that models the method of

> Y. Wang, P. Huang, J. Luo, *Predicting Stock Prices Based on Machine Learning to Build Self-adaptive Trading Strategy*,
> Computational Economics 68, 523–547 (2026), published online 18 August 2025. [doi:10.1007/s10614-025-11054-4](https://doi.org/10.1007/s10614-025-11054-4)

Earlier studies that trade on machine-learning forecasts of stock prices commit to one fixed trading rule. As the market
changes, a fixed rule loses its edge. The paper builds a *self-adaptive* strategy instead: machine-learning models forecast
each stock's next move from formulaic alpha factors, the forecasts drive a pool of trading rules, and the strategy keeps
moving to whichever rule currently works. It tests the idea on Hong Kong stocks.
This library is an independent implementation of that pipeline. It does not reproduce the text of the paper.

**Sources.** The article's full text is behind a paywall. This implementation is built from:
- the published abstract;
- the article's supplementary material, which defines the 23 alpha factors used as features;
- the methods the article cites (formulaic alphas, N-period min-max labelling, XGBoost, LightGBM, LSTM, long/short equity);
- the paper's description of the self-adaptive idea.

Where the paper's exact settings are unknown (model hyper-parameters, rule set, re-scoring schedule, costs), the library
makes them explicit, documented parameters with defaults. Every page of the web front end can change them.

The library has no dependencies beyond the C++17 standard library.

## Building

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build            # or ./build/sat_tests [name-filter]
./build/self_adaptive_trading_demo
```

Options: `-DSAT_BUILD_TESTS=OFF` and `-DSAT_BUILD_EXAMPLES=OFF`. `cmake --install` installs the
headers and a `sat::sat` CMake target.

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main`, on pull requests and on manual dispatch:

| Job | What it checks |
|---|---|
| `native (gcc)`, `native (clang)` | Release build with warnings as errors, `ctest`, example program |
| `native (ASan + UBSan)` | Debug build with address and undefined-behaviour sanitizers, `ctest` |
| `wasm` | Embind module built in `emscripten/emsdk:4.0.10`. Runs the Node checks of every entry point, and fails if the committed `web/wasm/` is out of date |
| `standalone` | Copy-deployable site and zip, uploaded as the `trading-sat-standalone` artifact |
| `e2e` | Every page in headless Chromium (WebGPU on SwiftShader): dev site over HTTP, standalone from `file://` and over HTTP. Screenshots are uploaded as an artifact |
| `docker` | `native` and `web` images with BuildKit layer caching, plus a smoke test of the nginx image |

GitHub Pages: `.github/workflows/pages.yml` deploys the standalone site to https://svr-net.github.io/trading/ after CI passes on every push to `main`. Pages must be enabled once under *Settings → Pages* with **Source: GitHub Actions** (done for this repository).

Releases: push a tag that matches the CMake project version (`git tag v0.1.0 && git push origin v0.1.0`), or run the **Release** workflow manually on `main`, which creates that tag itself. `.github/workflows/release.yml` runs the full CI, then publishes a GitHub release with `trading-sat-standalone-<tag>.zip` and its SHA-256.

## Standalone, copy-deployable build

`tools/standalone/build.mjs` turns `web/` into one folder of static files that runs anywhere you copy it.
- Open `index.html` straight from disk (`file://`), or upload the folder unchanged to any static host (S3, GitHub Pages, IIS, nginx...).
- There is no server code, no MIME setup and no build step on the target.
- The `.wasm` is embedded in the worker script, and every page is bundled into one classic script, so nothing is fetched at runtime.
- Asset names carry content hashes, so a host can cache them indefinitely.

```sh
docker build --target standalone-artifacts --output dist .     # -> dist/standalone/ + dist/trading-sat-standalone.zip
# or locally (needs web/wasm built):
npm --prefix tools/standalone ci && node tools/standalone/build.mjs
```

```
dist/standalone/
  index.html, core.html, ... gpu.html      10 pages
  assets/app.<hash>.js                     all pages (classic script, ~75 KiB)
  assets/sat-runtime.<hash>.js             WASM worker with embedded library (~685 KiB)
  assets/style.<hash>.css
  manifest.json, README.txt                build commit, file sizes and SHA-256
```

The whole site is about 770 KiB (about 300 KiB zipped). `node web/tests/e2e.mjs --root dist/standalone --file` opens every page from
disk. It checks that the pages make no network request at all, and that the WebGPU kernels still agree with WASM.

## Docker

The multi-stage `Dockerfile` builds everything from source. Each stage runs its tests, so a successful build means they passed.

```sh
docker build -t trading-sat .                      # WASM from source + checks -> standalone site on nginx (default)
docker run --rm -p 8080:80 trading-sat             # web front end at http://localhost:8080

docker build --target native -t trading-sat:native .  # native C++ build, ctest
docker run --rm trading-sat:native                    # end-to-end demo

docker build --target e2e .                        # every page in headless Chromium, WebGPU on SwiftShader:
                                                   # dev site over HTTP, standalone from file:// and over HTTP
docker build --target standalone-artifacts --output dist .  # export the standalone site and zip
docker build --target wasm-artifacts --output web/wasm .   # export sat.js / sat.wasm to the host
```

| Target | Base image | Contents |
|---|---|---|
| `native` | `ubuntu:24.04` | Library, unit tests and demo binary; installed headers and CMake package in `/opt/sat` |
| `wasm` | `emscripten/emsdk:4.0.10` | Embind module built from source and checked with `web/tests/wasm.test.mjs` |
| `wasm-artifacts` | `scratch` | Only `sat.js` and `sat.wasm`, for `--output` |
| `standalone` | `node:22-alpine` | Copy-deployable site built with the pinned esbuild |
| `standalone-artifacts` | `scratch` | `standalone/` folder and `trading-sat-standalone.zip`, for `--output` |
| `e2e` | `mcr.microsoft.com/playwright` | Browser test of all pages in three modes, including GPU-vs-WASM agreement |
| `web` (default) | `nginx:1.27-alpine` | The standalone site; hashed assets are cached as immutable |

If the network goes through a TLS-intercepting proxy, give the npm stages the proxy's CA so they can reach the registry:
`docker build --secret id=ca,src=/path/to/ca.pem ...`. Pass `--build-arg GIT_COMMIT=$(git rev-parse --short HEAD)` to stamp the commit into `manifest.json`.

## Web front end (WebAssembly + WebGPU)

`web/` contains one interactive page per library context. Each page runs the C++ library compiled to WebAssembly,
and the strategy search also runs as WebGPU compute kernels. The standalone build of these pages is deployed at
**https://svr-net.github.io/trading/** on every push to `main`.

```sh
python3 -m http.server -d web 8000      # any static server; file:// will not load WASM or module workers
# open http://localhost:8000
```

| Page | Context | Entry point |
|---|---|---|
| `core.html` | Operator algebra on real series, seeded generator, look-ahead check of every alpha | `coreDemo` |
| `data.html` | Regime-switching synthetic market or your own CSV of daily bars | `marketData` |
| `factors.html` | The 23 alphas: information coefficients, coverage, IC by regime, correlations | `factors` |
| `labels.html` | Next-day direction, excess over the median, N-period min-max labels | `labels` |
| `models.html` | Walk-forward training of every model: accuracy, AUC, ROC, calibration, importance | `models` |
| `strategies.html` | Every model × rule candidate, and the winner of each quarter | `strategies` |
| `adaptive.html` | The self-adaptive strategy against the best fixed rule and the market | `adaptive` |
| `robustness.html` | Look-back × adaptation-period × score grid, transaction costs, year by year | `robustness` |
| `gpu.html` | WebGPU kernels validated against WASM, with a benchmark | `gpuJobs` · `gpuAnalyse` (`validation`) |

The pages share one specification: market, factors, labels, models, walk-forward schedule, rules, costs, selector and
robustness grid. You edit it on any page, and the browser's local storage keeps it. An uploaded CSV is stored separately.
The WASM module runs in a module worker, so training and backtests don't block the page. The walk-forward predictions
are cached in the worker, so a page re-runs its strategy analysis without training the models again.

**Compute engine.** The Overview, Fixed strategies, Self-adaptive and Robustness pages have an *Engine* selector: **Auto** (the default) runs the WebGPU kernels whenever the browser supports WebGPU, on desktop and mobile alike, and WebAssembly otherwise; **WebGPU** or **WebAssembly** forces one. Unsupported settings (a selector that holds a mix of the top M candidates, more than 128 stocks or 8 models) and GPU errors fall back to WebAssembly, and the status line names the engine that ran and why. Views that need every candidate's daily returns (all equity curves, the quarterly winners, the average of all rules) are WebAssembly-only.

**All numerics are in C++.** Model training always runs in WebAssembly. Each strategy analysis (`strategies`, `adaptive`, `robustness`) is defined once in `wasm/bindings.cpp` as a list of jobs: a transaction cost and the selector settings to run over the candidate pool. A combine step then builds the page's result. On WebAssembly each job builds a `CandidateBook` and runs `evaluateGrid`. On WebGPU:

1. `gpuJobs(spec)` trains or reuses the models and compiles each job with `sat::gpu::compile` into a plan: one packed table, dispatch sizes and buffer sizes.
2. `web/js/gpu/engine.js` uploads each plan, dispatches the three kernels and reads back two arrays. The kernel sources come from the library (`gpuKernels`).
3. `gpuAnalyse(spec)` turns the read-backs into a `GridResult` (`sat::gpu::summarise`) and runs the same combine step as the CPU path.

The JavaScript layer (`web/js/gpu`) only routes calls and talks to WebGPU, and the pages only render results. `gpuEmulate(spec)` runs the kernels on the CPU instead, which is how Node tests the GPU path.

**WebGPU kernels** (`include/sat/gpu/fused_backtest.hpp`). `compile()` packs into one table:
- the candidates (model, rule, parameter, holding period) and the selector settings;
- the next-day returns;
- every model's out-of-sample probabilities and their cross-sectional ranks.

Three WGSL compute kernels (`src/gpu/fused_kernels.cpp`) then run on it:
1. **candidate-backtest** runs once per candidate. The 64 candidates of a workgroup walk the days in lockstep. For each day they cooperatively stage the universe's returns, probabilities and ranks in workgroup memory. Each candidate then rebalances on its schedule, books gross return and turnover, and extends prefix sums of its net return, squared return and squared loss.
2. **adaptive-select** runs once per selector setting. On each adaptation date it scores every candidate from the prefix sums in O(1). It holds the best one, or cash, and prices each switch from the two portfolios' weights, recomputed from the stored predictions.
3. **series-summary** accumulates the performance statistics of every series.

The kernels need at most 4 storage buffers and 8.5 KiB of workgroup memory, so they run on WebGPU compatibility-mode adapters (OpenGL ES) too. `runFusedReference()` (`src/gpu/fused_reference.cpp`) executes the same three kernels on the CPU in single precision, so the unit tests check the kernel algorithm natively against `evaluateGrid`. The backtests are deterministic, so the engines agree to f32 rounding. Only near-ties between candidate scores can resolve differently.

**Rebuilding the WASM module.** The built `web/wasm/sat.{js,wasm}` is committed. Rebuild it with Docker (`--target wasm-artifacts` above), or with a local Emscripten:

```sh
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build-wasm                 # writes web/wasm/sat.js and sat.wasm
node web/tests/wasm.test.mjs             # every entry point plus invariant checks, in Node
node web/tests/e2e.mjs                   # every page in headless Chromium (Playwright; WebGPU via SwiftShader)
```

## How the library maps onto the paper

| Step | Module | What is implemented |
|---|---|---|
| Data | `data/market_data.hpp`, `data/synthetic_market.hpp` | Daily OHLCV + VWAP panels, CSV loader (long format, gaps filled forward). A regime-switching synthetic market (hidden Markov chain of bull / bear / range-bound states; regime-dependent drift, volatility and short-term autocorrelation; reversal after abnormal volume) stands in for the paper's Hong Kong data |
| Factor algebra | `factors/operators.hpp` | `rank`, `scale`, `delay`, `delta`, `ts_sum/mean/min/max/argmax/rank/product/stddev`, rolling `corr`/`covariance`, `signedpower`, conditionals; no operator looks ahead |
| Features | `factors/alpha101.hpp`, `features/dataset.hpp` | The 23 formulaic alphas of the paper's factor table (Kakushadze, 2016): #1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 17, 20, 29, 33, 34, 35, 40, 41, 44, 62, 65, 81. Cross-sectional rank or z-score normalisation; lagged feature sequences for recurrent models |
| Labels | `features/dataset.hpp` | Up/down over a horizon, excess over the cross-sectional median, and N-period min-max labelling (Han, Kim and Enke, 2023), each with its look-ahead for purging |
| Prediction | `ml/*` | Logistic regression (IRLS), linear SVM (Pegasos + Platt scaling), CART, random forest, gradient-boosted trees with depth-wise (XGBoost) or leaf-wise (LightGBM) growth on histogram splits, MLP and LSTM (Adam, back-propagation through time). Accuracy, precision, recall, F1, AUC, log loss, ROC |
| Out-of-sample protocol | `ml/walk_forward.hpp` | Rolling re-fits on a trailing window, purged by the label look-ahead, with every prediction out of sample |
| Trading rules | `strategy/strategy.hpp` | Long top-k, dollar-neutral long-short, long above a probability threshold, probability-weighted. Holding periods; costs per unit of turnover; equal-weight market benchmark |
| Performance | `strategy/performance.hpp` | Annual return and volatility, Sharpe, Sortino, maximum drawdown, Calmar, win rate, turnover |
| Self-adaptive strategy | `adaptive/self_adaptive.hpp`, `adaptive/experiment.hpp` | The candidate pool (every model × every rule), re-scored every *k* days on its trailing out-of-sample record (Sharpe, Sortino or mean return). It holds the best candidate (or an equal mix of the top M), or cash when nothing scores above a minimum. Switches pay their turnover |
| Robustness | `adaptive/self_adaptive.hpp` (`evaluateGrid`) | The selector over a grid of look-backs, adaptation periods and scores, and over transaction costs, all on the same evaluation days |
| GPU strategy search | `gpu/fused_backtest.hpp` | Candidate backtests and the selector grid as WGSL kernels: plan compiler, kernel sources, read-back summary, and a CPU reference of the kernels |

## Conventions

- Panels are date × stock tables (row-major by date). NaN marks a missing value.
- Features on date *t* use data up to the close of *t*. A position set at the close of *t* earns the close-to-close return from *t* to *t + 1*.
- A strategy's weights depend only on the predictions of its last rebalance date, which is what lets the selector price a switch, on either engine, without replaying history.
- Turnover is the sum of absolute weight changes. Costs are charged per unit of turnover (default 10 bp, about Hong Kong's stamp duty on one side).
- Returns are annualised with 252 trading days and a zero risk-free rate.
- All strategies, including the self-adaptive one, are compared over the same evaluation days: those after the selector's first full look-back.

## Example

```cpp
#include "sat/sat.hpp"
using namespace sat;

MarketData data = generateSyntheticMarket(SyntheticMarketSpec{});  // or parseCsv(text)
ExperimentSpec spec;                                                // 23 alphas, next-day labels, six models
PredictionSet p = runPredictions(data, spec);                       // walk-forward, out-of-sample forecasts
Experiment e = runStrategies(p, spec);                              // every model x rule, then the selector

double adaptiveSharpe = e.adaptive.metrics.sharpe;
double bestFixedSharpe = e.candidateMetrics[e.bestFixed].sharpe;    // chosen in hindsight
double marketSharpe = e.benchmark.metrics.sharpe;
```

`examples/self_adaptive_trading_demo.cpp` runs the whole method on the synthetic market:
- the walk-forward forecasts of six model families;
- the best fixed rules;
- the self-adaptive strategy against the best fixed rule (chosen in hindsight) and the market;
- a grid of selector settings;
- the same grid on the GPU kernels' CPU reference.

## Modelling notes and limitations

- **The paper's data.** The Hong Kong data are available from the authors on request. The synthetic market gives every page realistic input, but its numbers are not the paper's results. Load real daily bars from a CSV file to study real stocks.
- **Unknown settings.** The paper's exact model hyper-parameters, rule set and re-scoring schedule are not public. The defaults here are reasonable choices, not the authors', and the robustness page shows how much they matter.
- **Execution.** Trades happen at the close that produced the signal, with no slippage or market impact beyond the cost per unit of turnover. Short positions pay no borrowing cost.
- **Selection bias.** The "best fixed rule" is chosen in hindsight over the evaluation period. No investor could have traded it, which is the point of comparing the self-adaptive strategy with it.
- **Alpha definitions.** Window lengths follow the paper's factor table, which rounds Kakushadze's fractional windows. Correlations over a constant window are undefined (NaN). Normalised features map them to the cross-sectional centre. Alphas #3 and #81 correlate ranks of price levels, which rarely move, so they are often undefined.
- **Models.** The networks are small so that they train in a browser in seconds. Tree ensembles use at most 32 histogram bins per feature.
