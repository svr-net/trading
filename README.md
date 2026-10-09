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

**Pipeline options from *Advances in Financial Machine Learning*.** A few techniques from M. López de Prado,
*Advances in Financial Machine Learning* (Wiley, 2018), are options of the same pipeline:
- triple-barrier labels;
- uniqueness-weighted training and CUSUM-sampled training events;
- fractional-differentiation and microstructure features;
- the bet-sized trading rule;
- the deflated Sharpe ratio and the probability of backtest overfitting, which assess the pool and the selector.

They are implemented from the published algorithms, in the library's own code.

**Self-adaptive forecast.** The paper's idea is applied one level down: the forecast the selector trades is itself
chosen self-adaptively.
- **Candidates.** Besides the two models, four *market-state specialists* are trained, each on calm, turbulent, rising
  or falling market days only.
- **Memory.** Each candidate's record grows day by day while the market stays in one state, and recedes to the start
  of the new state when the market changes (an ADWIN change detector on the market's moves).
- **Decision.** A specialist or model is used only while its lead over the average of all six is statistically
  significant; otherwise the average is.

On unseen synthetic markets this lifted the selector's Sharpe ratio from 1.57 to 2.00 and beat the previous default on
all six markets. See [Which forecast the selector trades](#which-forecast-the-selector-trades).

**Scope.** Version 0.6 keeps only the self-adaptive pipeline. Earlier versions also carried:
- information-driven bars, purged cross-validation, feature importance and hierarchical risk parity pages;
- hedging, pairs trading, trend following, regime switching and optimal execution;
- a strategy tournament with meta-allocation;
- a composite multi-signal strategy.

They remain in the git history and the v0.5.1 release.

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
  assets/app.<hash>.js                     all pages (classic script, ~95 KiB)
  assets/sat-runtime.<hash>.js             WASM worker with embedded library (~900 KiB)
  assets/style.<hash>.css
  manifest.json, README.txt                build commit, file sizes and SHA-256
```

The whole site is about 1 MiB (about 390 KiB zipped). `node web/tests/e2e.mjs --root dist/standalone --file` opens every page from
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
| `models.html` | Walk-forward training of every model and their composite forecast: accuracy, AUC, ROC, calibration, importance | `models` |
| `strategies.html` | Every model × rule candidate, and the winner of each quarter | `strategies` |
| `adaptive.html` | The self-adaptive strategy against the best fixed rule and the market | `adaptive` |
| `robustness.html` | Look-back × adaptation-period × score grid, transaction costs, year by year | `robustness` |
| `gpu.html` | WebGPU kernels validated against WASM, with a benchmark | `gpuJobs` · `gpuAnalyse` (`validation`) |

The pages share one specification: market, factors, labels, models, walk-forward schedule, rules, costs, selector and
robustness grid. You edit it on any page, and the browser's local storage keeps it. An uploaded CSV is stored separately.
The WASM module runs in a module worker, so training and backtests don't block the page. The walk-forward predictions
are cached in the worker, so a page re-runs its strategy analysis without training the models again.

The specification also holds the AFML options of the pipeline, under *Alpha factors*, *Labels* and *Walk-forward training*:
- extra features;
- triple-barrier labels;
- sample weights and CUSUM event sampling.

The composite forecast is set under *Composite forecast*, next to the models:
- the market-state specialists (on by default);
- the forecast traded: the separate models, the equal-weight average, or the self-adaptive forecast (the default);
- the self-adaptive forecast's memory and decision rule.

**Compute engine.** The Overview, Fixed strategies, Self-adaptive and Robustness pages have an *Engine* selector:
- **Auto**, the default, runs the WebGPU kernels whenever the browser has a usable WebGPU adapter, on desktop and mobile alike. Otherwise it runs the same kernels on the **emulated GPU**.
- **WebGPU**, **Emulated GPU** or **WebAssembly** forces one of them. A failing WebGPU run also falls back to the emulated GPU.

The other pages have no GPU kernel, and their status line says "WebAssembly (no GPU kernel for this analysis)".

Settings the kernels cannot express fall back to WebAssembly on any engine. Those are a selector that holds a mix of the top M candidates, more than 128 stocks, or more than 8 models. The status line names the engine that ran and why. Views that need every candidate's daily returns (all equity curves, the quarterly winners, the average of all rules) are WebAssembly-only.

**All numerics are in C++.** Model training always runs in WebAssembly. Each strategy analysis (`strategies`, `adaptive`, `robustness`) is defined once in `wasm/bindings.cpp` as a list of jobs: a transaction cost and the selector settings to run over the candidate pool. A combine step then builds the page's result. On WebAssembly each job builds a `CandidateBook` and runs `evaluateGrid`. On WebGPU:

1. `gpuJobs(spec)` trains or reuses the models and compiles each job with `sat::gpu::compile` into a plan: one packed table, dispatch sizes and buffer sizes.
2. `web/js/gpu/engine.js` uploads each plan, dispatches the three kernels and reads back two arrays. The kernel sources come from the library (`gpuKernels`).
3. `gpuAnalyse(spec)` turns the read-backs into a `GridResult` (`sat::gpu::summarise`) and runs the same combine step as the CPU path.

The JavaScript layer (`web/js/gpu`) only routes calls and talks to WebGPU, and the pages only render results.

**Emulated GPU.** `web/js/gpu/emulator.js` is a drop-in device with the same `run(plan)` interface as the WebGPU engine. Each plan from `gpuJobs` goes to `gpuRunPlan`, which runs it on `runFusedReference()` in the WebAssembly worker:
- the three kernels, invocation by invocation, in 32-bit floats;
- over the same header and table buffers that WebGPU uploads;
- returning the same `stats` and `adapt` read-back.

`gpuAnalyse` then treats the read-back exactly like a GPU's, so the pipeline and its results are unchanged when WebGPU is missing. The reference checks that a plan's header fits its tables, as WebGPU's bounds checks would. The WebGPU page always runs the emulated GPU next to WebGPU and the WebAssembly library: it reports the timings of all three, checks each kernel device against WASM and the two devices against each other, and benchmarks all three. The sidebar of every page shows whether the GPU emulator is on standby, selected, or in use because WebGPU is missing. `gpuEmulate(spec)` does compile, emulate and analyse in one call, which is how Node tests the kernel path.

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

## How the library maps onto *Advances in Financial Machine Learning*

| Chapter | Module | What is implemented |
|---|---|---|
| 2. Event sampling | `afml/labeling.hpp`, `afml/features.hpp` | Symmetric CUSUM filter. `ExperimentSpec::cusumMultiple` trains the models on CUSUM events only |
| 3. Labeling | `afml/labeling.hpp`, `features/dataset.hpp` | EWM volatility targets; triple-barrier method (profit taking, stop loss, vertical barrier); meta-labels for a given side. `LabelKind::TripleBarrier` labels every stock and date for the pipeline |
| 4. Sample weights | `afml/sampling.hpp`, `ml/walk_forward.hpp` | Concurrency, average uniqueness, return-attribution weights, time decay, the sequential bootstrap. `WalkForwardSpec::weighting` trains on a uniqueness-weighted (optionally decayed) bootstrap of each window |
| 5. Fractional differentiation | `afml/fracdiff.hpp` | Fixed-width-window weights, fractional differences, augmented Dickey-Fuller test, the scan for the minimum stationary *d*; the `ffd` feature |
| 10. Bet sizing | `afml/bet_sizing.hpp`, `strategy/strategy.hpp` | Size 2N(z) − 1 from the predicted probability, discretisation. `StrategyKind::BetSized` is a rule of the candidate pool, also in the GPU kernels |
| 11. The dangers of backtesting | `afml/backtest_stats.hpp` | Probability of backtest overfitting by combinatorially symmetric cross-validation, with the performance degradation and the probability of loss of the in-sample winner |
| 14. Backtest statistics | `afml/backtest_stats.hpp`, `afml/overfitting.hpp` | Probabilistic and deflated Sharpe ratios, expected maximum Sharpe ratio of unskilled trials, drawdown and time-under-water statistics, concentration of returns. The candidate pool is assessed as a multiple-testing problem |
| 16. Asset allocation | `afml/portfolio.hpp` | Hierarchical risk parity, inverse-variance and minimum-variance weights, used by the UK portfolio example |
| 19. Microstructural features | `afml/microstructure.hpp` | Roll spread, Corwin-Schultz spread, Amihud illiquidity and Kyle's lambda from daily bars, as optional features |

Note on these implementations:
- **The deflated Sharpe ratio** treats every candidate (model × rule) as a trial. The self-adaptive strategy is deflated by the same number of trials, which is conservative for a strategy that chooses out of sample.

## UK stocks from EODData

`tools/eoddata/fetch.mjs` downloads daily bars from the [EODData API](https://api.eoddata.com) into the library's CSV
format. The API key is read from the `EODDATA_API_KEY` environment variable and never written anywhere. For the London
Stock Exchange it keeps:
- the most traded company shares quoted in sterling (pence), so no warrants, ETFs, funds or international order-book lines;
- split-adjusted prices;
- only stocks with a long enough history and no unexplained daily jump above 60%.

Responses are cached under `data/`, which git ignores: EODData's licence does not allow redistributing the data.

`examples/uk_portfolio` builds a ten-stock portfolio from that file. Five constructions compete in a walk-forward backtest
with monthly rebalancing:
- the default models' ensemble forecast with HRP weights;
- 12-1 momentum with inverse-volatility weights;
- minimum variance;
- trailing Sharpe ratio with HRP weights;
- low correlation to the market with inverse-variance weights.

Each holds ten names, between 4% and 20% each. Every trade pays 10 bp for spread and commission, and every purchase also
pays 0.5% stamp duty. A selector holds the construction with the best trailing six-month Sharpe ratio. For each
construction, the program prints whole-share orders for a budget at the last close.

```sh
EODDATA_API_KEY=... node tools/eoddata/fetch.mjs --exchange LSE --universe 120 --years 6 --out data/LSE.csv
./build/uk_portfolio data/LSE.csv --stocks 10 --budget 100000 --report report.json
```

The **UK portfolio** workflow (`.github/workflows/uk-portfolio.yml`, run manually) does the same on GitHub Actions with
the `EODDATA_API_KEY` repository secret. It publishes only the results: the job summary and a report artifact.

On the 120 most traded LSE shares from July 2023 to October 2026, momentum had the best record (29.2% a year, Sharpe
1.50), against 13.0% for the equal-weight universe. The selector reached 16.6% (Sharpe 0.98); its monthly switching
costs stamp duty. The machine-learning forecasts had no skill on these stocks: AUC of 0.50 to 0.505.

These results are for research, not investment advice. Returns exclude dividends, the history is short, and the
universe is today's surviving stocks.

## Which models and rules earn their place

The default candidate pool has **two models × five rules**:
- models: logistic regression and random forest;
- rules: top 5 daily, top 10 held 5 days, long-short 3, P > 0.52, and probability-weighted.

This replaced the paper-style pool of six models × nine rules. The choice comes from `examples/pool_ablation`, which:
1. trains all eight model families on 8 ten-year synthetic markets;
2. measures what the self-adaptive strategy loses without each model and each rule;
3. removes, one at a time, the item whose absence costs least;
4. checks every step on 8 unseen markets.

| Pool | Candidates | Validation Sharpe | Max drawdown | Training |
|---|---|---|---|---|
| 8 models × 9 rules (everything) | 72 | 0.70 | 56% | 11.9 s per market |
| 6 models × 9 rules (previous default) | 54 | 0.81 | 51% | 11.0 s |
| **2 models × 5 rules (default)** | **10** | **1.22** | **49%** | **1.7 s** |
| 2 models × 1 rule | 2 | 1.52 | 46% | 1.7 s |

The deflated Sharpe ratio of the self-adaptive strategy rose from 41% with the previous pool to 72% with the default one, because there are fewer candidates and so fewer trials. The worst market's Sharpe ratio rose from −0.49 to −0.16.

What the ablation showed:
- **Logistic regression** has the best out-of-sample AUC (0.522), trains fastest, and costs the most when left out (−0.26 Sharpe).
- **The random forest** is the only other family that adds value.
- **The linear SVM** has no skill (AUC 0.499).
- **The rest add noise.** XGBoost, LightGBM, MLP, LSTM and the single tree mostly add candidates whose short winning streaks the selector chases.
- **Rules:** bet sizing, P > 0.55 and long-short 5 lowered the result.
- **Why stop at ten:** the two-candidate pool scores higher but leaves the selector almost nothing to choose from. The default stops where it still spans long-only and long-short rules, two horizons, and threshold and probability weighting.
- **Fewer trials:** a smaller pool also means fewer trials for the deflated Sharpe ratio, so the result is more credible.

Every family and rule kind is still in the editor.

**On real UK data** (`pool_ablation 0 --csv`, the 120 LSE shares, July 2023 to October 2026), no pool has value, because no model has skill: every family's AUC is 0.499 to 0.503.

| Pool | Self-adaptive Sharpe | Max drawdown | Training |
|---|---|---|---|
| Everything | −0.38 | 37% | 5.2 s |
| Previous default | −1.17 | 51% | 4.8 s |
| Default | −0.65 | 34% | 0.8 s |

The consolidated pool loses less than the previous default, with a smaller drawdown and 6× faster training. The model-and-rule approach still needs a market where the forecasts carry information.

Run `pool_ablation [markets] [days]` to repeat the study, or `pool_ablation 0 --csv data.csv` to compare the full and default pools on real bars.

## Which forecast the selector trades

The paper makes the *trading rule* self-adaptive. `examples/forecast_study` asks whether the *forecast* can be chosen the
same way, and what it needs to work. It trains the models on 6 ten-year synthetic markets (selection), 6 unseen ones
(validation) and the 120 LSE stocks, and runs the self-adaptive selector, with its default rules and settings, on each
forecast. Settings are chosen on the selection markets only.

### Step 1: the selector's own rule, one level down

`CompositeMethod::Adaptive` scores each candidate (each model, and their average) by its mean daily rank correlation
with the outcomes (information coefficient) and uses the best. It uses only outcomes known before each date; a unit test
checks this by scrambling future labels.

With the selector's settings (the last 63 days, every 21) it *lost* to the plain average: 1.41 against 1.57 on unseen
markets. The two models' scores differ by about 0.005, a quarter of the noise in a 63-day estimate, so the choice
followed noise.

### Step 2: memory that grows, and recedes with the market

`ScoringWindow` sets how long a candidate's record is remembered:

| Window | Memory |
|---|---|
| `Fixed` | the last 63 days, re-scored every 21 |
| `Exponential` | every day, weighted towards recent ones (a half-life, or none: an expanding window); updated daily |
| `Adwin` | grows day by day; recedes when the candidate's own record changes (ADWIN, Bifet and Gavaldà, 2007) |
| `MarketAdwin` | as `Adwin`, and also recedes to the start of the market's new state when the market changes |
| `SimilarState` | weights each past day by how much its market (volatility, trend) resembles today's |

`ScoringDecision::Evidence` leaves the average only while a candidate's lead over it has a t-statistic above 2.

**A bug found on the way.** The ADWIN bound in the paper assumes values between 0 and 1. With daily market moves (about
0.01) or information coefficients, its range term is far larger than any real change, so it never cut. The library uses
the scale-free form: two parts differ when their means differ by more than their standard error allows. The confidence
(1e-4) and minimum segment (30 days) were fixed on simulated noise alone, before any market result. With them it caught
an 8-fold volatility jump in about 19 days and a doubling in about 70, with no false cut in 25,000 days of noise.

With only the two generalist models, the evidence rule recovered the average's result (about 1.56–1.58 on unseen
markets) but could not beat it: the models never differ in a way worth switching for.

### Step 3: candidates that differ by market state

`ExperimentSpec::stateSpecialists` trains four more logistic regressions, each only on days of one market state:
- calm or turbulent (21-day market volatility below or above its median so far);
- rising or falling (63-day market trend above or below zero).

`marketStateMask` builds the masks from information available each day. Until a state has enough history, a specialist
trains on all days.

| Selector Sharpe ratio | Selection | Validation (unseen) | Beats the old default (validation) | UK (LSE) |
|---|---|---|---|---|
| Old default: average of the two models | 2.18 | 1.57 | — | −0.49 |
| Average of the six | 2.49 | 2.05 | 6 of 6 | −0.38 |
| **Self-adaptive: market-state ADWIN, evidence (new default)** | **2.50** | **2.00** | **6 of 6** | **−0.32** |
| Self-adaptive: similar state, evidence | 2.48 | 2.07 | 6 of 6 | −0.57 |
| Self-adaptive: expanding, evidence | 2.48 | 2.03 | 6 of 6 | −0.38 |
| Self-adaptive: market-state ADWIN, best | 2.20 | 1.56 | 4 of 6 | −0.52 |

- **The specialists drive the gain.** Every forecast built from all six beats the old default on every unseen market,
  by about 0.45 Sharpe.
- **The default is chosen by the rule.** Market-state ADWIN with the evidence rule scored highest on the selection
  markets, so it is the default. On unseen markets it is within noise of the average of the six (2.00 against 2.05), and
  on UK stocks it lost the least of the evidence variants.
- **Picking the best candidate outright is still worse.** It switches on noise; requiring evidence is what makes the
  self-adaptive forecast work.
- **UK.** No forecast has skill on these stocks (AUC about 0.50), so all of them lose money. The specialists improve the
  information coefficient a little (0.007 for the calm one against 0.002), but over about 2½ years the differences are
  within noise.

Run `forecast_study [markets] [days]` to repeat the study, or `forecast_study 0 --csv data.csv` on real bars.

**Tried and dropped (v0.5).**
- **Stacking and online aggregation.** A stacked meta-learner and Bernstein Online Aggregation were worse than the
  average.
- **A composite multi-signal strategy.** It combined the selector with momentum and trailing-Sharpe books under
  exponential weights. It raised the synthetic validation Sharpe ratio to 1.82, but trailed both the market and plain
  momentum on UK stocks after stamp duty.
- **Spherical (rotor) allocators.** They did not beat exponential weights on unseen markets.

All three are in the git history and the v0.5.1 release.

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
ExperimentSpec spec;                                                // 23 alphas, next-day labels, the default pool
PredictionSet p = runPredictions(data, spec);                       // walk-forward, out-of-sample forecasts
Experiment e = runStrategies(p, spec);                              // every model x rule, then the selector

double adaptiveSharpe = e.adaptive.metrics.sharpe;
double bestFixedSharpe = e.candidateMetrics[e.bestFixed].sharpe;    // chosen in hindsight
double marketSharpe = e.benchmark.metrics.sharpe;

// Advances in Financial Machine Learning options and diagnostics.
spec.label.kind = LabelKind::TripleBarrier;                         // barriers at +- 1 daily vol, 10 days
spec.label.horizon = 10;
spec.extraFeatures = {"ffd", "vol", "amihud"};
spec.walkForward.weighting = SampleWeighting::Uniqueness;
auto report = afml::assessOverfitting(e.book, e.adaptive.net, e.evalFrom);  // DSR, PBO
```

The composite forecast:

```cpp
spec.stateSpecialists = true;                                       // four market-state specialists
spec.composite.method = CompositeMethod::Adaptive;                  // the self-adaptive forecast
spec.composite.window = ScoringWindow::MarketAdwin;
spec.composite.decision = ScoringDecision::Evidence;
spec.composite.keepMembers = false;                                 // trade it in place of the models
PredictionSet q = runPredictions(data, spec);                       // the composite comes last in q.models
Experiment c = runStrategies(q, spec);
```

`examples/forecast_study.cpp` compares the forecasts the selector can trade, on selection, validation and real markets.
`examples/pool_ablation.cpp` finds which models and rules earn their place in the pool.

`examples/self_adaptive_trading_demo.cpp` runs the whole method on the synthetic market:
- the walk-forward forecasts of the default models (any of the eight families can be added);
- the best fixed rules;
- the self-adaptive strategy against the best fixed rule (chosen in hindsight) and the market;
- a grid of selector settings;
- the same grid on the GPU kernels' CPU reference;
- the deflated Sharpe ratios and the probability of backtest overfitting.

## Modelling notes and limitations

- **The paper's data.** The Hong Kong data are available from the authors on request. The synthetic market gives every page realistic input, but its numbers are not the paper's results. Load real daily bars from a CSV file to study real stocks.
- **Unknown settings.** The paper's exact model hyper-parameters, rule set and re-scoring schedule are not public. The defaults here are reasonable choices, not the authors', and the robustness page shows how much they matter.
- **Execution.** Trades happen at the close that produced the signal, with no slippage or market impact beyond the cost per unit of turnover. Short positions pay no borrowing cost.
- **Selection bias.** The "best fixed rule" is chosen in hindsight over the evaluation period. No investor could have traded it, which is the point of comparing the self-adaptive strategy with it.
- **Alpha definitions.** Window lengths follow the paper's factor table, which rounds Kakushadze's fractional windows. Correlations over a constant window are undefined (NaN). Normalised features map them to the cross-sectional centre. Alphas #3 and #81 correlate ranks of price levels, which rarely move, so they are often undefined.
- **Models.** The networks are small so that they train in a browser in seconds. Tree ensembles use at most 32 histogram bins per feature.
