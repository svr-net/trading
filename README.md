# UK Factor Model

A C++ factor model for UK shares whose every setting is derived from the data, with fused WebGPU
kernels (and their exact CPU emulation as fallback), a WebAssembly build and a small page that shows
each stock's expected return and a daily backtest against holding the market, after real costs.

## One kernel source for every backend

The maths runs in kernels written once, in WGSL (`kernels/model`: signal, rank, gram, expect;
`kernels/daytrade`: levels, trades, book). WebGPU runs the WGSL as written. At build time
`tools/wgsl2cpp.py` translates the same files into C++ templated on the float type, which the CPU
backends run invocation by invocation: `float` is the emulated GPU (the WebAssembly fallback where
WebGPU is missing), `double` the reference. A change to a kernel is one edit to its `.wgsl` file.

## The model

- **Signals** in five families (return, low volatility, nearness to the high, trend quality, small
  traded value) at dyadic horizons 2, 4, 8, ... days up to an eighth of the history.
- **Forced orthonormality**: each day the signals' rank z-scores are made exactly orthonormal by
  symmetric (Löwdin) orthogonalisation.
- **Self-adaptive premia**: every factor's daily record, shrunk by its own t-statistic
  (positive-part James–Stein). Weak evidence gives no tilt; the data choose the horizons and signs.
- **Expected returns**: the projection of the factors' premia on each stock, relative to the market.
- **Trading**: decisions at the close, fills at the next open. Start from the market; buy or sell a
  stock only when its expected return over a forecast's measured life beats the round-trip cost.
  Falling or overextended stocks are sold when their updated forecast says so (adaptive stops).
- **Futures hedge**: the index future most correlated with the market before trading starts; its
  own trend forecast, built the same way, decides a short hedge sized by the book's beta.
- **Inputs**: only the costs (UK: 10 bp dealing plus 50 bp stamp duty on purchases).

## Build and run

```sh
cmake -S . -B build -G Ninja && cmake --build build
./build/ofm_tests
./build/ofm_report                                   # sample market
./build/ofm_report --csv stocks.csv --series futures.csv --engine emulated --json report.json
```

WebAssembly (emsdk 4.0.10): `emcmake cmake -S . -B build-wasm && cmake --build build-wasm`, then serve
`web/` and open `index.html`. Engines: WebGPU, the emulated GPU (the kernels on the CPU in single
precision) and the double-precision reference. Tests: `node web/tests/wasm.test.mjs`,
`node web/tests/e2e.mjs` (Playwright; WebGPU on SwiftShader).

## UK data

The *UK portfolio* workflow downloads LSE bars and futures from EODData (key in the
`EODDATA_API_KEY` secret), runs the model and publishes results only: EODData's licence does not
allow redistributing prices. `model/` keeps the first Python version (fixed look-backs, monthly) as a
benchmark.

Research, not investment advice.
