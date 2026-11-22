# The Lab on the GPU (WebGPU)

The page's Lab runs the strategy engine in the browser (WebAssembly). With **The live settings in full** it runs the
market's live strategy with every parameter of `config/markets.json`: the nested Monte Carlo choosing the book and
learning each share's stop and buy-back. That is most of the run's work, so where the browser has WebGPU it runs on
the device's GPU.

## How it is built

All of it is C++; JavaScript only starts the engine.

| Part | File | What it does |
|---|---|---|
| Batch | `include/strategy/mc_gpu.hpp` | the Monte Carlo's trade kernel as one batch: G stop settings x S shares x P paths, each independent; `mcPack` builds it (the paths' days, each share's bars over the window as ratios, its start, the settings); `mcRunAny` and `mcSettingsAny` hand it to a back end that can run it and run it themselves otherwise |
| Kernel | `include/strategy/mc_wgsl.hpp` | the kernel in WGSL, one invocation per (setting, share, path), statement for statement `mcBatchRef<float>` in `mc_gpu.hpp` |
| Back end | `backends/webgpu.cpp` | `--backend webgpu`: WebGPU's C API (`webgpu.h`): the adapter and device opened once, the kernel compiled once, each batch uploaded, dispatched and read back in one wait; the engine waits for the GPU under Asyncify. No GPU, a validation error, a buffer over the device's limits, a lost device: that batch runs on the CPU |
| Build | `docker/wasm/Dockerfile` | `ofm_strategy.js` (the CPU engine) and `ofm_strategy_gpu.js` (`-DOFM_WEBGPU -sUSE_WEBGPU -sASYNCIFY`) |
| Page | `web/lab/worker.js`, `web/app.js` | the GPU build where `navigator.gpu` exists; the status says where the batches ran (`webgpu: N Monte Carlo batches on <adapter>`) |

## Precision, and how it is checked

WebGPU has no doubles: the GPU computes in single precision, the CPU engine in double. CI checks that this changes
nothing the model decides:

- `tools/check/mcgpu_check.sh` (native): every batch of a run with the UK's live settings, with a volatility stop
  and with a fixed stop is also run by the kernel's references (`X_MCGPU_CHECK`) and compared with the engine's own
  results: in single precision at most 0.05% of the slots more than 1e-4 apart (seen: 0.002%) and at most 0.01% of the
  stop decisions different (seen: 17 of 3.7 million, decisions sitting exactly on a boundary); and the whole run in
  the GPU's precision (`X_MCGPU_EMULATE`) makes the same orders and the same backtest trades (the live settings:
  416 trades, Sharpe 1.141 in double, 1.144 in single).
- `web/tests/labgpu.mjs` (headless Chromium, its software GPU SwiftShader): the Lab's worker runs the GPU build with
  the live settings on a generated market: every batch on the GPU, the backtest identical to the native engine's in
  single precision, and the same orders as the CPU build, whose results are the native engine's.

The CPU engine, the native engine and the daily scans are unchanged: their results stay bit for bit the same on every
back end (`tools/check/strategy_check.sh`).

## The adaptive nested Monte Carlo's network

The same back end trains the adaptive nested Monte Carlo's differential network (docs/adaptive-nested-mc.md): it sets
`anmc::diffnetOffload()`, and on the device it opened for the Monte Carlo runs a fit's fused kernel (`diffnetFused`,
`include/strategy/diffnet_wgsl.hpp`; its reference in `include/strategy/diffnet_gpu.hpp`), 64 steps a command buffer.
`docker/wasm/Dockerfile` links it into `ofm_anmc_gpu.js` beside the CPU build `ofm_anmc.js`; `web/tests/anmclab.mjs`
runs both and the native build on the same basket check. The strategy engine's builds carry the hook but never train a
network: their results, and the report line `strategy: webgpu: N Monte Carlo batches on ...`, are unchanged.

## Speed

On a GPU each batch is one dispatch of thousands of independent paths. Each batch also waits once for the browser
(a timer of a millisecond or more), so a run of many small batches is bound by those waits; the stop learning sends
one batch per share, which is the next thing to merge if a device's timings call for it. The software GPU in CI
(SwiftShader, on the CPU) says nothing about a real GPU's speed: there the CPU build takes 3.4 s and the GPU build
12.8 s for the same run.
