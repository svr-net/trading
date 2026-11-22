# Adaptive nested Monte Carlo

`include/strategy/adaptive_nested.hpp` (header-only, C++17) implements *Adaptive Nested Monte Carlo: How It Works*
(22 Nov 2026). It estimates risk statistics that need a conditional value at a horizon — VaR, ES and EPE —
from one ordinary run of an engine plus inner simulations only where the statistic needs them, instead of
brute-force nesting (N outer x M inner paths).

## How it works

An ordinary path from 0 to T passes through the horizon t; its own payoff discounted to t, Y, is an unbiased
one-sample estimate of the value V at that path's horizon state. The library turns that into a usable estimate:

| Step | What | Where |
|---|---|---|
| 1 split | seeded shuffle (SplitMix64, `derive(seed, -1, 0)`): half train the proxy, half are evaluated | `AdaptiveNestedEstimator::estimate` |
| 2 proxy | OLS on Hermite polynomials (total degree ≤ D, ≤ 2,000 functions) of the standardized state, tiny relative ridge, Cholesky; HC0 sandwich variance v(x) for every prediction. Injectable (`Options::proxy`): the [differential network](#the-differential-network-proxy) for states of many dimensions | `HermiteBasis`, `FittedProxy`, `Proxy` |
| 3 bins | 14 bins over the evaluation proxy values (cut points 0.5% ... 99.5%); payoff noise r²_b per bin from the training residuals | `estimate` |
| 4 scenarios | per evaluation scenario: μ, v, bin; fold 0 = Y and the odd inner batches, fold 1 = the even ones (Chan et al. merges); the precision blend Ṽ = μ + w(m̄ − μ), its standard error and the refinement gain G | `ScenarioSet` |
| 5 audit | 16 random scenarios per bin get 64 inner paths; the proxy's squared bias τ²_b per bin, measured again after every round | `measureBias` |
| 6 statistics | value, standard error and an urgency per scenario: VaR (wrong-side probability × G), ES (tail noise and cutoff misclassification), EPE (sign misclassification), with the error-budget rule | `ValueAtRisk`, `ExpectedShortfall`, `PositiveExposure`, `markLargestContributors` |
| 7 loop | the most urgent scenarios (≤ 2,000 a round) get 64 inner paths each, in parallel, seeded by (original index, batch number); the extrapolation guard refines scenarios outside the training range; stops when nothing is urgent, the budget or the 50 rounds run out | `estimate`, `run` |

Selection and value come from independent folds (ES and EPE select on the proxy plus fold 0, value from fold 1), so
a scenario refined near a cutoff carries no winner's curse. The same inputs and seed give the same result on any
number of threads.

### Integration

The engine enters at two points:

```cpp
anmc::OuterScenarios outer;           // n paths: the horizon state (d numbers, Markov) and the payoff discounted to t
anmc::InnerSimulator inner = [](const double* state, std::size_t d, double* payoffs, std::size_t count, std::uint64_t seed) {
  // the same engine restarted at `state`: `count` payoffs discounted to t, every random number from `seed`
};
anmc::Options opt;                    // TotalInnerBudget 2,000,000, InnerBatchSize 64, ... (the document's defaults)
opt.hermiteDegree = 3;
auto res = anmc::AdaptiveNestedEstimator(opt).estimate(outer, inner, anmc::ExpectedShortfall(0.975, v0));
std::puts(res.toString().c_str());   // estimate, regression-only baseline, paths, stop reason, bins
```

For a horizon state of many numbers, the [differential network](#the-differential-network-proxy) takes the Hermite
regression's place; it learns from each outer payoff's pathwise derivative by the state as well:

```cpp
outer.gradient = ...;                 // n x d: dY/dstate of each outer path (empty: values only)
opt.proxy = anmc::differentialProxy();         // include/strategy/differential_proxy.hpp
opt.proxy = anmc::fusedDifferentialProxy();    // the same, trained by the fused kernel (include/strategy/diffnet_gpu.hpp)
```

The requirements are the document's: a Markov horizon state, the same model and measure after t in the outer
continuation and the inner restarts, payoffs discounted to t, a base on the same footing as V, all randomness from the
seed, a thread-safe simulator and finite payoffs (a non-finite one throws, naming the scenario).

## Validation

`ofm_anmc` (apps/ofm_anmc.cpp) repeats the document's setup: GBM S0 = 100, r = 3%, σ = 25%, a real-world drift of 8%
to t = 10 trading days, risk-neutral to T = 0.5; 100,000 outer paths (50,000 evaluated); a short straddle at 100 for
99% VaR and ES, a risk reversal (long 105 call, short 95 put) for EPE; exact Black-Scholes values on the same
evaluation scenarios. `ofm_anmc --reps 6`, the library's defaults (RMSE as % of the exact value):

| Statistic, degree | Regression only | Adaptive | Cut | Inner paths | Document |
|---|---|---|---|---|---|
| VaR 99%, 1 | 58.5% | 1.9% | 31× | 1.67M | 40×, 1.66M |
| VaR 99%, 2 | 4.6% | 2.2% | 2.1× | 1.09M | 2.4×, 1.10M |
| VaR 99%, 4 | 2.4% | 1.2% | 1.9× | 1.13M | 3.2×, 1.17M |
| ES 99%, 1 | 17.3% | 2.0% | 8.6× | 1.99M (budget) | 6.3×, 2.00M |
| ES 99%, 2 | 17.0% | 0.5% | 32× | 0.87M | 27×, 0.96M |
| ES 99%, 4 | 17.5% | 0.6% | 28× | 0.92M | 21×, 1.26M |
| EPE, 1 / 2 / 4 | 2.2% | 2.1–2.3% | 1.0× | 0.10M | 0.9×, 14k |

VaR and ES reproduce the document (ES at degrees 2 and 4 within its 0.4–0.6%). EPE stays at the noise floor of one
outer payoff per evaluation scenario for this risk reversal (2.0%, measured separately), which inner paths for sign
decisions cannot remove; the document's lower figure points to a different EPE setup or error definition.

**One change from the document.** EPE's refinement criterion uses the selection estimate (the proxy alone until fold 1
has samples) and its standard error, not the pooled blend Ṽ. For a scenario without inner paths Ṽ holds its own outer
payoff Y, so choosing which scenarios to refine from Ṽ makes the choice depend on the Y values the unrefined scenarios
keep as their values, and biases them: in validation one round of 2,000 refined scenarios moved EPE by up to 3 points
(12 repetitions: 3.1% against 2.6% now). It is the document's own rule (selection independent of value) applied to the
refinement choice. ES keeps the document's cutoff rule on Ṽ: a degree-1 proxy ranks a two-sided tail wrongly and only
the outer payoffs in Ṽ find the tail scenarios it ranked low (8.5% at degree 1 on the selection alone, 2.0% on Ṽ).

## The differential network proxy

`include/strategy/differential_proxy.hpp` (header-only, C++17) is differential machine learning (Huge and Savine 2020)
as the proxy. A regression on a polynomial basis runs out of reach as the state grows: 231 functions at degree 2 and
20 numbers, 1,771 at degree 3. A network does not, and the pathwise derivatives make it learn from few paths.

| Part | What |
|---|---|
| labels | every training path's payoff Y and its pathwise derivative by each state number (`OuterScenarios::gradient`): E[dY/dx \| x] = dV/dx, so one path carries d + 1 labels |
| network | 3 hidden layers of 32, softplus; its derivative by its inputs by backpropagation (the twin network) |
| loss | α·MSE(value) + (1 − α)·Σ_j λ_j·MSE(derivative j)/d on the standardized labels, α = 1/(1 + d), λ_j = 1/mean(g_j²), as the paper |
| training | Adam, 60 epochs of minibatches of 256, the learning rate decayed by a cosine from 0.01 to a tenth; the weight gradient is exact, second order included (it differentiates through the twin network's backpropagation; checked against central differences) |
| estimation variance | the output layer is a regression on the last hidden layer's 32 features: its HC0 sandwich gives v(x), as the Hermite proxy's does. It leaves out the hidden layers' own uncertainty, as the Hermite proxy's leaves out the basis' |
| reproducible | the initial weights and the batch order from SplitMix64 (no `std::` distribution, whose numbers differ between standard libraries); each minibatch's gradient the sum of 8 fixed chunks in order: the same fit on any number of threads and on every platform |

A fit is laid out as a `DifferentialProblem` (standardized rows, initial weights, the batch order and the learning rate
of every step) and trained on it, so the fused kernels below train on exactly what the double-precision fit does.

### Validation

`ofm_anmc --basket D`: d correlated lognormal assets (S0 = 100, volatilities 15% to 35%, correlation 0.3), a
real-world drift of 8% to the horizon of 10 trading days, risk-neutral to expiry at 6 months. The book is short 3
straddles at 100 on the basket's geometric average plus risk reversals (long the 105 call, short the 95 put, ±0.5,
alternating) on every asset, so its value is not additive across assets and is exact at the horizon (Black-Scholes for
each asset and for the lognormal geometric average). The state is the d log-spots; the statistics are the P&L's 99% VaR
and ES and E[max(dV, 0)] (EPE). 20,000 outer paths (10,000 evaluated), a budget of 1,000,000 inner paths; RMSE as % of
the exact statistic on the same scenarios, regression only → adaptive (inner paths):

**10 assets** (`--reps 5`)

| Proxy | VaR 99% | ES 99% | EPE |
|---|---|---|---|
| Hermite, degree 2 (66 functions) | 6.08 → 1.29 (410k) | 18.06 → 1.33 (777k) | 7.03 → 4.08 (1.00M) |
| Hermite, degree 3 (286 functions) | 37.07 → 1.32 (515k) | 48.31 → 11.03 (985k) | 16.39 → 9.79 (1.00M) |
| network on values only | 44.02 → 2.83 (523k) | 40.50 → 11.50 (984k) | 10.90 → 9.82 (1.00M) |
| **differential network** | **2.87 → 1.28 (368k)** | **9.67 → 1.30 (603k)** | 6.14 → 5.96 (1.00M) |
| differential, no estimation variance | 2.57 → 2.57 (30k) | 9.67 → 0.58 (326k) | 6.14 → 5.32 (74k) |

**20 assets** (`--reps 3`; Hermite degree 3, 1,771 functions, skipped)

| Proxy | VaR 99% | ES 99% | EPE |
|---|---|---|---|
| Hermite, degree 2 (231 functions) | 22.13 → 0.77 (492k) | 19.28 → 0.55 (967k) | 19.09 → 16.91 (1.00M) |
| network on values only | 89.09 → 3.09 (552k) | 33.79 → 11.61 (1.00M) | 33.21 → 26.76 (1.00M) |
| **differential network** | **2.20 → 0.57 (376k)** | **10.27 → 0.50 (535k)** | **7.00 → 4.02 (1.00M)** |
| differential, no estimation variance | 2.33 → 2.33 (28k) | 10.27 → 0.38 (258k) | 7.00 → 7.81 (101k) |

What the numbers say:

- **The derivatives are what make it work.** The same network on the values alone is 2 to 40 times worse before
  refinement, and worse than the Hermite regression after it.
- **As a proxy it beats the regression, more so as the state grows.** Regression-only, the differential network's VaR
  error is 2.9% against 6.1% at 10 assets, and 2.2% against 22% at 20.
- **After refinement it is as good at 10 assets and better at 20.** At 10 assets VaR and ES are the regression's
  (1.28% and 1.30% against 1.29% and 1.33%) with 10% and 22% fewer inner paths, EPE worse (6.0% against 4.1%, both
  near its noise floor). At 20 assets: VaR 0.57% against 0.77%; ES 0.50% against 0.55% with 55% of the paths; EPE 4.0%
  against 16.9%. EPE's value is each scenario's own samples; the proxy only decides the signs, which a poor proxy gets
  wrong.
- **The estimation variance is a trade-off, on by default.** A network that claims to be exact (v = 0) leaves the audit
  to find its bias, and a bias below the payoff noise of 16 audited scenarios a bin goes unseen: the loop converges after
  30,000 paths with VaR at the proxy's own 2.3–2.6%. Its ES is the best of all (0.38–0.58%), because its selection is
  then the proxy's ranking alone, which is good. With v(x), the scenarios near the cutoffs blend in their outer payoffs
  and get refined: VaR as good as or better than the regression's, ES as good, and honest standard errors.
  `DifferentialOptions::estimationVariance = false` gives the other side.
- **EPE stays near the noise floor of the outer payoffs** the unrefined scenarios keep as their values (4–6% here), as
  in the one-asset validation above.

A fit takes about 3 s (10,000 rows, 60 epochs, 2 cores), against under a second for the Hermite regression of degree 2.

### Fused kernels

The fit also runs as fused GPU kernels: WGSL in `include/strategy/diffnet_wgsl.hpp`, with their plan compiler and
reference in `include/strategy/diffnet_gpu.hpp`, beside each other as `mc_wgsl.hpp` and `mc_gpu.hpp` are.

| Form | What |
|---|---|
| two kernels | a fit as one command buffer of (`diffnetGrad`, `diffnetAdam`) pairs, a step each, its record by a dynamic uniform offset; `diffnetGrad`, one invocation per sample, fuses the forward pass, the twin network, the loss on value and derivatives and the exact second-order adjoint into the sample's own gradient row (no atomics); `diffnetAdam`, one per weight, sums the rows in sample order and steps Adam; then `diffnetPredict` |
| one kernel | `diffnetFused`: the whole fit and the predictions in ONE dispatch of one workgroup of 256 threads. WebGPU has no barrier across workgroups, so every step synchronizes the workgroup with `storageBarrier()`: thread t adds the gradient rows of samples t, t + 256, ... into its own row, then sums the 256 rows in thread order for weights t, t + 256, ... and steps Adam. A uniform (first, last, predict) splits a long fit into several dispatches of the same kernel, for a browser's GPU watchdog |

`compileDifferentialFit` and `compileDifferentialFused` lay a `DifferentialProblem` out as flat f32/u32 buffers. The
references run the kernels in C++ in single precision, statement for statement and in the shader's order (WebGPU has no
doubles). `FusedDifferentialProxy` (`anmc::fusedDifferentialProxy()`) trains through the one-kernel plan: a back end
that sets `diffnetOffload()` runs it on a GPU, as `mcOffload()` does the Lab's Monte Carlo, otherwise the reference does,
so results never depend on whether a GPU was there, only on its single precision. In the tables above the kernel-trained
network gives the same errors as the double-precision one to two decimals.

`ofm_anmc --kernels --check` (10 inputs, 4,000 rows, 2,497 weights, 128 steps; one row in seven without derivatives,
one in five weighted 3):

| Check | Seen | Bound |
|---|---|---|
| exact weight gradient against central differences | 1.8e-6 | 1e-5 |
| one step's per-sample gradients (f32) against double precision | 4.7e-7 | 1e-4 |
| a whole fit's weights (two kernels, f32) against double precision | 2.8e-6 | 1e-3 |
| one kernel against two kernels, weights and predictions, batch 256 | 0 (bit for bit) | 0 |
| one kernel against two kernels, weights, batch 512 | 3.6e-7 | 1e-5 |

`web/tests/anmcgpu.mjs` runs the WGSL on WebGPU in headless Chromium (SwiftShader) from the plan `--dump` writes: both
forms, the one kernel in one dispatch and in 8, give the reference's weights, values and gradients within 2.5e-7 and
step losses within 1.3e-7, and the one kernel's weights equal the two kernels' on the GPU, bit for bit. SwiftShader runs
on the CPU, so its timings (2.9 s for 257 dispatches, 4.5 s for one) say nothing of a GPU's. In principle the one kernel
wins for small networks and batches, where launching hundreds of dispatches costs more than the work, and the two
kernels for large batches, which they spread over the whole GPU.

## Performance

The hot path is the per-round bookkeeping over every evaluation scenario, so it is one fused kernel
(`ScenarioSet::refresh`): a parallel pass computes each scenario's pooled sample, payoff noise and bias term, τ²_b is
reduced from them in index order, and a second pass computes the blend, its variance, the gain, the selection with its
standard error and the value into flat arrays that every statistic reads. It is incremental: only scenarios a batch
reached, or whose bin's τ² changed, are recomputed. Further: VaR's urgency skips the normal CDF where the result cannot
exceed 1 (d > z·se), the misclassification error skips it where it underflows to exactly 0 (g > 40·se), the error-budget
rule pops a heap only as far as the excess needs instead of sorting every candidate, proxy predictions reuse their
buffers, and the proxy fit accumulates its normal equations and HC0 meat in fixed chunks of 4,096 rows in parallel,
added in chunk order. Each step was checked to give the same output as before it, and the output is identical on any
number of threads. The validation's engine is a fused path kernel too (both normals of each Box-Muller draw, the
constants once a batch). `ofm_anmc --reps 3`: 6.2 s before, 3.0 s with the fused kernels, 2.5 s with the spin pool
(4 cores).

**Parallelism without locks or blocking.** The kernels run on a pool started once per estimate (`detail::SpinPool`):
no mutex, condition variable or join inside a kernel. A task lives in one atomic 64-bit word (generation, chunk count,
next chunk); the caller writes the task into the slot of its generation and publishes the word, the workers spin-wait
(the CPU's pause hint, a yield after 2,048 spins) for a new generation and claim chunks by compare-exchange on the
word, so a claim succeeds only against the live task and a late worker can never read a slot being rewritten. Finished
chunks are counted in an atomic the caller spin-waits on (it claims chunks too); the first exception is kept by
compare-exchange and rethrown in the caller. The output is bit for bit the same as before the pool and on 1, 3 and 4
threads, and ThreadSanitizer finds no race in the whole validation or after a simulator that throws.

CI runs `ofm_anmc --check` (one repetition, deterministic): the adaptive error must stay within 6% (VaR), 3% (ES) and
3% (EPE), and refinement must cut the regression-only error for VaR at degree 1 and for ES at every degree. It also
runs `ofm_anmc --kernels --check` (the bounds above), `ofm_anmc --basket 10 --check` (10 assets, one repetition: the
differential network's regression-only VaR and ES errors below half the values-only network's, its adaptive errors
within 3% (VaR, ES) and 8% (EPE), and the fused kernel's errors within 0.25 points of the double-precision network's),
and the WebGPU test.

## Limitations

The differential network's estimation variance leaves out its hidden layers' uncertainty, and without it the loop
trusts the network too far (above). Its kernels have a GPU hook (`diffnetOffload()`) but no back end sets it yet: the
Lab's webgpu back end runs the strategy's Monte Carlo only. The kernels take at most 32 inputs, a width of 32 and 4
hidden layers.

As in the document: τ² is constant within a bin of proxy values; VaR has no standard error; ES runs slightly low when
the proxy ranks the tail imperfectly; quantiles of noisy estimates carry the usual nested-simulation bias; one horizon
per run. Not yet wired into the strategy engine: its nested Monte Carlo (include/strategy/montecarlo.hpp) chooses the
book on simulated futures; a natural first use is the book's VaR and ES at the next reset, with the engine's own
simulator as the inner restart.
