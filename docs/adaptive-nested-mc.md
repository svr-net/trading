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
| 2 proxy | OLS on Hermite polynomials (total degree ≤ D, ≤ 2,000 functions) of the standardized state, tiny relative ridge, Cholesky; HC0 sandwich variance v(x) for every prediction | `HermiteBasis`, `FittedProxy` |
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
constants once a batch). `ofm_anmc --reps 3`: 6.2 s before, 3.0 s now (4 cores).

CI runs `ofm_anmc --check` (one repetition, deterministic): the adaptive error must stay within 6% (VaR), 3% (ES) and
3% (EPE), and refinement must cut the regression-only error for VaR at degree 1 and for ES at every degree.

## Limitations

As in the document: τ² is constant within a bin of proxy values; VaR has no standard error; ES runs slightly low when
the proxy ranks the tail imperfectly; quantiles of noisy estimates carry the usual nested-simulation bias; one horizon
per run. Not yet wired into the strategy engine: its nested Monte Carlo (include/strategy/montecarlo.hpp) chooses the
book on simulated futures; a natural first use is the book's VaR and ES at the next reset, with the engine's own
simulator as the inner restart.
