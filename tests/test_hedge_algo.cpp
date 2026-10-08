#include <algorithm>
#include <cmath>
#include <numeric>

#include "sat/algo/execution.hpp"
#include "sat/algo/pairs.hpp"
#include "sat/algo/regimes.hpp"
#include "sat/algo/trend.hpp"
#include "sat/core/random.hpp"
#include "sat/core/stats.hpp"
#include "sat/data/synthetic_market.hpp"
#include "sat/hedge/hedging.hpp"
#include "sat/hedge/options.hpp"
#include "sat/strategy/performance.hpp"
#include "test_framework.hpp"

using namespace sat;

namespace {

double variance(const std::vector<double>& x) {
  const double s = stdev(x);
  return s * s;
}

// Equal-weight daily market return of a synthetic market (row 0 skipped).
std::vector<double> marketReturns(const MarketData& m) {
  const Panel r = m.returns();
  std::vector<double> out;
  for (std::size_t t = 1; t < r.dates(); ++t) {
    double s = 0;
    for (std::size_t i = 0; i < r.assets(); ++i) s += r(t, i);
    out.push_back(s / static_cast<double>(r.assets()));
  }
  return out;
}

}  // namespace

TEST(hedge_minimum_variance_ratio_reduces_variance) {
  Rng rng(3);
  std::vector<double> x(2000), y(2000);
  for (std::size_t t = 0; t < x.size(); ++t) {
    x[t] = 0.01 * rng.normal();
    y[t] = 1.3 * x[t] + 0.005 * rng.normal();
  }
  const double h = hedge::minimumVarianceHedgeRatio(y, x);
  CHECK_NEAR(h, 1.3, 0.05);
  const auto hedged = hedge::applyHedge(y, x, std::vector<double>(x.size(), h));
  CHECK(variance(hedged) < 0.2 * variance(y));
  const auto rb = hedge::rollingBeta(y, x, 60);
  CHECK(std::isnan(rb[10]));  // no estimate before a full window
  CHECK_NEAR(rb[1500], 1.3, 0.25);
}

TEST(hedge_kalman_beta_tracks_a_break) {
  Rng rng(5);
  std::vector<double> x(1200), y(1200);
  for (std::size_t t = 0; t < x.size(); ++t) {
    x[t] = 0.01 * rng.normal();
    y[t] = (t < 600 ? 0.5 : 1.5) * x[t] + 0.002 * rng.normal();
  }
  const auto k = hedge::kalmanRegression(y, x, 1e-3, 4e-6);
  CHECK_NEAR(k.beta[590], 0.5, 0.2);
  CHECK_NEAR(k.beta[1190], 1.5, 0.2);
  // One-step-ahead: the estimate used at t is formed before y[t] is seen.
  CHECK(k.beta.size() == y.size() && k.forecastError.size() == y.size());
}

TEST(hedge_volatility_target_hits_target) {
  Rng rng(9);
  std::vector<double> r(3000);
  for (std::size_t t = 0; t < r.size(); ++t) r[t] = (t % 1000 < 500 ? 0.005 : 0.02) * rng.normal();
  std::vector<double> lev;
  const auto scaled = hedge::volatilityTarget(r, 0.10, 20, 10.0, &lev);
  std::vector<double> tail(scaled.begin() + 100, scaled.end());
  CHECK_NEAR(stdev(tail) * std::sqrt(252.0), 0.10, 0.025);
  CHECK(*std::max_element(lev.begin(), lev.end()) <= 10.0 + 1e-12);
  const auto kelly = hedge::kellyScale(r, 252, 0.5, 2.0);
  CHECK(kelly.size() == r.size());
}

TEST(options_black_scholes_parity_and_delta) {
  const double S = 100, K = 95, T = 0.5, r = 0.03, v = 0.25;
  const double c = hedge::blackScholes(S, K, T, r, v, true), p = hedge::blackScholes(S, K, T, r, v, false);
  CHECK_NEAR(c - p, S - K * std::exp(-r * T), 1e-10);
  const double h = 1e-4;
  const double fd = (hedge::blackScholes(S + h, K, T, r, v, true) - hedge::blackScholes(S - h, K, T, r, v, true)) / (2 * h);
  CHECK_NEAR(hedge::blackScholesDelta(S, K, T, r, v, true), fd, 1e-6);
  CHECK_NEAR(hedge::blackScholesDelta(S, K, T, r, v, true) - hedge::blackScholesDelta(S, K, T, r, v, false), 1.0, 1e-12);
}

TEST(options_delta_hedge_error_shrinks_with_frequency) {
  hedge::DeltaHedgeSpec s;
  s.paths = 600;
  const auto fine = hedge::simulateDeltaHedge(s, 1), coarse = hedge::simulateDeltaHedge(s, 32);
  CHECK(fine.sd < 0.5 * coarse.sd);
  CHECK(std::fabs(fine.mean) < 0.05);  // fair-priced and hedged: no edge either way
  // Hedging error scales roughly as the square root of the rebalancing interval.
  CHECK_NEAR(coarse.sd / fine.sd, std::sqrt(32.0), 2.5);
}

TEST(options_protective_put_cuts_drawdown) {
  SyntheticMarketSpec ms;
  ms.numDates = 1500;
  const auto m = generateSyntheticMarket(ms);
  const auto r = marketReturns(m);
  std::vector<double> price{100.0};
  for (double x : r) price.push_back(price.back() * (1 + x));
  const auto o = hedge::optionOverlay(price, {});
  CHECK(o.rolls > 50);
  const auto u = evaluatePerformance(o.unhedged), p = evaluatePerformance(o.protectivePut),
             c = evaluatePerformance(o.collar);
  CHECK(p.maxDrawdown < u.maxDrawdown);
  CHECK(c.annualVolatility < u.annualVolatility);
  CHECK(o.averagePutCost > 0 && o.averageCallIncome > 0);
}

TEST(pairs_cointegration_and_half_life) {
  algo::PairSpec ps;
  ps.days = 2000;
  std::vector<double> y, x;
  algo::generatePair(ps, y, x);
  const auto c = algo::engleGranger(y, x);
  CHECK(c.cointegrated());
  CHECK_NEAR(c.hedgeRatio, ps.beta, 0.15);
  // The estimate matches the half-life of the true spread in this sample, and the
  // generating half-life within sampling error.
  std::vector<double> spread(y.size());
  for (std::size_t t = 0; t < y.size(); ++t) spread[t] = y[t] - ps.beta * x[t] - 5.0;
  CHECK_NEAR(c.halfLife, algo::halfLife(spread), 0.1 * algo::halfLife(spread));
  CHECK_NEAR(c.halfLife, ps.halfLifeDays, 0.6 * ps.halfLifeDays);

  // Two independent random walks are not cointegrated (in the vast majority of draws).
  int rejected = 0;
  for (std::uint64_t seed = 1; seed <= 20; ++seed) {
    Rng rng(seed);
    std::vector<double> a{0}, b{0};
    for (int t = 1; t < 1000; ++t) {
      a.push_back(a.back() + rng.normal());
      b.push_back(b.back() + rng.normal());
    }
    rejected += algo::engleGranger(a, b).cointegrated();
  }
  CHECK(rejected <= 3);
}

TEST(pairs_kalman_strategy_profits_on_a_cointegrated_pair) {
  algo::PairSpec ps;
  ps.days = 1500;
  std::vector<double> y, x;
  algo::generatePair(ps, y, x);
  const auto res = algo::kalmanPairs(y, x, {});
  CHECK(res.trades > 10);
  CHECK(evaluatePerformance(res.returns).sharpe > 1.0);
  CHECK_NEAR(res.beta.back(), ps.beta, 0.3);
}

TEST(trend_weights_and_scaling) {
  SyntheticMarketSpec ms;
  ms.numAssets = 10;
  ms.numDates = 1000;
  const auto m = generateSyntheticMarket(ms);
  algo::TrendSpec ts;
  const auto res = algo::trendFollowing(m.close, ts);
  CHECK(res.weights.size() == ts.rules.size());
  for (std::size_t k = 0; k < res.weights[0].size(); k += 37) {
    double s = 0;
    for (const auto& w : res.weights) s += w[k];
    CHECK_NEAR(s, 1.0, 1e-9);
  }
  // Forecast scaling targets an average absolute forecast of 10, before the cap.
  const double avg = mean(res.combinedForecastMean);
  CHECK(avg > 4 && avg < 16);
  std::vector<double> tail(res.returns.begin() + static_cast<std::ptrdiff_t>(res.start), res.returns.end());
  const double vol = stdev(tail) * std::sqrt(252.0);
  CHECK(vol > 0.05 && vol < 0.35);
}

TEST(regimes_hmm_recovers_volatility_states) {
  SyntheticMarketSpec ms;
  ms.numDates = 2500;
  const auto m = generateSyntheticMarket(ms);
  const auto r = marketReturns(m);
  const auto model = algo::fitHmm(r, 2);
  CHECK(model.sd[1] > 1.5 * model.sd[0]);
  for (const auto& row : model.transition) CHECK_NEAR(std::accumulate(row.begin(), row.end(), 0.0), 1.0, 1e-9);
  const auto f = algo::filterHmm(model, r);
  std::size_t hit = 0;
  for (std::size_t t = 0; t < r.size(); ++t) hit += (f[t][1] > 0.5) == (m.regime[t + 1] == 1);
  CHECK(static_cast<double>(hit) / static_cast<double>(r.size()) > 0.75);

  algo::RegimeSwitchSpec rs;
  const auto sw = algo::regimeSwitch(r, rs);
  CHECK(sw.returns.size() == r.size() - sw.start && sw.exposure.size() == sw.returns.size());
  const std::vector<double> a(r.begin() + static_cast<std::ptrdiff_t>(sw.start), r.end());
  CHECK(evaluatePerformance(sw.returns).maxDrawdown < evaluatePerformance(a).maxDrawdown);
}

TEST(execution_almgren_chriss) {
  algo::ExecutionSpec s;
  s.riskAversion = 0.0;
  const auto twap = algo::almgrenChriss(s);
  for (double n : twap.trades) CHECK_NEAR(n, s.shares / static_cast<double>(s.periods), 1e-6);
  s.riskAversion = 1e-6;
  const auto ac = algo::almgrenChriss(s);
  CHECK(ac.trades.front() > ac.trades.back());  // risk aversion front-loads the trading
  CHECK_NEAR(std::accumulate(ac.trades.begin(), ac.trades.end(), 0.0), s.shares, 1e-6);
  CHECK(ac.expectedCost > twap.expectedCost && ac.variance < twap.variance);

  // Frontier: higher risk aversion buys lower risk at higher expected cost.
  const auto fr = algo::efficientFrontier(s, 1e-8, 1e-4, 12);
  for (std::size_t i = 1; i < fr.size(); ++i) {
    CHECK(fr[i].expectedCost >= fr[i - 1].expectedCost - 1e-6);
    CHECK(fr[i].sd <= fr[i - 1].sd + 1e-6);
  }

  // Monte Carlo agrees with the closed form.
  const auto sim = algo::simulateShortfall(s, ac.holdings, 20000, 4);
  CHECK_NEAR(mean(sim), ac.expectedCost, 0.03 * ac.expectedCost + 3 * std::sqrt(ac.variance / 20000.0));
  CHECK_NEAR(stdev(sim), std::sqrt(ac.variance), 0.03 * std::sqrt(ac.variance));
  // The optimum minimises E + lambda Var among the three schedules.
  double ce, cv;
  algo::scheduleCostVariance(s, twap.holdings, ce, cv);
  CHECK(ac.expectedCost + s.riskAversion * ac.variance <= ce + s.riskAversion * cv);
}
