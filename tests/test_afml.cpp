#include <algorithm>
#include <cmath>
#include <numeric>

#include "sat/adaptive/experiment.hpp"
#include "sat/afml/backtest_stats.hpp"
#include "sat/afml/bars.hpp"
#include "sat/afml/bet_sizing.hpp"
#include "sat/afml/features.hpp"
#include "sat/afml/fracdiff.hpp"
#include "sat/afml/importance.hpp"
#include "sat/afml/labeling.hpp"
#include "sat/afml/microstructure.hpp"
#include "sat/afml/overfitting.hpp"
#include "sat/afml/portfolio.hpp"
#include "sat/afml/sampling.hpp"
#include "sat/core/random.hpp"
#include "sat/core/stats.hpp"
#include "sat/data/synthetic_market.hpp"
#include "sat/gpu/fused_backtest.hpp"
#include "test_framework.hpp"

using namespace sat;
using namespace sat::afml;

TEST(normal_distribution_helpers) {
  CHECK_NEAR(normalCdf(0.0), 0.5, 1e-15);
  CHECK_NEAR(normalCdf(1.959963984540054), 0.975, 1e-12);
  for (double p : {1e-6, 0.01, 0.2, 0.5, 0.9, 0.999999}) CHECK_NEAR(normalCdf(normalQuantile(p)), p, 1e-12 * std::max(1.0, 1 / p));
  std::vector<double> x = {1, 2, 3, 4, 100};
  CHECK(skewness(x) > 1.0 && kurtosis(x) > 3.0);
  // OLS recovers a line.
  std::vector<std::vector<double>> X;
  std::vector<double> y, se;
  for (int k = 0; k < 50; ++k) {
    X.push_back({1.0, static_cast<double>(k)});
    y.push_back(2.0 + 0.5 * k + (k % 2 ? 0.01 : -0.01));
  }
  const auto b = ols(X, y, &se);
  CHECK_NEAR(b[0], 2.0, 0.01);
  CHECK_NEAR(b[1], 0.5, 0.001);
  CHECK(se[1] > 0 && se[1] < 1e-3);
}

TEST(information_bars_are_closer_to_normal_than_time_bars) {
  TradeStreamSpec s;
  s.days = 120;
  const auto trades = generateTrades(s);
  CHECK(trades.size() > 100000);
  for (std::size_t k = 1; k < trades.size(); ++k) CHECK(trades[k].time >= trades[k - 1].time);
  double dollars = 0;
  for (const auto& t : trades) dollars += t.price * t.volume;
  const std::size_t perDay = 20;
  const auto time = timeBars(trades, 1.0 / perDay);
  const auto ticks = tickBars(trades, trades.size() / (s.days * perDay));
  const auto dollar = dollarBars(trades, dollars / (s.days * perDay));
  const auto imb = tickImbalanceBars(trades, 50);
  CHECK(std::fabs(static_cast<double>(time.size()) - s.days * perDay) < 5);
  CHECK(std::fabs(static_cast<double>(dollar.size()) - s.days * perDay) < 0.1 * s.days * perDay);
  CHECK(imb.size() > 100);
  const auto st = barStatistics(time, s.days), sd = barStatistics(dollar, s.days), sk = barStatistics(ticks, s.days);
  // Activity clustering fattens the tails of clock-time returns; trade-based sampling undoes it.
  CHECK(st.jarqueBera > 5 * sd.jarqueBera);
  CHECK(st.jarqueBera > 5 * sk.jarqueBera);
  CHECK(sd.barsPerDaySd > st.barsPerDaySd);  // dollar bars follow activity
  // Bars conserve volume.
  double v = 0;
  for (const auto& b : tickBars(trades, 100)) v += b.volume;
  double all = 0;
  for (std::size_t k = 0; k < trades.size() / 100 * 100; ++k) all += trades[k].volume;
  CHECK_NEAR(v, all, 1e-6 * all);
  CHECK_THROWS(dollarBars(trades, 0));
}

TEST(fractional_differentiation) {
  // d = 1 is the first difference and d = 0 the identity.
  const auto w1 = fracDiffWeights(1.0);
  CHECK(w1.size() == 2 && w1[0] == 1.0 && w1[1] == -1.0);
  CHECK(fracDiffWeights(0.0).size() == 1);
  const auto w = fracDiffWeights(0.5, 1e-3);
  CHECK_NEAR(w[1], -0.5, 1e-15);
  CHECK_NEAR(w[2], -0.125, 1e-15);
  // A random walk is not stationary; its first difference is; some d in between suffices
  // while keeping most of the correlation with the level.
  Rng rng(1);
  std::vector<double> walk(1500);
  double x = 0;
  for (auto& v : walk) v = x += rng.normal();
  CHECK(!adfTest(walk).stationaryAt5());
  const auto diffed = fracDiff(walk, 1.0);
  CHECK(std::isnan(diffed[0]) && std::fabs(diffed[5] - (walk[5] - walk[4])) < 1e-12);
  CHECK(adfTest(std::vector<double>(diffed.begin() + 1, diffed.end())).stationaryAt5());
  const auto scan = scanFracDiff(walk, 0.1, 1e-3);
  CHECK(scan.d.size() == 11 && scan.minimumD > 0.0 && scan.minimumD <= 1.0);
  CHECK(scan.correlation[0] > 0.999);
  CHECK(scan.adf.back() < AdfResult::critical5);
  CHECK_THROWS(adfTest(std::vector<double>(10, 1.0)));
}

TEST(cusum_and_triple_barrier) {
  std::vector<double> lp = {0, 0.01, 0.02, 0.03, 0.02, 0.0, -0.02, -0.03};
  const auto ev = cusumFilter(lp, 0.025);  // +3% by day 3, then -3% by day 5 and again by day 7
  CHECK(ev.size() == 3 && ev[0] == 3 && ev[1] == 5 && ev[2] == 7);
  // Up 1% a day: the upper barrier at 2.5% is hit on day 3.
  std::vector<double> close = {100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 99};
  std::vector<double> trg(close.size(), 0.025);
  BarrierSpec b;
  b.profitTaking = b.stopLoss = 1.0;
  b.maxHolding = 5;
  const auto up = tripleBarrier(close, {0}, trg, b);
  CHECK(up.size() == 1 && up[0].t1 == 3 && up[0].barrier == 1 && up[0].label == 1);
  const auto down = tripleBarrier(close, {5}, trg, b);
  CHECK(down[0].barrier == -1 && down[0].label == -1 && down[0].t1 == 8);
  // Vertical barrier: a narrow window that never reaches 2.5%.
  b.maxHolding = 1;
  const auto vert = tripleBarrier(close, {0}, trg, b);
  CHECK(vert[0].barrier == 0 && vert[0].t1 == 1 && vert[0].label == 1);
  b.zeroOnVertical = true;
  CHECK(tripleBarrier(close, {0}, trg, b)[0].label == 0);
  // Meta-labels: a short from day 0 loses, from day 5 wins.
  b.maxHolding = 5;
  b.zeroOnVertical = false;
  const std::vector<int> sides = {-1, -1};
  const auto meta = tripleBarrier(close, {0, 5}, trg, b, &sides);
  CHECK(meta[0].label == 0 && meta[0].barrier == -1 && meta[1].label == 1 && meta[1].barrier == 1);
  // Events too close to the end are dropped.
  CHECK(tripleBarrier(close, {10}, trg, b).empty());
  const auto vol = ewmVolatility(close, 5);
  CHECK(std::isnan(vol[0]) && vol[3] >= 0 && std::isfinite(vol[11]));
}

TEST(uniqueness_and_sequential_bootstrap) {
  // Three labels: [0,2], [2,3], [4,5] on 6 dates.
  const std::vector<Span> spans = {{0, 2}, {2, 3}, {4, 5}};
  const auto c = concurrency(spans, 6);
  CHECK(c[0] == 1 && c[2] == 2 && c[3] == 1 && c[5] == 1);
  const auto u = averageUniqueness(spans, c);
  CHECK_NEAR(u[0], (1 + 1 + 0.5) / 3.0, 1e-12);
  CHECK_NEAR(u[1], (0.5 + 1) / 2.0, 1e-12);
  CHECK_NEAR(u[2], 1.0, 1e-12);
  const auto d = timeDecay(u, 0.5);
  CHECK(d.back() == 1.0 && d[0] < d[1] && d[0] >= 0.5);
  CHECK(timeDecay(u, -0.5)[0] == 0.0);
  const auto aw = returnAttributionWeights(spans, c, {0, 0.01, 0.02, -0.01, 0.0, 0.03});
  CHECK_NEAR(std::accumulate(aw.begin(), aw.end(), 0.0), 3.0, 1e-12);
  // Random overlapping spans: the sequential bootstrap draws more unique samples.
  Rng g(11), rng(1), rng2(2);
  double seq = 0, plain = 0;
  for (int rep = 0; rep < 100; ++rep) {
    std::vector<Span> many;
    for (int k = 0; k < 40; ++k) {
      const std::size_t t0 = g.below(100);
      many.push_back({t0, t0 + g.below(10)});
    }
    seq += sampleUniqueness(many, sequentialBootstrap(many, 110, 40, rng), 110);
    std::vector<std::size_t> s(40);
    for (auto& k : s) k = rng2.below(many.size());
    plain += sampleUniqueness(many, s, 110);
  }
  CHECK(seq > plain * 1.04);
}

TEST(purged_cross_validation_removes_leakage) {
  std::vector<Span> spans;
  for (std::size_t t = 0; t < 100; ++t) spans.push_back({t, std::min<std::size_t>(t + 5, 99)});
  const auto folds = purgedKFold(spans, 100, 5, 2);
  CHECK(folds.size() == 5);
  for (const auto& f : folds) {
    std::size_t lo = 1000, hi = 0;
    for (std::size_t k : f.test) {
      lo = std::min(lo, spans[k].first);
      hi = std::max(hi, spans[k].second);
    }
    for (std::size_t k : f.train) {
      CHECK(!(spans[k].first <= hi && spans[k].second >= lo));        // no overlap
      CHECK(!(spans[k].first > hi && spans[k].first <= hi + 2));       // embargo
    }
    CHECK(f.test.size() == 20);
  }
  const auto plain = purgedKFold(spans, 100, 5, 0, false);
  CHECK(plain[2].train.size() == 80 && folds[2].train.size() < 80);
  const auto c = combinatorialPurgedSplits(spans, 100, 6, 2, 1);
  CHECK(c.splits.size() == 15 && c.paths == 5);
  // Every path covers every group exactly once, with a split that tests that group.
  for (const auto& path : c.pathSplit)
    for (std::size_t g = 0; g < 6; ++g) {
      const auto& groups = c.testGroupsOf[path[g]];
      CHECK(std::find(groups.begin(), groups.end(), g) != groups.end());
    }
  CHECK_THROWS(combinatorialPurgedSplits(spans, 100, 4, 4, 0));
}

TEST(feature_importance_and_cv) {
  // Feature 0 drives the label, feature 1 is noise.
  Rng rng(2);
  const std::size_t n = 1200;
  Matrix X(n, 2);
  std::vector<double> y(n);
  std::vector<Span> spans;
  for (std::size_t r = 0; r < n; ++r) {
    X(r, 0) = rng.normal();
    X(r, 1) = rng.normal();
    y[r] = X(r, 0) + 0.5 * rng.normal() > 0 ? 1 : 0;
    spans.push_back({r, r});
  }
  const auto splits = purgedKFold(spans, n, 4, 0);
  ModelSpec m;
  m.type = "xgboost";
  m.trees = 30;
  const auto cv = crossValidate(m, X, y, splits);
  CHECK(cv.accuracy > 0.75 && cv.foldAccuracy.size() == 4);
  const auto mdi = meanDecreaseImpurity(m, X, y);
  CHECK(mdi.mean[0] > mdi.mean[1]);
  const auto mda = meanDecreaseAccuracy(m, X, y, splits);
  CHECK(mda.mean[0] > 0.1 && std::fabs(mda.mean[1]) < 0.05);
  ModelSpec lr;
  lr.type = "logistic";
  const auto sfi = singleFeatureImportance(lr, X, y, splits);
  CHECK(sfi.mean[0] > 0.8 && std::fabs(sfi.mean[1] - 0.5) < 0.1);
}

TEST(bet_sizing) {
  CHECK_NEAR(betSize(0.5), 0.0, 1e-12);
  CHECK(betSize(0.6) > 0 && betSize(0.4) < 0);
  CHECK_NEAR(betSize(0.6), -betSize(0.4), 1e-12);
  CHECK(betSize(0.9) > betSize(0.6) && betSize(0.99) < 1.0 && betSize(0.999999) <= 1.0);
  CHECK_NEAR(betSize(0.6), 2 * normalCdf(0.1 / std::sqrt(0.24)) - 1, 1e-12);
  CHECK(discretizeBet(0.37, 0.25) == 0.25 && discretizeBet(0.9, 0.25) == 1.0);
  // The bet-sized rule: gross exposure 1, sign of the forecast.
  const double p[4] = {0.7, 0.45, 0.5, 0.2};
  const auto rank = rankRow(p, 4);
  double w[4];
  strategyWeights({StrategyKind::BetSized, 0.0, 1}, p, rank.data(), 4, w);
  CHECK_NEAR(std::fabs(w[0]) + std::fabs(w[1]) + std::fabs(w[2]) + std::fabs(w[3]), 1.0, 1e-12);
  CHECK(w[0] > 0 && w[1] < 0 && w[2] == 0 && w[3] < w[1]);
  strategyWeights({StrategyKind::BetSized, 0.2, 1}, p, rank.data(), 4, w);
  CHECK(w[1] == 0.0);  // |size| of 0.45 is below 0.2
  CHECK(parseStrategyKind("betsize") == StrategyKind::BetSized);
}

TEST(sharpe_ratio_statistics) {
  // PSR: same Sharpe ratio, more observations, more confidence; fat tails, less.
  CHECK(probabilisticSharpe(0.1, 0.0, 1000, 0, 3) > probabilisticSharpe(0.1, 0.0, 100, 0, 3));
  CHECK(probabilisticSharpe(0.1, 0.0, 250, -1, 10) < probabilisticSharpe(0.1, 0.0, 250, 0, 3));
  CHECK_NEAR(probabilisticSharpe(0.1, 0.1, 250, 0, 3), 0.5, 1e-12);
  // The expected maximum grows with the number of trials; DSR <= PSR.
  CHECK(expectedMaxSharpe(100, 0.01) > expectedMaxSharpe(10, 0.01));
  CHECK(expectedMaxSharpe(1, 0.01) == 0.0);
  CHECK(deflatedSharpe(0.1, 250, 0, 3, 100, 0.002) < probabilisticSharpe(0.1, 0.0, 250, 0, 3));
  // Drawdowns and concentration.
  const auto dd = drawdownStats({0.1, -0.5, 0.2, 0.2, 0.5, -0.1});
  CHECK_NEAR(dd.maxDrawdown, 0.5, 1e-12);
  CHECK(dd.episodes == 2 && dd.longestUnderWater == 3);
  CHECK_NEAR(returnConcentration({0.01, 0.01, 0.01, -0.02}), 0.0, 1e-12);
  CHECK(returnConcentration({0.01, 0.0001, 0.0001, 0.5}) > 0.9);
}

TEST(probability_of_backtest_overfitting) {
  // Pure noise: the in-sample winner is a coin flip out of sample, PBO near 1/2.
  Rng rng(6);
  Matrix noise(800, 40);
  for (auto& v : noise.data()) v = 0.01 * rng.normal();
  const auto pn = probabilityOfBacktestOverfitting(noise, 10);
  CHECK(pn.combinations == 252);
  CHECK(pn.pbo > 0.25 && pn.pbo < 0.75);
  // One strategy with real skill: it wins in and out of sample, PBO near 0.
  Matrix skill = noise;
  for (std::size_t t = 0; t < skill.rows(); ++t) skill(t, 7) += 0.004;
  const auto ps = probabilityOfBacktestOverfitting(skill, 10);
  CHECK(ps.pbo < 0.05 && ps.probabilityOfLoss < 0.05);
  CHECK_THROWS(probabilityOfBacktestOverfitting(noise, 7));
}

TEST(hierarchical_risk_parity) {
  // Two blocks of correlated assets.
  Rng rng(8);
  Matrix R(500, 6);
  for (std::size_t t = 0; t < 500; ++t) {
    const double a = rng.normal(), b = rng.normal();
    for (std::size_t i = 0; i < 6; ++i) R(t, i) = 0.01 * (i < 3 ? 1.0 : 2.0) * ((i % 2 ? a : b) * 0.0 + (i < 3 ? a : b) * 0.8 + 0.6 * rng.normal());
  }
  const Matrix cov = covarianceMatrix(R), corr = correlationFromCovariance(cov);
  CHECK_NEAR(corr(0, 0), 1.0, 1e-12);
  const auto L = clusterAssets(corr);
  CHECK(L.order.size() == 6 && L.height.size() == 5);
  // Quasi-diagonalisation keeps each block together.
  std::size_t firstBlock = 0;
  for (std::size_t k = 0; k < 3; ++k) firstBlock += L.order[k] < 3;
  CHECK(firstBlock == 0 || firstBlock == 3);
  const auto h = hierarchicalRiskParity(cov, L.order), ivp = inverseVarianceWeights(cov), mv = minimumVarianceWeights(cov);
  for (const auto* w : {&h, &ivp, &mv}) CHECK_NEAR(std::accumulate(w->begin(), w->end(), 0.0), 1.0, 1e-9);
  for (double v : h) CHECK(v > 0);
  // The low-volatility block gets more weight.
  CHECK(h[0] + h[1] + h[2] > h[3] + h[4] + h[5]);
  CHECK(portfolioVariance(cov, mv) <= portfolioVariance(cov, h) + 1e-12);
  AllocationTrial trial;
  trial.trials = 30;
  const auto cmp = compareAllocations(trial);
  CHECK(cmp.hrpVariance.size() == 30 && mean(cmp.hrpVariance) > 0);
}

TEST(microstructure_features) {
  // Bid-ask bounce: trades at the bid or ask of a constant mid, at random. Roll's estimate
  // 2 sqrt(-cov) recovers the spread of 0.1 (0.1% of the price).
  const std::size_t T = 2000;
  Rng rng(12);
  Panel close(T, 1), high(T, 1), low(T, 1), vol(T, 1, 1000.0);
  for (std::size_t t = 0; t < T; ++t) {
    close(t, 0) = 100.0 + (rng.uniform() < 0.5 ? 0.05 : -0.05);
    high(t, 0) = 100.1;
    low(t, 0) = 99.9;
  }
  const Panel roll = rollSpread(close, 1500);
  CHECK(std::isnan(roll(5, 0)));
  CHECK_NEAR(roll(1990, 0), 0.1 / 100.0, 1e-4);
  const Panel cs = corwinSchultzSpread(high, low, 10);
  CHECK(cs(30, 0) >= 0 && std::isfinite(cs(30, 0)));
  const Panel am = amihudIlliquidity(close, vol, 10);
  CHECK(am(30, 0) > 0);
  CHECK(std::isfinite(kyleLambda(close, vol, 10)(30, 0)));
  CHECK_THROWS(rollSpread(close, 2));
}

TEST(pipeline_with_afml_options) {
  SyntheticMarketSpec ms;
  ms.numAssets = 12;
  ms.numDates = 560;
  const MarketData d = generateSyntheticMarket(ms);
  ExperimentSpec spec;
  spec.alphaIds = {2, 12, 33, 41};
  spec.extraFeatures = {"ffd", "vol", "roll", "corwin", "amihud", "kyle"};
  spec.label.kind = LabelKind::TripleBarrier;
  spec.label.horizon = 5;
  spec.cusumMultiple = 1.0;
  spec.walkForward.trainWindow = 250;
  spec.walkForward.weighting = SampleWeighting::UniquenessDecay;
  spec.models.resize(1);
  spec.models[0].type = "logistic";
  spec.strategies = {{StrategyKind::BetSized, 0.0, 1}, {StrategyKind::LongShort, 3, 1}};
  const PredictionSet p = runPredictions(d, spec);
  CHECK(p.features.size() == 10 && p.features.names[4] == "ffd");
  CHECK(p.features.warmup >= 55);
  CHECK(p.trainMask.countFinite() == d.numDates() * d.numAssets());
  double events = 0;
  for (double v : p.trainMask.data()) events += v;
  CHECK(events > 100 && events < 0.6 * d.numDates() * d.numAssets());
  // Triple-barrier label ends lie within the vertical barrier.
  for (std::size_t t = 0; t < d.numDates(); ++t)
    for (std::size_t i = 0; i < d.numAssets(); ++i)
      if (std::isfinite(p.labels(t, i))) CHECK(p.labelEnds(t, i) > t && p.labelEnds(t, i) <= t + 5);
  CHECK(p.models[0].oos.count > 500);
  const Experiment e = runStrategies(p, spec);
  const auto rep = assessOverfitting(e.book, e.adaptive.net, e.evalFrom, 8);
  CHECK(rep.trials == 2 && rep.pbo.combinations == 70);
  CHECK(rep.adaptive.dsr >= 0 && rep.adaptive.dsr <= 1 && rep.best.psr >= rep.best.dsr - 1e-12);
  // The GPU kernels price the bet-sized rule like the CPU.
  const auto plan = gpu::compile(p.models, spec.strategies, p.nextReturns, spec.costBps, {}, 10);
  const auto g = gpu::summarise(plan, gpu::runFusedReference(plan));
  const CandidateBook book(p.models, spec.strategies, p.nextReturns, spec.costBps);
  const auto cpu = evaluatePerformance(book.netSeries(0, 10), book.turnoverSeries(0, 10));
  CHECK_NEAR(g.candidates[0].sharpe, cpu.sharpe, 1e-3);
  CHECK_NEAR(g.candidates[0].averageTurnover, cpu.averageTurnover, 1e-4);
}
