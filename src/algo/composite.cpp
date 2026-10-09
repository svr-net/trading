#include "sat/algo/composite.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "sat/adaptive/composite.hpp"
#include "sat/algo/tournament.hpp"
#include "sat/core/stats.hpp"

namespace sat::algo {

SignalWeighting parseSignalWeighting(const std::string& name) {
  if (name == "equal") return SignalWeighting::Equal;
  if (name == "adaptive") return SignalWeighting::Adaptive;
  throw std::invalid_argument("unknown signal weighting '" + name + "' (equal, adaptive)");
}

std::string signalWeightingName(SignalWeighting w) { return w == SignalWeighting::Adaptive ? "adaptive" : "equal"; }

namespace {

// Cross-sectional percentile ranks in [0, 1] (average ranks for ties); NaN stays NaN.
void percentileRanks(const std::vector<double>& x, std::vector<double>& out) {
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < x.size(); ++i)
    if (std::isfinite(x[i])) idx.push_back(i);
  out.assign(x.size(), Panel::kMissing);
  if (idx.empty()) return;
  if (idx.size() == 1) {
    out[idx[0]] = 0.5;
    return;
  }
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  const double scale = 1.0 / static_cast<double>(idx.size() - 1);
  for (std::size_t k = 0; k < idx.size();) {
    std::size_t j = k;
    while (j + 1 < idx.size() && x[idx[j + 1]] == x[idx[k]]) ++j;
    const double r = 0.5 * static_cast<double>(k + j) * scale;
    for (std::size_t q = k; q <= j; ++q) out[idx[q]] = r;
    k = j + 1;
  }
}

// Pearson correlation of two rank vectors over the entries finite in both.
double rankIc(const std::vector<double>& rank, const std::vector<double>& next) {
  std::vector<double> a, b;
  for (std::size_t i = 0; i < rank.size(); ++i)
    if (std::isfinite(rank[i]) && std::isfinite(next[i])) a.push_back(rank[i]), b.push_back(next[i]);
  if (a.size() < 4) return Panel::kMissing;
  std::vector<double> rb;
  percentileRanks(b, rb);
  const double c = correlation(a, rb);
  return std::isfinite(c) ? c : Panel::kMissing;
}

}  // namespace

MultiSignalResult multiSignalSelection(const MarketData& data, const Panel* probability, const MultiSignalSpec& spec,
                                       std::size_t start, std::size_t end) {
  const std::size_t T = data.numDates(), N = data.numAssets();
  if (end == 0 || end > T - 1) end = T - 1;
  if (start < 1 || start >= end) throw std::invalid_argument("multi-signal selection: empty date range");
  if (!spec.useMl && !spec.useMomentum && !spec.useTrailingSharpe) throw std::invalid_argument("multi-signal selection: no signal selected");
  if (spec.useMl && (!probability || probability->dates() != T || probability->assets() != N))
    throw std::invalid_argument("multi-signal selection: the machine-learning signal needs a probability panel of the market's shape");
  if (spec.holdings < 1 || spec.rebalanceEvery < 1 || spec.volLookback < 2 || spec.momentumLookback <= spec.momentumSkip ||
      spec.sharpeLookback < 2 || spec.icLookback < 2)
    throw std::invalid_argument("multi-signal selection: holdings, rebalance >= 1; look-backs >= 2; momentum look-back > skip");
  if (!(spec.targetVol >= 0) || !(spec.maxLeverage > 0)) throw std::invalid_argument("multi-signal selection: target volatility >= 0, leverage > 0");

  const Panel r = data.returns();
  const Panel next = data.forwardReturns(1);
  MultiSignalResult out;
  out.start = start;
  if (spec.useMl) out.signals.push_back("Machine learning");
  if (spec.useMomentum) out.signals.push_back("Momentum");
  if (spec.useTrailingSharpe) out.signals.push_back("Trailing Sharpe");
  const std::size_t K = out.signals.size();

  // Regime gate from the equal-weight market (exposure known at the close of each date).
  std::vector<double> gate(T, 1.0);
  if (spec.regimeGate) {
    const auto mr = equalWeightMarket(data);
    if (mr.size() > spec.regimes.window + 2) {
      const auto sw = regimeSwitch(mr, spec.regimes);
      for (std::size_t k = 0; k < sw.exposure.size(); ++k) gate[sw.start + k] = sw.exposure[k];
    }
  }

  // Raw signal values of date t.
  auto signalsAt = [&](std::size_t t, std::vector<std::vector<double>>& raw) {
    raw.assign(K, std::vector<double>(N, Panel::kMissing));
    std::size_t k = 0;
    if (spec.useMl) {
      for (std::size_t i = 0; i < N; ++i) raw[k][i] = (*probability)(t, i);
      ++k;
    }
    if (spec.useMomentum) {
      if (t >= spec.momentumLookback)
        for (std::size_t i = 0; i < N; ++i)
          raw[k][i] = data.close(t - spec.momentumSkip, i) / data.close(t - spec.momentumLookback, i) - 1.0;
      ++k;
    }
    if (spec.useTrailingSharpe) {
      if (t >= spec.sharpeLookback)
        for (std::size_t i = 0; i < N; ++i) {
          double s1 = 0, s2 = 0, n = 0;
          for (std::size_t u = t + 1 - spec.sharpeLookback; u <= t; ++u)
            if (std::isfinite(r(u, i))) s1 += r(u, i), s2 += r(u, i) * r(u, i), ++n;
          if (n < 2) continue;
          const double m = s1 / n, v = std::max(0.0, s2 / n - m * m);
          raw[k][i] = v > 0 ? m / std::sqrt(v) : 0.0;
        }
    }
  };
  auto volatility = [&](std::size_t t, std::size_t i) {
    double s1 = 0, s2 = 0, n = 0;
    for (std::size_t u = t >= spec.volLookback ? t + 1 - spec.volLookback : 1; u <= t; ++u)
      if (std::isfinite(r(u, i))) s1 += r(u, i), s2 += r(u, i) * r(u, i), ++n;
    if (n < 2) return Panel::kMissing;
    const double m = s1 / n;
    return std::sqrt(std::max(0.0, s2 / n - m * m));
  };

  // Daily information coefficients, needed for the adaptive weights; warm them up before start.
  std::vector<std::vector<double>> ic(K);
  const std::size_t icFrom = spec.weighting == SignalWeighting::Adaptive ? (start > spec.icLookback ? start - spec.icLookback : 1) : start;

  std::vector<double> sigW(K, 1.0 / static_cast<double>(K));
  std::vector<double> base(N, 0.0), held(N, 0.0), baseHistory;
  double scale = spec.targetVol > 0 ? 0.0 : 1.0;
  std::vector<std::vector<double>> raw, ranks(K);
  for (std::size_t t = icFrom; t < end; ++t) {
    const bool trade = t >= start;
    const bool rebalance = trade && (t - start) % spec.rebalanceEvery == 0;
    if (rebalance || spec.weighting == SignalWeighting::Adaptive) {
      signalsAt(t, raw);
      for (std::size_t k = 0; k < K; ++k) percentileRanks(raw[k], ranks[k]);
    }
    if (rebalance) {
      if (spec.weighting == SignalWeighting::Adaptive) {
        // IC of date u is known once next(u) is, at the close of u + 1 <= t.
        std::vector<double> score(K, 0.0);
        bool enough = true;
        for (std::size_t k = 0; k < K; ++k) {
          std::vector<double> win;
          for (std::size_t q = ic[k].size(); q-- > 0 && win.size() < spec.icLookback;)
            if (std::isfinite(ic[k][q])) win.push_back(ic[k][q]);
          if (win.size() < 20) {
            enough = false;
            break;
          }
          // t-statistic of the mean IC (the floor keeps a constant IC finite)
          const double sd = std::max(0.01, stdev(win));
          score[k] = mean(win) / sd * std::sqrt(static_cast<double>(win.size()));
        }
        if (enough) {
          const double top = *std::max_element(score.begin(), score.end());
          double sum = 0;
          for (std::size_t k = 0; k < K; ++k) sum += (sigW[k] = std::exp(spec.eta * (score[k] - top)));
          for (auto& v : sigW) v /= sum;
        }
      }
      // Blend the ranks and pick the top names.
      std::vector<double> blend(N, Panel::kMissing);
      for (std::size_t i = 0; i < N; ++i) {
        double s = 0, w = 0;
        for (std::size_t k = 0; k < K; ++k)
          if (std::isfinite(ranks[k][i])) s += sigW[k] * ranks[k][i], w += sigW[k];
        if (w > 0 && std::isfinite(next(t, i))) blend[i] = s / w;
      }
      out.lastScore = blend;
      std::vector<std::size_t> order;
      for (std::size_t i = 0; i < N; ++i)
        if (std::isfinite(blend[i])) order.push_back(i);
      std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return blend[a] > blend[b]; });
      if (order.size() > spec.holdings) order.resize(spec.holdings);
      std::fill(base.begin(), base.end(), 0.0);
      double sum = 0;
      for (auto i : order) {
        const double v = volatility(t, i);
        base[i] = v > 0 ? 1.0 / v : 0.0;
        sum += base[i];
      }
      if (sum > 0)
        for (auto& v : base) v /= sum;
      else
        for (auto i : order) base[i] = 1.0 / static_cast<double>(order.size());
      // Volatility target on the book's own realised volatility (known returns only).
      if (spec.targetVol > 0) {
        std::vector<double> recent;
        for (std::size_t q = baseHistory.size(); q-- > 0 && recent.size() < spec.volLookback;) recent.push_back(baseHistory[q]);
        if (recent.size() >= 20) {
          const double vol = stdev(recent) * std::sqrt(252.0);
          scale = vol > 0 ? std::min(spec.maxLeverage, spec.targetVol / vol) : spec.maxLeverage;
        } else {
          scale = std::min(spec.maxLeverage, 1.0);  // no record yet: unlevered
        }
      }
    }
    if (trade) {
      const double exposure = scale * gate[t];
      double gross = 0, turnover = 0, baseRet = 0;
      for (std::size_t i = 0; i < N; ++i) {
        const double w = exposure * base[i];
        turnover += std::fabs(w - held[i]);
        held[i] = w;
        if (std::isfinite(next(t, i))) gross += w * next(t, i), baseRet += base[i] * next(t, i);
      }
      baseHistory.push_back(baseRet);
      out.gross.push_back(gross);
      out.turnover.push_back(turnover);
      out.exposure.push_back(exposure);
      out.returns.push_back(gross - spec.costBps * 1e-4 * turnover);
      out.signalWeights.push_back(sigW);
    }
    if (spec.weighting == SignalWeighting::Adaptive) {
      std::vector<double> nx(N);
      for (std::size_t i = 0; i < N; ++i) nx[i] = next(t, i);
      for (std::size_t k = 0; k < K; ++k) ic[k].push_back(rankIc(ranks[k], nx));
    }
  }
  out.holdings = held;
  out.metrics = evaluatePerformance(out.returns, out.turnover);
  return out;
}

}  // namespace sat::algo

namespace sat::algo {

CompositeStrategyResult runCompositeStrategy(const MarketData& data, const PredictionSet& predictions, const ExperimentSpec& experiment,
                                             const CompositeStrategySpec& spec) {
  if (predictions.models.empty()) throw std::invalid_argument("composite strategy: no model predictions");
  // The composite model: the experiment's own, or the equal-weight average of its models.
  if (!predictions.compositeWeights.empty() || predictions.models.size() == 1)
    return runCompositeStrategy(data, predictions.models.back(), predictions.nextReturns, experiment, spec);
  CompositeSpec cs;
  cs.method = CompositeMethod::Average;
  const auto composite = compositePredictions(predictions.models, predictions.labels, predictions.labelEnds, cs);
  return runCompositeStrategy(data, composite.model, predictions.nextReturns, experiment, spec);
}

CompositeStrategyResult runCompositeStrategy(const MarketData& data, const ModelPredictions& composite, const Panel& nextReturns,
                                             const ExperimentSpec& experiment, const CompositeStrategySpec& spec) {
  const auto rules = experiment.strategies.empty() ? ExperimentSpec::defaultStrategies() : experiment.strategies;
  const CandidateBook book({composite}, rules, nextReturns, experiment.costBps);
  const std::size_t D = book.days();
  const std::size_t evalFrom = std::min(experiment.selector.lookback, D - 1);
  const std::size_t L = spec.allocation.lookback;
  if (D <= evalFrom + L + 20) throw std::invalid_argument("composite strategy: too few out-of-sample days for the selector and allocation look-backs");

  CompositeStrategyResult out;
  for (const auto& c : book.candidates()) out.selectorCandidates.push_back(c.label);
  out.selector = runSelector(book, experiment.selector, evalFrom);
  const std::size_t first = book.start() + evalFrom;
  out.multiSignal = multiSignalSelection(data, &composite.probability, spec.signals, first, book.end());
  const auto& a = out.selector.net;
  const auto& b = out.multiSignal.returns;
  if (a.size() != b.size()) throw std::logic_error("composite strategy: sleeves of different lengths");
  const auto alloc = allocate({a, b}, spec.allocation);
  out.start = first + L;
  const auto bench = equalWeightBenchmark(nextReturns, out.start, book.end(), experiment.costBps);
  out.names = {"Self-adaptive selector (" + composite.name + ")", "Multi-signal selection", "Composite strategy", "Equal-weight market"};
  out.returns = {std::vector<double>(a.begin() + static_cast<std::ptrdiff_t>(L), a.end()),
                 std::vector<double>(b.begin() + static_cast<std::ptrdiff_t>(L), b.end()), alloc.returns, bench.series.net};
  for (const auto& r : out.returns) {
    if (r.size() != alloc.returns.size()) throw std::logic_error("composite strategy: series of different lengths");
    out.metrics.push_back(evaluatePerformance(r));
  }
  out.weights = alloc.weights;
  out.cash = alloc.cash;
  return out;
}

}  // namespace sat::algo
