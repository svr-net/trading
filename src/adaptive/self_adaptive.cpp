#include "sat/adaptive/self_adaptive.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <tuple>

namespace sat {

std::pair<std::size_t, std::size_t> commonRange(const std::vector<ModelPredictions>& models) {
  std::size_t start = 0, end = SIZE_MAX;
  for (const auto& m : models) {
    start = std::max(start, m.start);
    end = std::min(end, m.end);
  }
  if (models.empty() || start >= end) throw std::invalid_argument("the models share no out-of-sample dates");
  return {start, end};
}

CandidateBook::CandidateBook(const std::vector<ModelPredictions>& models, const std::vector<StrategySpec>& strategies,
                             const Panel& nextReturns, double costBps)
    : nextReturns_(nextReturns), costBps_(costBps) {
  if (models.empty() || strategies.empty()) throw std::invalid_argument("candidate book needs models and strategies");
  std::tie(start_, end_) = commonRange(models);
  end_ = std::min(end_, nextReturns.dates());
  if (start_ >= end_) throw std::invalid_argument("the models share no out-of-sample dates");
  for (std::size_t m = 0; m < models.size(); ++m) {
    modelNames_.push_back(models[m].name);
    probs_.push_back(models[m].probability);
    ranks_.push_back(rankTable(models[m].probability));
    for (const auto& s : strategies) candidates_.push_back({m, s, models[m].name + " · " + s.label()});
  }
  gross_ = Matrix(candidates_.size(), days());
  turnover_ = Matrix(candidates_.size(), days());
  for (std::size_t c = 0; c < candidates_.size(); ++c) {
    const auto& cand = candidates_[c];
    const auto r = backtest(cand.strategy, probs_[cand.model], ranks_[cand.model], nextReturns_, start_, end_, costBps_);
    std::copy(r.series.gross.begin(), r.series.gross.end(), gross_.row(c));
    std::copy(r.series.turnover.begin(), r.series.turnover.end(), turnover_.row(c));
  }
}

std::vector<double> CandidateBook::netSeries(std::size_t c, std::size_t from) const {
  std::vector<double> out;
  for (std::size_t d = from; d < days(); ++d) out.push_back(net(c, d));
  return out;
}

std::vector<double> CandidateBook::turnoverSeries(std::size_t c, std::size_t from) const {
  std::vector<double> out;
  for (std::size_t d = from; d < days(); ++d) out.push_back(turnover(c, d));
  return out;
}

void CandidateBook::weights(std::size_t c, std::size_t d, double* w) const {
  const auto& cand = candidates_.at(c);
  const std::size_t t = start_ + rebalanceOffset(d, cand.strategy.holding);
  const std::size_t N = assets();
  strategyWeights(cand.strategy, probs_[cand.model].row(t), ranks_[cand.model].data() + t * N, N, w);
}

ScoreMetric parseScoreMetric(const std::string& name) {
  if (name == "return") return ScoreMetric::Return;
  if (name == "sharpe") return ScoreMetric::Sharpe;
  if (name == "sortino") return ScoreMetric::Sortino;
  throw std::invalid_argument("unknown score metric '" + name + "' (return, sharpe, sortino)");
}

std::string scoreMetricName(ScoreMetric m) {
  switch (m) {
    case ScoreMetric::Return: return "return";
    case ScoreMetric::Sharpe: return "sharpe";
    case ScoreMetric::Sortino: return "sortino";
  }
  return "?";
}

std::string SelectorSpec::label() const {
  char buf[96];
  std::snprintf(buf, sizeof buf, "Self-adaptive (%s, %zud look-back, every %zud%s)", scoreMetricName(metric).c_str(), lookback,
                adaptEvery, topM > 1 ? (", top " + std::to_string(topM)).c_str() : "");
  return buf;
}

double windowScore(ScoreMetric metric, double n, double s1, double s2, double sd) {
  if (n < 2) return -1e30;
  const double mean = s1 / n;
  switch (metric) {
    case ScoreMetric::Return: return mean;
    case ScoreMetric::Sharpe: return mean / std::sqrt(std::max(0.0, (s2 - s1 * s1 / n) / (n - 1)) + 1e-10);
    case ScoreMetric::Sortino: return mean / std::sqrt(sd / n + 1e-10);
  }
  return -1e30;
}

AdaptiveResult runSelector(const CandidateBook& book, const SelectorSpec& spec, std::size_t evalFrom) {
  const std::size_t C = book.size(), D = book.days(), N = book.assets();
  if (spec.lookback < 2 || spec.adaptEvery < 1) throw std::invalid_argument("selector look-back must be >= 2 and step >= 1");
  if (evalFrom == SIZE_MAX) evalFrom = std::min(spec.lookback, D - 1);
  if (evalFrom >= D) throw std::invalid_argument("selector: nothing left to trade after the look-back");
  const std::size_t M = std::clamp<std::size_t>(spec.topM, 1, C);
  const double cost = book.costBps() * 1e-4;

  // Prefix sums of each candidate's net returns, squares and squared losses.
  Matrix p1(C, D + 1, 0.0), p2(C, D + 1, 0.0), pd(C, D + 1, 0.0);
  for (std::size_t c = 0; c < C; ++c)
    for (std::size_t d = 0; d < D; ++d) {
      const double r = book.net(c, d);
      p1(c, d + 1) = p1(c, d) + r;
      p2(c, d + 1) = p2(c, d) + r * r;
      pd(c, d + 1) = pd(c, d) + (r < 0 ? r * r : 0.0);
    }

  AdaptiveResult out;
  out.spec = spec;
  out.evalFrom = evalFrom;
  out.share.assign(C + 1, 0.0);
  std::vector<int> held, prevHeld;  // empty = cash
  std::vector<double> w(N), wMeta(N, 0.0), wPrev(N, 0.0), scores(C);
  std::vector<std::size_t> order(C);
  for (std::size_t d = evalFrom; d < D; ++d) {
    if ((d - evalFrom) % spec.adaptEvery == 0) {
      out.adaptations.push_back(d);
      const std::size_t from = d > spec.lookback ? d - spec.lookback : 0;
      const double n = static_cast<double>(d - from);
      for (std::size_t c = 0; c < C; ++c)
        scores[c] = windowScore(spec.metric, n, p1(c, d) - p1(c, from), p2(c, d) - p2(c, from), pd(c, d) - pd(c, from));
      for (std::size_t c = 0; c < C; ++c) order[c] = c;
      std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
      held.clear();
      for (std::size_t k = 0; k < M; ++k)
        if (!(spec.allowCash && !(scores[order[k]] > spec.minScore)) && scores[order[k]] > -1e29) held.push_back(static_cast<int>(order[k]));
      if (held.empty() && !spec.allowCash) held.push_back(static_cast<int>(order[0]));
    }
    double gross = 0.0, turnover = 0.0;
    const bool same = held == prevHeld && d > evalFrom;
    for (int c : held) gross += book.gross(static_cast<std::size_t>(c), d) / static_cast<double>(held.size());
    if (same && held.size() == 1) {
      turnover = book.turnover(static_cast<std::size_t>(held[0]), d);
    } else if (!(same && held.empty())) {
      std::fill(wMeta.begin(), wMeta.end(), 0.0);
      for (int c : held) {
        book.weights(static_cast<std::size_t>(c), d, w.data());
        for (std::size_t i = 0; i < N; ++i) wMeta[i] += w[i] / static_cast<double>(held.size());
      }
      std::fill(wPrev.begin(), wPrev.end(), 0.0);
      if (d > evalFrom)
        for (int c : prevHeld) {
          book.weights(static_cast<std::size_t>(c), d - 1, w.data());
          for (std::size_t i = 0; i < N; ++i) wPrev[i] += w[i] / static_cast<double>(prevHeld.size());
        }
      for (std::size_t i = 0; i < N; ++i) turnover += std::fabs(wMeta[i] - wPrev[i]);
    }
    if (d > evalFrom && held != prevHeld) ++out.switches;
    out.gross.push_back(gross);
    out.turnover.push_back(turnover);
    out.net.push_back(gross - cost * turnover);
    out.selection.push_back(held.empty() ? -1 : held[0]);
    out.share[held.empty() ? C : static_cast<std::size_t>(held[0])] += 1.0;
    prevHeld = held;
  }
  for (double& s : out.share) s /= static_cast<double>(D - evalFrom);
  out.metrics = evaluatePerformance(out.net, out.turnover);
  return out;
}

std::size_t GridResult::switches(std::size_t s) const {
  std::size_t n = 0;
  for (std::size_t d = 1; d < selection[s].size(); ++d) n += selection[s][d] != selection[s][d - 1];
  return n;
}

GridResult evaluateGrid(const CandidateBook& book, const std::vector<SelectorSpec>& selectors, std::size_t evalFrom) {
  const auto t0 = std::chrono::steady_clock::now();
  GridResult g;
  g.start = book.start();
  g.days = book.days();
  g.evalFrom = evalFrom;
  if (evalFrom >= book.days()) throw std::invalid_argument("evaluation starts after the last out-of-sample date");
  for (std::size_t c = 0; c < book.size(); ++c)
    g.candidates.push_back(evaluatePerformance(book.netSeries(c, evalFrom), book.turnoverSeries(c, evalFrom)));
  for (const auto& s : selectors) {
    AdaptiveResult a = runSelector(book, s, evalFrom);
    g.selectors.push_back(a.metrics);
    g.adaptiveNet.push_back(std::move(a.net));
    g.adaptiveTurnover.push_back(std::move(a.turnover));
    g.selection.push_back(std::move(a.selection));
  }
  g.elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return g;
}

}  // namespace sat
