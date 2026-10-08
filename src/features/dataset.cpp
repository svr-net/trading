#include "sat/features/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sat/afml/labeling.hpp"
#include "sat/core/stats.hpp"
#include "sat/factors/alpha101.hpp"

namespace sat {

std::size_t LabelSpec::lookahead() const { return kind == LabelKind::MinMax ? window / 2 : horizon; }

Normalisation parseNormalisation(const std::string& name) {
  if (name == "none") return Normalisation::None;
  if (name == "rank") return Normalisation::Rank;
  if (name == "zscore") return Normalisation::ZScore;
  throw std::invalid_argument("unknown normalisation '" + name + "' (none, rank, zscore)");
}

LabelKind parseLabelKind(const std::string& name) {
  if (name == "direction") return LabelKind::Direction;
  if (name == "excess") return LabelKind::ExcessDirection;
  if (name == "minmax") return LabelKind::MinMax;
  if (name == "triple") return LabelKind::TripleBarrier;
  throw std::invalid_argument("unknown label kind '" + name + "' (direction, excess, minmax, triple)");
}

namespace {

void normaliseRow(double* x, std::size_t n, Normalisation norm) {
  if (norm == Normalisation::Rank) {
    std::vector<double> row(x, x + n);
    const auto r = averageRanks(row);
    std::size_t m = 0;
    for (double v : r) m += finite(v) ? 1 : 0;
    for (std::size_t i = 0; i < n; ++i) x[i] = finite(r[i]) && m > 1 ? (r[i] - 0.5) / static_cast<double>(m) - 0.5 : 0.0;
  } else if (norm == Normalisation::ZScore) {
    std::vector<double> row(x, x + n);
    const double mu = mean(row), sd = stdev(row);
    for (std::size_t i = 0; i < n; ++i)
      x[i] = finite(x[i]) && finite(sd) && sd > 0.0 ? std::clamp((x[i] - mu) / sd, -3.0, 3.0) : 0.0;
  } else {
    for (std::size_t i = 0; i < n; ++i)
      if (!finite(x[i])) x[i] = 0.0;
  }
}

}  // namespace

FeatureSet buildFeatures(const MarketData& data, const std::vector<int>& alphaIds, Normalisation norm) {
  if (alphaIds.empty()) throw std::invalid_argument("no alpha factors selected");
  const AlphaInputs in(data);
  FeatureSet fs;
  for (int id : alphaIds) {
    Panel p = computeAlpha(id, in);
    for (std::size_t t = 0; t < p.dates(); ++t) normaliseRow(p.row(t), p.assets(), norm);
    fs.names.push_back("alpha" + std::to_string(id));
    fs.panels.push_back(std::move(p));
    fs.warmup = std::max(fs.warmup, alphaLookback(id));
  }
  return fs;
}

namespace {

Panel labelsOnly(const MarketData& data, const LabelSpec& spec) {
  const std::size_t T = data.numDates(), N = data.numAssets();
  Panel y(T, N);
  if (spec.kind == LabelKind::MinMax) {
    if (spec.window < 2) throw std::invalid_argument("min-max window must be at least 2");
    const std::size_t h = spec.window / 2;
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t t = h; t + h < T; ++t) {
        const double c = data.close(t, i);
        bool isMin = true, isMax = true;
        for (std::size_t k = t - h; k <= t + h; ++k) {
          if (k == t) continue;
          if (data.close(k, i) <= c) isMin = false;
          if (data.close(k, i) >= c) isMax = false;
        }
        if (isMin) y(t, i) = 1.0;
        else if (isMax) y(t, i) = 0.0;
      }
    return y;
  }
  if (spec.horizon < 1) throw std::invalid_argument("label horizon must be at least 1 day");
  const Panel fwd = data.forwardReturns(spec.horizon);
  for (std::size_t t = 0; t < T; ++t) {
    double threshold = 0.0;
    if (spec.kind == LabelKind::ExcessDirection) {
      threshold = quantile(std::vector<double>(fwd.row(t), fwd.row(t) + N), 0.5);
      if (!finite(threshold)) continue;
    }
    for (std::size_t i = 0; i < N; ++i)
      if (finite(fwd(t, i))) y(t, i) = fwd(t, i) > threshold ? 1.0 : 0.0;
  }
  return y;
}

}  // namespace

Panel makeLabels(const MarketData& data, const LabelSpec& spec, Panel* ends) {
  if (spec.kind == LabelKind::TripleBarrier) {
    if (spec.horizon < 1 || !(spec.barrierWidth > 0)) throw std::invalid_argument("triple barrier needs a horizon >= 1 and a positive width");
    afml::BarrierSpec b;
    b.profitTaking = b.stopLoss = spec.barrierWidth;
    b.maxHolding = spec.horizon;
    return afml::tripleBarrierLabels(data.close, b, spec.volSpan, ends);
  }
  Panel y = labelsOnly(data, spec);
  if (ends) {
    *ends = y.like();
    const std::size_t reach = spec.lookahead();
    for (std::size_t t = 0; t < y.dates(); ++t)
      for (std::size_t i = 0; i < y.assets(); ++i)
        if (finite(y(t, i))) (*ends)(t, i) = static_cast<double>(t + reach);
  }
  return y;
}

Dataset assemble(const FeatureSet& features, const Panel& labels, std::size_t from, std::size_t to, std::size_t lags,
                 bool labelledOnly) {
  const std::size_t F = features.size(), N = features.assets();
  to = std::min(to, features.dates());
  if (lags < 1) lags = 1;
  Dataset ds;
  ds.lags = lags;
  std::vector<std::size_t> dates, assets;
  for (std::size_t t = from; t < to; ++t)
    for (std::size_t i = 0; i < N; ++i)
      if (!labelledOnly || finite(labels(t, i))) {
        dates.push_back(t);
        assets.push_back(i);
      }
  ds.X = Matrix(dates.size(), F * lags);
  ds.y.resize(dates.size());
  for (std::size_t r = 0; r < dates.size(); ++r) {
    const std::size_t t = dates[r], i = assets[r];
    double* row = ds.X.row(r);
    for (std::size_t l = 0; l < lags; ++l) {
      const std::size_t back = lags - 1 - l;
      const std::size_t s = t >= back ? t - back : 0;
      for (std::size_t f = 0; f < F; ++f) row[l * F + f] = features.panels[f](s, i);
    }
    ds.y[r] = labels(t, i);
  }
  ds.date = std::move(dates);
  ds.asset = std::move(assets);
  return ds;
}

}  // namespace sat
