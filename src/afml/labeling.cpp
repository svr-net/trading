#include "sat/afml/labeling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sat::afml {

std::vector<double> ewmVolatility(const std::vector<double>& close, double span) {
  const double alpha = 2.0 / (span + 1.0);
  std::vector<double> out(close.size(), std::numeric_limits<double>::quiet_NaN());
  double m = 0.0, v = 0.0, weight = 0.0;
  for (std::size_t t = 1; t < close.size(); ++t) {
    const double r = close[t] / close[t - 1] - 1.0;
    if (!std::isfinite(r)) continue;
    // Bias-corrected exponentially weighted mean and variance.
    weight = (1 - alpha) * weight + alpha;
    const double delta = r - m;
    m += alpha * delta / weight;
    v = (1 - alpha) * (v + alpha * delta * delta / weight);
    out[t] = std::sqrt(v / weight);
  }
  return out;
}

std::vector<std::size_t> cusumFilter(const std::vector<double>& logPrice, double threshold) {
  if (!(threshold > 0)) throw std::invalid_argument("CUSUM threshold must be positive");
  std::vector<std::size_t> events;
  double up = 0.0, down = 0.0;
  for (std::size_t t = 1; t < logPrice.size(); ++t) {
    const double r = logPrice[t] - logPrice[t - 1];
    if (!std::isfinite(r)) continue;
    up = std::max(0.0, up + r);
    down = std::min(0.0, down + r);
    if (up > threshold || down < -threshold) {
      events.push_back(t);
      up = down = 0.0;
    }
  }
  return events;
}

std::vector<BarrierEvent> tripleBarrier(const std::vector<double>& close, const std::vector<std::size_t>& events,
                                        const std::vector<double>& target, const BarrierSpec& spec,
                                        const std::vector<int>* sides) {
  if (spec.maxHolding < 1) throw std::invalid_argument("the vertical barrier must be at least one bar away");
  if (sides && sides->size() != events.size()) throw std::invalid_argument("one side per event expected");
  std::vector<BarrierEvent> out;
  for (std::size_t k = 0; k < events.size(); ++k) {
    const std::size_t t0 = events[k];
    if (t0 >= close.size() || t0 + spec.maxHolding >= close.size()) continue;
    const double trg = target[t0];
    if (!std::isfinite(trg) || trg <= 0 || trg < spec.minTarget) continue;
    BarrierEvent e;
    e.t0 = t0;
    e.target = trg;
    e.side = sides ? ((*sides)[k] >= 0 ? 1 : -1) : 1;
    const double p0 = close[t0];
    e.t1 = t0 + spec.maxHolding;
    for (std::size_t t = t0 + 1; t <= t0 + spec.maxHolding; ++t) {
      const double r = e.side * (close[t] / p0 - 1.0);
      if (spec.profitTaking > 0 && r >= spec.profitTaking * trg) {
        e.t1 = t;
        e.barrier = 1;
        break;
      }
      if (spec.stopLoss > 0 && r <= -spec.stopLoss * trg) {
        e.t1 = t;
        e.barrier = -1;
        break;
      }
    }
    e.ret = e.side * (close[e.t1] / p0 - 1.0);
    if (sides) e.label = e.ret > 0 ? 1 : 0;
    else if (e.barrier != 0) e.label = e.barrier;
    else e.label = spec.zeroOnVertical ? 0 : (e.ret > 0 ? 1 : (e.ret < 0 ? -1 : 0));
    out.push_back(e);
  }
  return out;
}

Panel tripleBarrierLabels(const Panel& close, const BarrierSpec& spec, double volSpan, Panel* ends) {
  Panel y = close.like();
  if (ends) *ends = close.like();
  std::vector<std::size_t> all(close.dates());
  for (std::size_t t = 0; t < all.size(); ++t) all[t] = t;
  for (std::size_t i = 0; i < close.assets(); ++i) {
    const auto c = close.series(i);
    const auto vol = ewmVolatility(c, volSpan);
    for (const auto& e : tripleBarrier(c, all, vol, spec)) {
      if (e.label == 0) continue;  // vertical touch without a move
      y(e.t0, i) = e.label > 0 ? 1.0 : 0.0;
      if (ends) (*ends)(e.t0, i) = static_cast<double>(e.t1);
    }
  }
  return y;
}

}  // namespace sat::afml
