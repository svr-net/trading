#include "sat/afml/bars.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sat/core/random.hpp"
#include "sat/core/stats.hpp"

namespace sat::afml {

std::vector<Trade> generateTrades(const TradeStreamSpec& spec) {
  if (spec.days < 1 || spec.tradesPerDay < 10) throw std::invalid_argument("trade stream needs at least a day and 10 trades a day");
  Rng rng(spec.seed);
  std::vector<Trade> trades;
  // Every trade has the same variance, so a day's variance is proportional to its trade count.
  const double tradeVol = spec.dailyVolatility / std::sqrt(spec.tradesPerDay);
  double logPrice = std::log(spec.startPrice);
  int side = 1;
  for (std::size_t d = 0; d < spec.days; ++d) {
    const double activity = std::exp(spec.activityDispersion * rng.normal() - 0.5 * spec.activityDispersion * spec.activityDispersion);
    const auto n = std::max<std::size_t>(2, static_cast<std::size_t>(std::llround(spec.tradesPerDay * activity)));
    std::vector<double> times(n);
    for (auto& t : times) t = static_cast<double>(d) + rng.uniform();
    std::sort(times.begin(), times.end());
    for (std::size_t k = 0; k < n; ++k) {
      if (rng.uniform() > spec.persistence) side = rng.uniform() < 0.5 ? 1 : -1;
      const double shock = tradeVol * (spec.impact * side * std::sqrt(2.0 / 3.141592653589793) + std::sqrt(1.0 - spec.impact * spec.impact) * rng.normal());
      logPrice += shock;
      Trade t;
      t.time = times[k];
      t.price = std::exp(logPrice);
      t.volume = std::max(1.0, std::round(spec.meanSize * std::exp(0.8 * rng.normal() - 0.32)));
      t.side = side;
      trades.push_back(t);
    }
  }
  return trades;
}

namespace {

// Accumulates trades into one bar.
struct BarBuilder {
  Bar bar;
  bool open = false;
  void add(const Trade& t) {
    if (!open) {
      bar = Bar{};
      bar.timeOpen = t.time;
      bar.open = bar.high = bar.low = t.price;
      open = true;
    }
    bar.timeClose = t.time;
    bar.close = t.price;
    bar.high = std::max(bar.high, t.price);
    bar.low = std::min(bar.low, t.price);
    bar.volume += t.volume;
    bar.dollarVolume += t.volume * t.price;
    bar.ticks += 1;
    bar.imbalance += t.side;
  }
  Bar close() {
    open = false;
    bar.vwap = bar.volume > 0 ? bar.dollarVolume / bar.volume : bar.close;
    return bar;
  }
};

template <class Done>
std::vector<Bar> build(const std::vector<Trade>& trades, Done done) {
  std::vector<Bar> out;
  BarBuilder b;
  for (const auto& t : trades) {
    b.add(t);
    if (done(b.bar)) out.push_back(b.close());
  }
  return out;  // an unfinished last bar is dropped
}

}  // namespace

std::vector<Bar> timeBars(const std::vector<Trade>& trades, double interval) {
  if (!(interval > 0)) throw std::invalid_argument("time bars need a positive interval");
  std::vector<Bar> out;
  BarBuilder b;
  long current = -1;
  for (const auto& t : trades) {
    const long slot = static_cast<long>(std::floor(t.time / interval));
    if (b.open && slot != current) out.push_back(b.close());
    current = slot;
    b.add(t);
  }
  if (b.open) out.push_back(b.close());
  return out;
}

std::vector<Bar> tickBars(const std::vector<Trade>& trades, std::size_t count) {
  if (count < 1) throw std::invalid_argument("tick bars need at least one trade per bar");
  return build(trades, [count](const Bar& bar) { return bar.ticks >= count; });
}

std::vector<Bar> volumeBars(const std::vector<Trade>& trades, double threshold) {
  if (!(threshold > 0)) throw std::invalid_argument("volume bars need a positive threshold");
  return build(trades, [threshold](const Bar& bar) { return bar.volume >= threshold; });
}

std::vector<Bar> dollarBars(const std::vector<Trade>& trades, double threshold) {
  if (!(threshold > 0)) throw std::invalid_argument("dollar bars need a positive threshold");
  return build(trades, [threshold](const Bar& bar) { return bar.dollarVolume >= threshold; });
}

std::vector<Bar> tickImbalanceBars(const std::vector<Trade>& trades, std::size_t initialTicks, double span) {
  if (initialTicks < 2) throw std::invalid_argument("imbalance bars need an initial expectation of at least 2 ticks");
  const double alpha = 2.0 / (span + 1.0);
  double expectedTicks = static_cast<double>(initialTicks);
  // Expected sign from the first window of trades.
  double expectedSign = 0.0;
  const std::size_t warm = std::min(trades.size(), initialTicks);
  for (std::size_t k = 0; k < warm; ++k) expectedSign += trades[k].side;
  expectedSign = warm ? expectedSign / static_cast<double>(warm) : 0.0;
  const double lo = static_cast<double>(initialTicks) / 4.0, hi = static_cast<double>(initialTicks) * 4.0;
  // Variance of the imbalance per tick, from the warm-up window split into blocks of 10.
  double imbalanceRate = 1.0;
  if (warm >= 20) {
    double s2 = 0;
    std::size_t blocks = 0;
    for (std::size_t k = 0; k + 10 <= warm; k += 10, ++blocks) {
      double s = 0;
      for (std::size_t j = k; j < k + 10; ++j) s += trades[j].side;
      s2 += s * s / 10.0;
    }
    if (blocks) imbalanceRate = std::max(1.0, s2 / static_cast<double>(blocks));
  }
  std::vector<Bar> out;
  BarBuilder b;
  for (const auto& t : trades) {
    b.add(t);
    // E[T] |2P[b=1] - 1| = E[T] |E[b]|. With balanced flow |E[b]| is near zero and the
    // threshold would collapse to one tick, so it is floored at sqrt(E[T] v), v being the
    // observed variance of the imbalance per tick (1 for independent signs, more when the
    // flow is persistent): the imbalance such a flow typically reaches in E[T] ticks.
    const double threshold = std::max(expectedTicks * std::fabs(expectedSign), std::sqrt(expectedTicks * imbalanceRate));
    if (std::fabs(b.bar.imbalance) >= threshold || static_cast<double>(b.bar.ticks) >= hi) {
      const Bar bar = b.close();
      const double ticks = static_cast<double>(bar.ticks);
      expectedTicks = std::clamp((1 - alpha) * expectedTicks + alpha * ticks, lo, hi);
      expectedSign = (1 - alpha) * expectedSign + alpha * bar.imbalance / ticks;
      imbalanceRate = (1 - alpha) * imbalanceRate + alpha * bar.imbalance * bar.imbalance / ticks;
      out.push_back(bar);
    }
  }
  return out;
}

std::vector<double> barReturns(const std::vector<Bar>& bars) {
  std::vector<double> r;
  for (std::size_t k = 1; k < bars.size(); ++k) r.push_back(std::log(bars[k].close / bars[k - 1].close));
  return r;
}

BarStatistics barStatistics(const std::vector<Bar>& bars, std::size_t days) {
  BarStatistics s;
  s.count = bars.size();
  if (days > 0) {
    std::vector<double> perDay(days, 0.0);
    for (const auto& b : bars) {
      const auto d = static_cast<std::size_t>(std::clamp(std::floor(b.timeClose), 0.0, static_cast<double>(days - 1)));
      perDay[d] += 1.0;
    }
    s.barsPerDayMean = mean(perDay);
    s.barsPerDaySd = stdev(perDay);
  }
  const auto r = barReturns(bars);
  if (r.size() < 20) return s;
  s.returnSd = stdev(r);
  s.skewness = skewness(r);
  s.kurtosis = kurtosis(r);
  const double n = static_cast<double>(r.size());
  s.jarqueBera = n / 6.0 * (s.skewness * s.skewness + (s.kurtosis - 3.0) * (s.kurtosis - 3.0) / 4.0);
  s.serialCorrelation = correlation(std::vector<double>(r.begin(), r.end() - 1), std::vector<double>(r.begin() + 1, r.end()));
  // Variance in 10 consecutive subsamples, relative to the average variance.
  std::vector<double> vars;
  const std::size_t chunk = r.size() / 10;
  for (std::size_t k = 0; k < 10; ++k) {
    const std::vector<double> part(r.begin() + static_cast<std::ptrdiff_t>(k * chunk), r.begin() + static_cast<std::ptrdiff_t>((k + 1) * chunk));
    const double sd = stdev(part);
    vars.push_back(sd * sd);
  }
  const double mv = mean(vars);
  s.varianceOfVariance = mv > 0 ? stdev(vars) / mv : 0.0;
  return s;
}

}  // namespace sat::afml
