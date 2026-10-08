#include "sat/afml/features.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sat/afml/fracdiff.hpp"
#include "sat/afml/labeling.hpp"
#include "sat/afml/microstructure.hpp"
#include "sat/core/stats.hpp"

namespace sat::afml {

const std::vector<std::string>& extraFeatureNames() {
  static const std::vector<std::string> names = {"ffd", "vol", "roll", "corwin", "amihud", "kyle"};
  return names;
}

void appendExtraFeatures(FeatureSet& fs, const MarketData& data, const std::vector<std::string>& names, Normalisation norm, double ffdD) {
  const std::size_t T = data.numDates(), N = data.numAssets();
  const std::size_t window = 20;
  for (const auto& name : names) {
    Panel p(T, N);
    std::size_t warm = window + 2;
    if (name == "ffd") {
      for (std::size_t i = 0; i < N; ++i) {
        std::vector<double> lp(T);
        for (std::size_t t = 0; t < T; ++t) lp[t] = std::log(data.close(t, i));
        const auto f = fracDiff(lp, ffdD, 1e-3);
        for (std::size_t t = 0; t < T; ++t) p(t, i) = f[t];
      }
      warm = fracDiffWeights(ffdD, 1e-3).size();
    } else if (name == "vol") {
      for (std::size_t i = 0; i < N; ++i) {
        const auto v = ewmVolatility(data.close.series(i), 20.0);
        for (std::size_t t = 0; t < T; ++t) p(t, i) = t >= window ? v[t] : Panel::kMissing;
      }
    } else if (name == "roll") {
      p = rollSpread(data.close, window);
    } else if (name == "corwin") {
      p = corwinSchultzSpread(data.high, data.low, window);
    } else if (name == "amihud") {
      p = amihudIlliquidity(data.close, data.volume, window);
    } else if (name == "kyle") {
      p = kyleLambda(data.close, data.volume, window);
    } else {
      throw std::invalid_argument("unknown extra feature '" + name + "' (ffd, vol, roll, corwin, amihud, kyle)");
    }
    // Same cross-sectional normalisation as the alphas; NaN becomes the centre.
    std::vector<double> row(N);
    for (std::size_t t = 0; t < T; ++t) {
      std::copy(p.row(t), p.row(t) + N, row.begin());
      if (norm == Normalisation::Rank) {
        const auto r = averageRanks(row);
        std::size_t m = 0;
        for (double v : r) m += finite(v) ? 1 : 0;
        for (std::size_t i = 0; i < N; ++i) p(t, i) = finite(r[i]) && m > 1 ? (r[i] - 0.5) / static_cast<double>(m) - 0.5 : 0.0;
      } else if (norm == Normalisation::ZScore) {
        const double mu = mean(row), sd = stdev(row);
        for (std::size_t i = 0; i < N; ++i)
          p(t, i) = finite(row[i]) && finite(sd) && sd > 0 ? std::clamp((row[i] - mu) / sd, -3.0, 3.0) : 0.0;
      } else {
        for (std::size_t i = 0; i < N; ++i)
          if (!finite(p(t, i))) p(t, i) = 0.0;
      }
    }
    fs.names.push_back(name);
    fs.panels.push_back(std::move(p));
    fs.warmup = std::max(fs.warmup, warm);
  }
}

Panel cusumEventMask(const MarketData& data, double multiple) {
  if (!(multiple > 0)) throw std::invalid_argument("CUSUM multiple must be positive");
  Panel mask(data.numDates(), data.numAssets(), 0.0);
  for (std::size_t i = 0; i < data.numAssets(); ++i) {
    const auto c = data.close.series(i);
    std::vector<double> lp(c.size());
    for (std::size_t t = 0; t < c.size(); ++t) lp[t] = std::log(c[t]);
    std::vector<double> r;
    for (std::size_t t = 1; t < c.size(); ++t) r.push_back(std::fabs(c[t] / c[t - 1] - 1.0));
    const double scale = quantile(r, 0.5) * 1.4826;  // median absolute return as a robust volatility
    if (!(scale > 0)) continue;
    for (std::size_t t : cusumFilter(lp, multiple * scale)) mask(t, i) = 1.0;
  }
  return mask;
}

}  // namespace sat::afml
