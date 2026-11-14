// Time-series momentum, or trend following (Moskowitz, Ooi and Pedersen 2012; Hurst, Ooi and Pedersen 2017:
// positive in 67 markets back to 1880): an asset is held while its own return over the lookback is positive,
// the strongest trends (the return over its volatility) first; long only, cash otherwise. Rebalanced weekly.
#include <cmath>
#include <cstdio>

#include "strategy/strategy.hpp"

namespace {

class Trend : public strat::Strategy {
 public:
  std::string about() const override { return "trend following: held while its own 12-month return is positive, strongest trends first, rebalanced weekly (Hurst, Ooi and Pedersen 2017)"; }
  strat::Params defaults() const override {
    return {{"lookback", 252}, {"rebalance", 5}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 260}};
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t t = v.t(), L = std::size_t(p.at("lookback"));
    out.assign(v.N(), strat::kNaN);
    if (t < L) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i) {
        if (!v.tradable(i)) continue;
        const double r = v.change(t - L, t, i), s = v.sigma(t, i);
        if (std::isfinite(r) && r > 0 && s > 0) out[i] = r / (s * std::sqrt(double(L)));
      }
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    const std::size_t t = v.t(), L = std::size_t(p.at("lookback"));
    const double r = v.change(t - L, t, i), s = v.sigma(t, i);
    char b[220];
    std::snprintf(b, sizeof b, "In an uptrend: %+.1f%% over the last 12 months, %.1f times its volatility over that time (a daily move of %.1f%%).",
                  100 * r, r / (s * std::sqrt(double(L))), 100 * s);
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("trend", Trend);
