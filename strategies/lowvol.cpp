// The low-volatility anomaly (Ang, Hodrick, Xing and Zhang 2006; Frazzini and Pedersen 2014): the shares with
// the lowest volatility over the last year have earned as much as the riskiest, with far less risk.
// Rebalanced monthly.
#include <cmath>
#include <cstdio>

#include "strategy/strategy.hpp"

namespace {

class LowVol : public strat::Strategy {
 public:
  std::string about() const override { return "low volatility: the calmest shares over the last year, rebalanced monthly (Frazzini and Pedersen 2014)"; }
  strat::Params defaults() const override {
    return {{"window", 252}, {"rebalance", 21}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 260}};
  }
  static double vol(const strat::View& v, std::size_t i, std::size_t W) {
    const std::size_t t = v.t();
    double s1 = 0, s2 = 0, n = 0;
    for (std::size_t s = t + 1 - W; s <= t; ++s) {
      const double r = v.ret(s, i);
      if (std::isfinite(r)) s1 += r, s2 += r * r, n += 1;
    }
    return n >= 0.8 * double(W) ? std::sqrt(std::max(0.0, s2 / n - (s1 / n) * (s1 / n))) : strat::kNaN;
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t W = std::size_t(p.at("window"));
    out.assign(v.N(), strat::kNaN);
    if (v.t() + 1 < W) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) {
          const double s = vol(v, i, W);
          if (s > 0) out[i] = -s;
        }
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    char b[200];
    std::snprintf(b, sizeof b, "One of the calmest shares: a daily move of %.2f%% over the last year.", 100 * vol(v, i, std::size_t(p.at("window"))));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("lowvol", LowVol);
