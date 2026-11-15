// Long-term reversal (De Bondt and Thaler 1985): the shares that fell most over the three years to a year ago. Losers
// over long horizons are oversold and recover (an overreaction that corrects over years); it holds in most markets,
// strongest among smaller shares. The scenario momentum misses: a market where the trend turns. Held a month.
#include <cstdio>

#include "strategy/strategy.hpp"

namespace {

class LtRev : public strat::Strategy {
 public:
  std::string about() const override { return "long-term reversal: the losers of the 3 years to a year ago, rebalanced monthly (De Bondt and Thaler 1985)"; }
  strat::Params defaults() const override {
    return {{"from", 756}, {"to", 252}, {"rebalance", 21}, {"warmup", 760}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 260}};
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t t = v.t(), F = std::size_t(p.at("from")), T = std::size_t(p.at("to"));
    out.assign(v.N(), strat::kNaN);
    if (t < F) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) {
          const double r = v.change(t - F, t - T, i);
          out[i] = std::isfinite(r) ? -r : strat::kNaN;
        }
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    const std::size_t t = v.t();
    char b[200];
    std::snprintf(b, sizeof b, "It moved %+.1f%% over the three years to a year ago: long-term losers tend to recover.",
                  100 * v.change(t - std::size_t(p.at("from")), t - std::size_t(p.at("to")), i));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("ltrev", LtRev);
