// The 52-week high (George and Hwang 2004): the shares closest to their highest close of the last year. Investors
// anchor on that level and are slow to price news that lifts a share near it, so the nearest keep rising; it holds
// across markets and explains much of momentum's return, without momentum's crashes (it does not buy last year's
// losers' rebound). Held a month.
#include <algorithm>
#include <cstdio>

#include "strategy/strategy.hpp"

namespace {

class High52 : public strat::Strategy {
 public:
  std::string about() const override { return "closest to the 52-week high, rebalanced monthly (George and Hwang 2004)"; }
  strat::Params defaults() const override {
    return {{"window", 252}, {"rebalance", 21}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 260}};
  }
  static double ratio(const strat::View& v, std::size_t t, std::size_t W, std::size_t i) {
    double hi = 0;
    for (std::size_t s = t + 1 - W; s <= t; ++s) hi = std::max(hi, v.close(s, i) > 0 ? v.close(s, i) : 0.0);
    return hi > 0 && v.close(t, i) > 0 ? v.close(t, i) / hi : strat::kNaN;
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t t = v.t(), W = std::size_t(p.at("window"));
    out.assign(v.N(), strat::kNaN);
    if (t < W) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) out[i] = ratio(v, t, W, i);
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    char b[160];
    std::snprintf(b, sizeof b, "Its close is %.1f%% under its highest of the last 52 weeks: shares near that level tend to keep rising.",
                  100 * (1 - ratio(v, v.t(), std::size_t(p.at("window")), i)));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("high52", High52);
