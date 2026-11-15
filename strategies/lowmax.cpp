// The lottery effect (Bali, Cakici and Whitelaw 2011): the shares whose largest daily gain of the last month was the
// smallest. Investors overpay for lottery-like shares (a few huge up days), which then lag; avoiding them earns a
// premium in the US and internationally, and protects in speculative markets. Held a month.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "strategy/strategy.hpp"

namespace {

class LowMax : public strat::Strategy {
 public:
  std::string about() const override { return "no lottery shares: the smallest largest daily gain of the last month, rebalanced monthly (Bali, Cakici and Whitelaw 2011)"; }
  strat::Params defaults() const override {
    return {{"window", 21}, {"rebalance", 21}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 260}};
  }
  static double maxUp(const strat::View& v, std::size_t t, std::size_t W, std::size_t i) {
    double m = -1e300, n = 0;
    for (std::size_t s = t + 1 - W; s <= t; ++s)
      if (std::isfinite(v.ret(s, i))) m = std::max(m, v.ret(s, i)), n += 1;
    return n >= double(W) / 2 ? m : strat::kNaN;
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t t = v.t(), W = std::size_t(p.at("window"));
    out.assign(v.N(), strat::kNaN);
    if (t < W) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) {
          const double m = maxUp(v, t, W, i);
          out[i] = std::isfinite(m) ? -m : strat::kNaN;
        }
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    char b[160];
    std::snprintf(b, sizeof b, "Its largest daily gain of the last month was only %+.1f%%: no lottery-like jumps, which tend to lag after.",
                  100 * (std::exp(maxUp(v, v.t(), std::size_t(p.at("window")), i)) - 1));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("lowmax", LowMax);
