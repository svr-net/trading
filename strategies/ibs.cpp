// The internal-bar-strength effect (Pagonidis 2013, "The IBS Effect: Mean Reversion in Equity ETFs"; confirmed on
// later data by Pandey and Joshi 2023, arXiv 2306.12434): a close near the day's low (IBS = (close - low) /
// (high - low) under 0.2) tends to be followed by a rise the next day, and the effect lasts about a day. Bought at
// the next open and held one session; the lowest IBS first, ties broken by the day's fall in volatilities.
#include <cstdio>

#include "strategy/daily.hpp"

namespace {
using namespace strat::daily;

class Ibs : public strat::Strategy {
 public:
  std::string about() const override { return "internal bar strength: closes near the day's low (IBS under 0.2), held a day (Pagonidis 2013; Pandey and Joshi 2023)"; }
  strat::Params defaults() const override {
    return {{"entry", 0.2}, {"trend", 0}, {"rebalance", 1}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 3}};
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t t = v.t(), T = std::size_t(p.at("trend"));
    out.assign(v.N(), strat::kNaN);
    if (t < T + 1) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) {
          const double x = ibs(v, t, i), r = v.ret(t, i), sg = v.sigma(t, i);
          if (!(x < p.at("entry")) || !std::isfinite(r) || !(sg > 0)) continue;
          if (T > 0 && !(v.close(t, i) > sma(v, t, T, i))) continue;  // trend: only over the T-day average (0: any)
          out[i] = (p.at("entry") - x) - 0.01 * r / sg;
        }
    });
  }
  std::string reason(const strat::View& v, const strat::Params&, std::size_t i) const override {
    const std::size_t t = v.t();
    char b[200];
    std::snprintf(b, sizeof b, "It closed at %.0f%% of the day's range (%+.1f%% on the day): closes near the low have tended to bounce the next day.",
                  100 * ibs(v, t, i), 100 * (std::exp(v.ret(t, i)) - 1));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("ibs", Ibs);
