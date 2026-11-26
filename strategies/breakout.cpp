// Channel breakout (Donchian; the Turtle rules, Faith 2007; for shares, a close at a new high predicts gains:
// George and Hwang 2004, and the trend-following evidence of Hurst, Ooi and Pedersen 2017): a share in a long uptrend
// that closed at its highest of `entry` sessions within the last `hold` sessions, and is still above its lowest
// close of the last `exit` sessions, is held; the strongest breakouts (the move over the channel in volatilities)
// first. Re-decided after every close: the trend-following side of a daily book.
#include <cmath>
#include <cstdio>

#include "strategy/daily.hpp"

namespace {
using namespace strat::daily;

class Breakout : public strat::Strategy {
 public:
  std::string about() const override { return "20-day breakout: a new 20-day closing high in an uptrend, out under the 10-day low; daily (Donchian; Faith 2007)"; }
  strat::Params defaults() const override {
    return {{"entry", 20}, {"exit", 10}, {"hold", 5}, {"trend", 200}, {"rebalance", 1}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 20}};
  }
  // Sessions since the last new `entry`-session closing high (within `hold`), or -1.
  static long since(const strat::View& v, const strat::Params& p, std::size_t i) {
    const std::size_t t = v.t(), E = std::size_t(p.at("entry")), H = std::size_t(p.at("hold"));
    for (std::size_t k = 0; k < H && t >= k + E; ++k) {
      const double c = v.close(t - k, i), m = maxClose(v, t - k - 1, E, i);
      if (c > 0 && c > m) return long(k);
    }
    return -1;
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t t = v.t(), E = std::size_t(p.at("entry")), X = std::size_t(p.at("exit")), T = std::size_t(p.at("trend"));
    out.assign(v.N(), strat::kNaN);
    if (t < std::max(T, E) + 1) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i) {
        if (!v.tradable(i)) continue;
        const double c = v.close(t, i), sg = v.sigma(t, i);
        if (!(c > sma(v, t, T, i)) || !(c > minClose(v, t - 1, X, i)) || !(sg > 0) || since(v, p, i) < 0) continue;
        out[i] = std::log(c / maxClose(v, t - E, E, i)) / (sg * std::sqrt(double(E)));
      }
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    const std::size_t t = v.t();
    char b[200];
    std::snprintf(b, sizeof b, "It made a new 20-day closing high %ld session(s) ago (%+.1f%% over 20 sessions) above its 200-day average: a breakout, held until it closes under its 10-day low.",
                  since(v, p, i), 100 * v.change(t - std::size_t(p.at("entry")), t, i));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("breakout", Breakout);
