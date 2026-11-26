// Daily residual reversal (Lehmann 1990; Lo and MacKinlay 1990; Nagel 2012, "Evaporating Liquidity": short-term
// reversal returns are the returns to providing liquidity, highest when markets are stressed): the shares that
// fell most over the last `window` sessions net of their beta times the market's, in volatilities, are bought at the
// next open and held a session; re-decided after every close.
#include <cmath>
#include <cstdio>

#include "strategy/daily.hpp"

namespace {
using namespace strat::daily;

class RevDay : public strat::Strategy {
 public:
  std::string about() const override { return "daily residual reversal: the day's largest falls against the market, held a day (Lehmann 1990; Nagel 2012)"; }
  strat::Params defaults() const override {
    return {{"window", 1}, {"beta", 252}, {"entry", 1.5}, {"rebalance", 1}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 3}};
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t W = std::size_t(p.at("window")), B = std::size_t(p.at("beta"));
    out.assign(v.N(), strat::kNaN);
    if (v.t() + 1 < B) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) {
          const double z = residual(v, v.t(), W, B, i);
          if (z < -p.at("entry")) out[i] = -z;
        }
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    const std::size_t W = std::size_t(p.at("window")), t = v.t();
    char b[220];
    std::snprintf(b, sizeof b, "It fell %+.1f%% over the last %zu session(s), %.1f volatilities more than the market explains: such falls have tended to recover within days.",
                  100 * v.change(t - W, t, i), W, -residual(v, t, W, std::size_t(p.at("beta")), i));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("revday", RevDay);
