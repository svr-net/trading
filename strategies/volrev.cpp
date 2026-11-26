// Volume-conditioned reversal (Conrad, Hameed and Niden 1994: heavily traded shares reverse, lightly traded ones
// continue; Campbell, Grossman and Wang 1993; Avramov, Chordia and Goyal 2006: the reversal sits in high-turnover,
// illiquid shares): the shares whose last `window` sessions fell most against the market on volume at least
// `minvol` times their usual, scored by the fall times the volume (capped), bought at the next open and held while
// still among the best; re-decided after every close.
#include <algorithm>
#include <cstdio>

#include "strategy/daily.hpp"

namespace {
using namespace strat::daily;

class VolRev : public strat::Strategy {
 public:
  std::string about() const override { return "volume reversal: the 3-day falls against the market on heavy volume, held days (Conrad, Hameed and Niden 1994; Avramov, Chordia and Goyal 2006)"; }
  strat::Params defaults() const override {
    return {{"window", 3}, {"beta", 252}, {"base", 60}, {"minvol", 1.5}, {"entry", 1.0}, {"rebalance", 1}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 5}};
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t W = std::size_t(p.at("window")), B = std::size_t(p.at("beta")), V = std::size_t(p.at("base"));
    out.assign(v.N(), strat::kNaN);
    if (v.t() + 1 < std::max(B, W + V)) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) {
          const double z = residual(v, v.t(), W, B, i), rv = relVolume(v, v.t(), W, V, i);
          if (z < -p.at("entry") && rv >= p.at("minvol")) out[i] = -z * std::min(rv, 3.0);
        }
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    const std::size_t W = std::size_t(p.at("window")), t = v.t();
    char b[240];
    std::snprintf(b, sizeof b, "It fell %+.1f%% over %zu sessions (%.1f volatilities against the market) on %.1f times its usual volume: heavy-volume falls have tended to reverse.",
                  100 * v.change(t - W, t, i), W, -residual(v, t, W, std::size_t(p.at("beta")), i), relVolume(v, t, W, std::size_t(p.at("base")), i));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("volrev", VolRev);
