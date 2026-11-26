// The high-volume return premium (Gervais, Kaniel and Mingelgrin 2001; international: Kaniel, Ozoguz and Starks
// 2012): a share whose day's volume is far above its usual tends to rise over the following weeks, as the
// attention brings it new investors. A share is wanted for `hold` sessions after a day of at least `minvol` times
// its median volume of the 49 sessions before, if that day did not close lower (the volume shock, not a sell-off),
// the most extreme shock first; re-decided after every close.
#include <algorithm>
#include <cstdio>

#include "strategy/daily.hpp"

namespace {
using namespace strat::daily;

class Hvp : public strat::Strategy {
 public:
  std::string about() const override { return "high-volume return premium: days of extreme volume, held a week (Gervais, Kaniel and Mingelgrin 2001)"; }
  strat::Params defaults() const override {
    return {{"base", 49}, {"minvol", 2.5}, {"hold", 5}, {"rebalance", 1}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 10}};
  }
  // The largest volume shock (volume over the usual) of the last `hold` sessions on a day that did not close lower, and how long ago.
  static double shock(const strat::View& v, const strat::Params& p, std::size_t i, std::size_t* ago = nullptr) {
    const std::size_t t = v.t(), Bv = std::size_t(p.at("base")), H = std::size_t(p.at("hold"));
    double best = strat::kNaN;
    for (std::size_t k = 0; k < H && t >= k + Bv + 1; ++k) {
      const double r = v.ret(t - k, i), x = relVolume(v, t - k, 1, Bv, i);
      if (std::isfinite(r) && r >= 0 && x >= p.at("minvol") && !(x <= best)) best = x, ago && (*ago = k);
    }
    return best;
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    out.assign(v.N(), strat::kNaN);
    if (v.t() < std::size_t(p.at("base") + p.at("hold")) + 1) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) out[i] = std::log(shock(v, p, i));
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    std::size_t ago = 0;
    const double x = shock(v, p, i, &ago);
    char b[200];
    std::snprintf(b, sizeof b, "%zu session(s) ago it traded %.1f times its usual volume without falling: extreme-volume days have tended to be followed by gains.", ago, x);
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("hvp", Hvp);
