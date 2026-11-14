// Short-term residual reversal (Blitz, Huij, Lansdorp and Verbeek 2013): the shares that fell most over the last
// week against the market (their return less their beta times the market's), in volatilities, are bought and
// held a week. The residual form keeps a return after reasonable trading costs where the plain one does not.
#include <cmath>
#include <cstdio>

#include "strategy/strategy.hpp"

namespace {

class Reversal : public strat::Strategy {
 public:
  std::string about() const override { return "residual reversal: the week's largest falls against the market, held a week (Blitz et al. 2013)"; }
  strat::Params defaults() const override {
    return {{"window", 5}, {"beta", 252}, {"rebalance", 5}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 260}};
  }
  // The share's residual return over the last `window` sessions, in its volatilities over them.
  static double resid(const strat::View& v, std::size_t i, std::size_t W, std::size_t B) {
    const std::size_t t = v.t();
    double sxy = 0, sxx = 0, sx = 0, sy = 0, n = 0;
    for (std::size_t s = t + 1 - B; s <= t; ++s) {
      const double y = v.ret(s, i), x = v.market(s);
      if (std::isfinite(y)) sxy += x * y, sxx += x * x, sx += x, sy += y, n += 1;
    }
    if (n < 0.8 * double(B)) return strat::kNaN;
    const double vx = sxx / n - (sx / n) * (sx / n), beta = vx > 0 ? (sxy / n - (sx / n) * (sy / n)) / vx : 1.0;
    double r = 0;
    for (std::size_t s = t + 1 - W; s <= t; ++s) {
      const double y = v.ret(s, i);
      if (!std::isfinite(y)) return strat::kNaN;
      r += y - beta * v.market(s);
    }
    const double sg = v.sigma(t, i);
    return sg > 0 ? r / (sg * std::sqrt(double(W))) : strat::kNaN;
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t W = std::size_t(p.at("window")), B = std::size_t(p.at("beta"));
    out.assign(v.N(), strat::kNaN);
    if (v.t() + 1 < B) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) {
          const double z = resid(v, i, W, B);
          if (std::isfinite(z) && z < 0) out[i] = -z;
        }
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    const std::size_t W = std::size_t(p.at("window")), t = v.t();
    char b[220];
    std::snprintf(b, sizeof b, "It fell %+.1f%% over the last week, %.1f volatilities more than the market explains: such falls have tended to recover.",
                  100 * v.change(t - W, t, i), -resid(v, i, W, std::size_t(p.at("beta"))));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("reversal", Reversal);
