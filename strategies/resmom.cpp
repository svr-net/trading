// Residual momentum (Blitz, Huij and Martens 2011): momentum in the part of a share's return the market does not
// explain. Each share's daily moves over the last year regressed on the market's; the residuals of the 12 months to a
// month ago summed and divided by their volatility. About twice raw momentum's risk-adjusted return, and far smaller
// crashes when the market turns (raw momentum then holds the high-beta shares).
#include <cmath>
#include <cstdio>

#include "strategy/strategy.hpp"

namespace {

class ResMom : public strat::Strategy {
 public:
  std::string about() const override { return "residual momentum: the share's own trend net of the market's, rebalanced monthly (Blitz, Huij and Martens 2011)"; }
  strat::Params defaults() const override {
    return {{"lookback", 252}, {"skip", 21}, {"rebalance", 21}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 260}};
  }
  // The residuals' sum over [t-L, t-S] over their volatility, and the share's beta; NaN without enough days.
  static double resid(const strat::View& v, std::size_t t, std::size_t L, std::size_t S, std::size_t i, double* betaOut = nullptr) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
    for (std::size_t s = t + 1 - L; s <= t - S; ++s) {
      const double y = v.ret(s, i), x = v.market(s);
      if (!std::isfinite(y) || !std::isfinite(x)) continue;
      sx += x, sy += y, sxx += x * x, sxy += x * y, n += 1;
    }
    if (n < 120) return strat::kNaN;
    const double vx = sxx / n - (sx / n) * (sx / n), beta = vx > 0 ? (sxy / n - sx / n * sy / n) / vx : 1.0, alpha = sy / n - beta * sx / n;
    double se = 0, se2 = 0;
    for (std::size_t s = t + 1 - L; s <= t - S; ++s) {
      const double y = v.ret(s, i), x = v.market(s);
      if (!std::isfinite(y) || !std::isfinite(x)) continue;
      const double e = y - alpha - beta * x;
      se += e, se2 += e * e;
    }
    const double sd = std::sqrt(std::max(0.0, se2 / n - (se / n) * (se / n)));
    if (betaOut) *betaOut = beta;
    return sd > 0 ? (se + n * alpha) / (sd * std::sqrt(n)) : strat::kNaN;  // the residuals with their mean (the share's own drift)
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t t = v.t(), L = std::size_t(p.at("lookback")), S = std::size_t(p.at("skip"));
    out.assign(v.N(), strat::kNaN);
    if (t < L) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) out[i] = resid(v, t, L, S, i);
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    double beta = 1;
    const double z = resid(v, v.t(), std::size_t(p.at("lookback")), std::size_t(p.at("skip")), i, &beta);
    char b[200];
    std::snprintf(b, sizeof b, "Its own trend over the 12 months to a month ago, with the market's moves taken out (beta %.2f): %.1f standard deviations.", beta, z);
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("resmom", ResMom);
