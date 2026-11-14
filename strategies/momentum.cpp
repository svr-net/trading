// Cross-sectional momentum (Jegadeesh and Titman 1993): the shares with the highest return over the last 12
// months, skipping the latest month (its short-term reversal), held a month. Among the anomalies that replicate
// best (Hou, Xue and Zhang 2020; Jensen, Kelly and Pedersen 2023); with a monthly rebalance its turnover is low
// enough to keep a return after costs (Novy-Marx and Velikov 2016).
#include <cstdio>

#include "strategy/strategy.hpp"

namespace {

class Momentum : public strat::Strategy {
 public:
  std::string about() const override { return "12-month momentum skipping the last month, rebalanced monthly (Jegadeesh and Titman 1993)"; }
  strat::Params defaults() const override {
    return {{"lookback", 252}, {"skip", 21}, {"rebalance", 21}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 260}};
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t t = v.t(), L = std::size_t(p.at("lookback")), S = std::size_t(p.at("skip"));
    out.assign(v.N(), strat::kNaN);
    if (t < L) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) out[i] = v.change(t - L, t - S, i);
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    const std::size_t t = v.t(), L = std::size_t(p.at("lookback")), S = std::size_t(p.at("skip"));
    char b[200];
    std::snprintf(b, sizeof b, "Its return over the 12 months to a month ago: %+.1f%%; over the last month %+.1f%% (left out: short-term moves tend to reverse).",
                  100 * v.change(t - L, t - S, i), 100 * v.change(t - S, t, i));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("momentum", Momentum);
