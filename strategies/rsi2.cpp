// Connors' 2-day RSI pullback (Connors and Alvarez 2009, "Short Term Trading Strategies That Work"; StockCharts'
// RSI(2) and many later tests): a share in a long uptrend (its close over its 200-day average) whose 2-day RSI has
// fallen under 10 is bought at the next open and held until it closes back over its 5-day average (at most
// `maxdays` sessions). Daily: the book is decided after every close. The deepest pullbacks (lowest RSI) first.
#include <algorithm>
#include <cstdio>

#include "strategy/daily.hpp"

namespace {
using namespace strat::daily;

class Rsi2 : public strat::Strategy {
 public:
  std::string about() const override { return "RSI(2) pullback in an uptrend: RSI(2) under 10 over the 200-day average, out over the 5-day average; daily (Connors and Alvarez 2009)"; }
  strat::Params defaults() const override {
    return {{"entry", 10}, {"trend", 200}, {"exit", 5}, {"hold", 5}, {"rebalance", 1}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 10}};
  }
  // The lowest 2-day RSI since the trade opened (within `hold` sessions), while the share has stayed under its
  // exit average (the trade still open); NaN when no trade is open.
  static double open(const strat::View& v, const strat::Params& p, std::size_t i) {
    const std::size_t t = v.t(), T = std::size_t(p.at("trend")), X = std::size_t(p.at("exit")), H = std::size_t(p.at("hold"));
    const double c = v.close(t, i), m200 = sma(v, t, T, i), m5 = sma(v, t, X, i);
    if (!(c > 0) || !(c > m200) || !(c < m5)) return strat::kNaN;
    double lo = strat::kNaN;
    for (std::size_t k = 0; k < H && k + 31 < t; ++k) {
      if (k > 0 && v.close(t - k, i) > sma(v, t - k, X, i)) break;  // closed over its exit average then: an earlier trade ended
      const double r = rsi(v, t - k, 2, i);
      if (std::isfinite(r)) lo = std::isfinite(lo) ? std::min(lo, r) : r;
    }
    return lo < p.at("entry") ? lo : strat::kNaN;
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    out.assign(v.N(), strat::kNaN);
    if (v.t() < std::size_t(p.at("trend")) + 31) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i)) {
          const double r = open(v, p, i);
          if (std::isfinite(r)) out[i] = p.at("entry") - r;
        }
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    const std::size_t t = v.t();
    char b[220];
    std::snprintf(b, sizeof b, "Over its 200-day average (%+.1f%%) but its 2-day RSI fell to %.1f: a pullback in an uptrend, held until it closes over its 5-day average.",
                  100 * (v.close(t, i) / sma(v, t, std::size_t(p.at("trend")), i) - 1), open(v, p, i));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("rsi2", Rsi2);
