// Injectable strategies for the daily scan (apps/ofm_strategy).
//
// A strategy is one source file in strategies/ that defines a class derived from strat::Strategy and
// registers it with REGISTER_STRATEGY. The build compiles every file there into apps/ofm_strategy (CMake
// globs the folder), and config/markets.json chooses a market's strategy by name (--strategy NAME), with
// its parameters overridable (--param name=value). Nothing else changes to add one.
//
// A strategy only scores the day's shares. It sees the market through a View, which ends at the day's close:
// a read of any later day throws, so no strategy can look ahead. The engine does the rest, the same for
// every strategy: the tradable shares, the book of `slots` positions rebalanced every `rebalance` sessions
// (the highest scores held), the buys in the opening auction under a price cap, the walk-forward backtest
// (before any cost), each order's expected return from the strategy's own past trades, and, at the very end,
// the broker's costs applied to the finished order list.
#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "strategy/backend.hpp"

namespace strat {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// The market as known after the close of day t: every read of a later day throws.
struct Data {
  std::size_t T = 0, N = 0;
  std::vector<std::string> days, tickers;
  std::vector<double> O, H, L, C, V;  // the bars (NaN: no bar), row-major [day * N + share]
  std::vector<double> Cf;             // the close carried forward over a few missing bars (the signals' price)
  std::vector<double> ret;            // the day's log return of Cf (NaN before a share's first bar)
  std::vector<double> sigma;          // the daily volatility: 60 sessions of ret
  std::vector<double> value;          // the median traded value of the last 20 bars (the market's currency)
  std::vector<double> mkt;            // the day's mean return of the tradable shares
  std::vector<std::size_t> first;     // each share's first bar
};

class View {
 public:
  View(const Data& d, std::size_t t, const char* tradable, const Backend& b) : d_(d), t_(t), in_(tradable), b_(b) {}
  // The chosen back end's parallel loop over [0, n), for a strategy's own per-share work (write only index i's output).
  void parallel(std::size_t n, const std::function<void(std::size_t, std::size_t)>& f) const { b_.parallelFor(n, f); }
  std::size_t t() const { return t_; }
  std::size_t N() const { return d_.N; }
  const std::string& ticker(std::size_t i) const { return d_.tickers[i]; }
  const std::string& day(std::size_t s) const { return d_.days[check(s)]; }
  bool tradable(std::size_t i) const { return in_[i] != 0; }             // in the day's universe
  bool bar(std::size_t s, std::size_t i) const { return std::isfinite(d_.C[at(s, i)]); }  // a real bar on day s
  double close(std::size_t s, std::size_t i) const { return d_.Cf[at(s, i)]; }
  double open(std::size_t s, std::size_t i) const { return d_.O[at(s, i)]; }
  double high(std::size_t s, std::size_t i) const { return d_.H[at(s, i)]; }    // NaN: no bar
  double low(std::size_t s, std::size_t i) const { return d_.L[at(s, i)]; }     // NaN: no bar
  double volume(std::size_t s, std::size_t i) const { return d_.V[at(s, i)]; }  // NaN: no bar (0: no volumes)
  double ret(std::size_t s, std::size_t i) const { return d_.ret[at(s, i)]; }
  double sigma(std::size_t s, std::size_t i) const { return d_.sigma[at(s, i)]; }
  double value(std::size_t s, std::size_t i) const { return d_.value[at(s, i)]; }
  double market(std::size_t s) const { return d_.mkt[check(s)]; }
  std::size_t first(std::size_t i) const { return d_.first[i]; }
  // The return of the carried-forward close from day a to day b (a < b <= t), NaN when either is missing.
  double change(std::size_t a, std::size_t b, std::size_t i) const {
    const double x = close(a, i), y = close(b, i);
    return x > 0 && y > 0 ? y / x - 1 : kNaN;
  }

 private:
  std::size_t check(std::size_t s) const {
    if (s > t_) throw std::logic_error("look-ahead: a strategy read a day after the close it decides on");
    return s;
  }
  std::size_t at(std::size_t s, std::size_t i) const { return check(s) * d_.N + i; }
  const Data& d_;
  std::size_t t_;
  const char* in_;
  const Backend& b_;
};

using Params = std::map<std::string, double>;

class Strategy {
 public:
  virtual ~Strategy() = default;
  virtual std::string about() const = 0;  // one line: what it does and where the evidence is
  // Its parameters and their defaults. The engine reads: rebalance (sessions between rebalances), warmup
  // (sessions of history a share needs), capsd and capmax (the buy cap over the close: capsd daily
  // volatilities, at most capmax), maxdays (the longest a position is held, a safety).
  virtual Params defaults() const = 0;
  // The day's score of every share (higher: more wanted); NaN where it is not wanted at all. Only the
  // tradable shares with a finite score are considered; the book holds the `slots` highest.
  virtual void score(const View& v, const Params& p, std::vector<double>& out) const = 0;
  // Why share i scores as it does, in a sentence (the page's pop-up).
  virtual std::string reason(const View& v, const Params& p, std::size_t i) const = 0;
};

using Factory = std::function<std::unique_ptr<Strategy>()>;

inline std::map<std::string, Factory>& registry() {
  static std::map<std::string, Factory> r;
  return r;
}

struct Register {
  Register(const char* name, Factory f) { registry()[name] = std::move(f); }
};

}  // namespace strat

#define REGISTER_STRATEGY(NAME, CLASS) \
  static const strat::Register strat_register_##CLASS(NAME, [] { return std::unique_ptr<strat::Strategy>(new CLASS()); })
