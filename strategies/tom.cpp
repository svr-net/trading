// The turn of the month (Lakonishok and Smidt 1988; McConnell and Xu 2008: in US data 1926-2005 the market's whole
// return came in the four sessions from the month's last to its third, in 31 of 35 countries): shares are held
// only for the session the window covers, the highest betas first (they carry most of the market's move), among
// those over their 200-day average. The window is read from the calendar and the sessions seen so far, never a
// later day: the next session is the month's last when it is the last weekday before the month ends (holidays
// can shift it a day), or among its first `days` when at most `days` - 1 of the month's sessions have passed.
#include <cstdio>
#include <cstdlib>

#include "strategy/daily.hpp"

namespace {
using namespace strat::daily;

// Days since 1970-01-01 of a civil date (Hinnant's algorithm).
long civil(int y, int m, int d) {
  y -= m <= 2;
  const long era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = unsigned(y - era * 400), doy = unsigned((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1), doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + long(doe) - 719468;
}
struct Ymd { int y, m, d; };
Ymd parse(const std::string& s) { return {std::atoi(s.substr(0, 4).c_str()), std::atoi(s.substr(5, 2).c_str()), std::atoi(s.substr(8, 2).c_str())}; }
Ymd fromDays(long z) {
  z += 719468;
  const long era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = unsigned(z - era * 146097), yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100), mp = (5 * doy + 2) / 153, d = doy - (153 * mp + 2) / 5 + 1, m = mp < 10 ? mp + 3 : mp - 9;
  return {int(long(yoe) + era * 400 + (m <= 2)), int(m), int(d)};
}
bool weekday(long z) { const long w = ((z % 7) + 11) % 7; return w < 5; }  // 0 = Monday (1970-01-01 was a Thursday)
long nextWeekday(long z) { do ++z; while (!weekday(z)); return z; }

class Tom : public strat::Strategy {
 public:
  std::string about() const override { return "turn of the month: held from the month's last session to its third, high betas first (Lakonishok and Smidt 1988; McConnell and Xu 2008)"; }
  strat::Params defaults() const override {
    return {{"days", 3}, {"trend", 200}, {"beta", 126}, {"rebalance", 1}, {"warmup", 260}, {"capsd", 1.5}, {"capmax", 0.03}, {"maxdays", 6}};
  }
  // Whether the session after day t falls in the window (0: no; 1: the month's last; 2: among its first `days`).
  static int window(const strat::View& v, std::size_t t, std::size_t days) {
    const Ymd a = parse(v.day(t));
    const long z = civil(a.y, a.m, a.d), n1 = nextWeekday(z), n2 = nextWeekday(n1);
    const Ymd b = fromDays(n1), c = fromDays(n2);
    if (b.m != a.m) return 2;               // the next session opens a new month
    if (c.m != b.m) return 1;               // the next session is the month's last weekday
    std::size_t seen = 0;                   // the month's sessions up to t
    for (std::size_t s = t + 1; s-- > 0 && seen <= days;) {
      const Ymd q = parse(v.day(s));
      if (q.m != a.m || q.y != a.y) break;
      ++seen;
    }
    return seen < days ? 2 : 0;
  }
  void score(const strat::View& v, const strat::Params& p, std::vector<double>& out) const override {
    const std::size_t t = v.t(), T = std::size_t(p.at("trend")), B = std::size_t(p.at("beta"));
    out.assign(v.N(), strat::kNaN);
    if (t + 1 < std::max(T, B) || !window(v, t, std::size_t(p.at("days")))) return;
    v.parallel(v.N(), [&](std::size_t a, std::size_t e) {
      for (std::size_t i = a; i < e; ++i)
        if (v.tradable(i) && (T == 0 || v.close(t, i) > sma(v, t, T, i))) out[i] = beta(v, t, B, i);
    });
  }
  std::string reason(const strat::View& v, const strat::Params& p, std::size_t i) const override {
    const std::size_t t = v.t();
    char b[220];
    std::snprintf(b, sizeof b, "The next session is in the turn of the month (%s), when markets have earned most of their return; its beta of %.2f carries more of that move.",
                  window(v, t, std::size_t(p.at("days"))) == 1 ? "the month's last" : "the month's first days", beta(v, t, std::size_t(p.at("beta")), i));
    return b;
  }
};

}  // namespace

REGISTER_STRATEGY("tom", Tom);
