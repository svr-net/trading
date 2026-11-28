// The technical signals: every share, every day, before any filter (the universe's liquidity, the volatility
// filter, the stops' recovery): the classic indicators' buy and sell events on its daily bars, and what each has
// been worth since, learnt walk-forward on the next session (its open to its close: the same-day trade), in the
// share's daily volatilities.
//
// The events (the indicators' textbook settings; what they are worth is learnt, nothing assumed):
//   RSI(14) up through 30 / down through 70          RSI(2) under 10 over its 200-day mean / over 90 under it
//   the close back over the lower Bollinger band (20, 2) / back under the upper
//   the Parabolic SAR (0.02, 0.2) turning up / down  MACD (12, 26, 9) over its signal line / under it
//   +DI over -DI with ADX(14) over 20 / -DI over +DI   a close over the last 20 sessions' high / under their low
//   the 50-day mean over the 200-day / under it
//
// The chance of an up session given the day's events (naive Bayes: each event's own odds over the market's, shrunk
// to the market's by 200 cases) uses only the outcomes known on the day (events up to the day before).
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "strategy/strategy.hpp"

namespace strat {

struct Signals {
  static constexpr int K = 16;
  static constexpr std::array<const char*, K> name = {
      "RSI(14) up through 30",  "RSI(2) under 10 in an uptrend", "back over the lower Bollinger band", "Parabolic SAR turned up",
      "MACD over its signal",   "+DI over -DI, ADX over 20",     "20-session high",                    "50-day mean over the 200-day",
      "RSI(14) down through 70", "RSI(2) over 90 in a downtrend", "back under the upper Bollinger band", "Parabolic SAR turned down",
      "MACD under its signal",  "-DI over +DI, ADX over 20",     "20-session low",                     "50-day mean under the 200-day"};
  static bool buy(int e) { return e < K / 2; }

  std::size_t T = 0, N = 0;
  std::vector<std::uint32_t> ev;  // [t * N + i]: the events of share i at day t's close
  // cum[t]: the outcomes known at day t's close (events of days < t): per event n, ups, sum and sum of squares of
  // the next session's move in volatilities; then the same for every share-day (the market's base)
  std::vector<std::array<double, 4 * (K + 1)>> cum;

  void compute(const Data& d) {
    T = d.T, N = d.N;
    ev.assign(T * N, 0u);
    for (std::size_t i = 0; i < N; ++i) share(d, i);
    cum.assign(T + 1, {});
    std::array<double, 4 * (K + 1)> run{};
    for (std::size_t t = 0; t < T; ++t) {
      cum[t] = run;  // known at t: events of days < t (their next session is <= t)
      if (t + 1 >= T) continue;
      for (std::size_t i = 0; i < N; ++i) {
        const double s = d.sigma[t * N + i], o = d.O[(t + 1) * N + i], c = d.C[(t + 1) * N + i];
        if (!(std::isfinite(s) && s > 0 && o > 0 && c > 0)) continue;
        const double z = std::max(-10.0, std::min(10.0, (c / o - 1) / s));
        const auto add = [&](int e) { double* a = &run[4 * std::size_t(e)]; a[0] += 1, a[1] += z > 0, a[2] += z, a[3] += z * z; };
        add(K);
        for (int e = 0; e < K; ++e) if (ev[t * N + i] >> e & 1u) add(e);
      }
    }
    cum[T] = run;
  }

  // P(an up session | share i's events at day t's close), from the outcomes known at t
  double pUp(std::size_t t, std::size_t i) const {
    const auto& c = cum[t];
    const double nB = c[4 * K], uB = c[4 * K + 1];
    if (nB < 1000) return kNaN;
    const double pB = uB / nB, lB = logit(pB);
    double l = lB;
    for (int e = 0; e < K; ++e)
      if (ev[t * N + i] >> e & 1u) l += logit((c[4 * e + 1] + 200 * pB) / (c[4 * e] + 200)) - lB;
    return 1 / (1 + std::exp(-l));
  }
  // each event's record at day t: n, P(up), the market's P(up), mean move (volatilities), its standard error
  std::array<double, 5> record(std::size_t t, int e) const {
    const auto& c = cum[t];
    const double n = c[4 * e], nB = c[4 * K];
    const double m = n > 0 ? c[4 * e + 2] / n : kNaN, v = n > 1 ? (c[4 * e + 3] - n * m * m) / (n - 1) : kNaN;
    return {n, n > 0 ? c[4 * e + 1] / n : kNaN, nB > 0 ? c[4 * K + 1] / nB : kNaN, m, n > 1 ? std::sqrt(std::max(0.0, v) / n) : kNaN};
  }
  std::string names(std::size_t t, std::size_t i) const {
    std::string s;
    for (int e = 0; e < K; ++e) if (ev[t * N + i] >> e & 1u) s += (s.empty() ? "" : "; ") + std::string(name[std::size_t(e)]);
    return s;
  }

 private:
  static double logit(double p) { p = std::min(1 - 1e-4, std::max(1e-4, p)); return std::log(p / (1 - p)); }

  void share(const Data& d, std::size_t i) {
    std::vector<double> cl;  // the share's closes so far (its bars only)
    double g14 = 0, l14 = 0, g2 = 0, l2 = 0, rsi14p = kNaN;
    double e12 = 0, e26 = 0, sig = 0, macdP = kNaN, sigP = kNaN;
    double tr = 0, pdm = 0, mdm = 0, adx = 0, pdiP = kNaN, mdiP = kNaN;
    double sar = 0, ep = 0, af = 0.02;
    bool up = true;
    double pc = kNaN, ph = kNaN, pl = kNaN, sma50P = kNaN, sma200P = kNaN, lowerP = kNaN, upperP = kNaN;
    std::vector<double> hi, lo;
    std::size_t n = 0;
    for (std::size_t t = 0; t < d.T; ++t) {
      const double H = d.H[t * N + i], L = d.L[t * N + i], C = d.C[t * N + i];
      if (!(H > 0 && L > 0 && C > 0)) continue;
      std::uint32_t m = 0;
      ++n;
      if (n == 1) {
        e12 = e26 = C, sar = L, ep = H;
        cl.push_back(C), hi.push_back(H), lo.push_back(L), pc = C, ph = H, pl = L;
        continue;
      }
      // RSI(14) and RSI(2), Wilder's smoothing
      const double ch = C - pc, gn = std::max(0.0, ch), ls = std::max(0.0, -ch);
      g14 = n <= 15 ? g14 + gn / 14 : (g14 * 13 + gn) / 14, l14 = n <= 15 ? l14 + ls / 14 : (l14 * 13 + ls) / 14;
      g2 = (g2 + gn) / 2, l2 = (l2 + ls) / 2;
      const double rsi14 = n > 15 ? (l14 > 0 ? 100 - 100 / (1 + g14 / l14) : 100) : kNaN;
      const double rsi2 = n > 3 ? (l2 > 0 ? 100 - 100 / (1 + g2 / l2) : 100) : kNaN;
      // the means and the Bollinger band
      cl.push_back(C);
      const auto mean = [&](std::size_t k) {
        if (cl.size() < k) return kNaN;
        double s = 0;
        for (std::size_t q = cl.size() - k; q < cl.size(); ++q) s += cl[q];
        return s / double(k);
      };
      const double sma20 = mean(20), sma50 = mean(50), sma200 = mean(200);
      double sd20 = kNaN;
      if (std::isfinite(sma20)) {
        double s = 0;
        for (std::size_t q = cl.size() - 20; q < cl.size(); ++q) s += (cl[q] - sma20) * (cl[q] - sma20);
        sd20 = std::sqrt(s / 20);
      }
      const double lower = sma20 - 2 * sd20, upper = sma20 + 2 * sd20;
      // MACD
      e12 += (C - e12) * 2 / 13, e26 += (C - e26) * 2 / 27;
      const double macd = e12 - e26;
      sig = n <= 2 ? macd : sig + (macd - sig) * 2 / 10;
      // +DI, -DI and the ADX (Wilder, 14)
      const double trd = std::max({H - L, std::fabs(H - pc), std::fabs(L - pc)});
      const double upm = H - ph, dnm = pl - L;
      const double pd = upm > dnm && upm > 0 ? upm : 0, md = dnm > upm && dnm > 0 ? dnm : 0;
      tr = n <= 15 ? tr + trd : tr - tr / 14 + trd, pdm = n <= 15 ? pdm + pd : pdm - pdm / 14 + pd, mdm = n <= 15 ? mdm + md : mdm - mdm / 14 + md;
      const double pdi = tr > 0 ? 100 * pdm / tr : kNaN, mdi = tr > 0 ? 100 * mdm / tr : kNaN;
      const double dx = pdi + mdi > 0 ? 100 * std::fabs(pdi - mdi) / (pdi + mdi) : 0;
      adx = n <= 15 ? dx : (adx * 13 + dx) / 14;
      // the Parabolic SAR (0.02, 0.2)
      bool turnUp = false, turnDn = false;
      sar = sar + af * (ep - sar);
      if (up) {
        sar = std::min({sar, pl, lo.size() > 1 ? lo[lo.size() - 2] : pl});
        if (L < sar) up = false, turnDn = true, sar = ep, ep = L, af = 0.02;
        else if (H > ep) ep = H, af = std::min(0.2, af + 0.02);
      } else {
        sar = std::max({sar, ph, hi.size() > 1 ? hi[hi.size() - 2] : ph});
        if (H > sar) up = true, turnUp = true, sar = ep, ep = H, af = 0.02;
        else if (L < ep) ep = L, af = std::min(0.2, af + 0.02);
      }
      // the last 20 sessions' high and low (before today)
      double hh = kNaN, ll = kNaN;
      if (hi.size() >= 20) {
        hh = 0, ll = 1e300;
        for (std::size_t q = hi.size() - 20; q < hi.size(); ++q) hh = std::max(hh, hi[q]), ll = std::min(ll, lo[q]);
      }
      if (n > 35) {
        const auto set = [&](int e, bool on) { if (on) m |= 1u << e; };
        set(0, rsi14p < 30 && rsi14 >= 30);
        set(1, std::isfinite(sma200) && rsi2 < 10 && C > sma200);
        set(2, std::isfinite(lowerP) && pc < lowerP && C >= lower);
        set(3, turnUp);
        set(4, macdP <= sigP && macd > sig);
        set(5, adx > 20 && pdiP <= mdiP && pdi > mdi);
        set(6, std::isfinite(hh) && C > hh);
        set(7, std::isfinite(sma200P) && sma50P <= sma200P && sma50 > sma200);
        set(8, rsi14p > 70 && rsi14 <= 70);
        set(9, std::isfinite(sma200) && rsi2 > 90 && C < sma200);
        set(10, std::isfinite(upperP) && pc > upperP && C <= upper);
        set(11, turnDn);
        set(12, macdP >= sigP && macd < sig);
        set(13, adx > 20 && mdiP <= pdiP && mdi > pdi);
        set(14, std::isfinite(ll) && C < ll);
        set(15, std::isfinite(sma200P) && sma50P >= sma200P && sma50 < sma200);
      }
      ev[t * N + i] = m;
      rsi14p = rsi14, macdP = macd, sigP = sig, pdiP = pdi, mdiP = mdi, sma50P = sma50, sma200P = sma200, lowerP = lower, upperP = upper;
      hi.push_back(H), lo.push_back(L), pc = C, ph = H, pl = L;
      if (cl.size() > 400) cl.erase(cl.begin(), cl.begin() + 200);
      if (hi.size() > 400) hi.erase(hi.begin(), hi.begin() + 200), lo.erase(lo.begin(), lo.begin() + 200);
    }
  }
};

}  // namespace strat
