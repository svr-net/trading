#include "ofm/intrabook.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

#include "ofm/liquidity.hpp"
#include "ofm/trade.hpp"

#include <cstdio>
#include <map>

namespace ofm {

namespace {

// Least squares by normal equations over every observation so far (as in the daily trades).
struct Fit {
  std::size_t k;
  std::vector<double> A, b;
  double n = 0;
  explicit Fit(std::size_t dim) : k(dim), A(dim * dim, 0.0), b(dim, 0.0) {}
  void add(const std::vector<double>& x, double y) {
    for (std::size_t i = 0; i < k; ++i) {
      b[i] += x[i] * y;
      for (std::size_t j = 0; j < k; ++j) A[i * k + j] += x[i] * x[j];
    }
    n += 1;
  }
  std::vector<double> solve() const {
    if (n <= double(k)) return {};
    std::vector<double> M = A, v = b;
    double tr = 0;
    for (std::size_t i = 0; i < k; ++i) tr += M[i * k + i];
    for (std::size_t i = 0; i < k; ++i) M[i * k + i] += 1e-12 * tr;
    for (std::size_t c = 0; c < k; ++c) {
      std::size_t piv = c;
      for (std::size_t r = c + 1; r < k; ++r)
        if (std::fabs(M[r * k + c]) > std::fabs(M[piv * k + c])) piv = r;
      if (!(std::fabs(M[piv * k + c]) > 0)) return {};
      if (piv != c) {
        for (std::size_t j = 0; j < k; ++j) std::swap(M[c * k + j], M[piv * k + j]);
        std::swap(v[c], v[piv]);
      }
      for (std::size_t r = c + 1; r < k; ++r) {
        const double q = M[r * k + c] / M[c * k + c];
        for (std::size_t j = c; j < k; ++j) M[r * k + j] -= q * M[c * k + j];
        v[r] -= q * v[c];
      }
    }
    std::vector<double> x(k);
    for (std::size_t c = k; c-- > 0;) {
      double s = v[c];
      for (std::size_t j = c + 1; j < k; ++j) s -= M[c * k + j] * x[j];
      x[c] = s / M[c * k + c];
    }
    return x;
  }
  double predict(const std::vector<double>& beta, const std::vector<double>& x) const {
    double s = 0;
    for (std::size_t i = 0; i < k; ++i) s += beta[i] * x[i];
    return s;
  }
};

// A trade bought at `entry` after bar k-1, sold at the stop, the take-profit or the last bar's
// close: net return. stopFirst: when one bar touched both, the stop is taken. exit: 0 close, 1 stop,
// 2 take-profit.
double pathTrade(const std::vector<day::Tick>& p, std::size_t k, double entry, double stop, double take, double buy, double sell, bool stopFirst,
                 int* exit) {
  double px = p.back().c;
  *exit = 0;
  for (std::size_t j = k; j < p.size(); ++j) {
    const bool hs = p[j].l <= stop, ht = p[j].h >= take;
    if (hs && (stopFirst || !ht)) {
      px = stop, *exit = 1;
      break;
    }
    if (ht) {
      px = take, *exit = 2;
      break;
    }
  }
  return (px / entry) * (1 - sell) / (1 + buy) - 1;
}

}  // namespace

IntraBook intraBook(const Market& m, const Forecast& f, const day::Paths& paths, double buy, double sell, std::size_t universe) {
  IntraBook book;
  const std::size_t T = m.T(), N = m.N();
  if (paths.empty() || f.horizons.empty()) return book;
  const std::size_t kEntry[4] = {1, 2, 4, 8}, NK = 4, BOOK = 10;
  // Stop and take-profit multiples of the volatility for the rest of the day, from the model's
  // horizons (h / h_mid), as in the daily trades.
  std::vector<double> mult;
  for (auto h : f.horizons) mult.push_back(double(h) / double(f.horizons[f.horizons.size() / 2]));
  const std::size_t G = mult.size(), P = G * G;
  const double halfLife = double(f.horizons[f.horizons.size() / 2]), lam = std::exp(-std::log(2.0) / halfLife);
  const Liquidity liq(m, halfLife);
  // Each stock's open-to-close volatility (exponentially weighted, through the last close).
  std::vector<double> var(N, 0.0), varN(N, 0.0);
  auto bar = [&](std::size_t t, std::size_t i) {
    return m.open(t, i) > 0 && m.high(t, i) > m.low(t, i) && m.low(t, i) > 0 && m.close(t, i) > 0;
  };
  const std::size_t F = 7;  // [1, opening move, market's opening move, gap, last day's open-to-close, close in its range, model forecast]
  book.featureNames = {"opening move", "market's opening move", "overnight gap", "last day's open-to-close", "last close in its range", "model forecast"};
  std::vector<Fit> fit(NK, Fit(F));
  // The inputs standardised by everything seen so far and cut at Chauvenet's bound, so a few wild
  // early values cannot dominate; no forecast before the fit has seen as many days as the model's
  // middle horizon (its learnt time scale).
  std::vector<day::Scaler> scal(NK, day::Scaler(F - 1));
  const std::size_t warmDays = f.horizons[f.horizons.size() / 2];
  std::vector<std::vector<Fit>> pairFit(NK, std::vector<Fit>(P, Fit(2)));
  std::vector<day::Adwin> rec(NK);
  for (auto& a : rec) a.horizon = double(T);
  for (auto& fs : book.featureSkill) fs.resize(F - 1);
  // The first day with a path: the walk starts there (the volatility learns from every day before).
  std::size_t first = T;
  for (const auto& kv : paths) first = std::min(first, kv.first / N);
  for (std::size_t d = 1; d < T; ++d) {
    const std::size_t t = d - 1;
    for (std::size_t i = 0; i < N; ++i)
      if (bar(t, i)) {
        const double r = std::log(m.close(t, i) / m.open(t, i));
        var[i] = varN[i] == 0 ? r * r : lam * var[i] + (1 - lam) * r * r, varN[i] += 1;
      }
    if (d < first) continue;
    // Candidates: the day's universe (the most traded at the close t), a volatility, a path on day d.
    const auto inU = liq.universeAt(t, universe);
    struct Cand {
      std::size_t i;
      const std::vector<day::Tick>* p;
      double sig, gap, lastOC, lastLoc, fc;
    };
    std::vector<Cand> cand;
    for (std::size_t i = 0; i < N; ++i) {
      if (!inU[i] || varN[i] < 2 || !(var[i] > 0) || !bar(t, i) || !(m.open(d, i) > 0)) continue;
      const auto it = paths.find(d * N + i);
      if (it == paths.end() || it->second.size() <= kEntry[NK - 1]) continue;
      const double sig = std::sqrt(var[i]), o = m.open(d, i);
      const double hl = m.high(t, i) - m.low(t, i);
      cand.push_back({i, &it->second, sig, std::log(o / m.close(t, i)) / sig, std::log(m.close(t, i) / m.open(t, i)) / sig,
                      (m.close(t, i) - m.low(t, i)) / hl - 0.5, std::isfinite(f.E(t, i)) ? f.E(t, i) / sig : 0.0});
    }
    if (cand.empty()) continue;
    book.candidates += cand.size();
    book.days.push_back(d);
    // Each entry's book.
    std::size_t use = 0;
    for (std::size_t k = 1; k < NK; ++k)
      if (rec[k].score() > rec[use].score()) use = k;
    const bool trade = rec[use].score() > 0;
    double dayRet = 0, dayOpt = 0;
    const bool warm = book.days.size() > warmDays;
    for (std::size_t kk = 0; kk < NK; ++kk) {
      const std::size_t k = kEntry[kk];
      const auto beta = warm ? fit[kk].solve() : std::vector<double>{};
      const double zf = day::chauvenet(fit[kk].n + 1);
      std::vector<std::vector<double>> ab(P);
      for (std::size_t q = 0; q < P; ++q) ab[q] = pairFit[kk][q].solve();
      // The market's opening move: the candidates' mean.
      double mo = 0;
      for (const auto& c : cand) mo += std::log((*c.p)[k - 1].c / m.open(d, c.i)) / c.sig;
      mo /= double(cand.size());
      // Each candidate's forecast, best pair and outcomes.
      std::vector<std::tuple<double, std::size_t, std::size_t, double>> v;  // (expected, cand, pair, entry)
      std::vector<std::vector<double>> xs(cand.size()), raw(cand.size());
      std::vector<double> yhat(cand.size(), 0.0);
      for (std::size_t c = 0; c < cand.size(); ++c) {
        const auto& C = cand[c];
        const double o = m.open(d, C.i), e = (*C.p)[k - 1].c;
        raw[c] = {std::log(e / o) / C.sig, mo, C.gap, C.lastOC, C.lastLoc, C.fc};
        const auto sc = scal[kk].apply(raw[c]);
        xs[c] = {1.0};
        xs[c].insert(xs[c].end(), sc.begin(), sc.end());
        if (beta.empty()) continue;
        yhat[c] = std::clamp(fit[kk].predict(beta, xs[c]), -zf, zf);
        const double rest = C.sig * std::sqrt(double((*C.p).size() - k) / double((*C.p).size()));
        double best = -1e300;
        std::size_t bq = 0;
        for (std::size_t q = 0; q < P; ++q)
          if (!ab[q].empty()) {
            const double ex = ab[q][0] + ab[q][1] * yhat[c];
            if (ex > best) best = ex, bq = q;
          }
        if (best > 0) v.emplace_back(best * rest, c, bq, e);
      }
      std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
      if (v.size() > BOOK) v.resize(BOOK);
      double r = 0, ro = 0;
      for (const auto& [ex, c, q, e] : v) {
        const auto& C = cand[c];
        const double rest = C.sig * std::sqrt(double((*C.p).size() - k) / double((*C.p).size()));
        const double stop = e * std::exp(-mult[q / G] * rest), take = e * std::exp(mult[q % G] * rest);
        int x;
        r += pathTrade(*C.p, k, e, stop, take, buy, sell, true, &x) / double(BOOK);
        if (trade && kk == use) ++book.trades, book.stops += x == 1, book.takes += x == 2;
        ro += pathTrade(*C.p, k, e, stop, take, buy, sell, false, &x) / double(BOOK);
      }
      if (trade && kk == use) dayRet = r, dayOpt = ro;
      rec[kk].push(std::log1p(r));
      // After the close: the outcomes, for the fits (every candidate) and the skill.
      std::vector<double> z(cand.size());
      double nz = 0;
      for (std::size_t c = 0; c < cand.size(); ++c) {
        const auto& C = cand[c];
        z[c] = std::log((*C.p).back().c / (*C.p)[k - 1].c) / C.sig;
      }
      const double zb = day::chauvenet(fit[kk].n + double(cand.size()));
      for (std::size_t c = 0; c < cand.size(); ++c) {
        const auto& C = cand[c];
        const double y = std::clamp(z[c], -zb, zb);
        if (!beta.empty()) book.skill[kk].add(yhat[c], y);
        for (std::size_t j = 1; j < F; ++j) book.featureSkill[kk][j - 1].add(xs[c][j], y);
        fit[kk].add(xs[c], y);
        if (beta.empty()) continue;
        const double e = (*C.p)[k - 1].c, rest = C.sig * std::sqrt(double((*C.p).size() - k) / double((*C.p).size()));
        for (std::size_t q = 0; q < P; ++q) {
          int x;
          const double out = pathTrade(*C.p, k, e, e * std::exp(-mult[q / G] * rest), e * std::exp(mult[q % G] * rest), buy, sell, true, &x);
          const double zq = day::chauvenet(pairFit[kk][q].n + 1);
          pairFit[kk][q].add({1.0, yhat[c]}, std::clamp(std::log1p(out) / rest, -zq, zq));
        }
        nz += 1;
      }
      (void)nz;
      for (const auto& x : raw) scal[kk].add(x);
    }
    book.ret.push_back(dayRet), book.optimistic.push_back(dayOpt), book.entryBars.push_back(trade ? int(kEntry[use]) : 0);
  }
  return book;
}

std::string intraText(const Market& m, const IntraBook& b, double costPerSide) {
  std::string o;
  char buf[512];
  o += "\nUS intraday strategy (5-minute bars; bought shortly after the open, sold the same day)\n";
  if (b.ret.empty()) return o + "  no days with intraday paths\n";
  const std::size_t n = b.ret.size(), h = n / 2;
  auto row = [&](const char* name, std::size_t lo, std::size_t hi) {
    const Metrics mt = metrics(std::vector<double>(b.ret.begin() + std::ptrdiff_t(lo), b.ret.begin() + std::ptrdiff_t(hi)));
    std::snprintf(buf, sizeof buf, "  %-12s %s .. %s  %+6.1f%% a year, volatility %5.1f%%, Sharpe %5.2f, max drawdown %5.1f%%\n", name, m.dates[b.days[lo]].c_str(),
                  m.dates[b.days[hi - 1]].c_str(), 100 * mt.annualReturn, 100 * mt.volatility, mt.sharpe, 100 * mt.maxDrawdown);
    o += buf;
  };
  row("whole", 0, n);
  row("first half", 0, h);
  row("second half", h, n);
  const Metrics wc = metrics(b.ret), op = metrics(b.optimistic);
  std::snprintf(buf, sizeof buf, "  costs %.2f%% per side; %.2f trades a day (%zu stops, %zu take-profits); %.0f candidates a day\n", 100 * costPerSide,
                double(b.trades) / double(n), b.stops, b.takes, double(b.candidates) / double(n));
  o += buf;
  std::snprintf(buf, sizeof buf, "  fill test (one 5-minute bar touched both levels): %+.1f%% a year with the stop first, %+.1f%% with the take-profit first\n",
                100 * wc.annualReturn, 100 * op.annualReturn);
  o += buf;
  std::map<int, std::size_t> used;
  for (int k : b.entryBars) ++used[k];
  o += "  entry used (days):";
  for (const auto& [k, c] : used) {
    std::snprintf(buf, sizeof buf, " %s %zu", k == 0 ? "cash" : (std::to_string(5 * k) + " min").c_str(), c);
    o += buf;
  }
  o += "\n  forecast skill (correlation with the move to the close) by entry:";
  const int ks[4] = {1, 2, 4, 8};
  for (int k = 0; k < 4; ++k) {
    std::snprintf(buf, sizeof buf, " %d min %+.4f (t %+.1f);", 5 * ks[k], b.skill[k].corr(), b.skill[k].corr() * std::sqrt(b.skill[k].n));
    o += buf;
  }
  o += "\n  inputs vs the move to the close (correlation at each entry: 5, 10, 20, 40 min):\n";
  for (std::size_t j = 0; j < b.featureNames.size(); ++j) {
    std::snprintf(buf, sizeof buf, "    %-26s", b.featureNames[j].c_str());
    o += buf;
    for (int k = 0; k < 4; ++k)
      if (j < b.featureSkill[k].size()) {
        std::snprintf(buf, sizeof buf, " %+.4f", b.featureSkill[k][j].corr());
        o += buf;
      }
    o += "\n";
  }
  std::map<std::string, std::vector<double>> years;
  for (std::size_t k = 0; k < n; ++k) years[m.dates[b.days[k]].substr(0, 4)].push_back(b.ret[k]);
  o += "  by year:";
  for (const auto& [y, r] : years) {
    double w = 1;
    for (double x : r) w *= 1 + x;
    std::snprintf(buf, sizeof buf, " %s %+.1f%%;", y.c_str(), 100 * (w - 1));
    o += buf;
  }
  return o + "\n";
}

}  // namespace ofm
