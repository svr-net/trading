#include "ofm/intrabook.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

#include "ofm/liquidity.hpp"
#include "ofm/trade.hpp"
#include "daykernels.hpp"

#include <cstdio>
#include <map>

namespace ofm {

namespace {

// Least squares by normal equations over every observation so far (as in the daily trades).
struct Fit {
  std::size_t k;
  std::vector<double> A, b;
  double n = 0, yy = 0;
  explicit Fit(std::size_t dim) : k(dim), A(dim * dim, 0.0), b(dim, 0.0) {}
  void add(const std::vector<double>& x, double y) {
    for (std::size_t i = 0; i < k; ++i) {
      b[i] += x[i] * y;
      for (std::size_t j = 0; j < k; ++j) A[i * k + j] += x[i] * x[j];
    }
    n += 1, yy += y * y;
  }
  std::vector<double> solve() const { return solve(b); }
  // A^-1 rhs (with a vanishing ridge), empty while there are too few observations.
  std::vector<double> solve(const std::vector<double>& rhs) const {
    if (n <= double(k)) return {};
    std::vector<double> M = A, v = rhs;
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
  // The prediction at x shrunk by its t statistic, as the records score: e max(0, 1 - 1/t^2), with
  // t = e / its standard error (the residual variance times x' A^-1 x).
  double shrunk(const std::vector<double>& beta, const std::vector<double>& x) const {
    const double e = predict(beta, x);
    const auto z = solve(x);
    if (z.empty() || n <= double(k) + 1) return 0;
    double bb = 0, xz = 0;
    for (std::size_t i = 0; i < k; ++i) bb += beta[i] * b[i], xz += x[i] * z[i];
    const double s2 = std::max(0.0, yy - bb) / (n - double(k)), se2 = s2 * xz;
    return e * std::max(0.0, 1 - se2 / std::max(e * e, 1e-300));
  }
};

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
  mult.push_back(-1);  // no stop / no take-profit: held to the close
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
  // Two forecasts compete: 0 the least squares on every input, 1 the one input whose correlation
  // with the outcome has been strongest so far (its own fit), so a strong single signal is not
  // blurred by weaker ones. Each (entry, forecast) has its own book and pair fits.
  const std::size_t NM = 2;
  std::vector<std::vector<Fit>> uni(NK, std::vector<Fit>(F - 1, Fit(2)));
  std::vector<std::vector<Fit>> pairFit(NK * NM, std::vector<Fit>(P, Fit(2)));
  day::AdwinBank rec(NK * NM, T, double(T));
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
    // Every candidate's trade at every entry and pair, in one fused kernel: out[((c NK + kk) P + q) 3].
    std::vector<double> ticks, cinfo, out;
    for (const auto& C : cand) {
      cinfo.insert(cinfo.end(), {double(ticks.size() / 3), double(C.p->size()), C.sig});
      for (const auto& b : *C.p) ticks.insert(ticks.end(), {b.h, b.l, b.c});
    }
    day::kernels::pairTrades(ticks, cinfo, std::vector<std::uint32_t>(kEntry, kEntry + NK), mult, buy, sell, out);
    auto at = [&](std::size_t c, std::size_t kk, std::size_t q) { return &out[((c * NK + kk) * P + q) * 3]; };
    book.candidates += cand.size();
    book.days.push_back(d);
    // Each (entry, forecast) book; the one whose record is best so far trades.
    std::size_t use = 0;
    for (std::size_t k = 1; k < NK * NM; ++k)
      if (rec.score(k) > rec.score(use)) use = k;
    const bool trade = rec.score(use) > 0;
    std::vector<std::size_t> recW;  // the books' values of the day, pushed together
    std::vector<double> recX;
    double dayRet = 0, dayOpt = 0;
    const bool warm = book.days.size() > warmDays;
    for (std::size_t kk = 0; kk < NK; ++kk) {
      const std::size_t k = kEntry[kk];
      const auto beta = warm ? fit[kk].solve() : std::vector<double>{};
      const double zf = day::chauvenet(fit[kk].n + 1);
      // The single strongest input so far and its fit.
      std::size_t js = 0;
      for (std::size_t j1 = 1; j1 + 1 < F; ++j1)
        if (std::fabs(book.featureSkill[kk][j1].corr()) > std::fabs(book.featureSkill[kk][js].corr())) js = j1;
      const auto betaU = warm ? uni[kk][js].solve() : std::vector<double>{};
      // The market's opening move: the candidates' mean.
      double mo = 0;
      for (const auto& c : cand) mo += std::log((*c.p)[k - 1].c / m.open(d, c.i)) / c.sig;
      mo /= double(cand.size());
      std::vector<std::vector<double>> xs(cand.size()), raw(cand.size());
      std::vector<double> yhat[NM] = {std::vector<double>(cand.size(), 0.0), std::vector<double>(cand.size(), 0.0)};
      for (std::size_t c = 0; c < cand.size(); ++c) {
        const auto& C = cand[c];
        const double o = m.open(d, C.i), e = (*C.p)[k - 1].c;
        raw[c] = {std::log(e / o) / C.sig, mo, C.gap, C.lastOC, C.lastLoc, C.fc};
        const auto sc = scal[kk].apply(raw[c]);
        xs[c] = {1.0};
        xs[c].insert(xs[c].end(), sc.begin(), sc.end());
        if (!beta.empty()) yhat[0][c] = std::clamp(fit[kk].predict(beta, xs[c]), -zf, zf);
        if (betaU.size() == 2) yhat[1][c] = std::clamp(betaU[0] + betaU[1] * xs[c][1 + js], -zf, zf);
      }
      for (std::size_t mdl = 0; mdl < NM; ++mdl) {
        const std::size_t bkI = kk * NM + mdl;
        if ((mdl == 0 && beta.empty()) || (mdl == 1 && betaU.size() != 2)) {
          recW.push_back(bkI), recX.push_back(0.0);
          continue;
        }
        std::vector<std::vector<double>> ab(P);
        for (std::size_t q = 0; q < P; ++q) ab[q] = pairFit[bkI][q].solve();
        std::vector<std::tuple<double, std::size_t, std::size_t, double>> v;  // (expected, cand, pair, entry)
        for (std::size_t c = 0; c < cand.size(); ++c) {
          const auto& C = cand[c];
          const double e = (*C.p)[k - 1].c, rest = C.sig * std::sqrt(double((*C.p).size() - k) / double((*C.p).size()));
          double best = -1e300;
          std::size_t bq = 0;
          // The pair whose expectation is best after shrinking by its own uncertainty: the plain best
          // of 81 noisy fits favours the luckiest (in practice the tight stops).
          for (std::size_t q = 0; q < P; ++q)
            if (!ab[q].empty()) {
              const double ex = pairFit[bkI][q].shrunk(ab[q], {1.0, yhat[mdl][c]});
              if (ex > best) best = ex, bq = q;
            }
          if (best > 0) v.emplace_back(best * rest, c, bq, e);
        }
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        if (v.size() > BOOK) v.resize(BOOK);
        double r = 0, ro = 0;
        for (const auto& [ex, c, q, e] : v) {
          const double* o = at(c, kk, q);
          r += o[0] / double(v.size()), ro += o[1] / double(v.size());
          if (trade && bkI == use) ++book.trades, book.stops += o[2] == 1, book.takes += o[2] == 2;
        }
        if (trade && bkI == use) dayRet = r, dayOpt = ro;
        recW.push_back(bkI), recX.push_back(std::log1p(r));
      }
      // After the close: the outcomes, for the fits (every candidate) and the skill.
      const double zb = day::chauvenet(fit[kk].n + double(cand.size()));
      for (std::size_t c = 0; c < cand.size(); ++c) {
        const auto& C = cand[c];
        const double y = std::clamp(std::log((*C.p).back().c / (*C.p)[k - 1].c) / C.sig, -zb, zb);
        if (!beta.empty()) book.skill[kk].add(yhat[0][c], y);
        for (std::size_t j1 = 1; j1 < F; ++j1) book.featureSkill[kk][j1 - 1].add(xs[c][j1], y), uni[kk][j1 - 1].add({1.0, xs[c][j1]}, y);
        fit[kk].add(xs[c], y);
        const double rest = C.sig * std::sqrt(double((*C.p).size() - k) / double((*C.p).size()));
        for (std::size_t mdl = 0; mdl < NM; ++mdl) {
          if ((mdl == 0 && beta.empty()) || (mdl == 1 && betaU.size() != 2)) continue;
          for (std::size_t q = 0; q < P; ++q) {
            auto& pf = pairFit[kk * NM + mdl][q];
            const double zq = day::chauvenet(pf.n + 1);
            pf.add({1.0, yhat[mdl][c]}, std::clamp(std::log1p(at(c, kk, q)[0]) / rest, -zq, zq));
          }
        }
      }
      for (const auto& x : raw) scal[kk].add(x);
    }
    rec.push(recW, recX);
    book.forecastUsed.push_back(trade ? int(use % NM) : -1);
    double mk = 0;
    for (const auto& C : cand) mk += m.close(d, C.i) > 0 ? m.close(d, C.i) / m.close(t, C.i) - 1 : 0.0;
    book.market.push_back(mk / double(cand.size()));
    book.ret.push_back(dayRet), book.optimistic.push_back(dayOpt), book.entryBars.push_back(trade ? int(kEntry[use / NM]) : 0);
  }
  return book;
}

std::string intraJson(const Market& m, const IntraBook& b) {
  std::string o = "{";
  char buf[512];
  const std::size_t n = b.ret.size(), h = n / 2;
  if (n < 2) return "{}";
  // The page's chart, laid out here as the daily trades' is (report.cpp): the strategy and the
  // candidates held, on a log scale, heights and positions as shares of the chart.
  {
    const std::size_t step = std::max<std::size_t>(1, n / 600);
    const std::pair<const char*, const std::vector<double>*> sr[2] = {{"intraday strategy", &b.ret}, {"market (bought and held)", &b.market}};
    std::vector<double> lv[2];
    double lo = 0, hi = 0;
    for (int s = 0; s < 2; ++s) {
      double w = 0;
      for (std::size_t k = 0; k < n; ++k) {
        w += std::log1p((*sr[s].second)[k]);
        if (k % step == 0) lv[s].push_back(w), lo = std::min(lo, w), hi = std::max(hi, w);
      }
    }
    const double span = std::max(hi - lo, 1e-9);
    o += "\"chart\":{\"series\":[";
    for (int s = 0; s < 2; ++s) {
      double w = 0;
      for (double x : *sr[s].second) w += std::log1p(x);
      std::snprintf(buf, sizeof buf, "%s{\"name\":\"%s\",\"final\":\"%.2f×\",\"y\":[", s ? "," : "", sr[s].first, std::exp(w));
      o += buf;
      for (std::size_t k = 0; k < lv[s].size(); ++k) std::snprintf(buf, sizeof buf, "%s%.4f", k ? "," : "", (lv[s][k] - lo) / span), o += buf;
      o += "]}";
    }
    o += "],\"ticks\":[";
    bool lead = true;
    for (double v : {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0}) {
      const double y = (std::log(v) - lo) / span;
      if (y < -1e-9 || y > 1 + 1e-9) continue;
      std::snprintf(buf, sizeof buf, "%s{\"label\":\"%g×\",\"y\":%.4f}", lead ? "" : ",", v, y), o += buf, lead = false;
    }
    o += "],\"years\":[";
    lead = true;
    const std::size_t npts = lv[0].size();
    for (std::size_t k = 0, j = 0; k < n; k += step, ++j)
      if (j == 0 || m.dates[b.days[k]].substr(0, 4) != m.dates[b.days[k - step]].substr(0, 4))
        std::snprintf(buf, sizeof buf, "%s{\"label\":\"%s\",\"x\":%.4f}", lead ? "" : ",", m.dates[b.days[k]].substr(0, 4).c_str(),
                      npts > 1 ? double(j) / double(npts - 1) : 0.0),
            o += buf, lead = false;
    o += "]},";
  }
  // The last 20 days: the strategy and the candidates held.
  o += "\"recent\":[";
  const std::size_t r0 = n > 20 ? n - 20 : 0;
  for (std::size_t k = r0; k < n; ++k)
    std::snprintf(buf, sizeof buf, "%s{\"date\":\"%s\",\"day\":%.6g,\"market\":%.6g}", k == r0 ? "" : ",", m.dates[b.days[k]].c_str(), b.ret[k],
                  b.market[k]),
        o += buf;
  o += "],";
  {
    const Metrics mk = metrics(b.market);
    std::snprintf(buf, sizeof buf, "\"market\":{\"annualReturn\":%.6g,\"sharpe\":%.6g,\"maxDrawdown\":%.6g},", mk.annualReturn, mk.sharpe, mk.maxDrawdown);
    o += buf;
  }
  auto part = [&](std::size_t lo, std::size_t hi) {
    const Metrics mt = metrics(std::vector<double>(b.ret.begin() + std::ptrdiff_t(lo), b.ret.begin() + std::ptrdiff_t(hi)));
    std::snprintf(buf, sizeof buf, "{\"from\":\"%s\",\"to\":\"%s\",\"annualReturn\":%.6g,\"volatility\":%.6g,\"sharpe\":%.6g,\"maxDrawdown\":%.6g}",
                  m.dates[b.days[lo]].c_str(), m.dates[b.days[hi - 1]].c_str(), mt.annualReturn, mt.volatility, mt.sharpe, mt.maxDrawdown);
    return std::string(buf);
  };
  o += "\"whole\":" + part(0, n) + ",\"halves\":[" + part(0, h) + "," + part(h, n) + "]";
  const Metrics op = metrics(b.optimistic);
  std::snprintf(buf, sizeof buf, ",\"tradesPerDay\":%.4g,\"stopShare\":%.4g,\"takeShare\":%.4g,\"fillOptimistic\":%.6g,\"candidatesPerDay\":%.4g,\"entryMinutes\":%d",
                double(b.trades) / double(n), b.trades ? double(b.stops) / double(b.trades) : 0.0, b.trades ? double(b.takes) / double(b.trades) : 0.0,
                op.annualReturn, double(b.candidates) / double(n), 5 * b.entryBars.back());
  o += buf;
  std::map<std::string, std::vector<double>> years;
  for (std::size_t k = 0; k < n; ++k) years[m.dates[b.days[k]].substr(0, 4)].push_back(b.ret[k]);
  o += ",\"years\":[";
  bool first = true;
  for (const auto& [y, r] : years) {
    double w = 1;
    for (double x : r) w *= 1 + x;
    std::snprintf(buf, sizeof buf, "%s{\"year\":\"%s\",\"ret\":%.6g,\"days\":%zu}", first ? "" : ",", y.c_str(), w - 1, r.size());
    o += buf, first = false;
  }
  return o + "]}";
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
  {
    std::size_t f0 = 0, f1 = 0;
    for (int x : b.forecastUsed) f0 += x == 0, f1 += x == 1;
    std::snprintf(buf, sizeof buf, "\n  forecast used (days): every input %zu, strongest single input %zu", f0, f1);
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
