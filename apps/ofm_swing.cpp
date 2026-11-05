// The swing model: which shares to hold, from when to when, learnt walk-forward from the past only.
//
// Each close t, every tradable share (apps/swing.hpp: the most traded, a price floor) gets the signals
// known then (short-term losers, momentum, the close near the day's low, ...), each ranked across the
// day's shares. For each holding window that orders can trade (bought in the next open's or close's
// auction, held overnight or for days, sold in an auction: no spread) a ridge regression learns from
// every past trade that has settled by t how a share's net return in that window, relative to the
// day's mean, follows from its signal ranks. Its settings are learnt too: the window, how fast old days
// fade (none, or a half-life), the ridge and the number of shares bought (5, 10, 20). Every setting
// trades on paper every day from the start; each close, the setting whose past daily results (net of
// the broker's costs: apps/swing.hpp) have the best Sharpe ratio, after a year of them, is the one
// traded the next session. Changing setting sells the whole book and buys the new one, at its costs.
// The headline is that walk-forward result (every choice from the past only), in full and its first
// and second half apart; the settings fixed in hindsight are shown beside it only for comparison.
//
//   ofm_swing --csv stocks.csv --costs US|UK|EU|APAC [--capital 5000] [--top 500] [--min-price 5]
//             [--venues tools/venues.txt] [--stamp-free aim.txt] [--orders orders.json]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "swing.hpp"

namespace {
using swing::kNaN;

// Solves (G + lambda * mean diagonal * I) x = b (symmetric, Gaussian elimination with pivoting).
std::vector<double> ridge(std::vector<double> G, std::vector<double> b, std::size_t n, double lambda) {
  double tr = 0;
  for (std::size_t k = 0; k < n; ++k) tr += G[k * n + k];
  const double add = lambda * (tr > 0 ? tr / double(n) : 1.0);
  for (std::size_t k = 1; k < n; ++k) G[k * n + k] += add;  // the intercept (0) is not shrunk
  for (std::size_t c = 0; c < n; ++c) {
    std::size_t piv = c;
    for (std::size_t r = c + 1; r < n; ++r) if (std::fabs(G[r * n + c]) > std::fabs(G[piv * n + c])) piv = r;
    if (std::fabs(G[piv * n + c]) < 1e-300) return std::vector<double>(n, 0.0);
    if (piv != c) {
      for (std::size_t k = 0; k < n; ++k) std::swap(G[c * n + k], G[piv * n + k]);
      std::swap(b[c], b[piv]);
    }
    for (std::size_t r = c + 1; r < n; ++r) {
      const double f = G[r * n + c] / G[c * n + c];
      for (std::size_t k = c; k < n; ++k) G[r * n + k] -= f * G[c * n + k];
      b[r] -= f * b[c];
    }
  }
  std::vector<double> x(n, 0.0);
  for (std::size_t c = n; c-- > 0;) {
    double s = b[c];
    for (std::size_t k = c + 1; k < n; ++k) s -= G[c * n + k] * x[k];
    x[c] = s / G[c * n + c];
  }
  return x;
}

struct Series {
  double sum = 0, sum2 = 0;
  long n = 0;
  void add(double x) { sum += x, sum2 += x * x, ++n; }
  double mean() const { return n ? sum / double(n) : 0; }
  double sd() const { return n > 1 ? std::sqrt(std::max(0.0, sum2 / double(n) - mean() * mean())) : 0; }
  double sharpe() const { return sd() > 0 ? mean() / sd() * std::sqrt(252.0) : 0; }
};

std::string fmt(const Series& s) {
  char b[96];
  std::snprintf(b, sizeof b, "%+7.1f%% a year (Sharpe %5.2f, t %+5.1f, %ld days)", 25200 * s.mean(), s.sharpe(),
                s.sd() > 0 ? s.mean() / s.sd() * std::sqrt(double(s.n)) : 0.0, s.n);
  return b;
}
}  // namespace

int main(int argc, char** argv) {
  std::string csv, market, venues, stampFree, ordersOut, traceOut;
  std::size_t top = 500;
  double capital = 5000, minPrice = 0;
  for (int k = 1; k + 1 < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv") csv = argv[++k];
    else if (a == "--costs") market = argv[++k];
    else if (a == "--capital") capital = std::atof(argv[++k]);
    else if (a == "--top") top = std::size_t(std::atol(argv[++k]));
    else if (a == "--min-price") minPrice = std::atof(argv[++k]);
    else if (a == "--venues") venues = argv[++k];
    else if (a == "--stamp-free") stampFree = argv[++k];
    else if (a == "--orders") ordersOut = argv[++k];
    else if (a == "--trace") traceOut = argv[++k];  // each close: the setting chosen and its picks (the no-look-ahead check)
  }
  if (csv.empty() || market.empty()) {
    std::fprintf(stderr, "usage: ofm_swing --csv stocks.csv --costs US|UK|EU|APAC [--capital 5000] [--top 500] [--min-price 5] [--venues f] [--stamp-free f] [--orders out.json]\n");
    return 2;
  }
  const swing::Panel p = swing::load(csv);
  const std::size_t T = p.T, N = p.N;
  const auto uni = swing::universe(p, top, minPrice);
  const swing::Costs cost = swing::costs(p, market, capital, venues, stampFree);
  const auto sig = swing::signals(p);
  const auto& win = swing::windows();
  const std::size_t S = sig.size(), K = S + 1, W = win.size();
  // Each day's features: the signals ranked across the day's shares (-0.5 to 0.5; none: 0), and 1.
  std::vector<std::vector<float>> X(T);
  for (std::size_t t = 0; t < T; ++t) {
    const auto& u = uni[t];
    X[t].assign(u.size() * K, 0.0f);
    for (std::size_t j = 0; j < u.size(); ++j) X[t][j * K] = 1.0f;
    for (std::size_t s = 0; s < S; ++s) {
      std::vector<std::pair<double, std::size_t>> z;
      for (std::size_t j = 0; j < u.size(); ++j) {
        const double v = sig[s].f(t, u[j]);
        if (std::isfinite(v)) z.emplace_back(v, j);
      }
      if (z.size() < 2) continue;
      std::sort(z.begin(), z.end());
      for (std::size_t r = 0; r < z.size(); ++r) X[t][z[r].second * K + 1 + s] = float(double(r) / double(z.size() - 1) - 0.5);
    }
  }
  // The learnt settings' grid.
  const std::vector<double> fades = {0.0, std::log(2.0) / 250};  // none, a half-life of 250 days
  const std::vector<double> lambdas = {1e-3, 1e-1, 10.0};
  const std::vector<std::size_t> picks = {5, 10, 20};
  const std::size_t nF = fades.size(), nL = lambdas.size(), nP = picks.size();
  const std::size_t nModels = W * nF * nL, nCfg = nModels * nP;
  // The regressions' sums, per window and fade: G (K x K) and b.
  std::vector<std::vector<double>> G(W * nF, std::vector<double>(K * K, 0.0)), B(W * nF, std::vector<double>(K, 0.0));
  // A day's settled trades enter the sums the day they settle (relative to that day's mean, clamped).
  const auto settledRows = [&](std::size_t w, std::size_t f) {
    const auto& u = uni[f];
    std::vector<double> y(u.size(), kNaN);
    double m = 0;
    std::size_t n = 0;
    for (std::size_t j = 0; j < u.size(); ++j) {
      const auto lg = swing::legs(p, cost, win[w], f, u[j]);
      if (lg.empty()) continue;
      double g1 = 1, c1 = 0;
      for (const auto& l : lg) g1 *= 1 + l.gross, c1 += l.cost;
      y[j] = g1 - 1 - c1, m += y[j], ++n;
    }
    if (n < 10) return;
    m /= double(n);
    for (std::size_t j = 0; j < u.size(); ++j) {
      if (!std::isfinite(y[j])) continue;
      const double yy = std::clamp(y[j] - m, -0.2, 0.2);
      const float* x = &X[f][j * K];
      for (std::size_t fd = 0; fd < nF; ++fd) {
        auto& g = G[w * nF + fd];
        auto& b = B[w * nF + fd];
        for (std::size_t a = 0; a < K; ++a) {
          b[a] += x[a] * yy;
          for (std::size_t c = 0; c < K; ++c) g[a * K + c] += double(x[a]) * x[c];
        }
      }
    }
  };
  // Each setting's book: its daily net result (the day a leg ends on), every setting from the start.
  std::vector<std::vector<double>> book(nCfg, std::vector<double>(T + 25, 0.0));
  std::vector<std::size_t> firstDay(nCfg, 0);
  std::vector<std::vector<std::size_t>> picksAt(nCfg);  // the last close's picks (the orders)
  std::vector<std::vector<double>> beta(nModels);
  const std::size_t warm = 260;  // signals need a year of bars (momentum, the 52-week high)
  // The walk-forward choice: each close, the setting with the best past Sharpe ratio (a year at least).
  std::vector<Series> past(nCfg);
  Series meta, metaH1, metaH2;
  std::vector<double> metaDays;
  long switches = 0;
  FILE* trace = traceOut.empty() ? nullptr : std::fopen(traceOut.c_str(), "w");
  std::size_t chosen = nCfg, metaStart = 0;
  std::vector<std::size_t> pending(W, 0);
  for (std::size_t t = 0; t + 1 < T; ++t) {
    // The sums: days settled by the close t, then the fade.
    for (std::size_t w = 0; w < W; ++w)
      for (std::size_t f = 0; f < nF; ++f) {
        auto& g = G[w * nF + f];
        auto& b = B[w * nF + f];
        if (fades[f] > 0) {
          const double d = std::exp(-fades[f]);
          for (auto& v : g) v *= d;
          for (auto& v : b) v *= d;
        }
      }
    for (std::size_t w = 0; w < W; ++w)
      for (std::size_t fd = pending[w]; fd < T && swing::settles(win[w], fd) <= t; ++fd) {
        pending[w] = fd + 1;
        if (fd >= warm) settledRows(w, fd);
      }
    // The day's results so far: the settings' books up to the close t (each day once).
    if (t > 0) {
      for (std::size_t c = 0; c < nCfg; ++c)
        if (firstDay[c] && t >= firstDay[c]) past[c].add(book[c][t]);
      if (chosen < nCfg) {
        meta.add(book[chosen][t]), metaDays.push_back(book[chosen][t]);
      }
    }
    if (t < warm + 60) continue;
    // The settings' picks at the close t, and their trades into the books.
    const auto& u = uni[t];
    if (u.size() < 2 * picks.back()) continue;
    for (std::size_t w = 0; w < W; ++w)
      for (std::size_t f = 0; f < nF; ++f)
        for (std::size_t l = 0; l < nL; ++l) {
          const std::size_t mdl = (w * nF + f) * nL + l;
          if (G[w * nF + f][0] <= 0) continue;
          beta[mdl] = ridge(G[w * nF + f], B[w * nF + f], K, lambdas[l]);
          std::vector<std::pair<double, std::size_t>> sc;
          for (std::size_t j = 0; j < u.size(); ++j) {
            double s = 0;
            for (std::size_t a = 0; a < K; ++a) s += beta[mdl][a] * X[t][j * K + a];
            sc.emplace_back(s, u[j]);
          }
          std::partial_sort(sc.begin(), sc.begin() + std::ptrdiff_t(picks.back()), sc.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
          const double held = double(std::max(1, win[w].days));  // cohorts held at once: each 1/held of the money
          std::vector<std::vector<swing::Leg>> lg(picks.back());
          for (std::size_t r = 0; r < picks.back(); ++r) lg[r] = swing::legs(p, cost, win[w], t, sc[r].second);
          for (std::size_t q = 0; q < nP; ++q) {
            const std::size_t c = mdl * nP + q, k = picks[q];
            if (!firstDay[c]) firstDay[c] = t + 1;
            picksAt[c].clear();
            for (std::size_t r = 0; r < k; ++r) {
              picksAt[c].push_back(sc[r].second);
              for (const auto& x : lg[r])
                if (x.day < book[c].size()) book[c][x.day] += (x.gross - x.cost) / double(k) / held;
            }
          }
        }
    // The walk-forward choice for the next session: the best past Sharpe ratio after a year.
    std::size_t best = nCfg;
    double bestSh = -1e300;
    for (std::size_t c = 0; c < nCfg; ++c)
      if (past[c].n >= 250 && past[c].sharpe() > bestSh) bestSh = past[c].sharpe(), best = c;
    if (best < nCfg && best != chosen) {
      if (chosen < nCfg) {  // the whole book sold and the new one bought: a round trip on all of it
        double rt = 0;
        for (std::size_t i : u) rt += cost.roundTrip(i, p.C[t * N + i], p.C[t * N + i]);
        rt /= double(u.size());
        if (!metaDays.empty()) meta.sum -= rt, meta.sum2 += rt * rt - 2 * rt * metaDays.back(), metaDays.back() -= rt;
        ++switches;
      }
      if (chosen == nCfg) metaStart = t + 1;
      chosen = best;
    }
    if (trace && chosen < nCfg) {
      std::fprintf(trace, "%s %zu", p.days[t].c_str(), chosen);
      for (std::size_t i : picksAt[chosen]) std::fprintf(trace, " %s", p.tickers[i].c_str());
      std::fprintf(trace, "\n");
    }
  }
  if (trace) std::fclose(trace);
  // The halves of the walk-forward days.
  for (std::size_t k = 0; k < metaDays.size(); ++k) (k < metaDays.size() / 2 ? metaH1 : metaH2).add(metaDays[k]);
  double peak = 0, eq = 0, dd = 0;
  for (double r : metaDays) eq += std::log1p(r), peak = std::max(peak, eq), dd = std::min(dd, eq - peak);
  const auto name = [&](std::size_t c) {
    const std::size_t q = c % nP, mdl = c / nP, l = mdl % nL, f = mdl / nL % nF, w = mdl / (nL * nF);
    char b[160];
    std::snprintf(b, sizeof b, "%s, %zu shares, fade %s, ridge %g", win[w].name, picks[q], fades[f] > 0 ? "half-life 250 days" : "none", lambdas[l]);
    return std::string(b);
  };
  std::printf("swing: %zu shares, %zu days (%s to %s); each day the %zu most traded%s; costs %s on %.0f a trade (auctions: no spread)\n", N, T,
              p.days.front().c_str(), p.days.back().c_str(), top, minPrice > 0 ? " at the price floor or above" : "", market.c_str(), capital);
  std::printf("swing: walk-forward (every choice from the past only): %s, from %s; max drawdown %.1f%%; %ld changes of setting\n", fmt(meta).c_str(),
              metaStart < T ? p.days[metaStart].c_str() : "-", 100 * (std::exp(dd) - 1), switches);
  std::printf("swing:   first half  %s\n", fmt(metaH1).c_str());
  std::printf("swing:   second half %s\n", fmt(metaH2).c_str());
  // For comparison only: each window's best setting chosen in hindsight over the same days.
  for (std::size_t w = 0; w < W; ++w) {
    std::size_t bc = nCfg;
    for (std::size_t c = 0; c < nCfg; ++c)
      if (c / nP / (nL * nF) == w && (bc == nCfg || past[c].sharpe() > past[bc].sharpe())) bc = c;
    if (bc < nCfg) std::printf("swing:   hindsight, %-30s %s (%s)\n", win[w].name, fmt(past[bc]).c_str(), name(bc).c_str());
  }
  if (chosen < nCfg) {
    const std::size_t mdl = chosen / nP;
    std::printf("swing: learnt for the next session: %s\n", name(chosen).c_str());
    std::printf("swing: its weights on the signal ranks:");
    for (std::size_t s = 0; s < S; ++s) std::printf(" %s %+.4f%s", sig[s].name, beta[mdl].empty() ? 0.0 : beta[mdl][1 + s], s + 1 < S ? "," : "\n");
    std::printf("swing: next session's buys (%s): ", win[chosen / nP / (nL * nF)].name);
    for (std::size_t k = 0; k < picksAt[chosen].size(); ++k) std::printf("%s%s", k ? ", " : "", p.tickers[picksAt[chosen][k]].c_str());
    std::printf("\n");
    if (!ordersOut.empty()) {
      const auto& w = win[chosen / nP / (nL * nF)];
      std::ofstream o(ordersOut);
      o << "{\"asOf\":\"" << p.days.back() << "\",\"model\":\"swing\",\"window\":\"" << w.name << "\",\"entry\":\"" << (w.entryAtClose ? "close" : "open")
        << "\",\"days\":" << w.days << ",\"orders\":[";
      for (std::size_t k = 0; k < picksAt[chosen].size(); ++k) o << (k ? "," : "") << "{\"ticker\":\"" << p.tickers[picksAt[chosen][k]] << "\"}";
      o << "]}\n";
    }
  }
  return 0;
}
