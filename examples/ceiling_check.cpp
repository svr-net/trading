// Ceiling check for an overlay built on orthonormal control portfolios.
//
//   ceiling_check [markets per set=6] [days=2520] [--csv data.csv]
//   ceiling_check 0 --csv data.csv     (real data only)
//
// Each date the control portfolios are the market (equal weight) and each of the seven
// forecasts' long-short portfolio (centred cross-sectional ranks), made orthonormal by
// Gram-Schmidt in that order. A position is a fixed combination of them, scaled to be fully
// invested (gross 1), moved towards each date's target at a fixed trade rate. The coefficients
// and the rate are chosen with hindsight to maximise the net Sharpe ratio over the evaluation
// days: the most any overlay with a fixed combination of these portfolios could reach.
//
// Decision (fixed in advance): build gate 1c only if this hindsight ceiling beats the default
// self-adaptive selector's mean Sharpe ratio on the selection markets and on the unseen markets
// at both cost levels, and on UK data at UK costs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "sat/sat.hpp"

using namespace sat;

namespace {

struct Costs {
  const char* name;
  double cost, stamp;
};
const Costs kCosts[] = {{"10 bp", 10, 0}, {"UK: 10 bp + 50 bp stamp duty on purchases", 10, 50}};
const double kRates[] = {1.0, 0.5, 0.2, 0.1, 0.05, 0.02};
constexpr std::size_t kRows = 5;
const char* kNames[kRows] = {"default (self-adaptive selector, online)", "best single rule (hindsight)",
                             "ceiling: market + 7 forecast axes", "ceiling: 7 forecast axes only", "market only, best rate"};

struct Fit {
  PerformanceMetrics metrics;
  std::vector<double> coef;
  double rate = 0;
};

// Orthonormal control portfolios per evaluation date: [day][axis][asset].
struct Axes {
  std::size_t K = 0, N = 0, D = 0;
  std::vector<double> e;  // D x K x N
  const double* at(std::size_t d, std::size_t k) const { return &e[(d * K + k) * N]; }
};

Axes buildAxes(const std::vector<ModelPredictions>& forecasts, std::size_t from, std::size_t to, std::size_t N) {
  Axes ax;
  ax.K = 1 + forecasts.size(), ax.N = N, ax.D = to - from;
  ax.e.assign(ax.D * ax.K * N, 0.0);
  for (std::size_t d = 0; d < ax.D; ++d) {
    const std::size_t t = from + d;
    std::vector<std::vector<double>> basis;
    auto add = [&](std::vector<double> v, std::size_t k) {
      for (const auto& b : basis) {
        double dot = 0;
        for (std::size_t i = 0; i < N; ++i) dot += v[i] * b[i];
        for (std::size_t i = 0; i < N; ++i) v[i] -= dot * b[i];
      }
      double norm = 0;
      for (double x : v) norm += x * x;
      norm = std::sqrt(norm);
      if (norm < 1e-9) return;  // nothing new: the axis stays zero today
      for (double& x : v) x /= norm;
      std::copy(v.begin(), v.end(), &ax.e[(d * ax.K + k) * N]);
      basis.push_back(std::move(v));
    };
    add(std::vector<double>(N, 1.0), 0);
    for (std::size_t f = 0; f < forecasts.size(); ++f) {
      std::vector<double> p, v(N, 0.0);
      std::vector<std::size_t> idx;
      for (std::size_t i = 0; i < N; ++i)
        if (std::isfinite(forecasts[f].probability(t, i))) p.push_back(forecasts[f].probability(t, i)), idx.push_back(i);
      if (p.size() >= 2) {
        const auto rk = averageRanks(p);
        for (std::size_t j = 0; j < p.size(); ++j) v[idx[j]] = (rk[j] - 1.0) / static_cast<double>(p.size() - 1) - 0.5;
      }
      add(v, 1 + f);
    }
  }
  return ax;
}

PerformanceMetrics simulate(const Axes& ax, const Panel& next, std::size_t from, const std::vector<double>& c, double rate, const Costs& k) {
  const std::size_t N = ax.N;
  std::vector<double> w(N, 0.0), v(N), net, turn;
  net.reserve(ax.D), turn.reserve(ax.D);
  for (std::size_t d = 0; d < ax.D; ++d) {
    std::fill(v.begin(), v.end(), 0.0);
    for (std::size_t a = 0; a < ax.K; ++a)
      if (c[a] != 0) {
        const double* e = ax.at(d, a);
        for (std::size_t i = 0; i < N; ++i) v[i] += c[a] * e[i];
      }
    double s = 0;
    for (double x : v) s += std::fabs(x);
    double gross = 0, turnover = 0, buys = 0;
    for (std::size_t i = 0; i < N; ++i) {
      const double target = s > 1e-12 ? v[i] / s : 0.0, dw = rate * (target - w[i]);
      w[i] += dw;
      turnover += std::fabs(dw), buys += std::max(0.0, dw);
      const double r = next(from + d, i);
      if (std::isfinite(r)) gross += w[i] * r;
    }
    net.push_back(gross - k.cost * 1e-4 * turnover - k.stamp * 1e-4 * buys);
    turn.push_back(turnover);
  }
  return evaluatePerformance(net, turn);
}

// Hindsight search over the coefficients (axes in `free`) and the trade rate.
Fit search(const Axes& ax, const Panel& next, std::size_t from, const std::vector<bool>& free, const Costs& k) {
  Fit best;
  best.metrics.sharpe = -1e30;
  Rng rng(17);
  for (double rate : kRates) {
    std::vector<std::vector<double>> starts;
    for (std::size_t a = 0; a < ax.K; ++a)
      if (free[a])
        for (double sgn : {1.0, -1.0}) {
          std::vector<double> c(ax.K, 0.0);
          c[a] = sgn;
          starts.push_back(c);
        }
    for (int r = 0; r < 24; ++r) {
      std::vector<double> c(ax.K, 0.0);
      for (std::size_t a = 0; a < ax.K; ++a)
        if (free[a]) c[a] = rng.normal();
      starts.push_back(c);
    }
    std::vector<std::pair<double, std::vector<double>>> scored;
    for (auto& c : starts) scored.push_back({simulate(ax, next, from, c, rate, k).sharpe, c});
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (std::size_t s = 0; s < std::min<std::size_t>(3, scored.size()); ++s) {
      auto [score, c] = scored[s];
      double l1 = 0;
      for (double x : c) l1 += std::fabs(x);
      for (double& x : c) x /= l1;
      for (double step = 0.5; step >= 0.02; step /= 2) {
        // The Sharpe ratio does not depend on the coefficients' scale: keep them at unit L1 norm
        // so the search cannot creep along that ridge.
        bool improved = true;
        for (int sweep = 0; improved && sweep < 20; ++sweep) {
          improved = false;
          for (std::size_t a = 0; a < ax.K; ++a) {
            if (!free[a]) continue;
            for (double dir : {step, -step}) {
              auto trial = c;
              trial[a] += dir;
              double l1 = 0;
              for (double x : trial) l1 += std::fabs(x);
              if (l1 < 1e-12) continue;
              for (double& x : trial) x /= l1;
              const double sc = simulate(ax, next, from, trial, rate, k).sharpe;
              if (sc > score + 1e-4) score = sc, c = trial, improved = true;
            }
          }
        }
      }
      if (score > best.metrics.sharpe) best = {simulate(ax, next, from, c, rate, k), c, rate};
    }
  }
  return best;
}

struct Market {
  std::string name;
  std::vector<std::vector<PerformanceMetrics>> out;  // [cost][row]
  std::vector<std::vector<Fit>> fits;                // [cost][ceiling rows]
};

Market prepare(const std::string& name, const MarketData& data) {
  ExperimentSpec e;
  e.stateSpecialists = true;
  e.composite.method = CompositeMethod::Adaptive;
  e.composite.window = ScoringWindow::MarketAdwin;
  e.composite.decision = ScoringDecision::Evidence;
  const PredictionSet p = runPredictions(data, e);
  std::size_t first = 0;
  for (const auto& mp : p.models) first = std::max(first, mp.start);
  first += SelectorSpec{}.lookback;
  Market m{name, {}, {}};
  const std::vector<ModelPredictions> forecast = {p.models.back()};
  for (const auto& k : kCosts) {
    const CandidateBook book(forecast, ExperimentSpec::defaultStrategies(), p.nextReturns, k.cost, k.stamp);
    const std::size_t from = first - book.start();
    std::vector<PerformanceMetrics> row(kRows);
    row[0] = runSelector(book, SelectorSpec{}, from).metrics;
    row[1].sharpe = -1e30;
    for (std::size_t c = 0; c < book.size(); ++c) {
      const auto pm = evaluatePerformance(book.netSeries(c, from), book.turnoverSeries(c, from));
      if (pm.sharpe > row[1].sharpe) row[1] = pm;
    }
    const Axes ax = buildAxes(p.models, first, book.end(), p.nextReturns.assets());
    std::vector<bool> all(ax.K, true), picks(ax.K, true), market(ax.K, false);
    picks[0] = false, market[0] = true;
    std::vector<Fit> fits = {search(ax, p.nextReturns, first, all, k), search(ax, p.nextReturns, first, picks, k)};
    // Market only: hold the equal-weight market, at the best rate.
    Fit mk;
    mk.metrics.sharpe = -1e30;
    for (double rate : kRates) {
      std::vector<double> c(ax.K, 0.0);
      c[0] = 1;
      const auto pm = simulate(ax, p.nextReturns, first, c, rate, k);
      if (pm.sharpe > mk.metrics.sharpe) mk = {pm, c, rate};
    }
    row[2] = fits[0].metrics, row[3] = fits[1].metrics, row[4] = mk.metrics;
    m.out.push_back(row);
    m.fits.push_back(fits);
  }
  std::fprintf(stderr, "  %s ready\n", name.c_str());
  return m;
}

std::vector<bool> report(const char* title, const std::vector<Market>& set) {
  const double n = static_cast<double>(set.size());
  std::vector<bool> pass;
  for (std::size_t c = 0; c < std::size(kCosts); ++c) {
    std::printf("\n%s, %s\n  %-42s %8s %8s %8s %8s %8s %6s\n", title, kCosts[c].name, "", "ann.ret", "Sharpe", "worst", "max DD", "turn/yr",
                "beats");
    std::vector<double> mean(kRows, 0.0);
    for (std::size_t r = 0; r < kRows; ++r) {
      double ret = 0, worst = 1e9, dd = 0, turn = 0;
      int beats = 0;
      for (const auto& m : set) {
        const auto& pm = m.out[c][r];
        ret += pm.annualReturn / n, mean[r] += pm.sharpe / n, dd += pm.maxDrawdown / n, turn += 252 * pm.averageTurnover / n;
        worst = std::min(worst, pm.sharpe);
        beats += r > 0 && pm.sharpe > m.out[c][0].sharpe + 1e-9 ? 1 : 0;
      }
      std::printf("  %-42s %7.1f%% %8.2f %8.2f %7.1f%% %7.0fx", kNames[r], 100 * ret, mean[r], worst, 100 * dd, turn);
      if (r > 0) std::printf(" %3d/%-2zu", beats, set.size());
      std::printf("\n");
    }
    // What the full ceiling used: mean |coefficient| share per axis, and the rate.
    const char* axes[] = {"market", "f1", "f2", "f3", "f4", "f5", "f6", "self-adaptive"};
    std::vector<double> share(8, 0.0);
    double rate = 0;
    for (const auto& m : set) {
      const auto& f = m.fits[c][0];
      double s = 0;
      for (double x : f.coef) s += std::fabs(x);
      for (std::size_t a = 0; a < f.coef.size() && a < 8; ++a) share[a] += (s > 0 ? f.coef[a] / s : 0) / n;
      rate += f.rate / n;
    }
    std::printf("  ceiling's mean signed weight per axis:");
    for (std::size_t a = 0; a < 8; ++a) std::printf(" %s %+.2f", axes[a], share[a]);
    std::printf("; trade rate %.2f\n", rate);
    pass.push_back(mean[2] > mean[0]);
  }
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::size_t markets = 6, days = 2520;
  std::string csv;
  std::vector<std::string> pos;
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv" && k + 1 < argc) csv = argv[++k];
    else pos.push_back(a);
  }
  if (pos.size() > 0) markets = std::strtoul(pos[0].c_str(), nullptr, 10);
  if (pos.size() > 1) days = std::strtoul(pos[1].c_str(), nullptr, 10);
  if (markets == 0 && csv.empty()) {
    std::fprintf(stderr, "ceiling_check 0 needs --csv\n");
    return 2;
  }
  bool go = true;
  auto verdict = [&](const std::string& set, const std::vector<bool>& pass, std::size_t only = SIZE_MAX) {
    for (std::size_t c = 0; c < pass.size(); ++c) {
      if (only != SIZE_MAX && c != only) continue;
      std::printf("Ceiling above the default on %s, %s: %s\n", set.c_str(), kCosts[c].name, pass[c] ? "yes" : "NO");
      go = go && pass[c];
    }
  };
  if (markets > 0) {
    auto build = [&](std::uint64_t base) {
      std::vector<Market> set;
      for (std::size_t k = 0; k < markets; ++k) {
        SyntheticMarketSpec ms;
        ms.seed = base + k;
        ms.numDates = days;
        set.push_back(prepare("seed " + std::to_string(ms.seed), generateSyntheticMarket(ms)));
      }
      return set;
    };
    std::fprintf(stderr, "selection markets\n");
    const auto sel = report("Selection markets", build(101));
    std::fprintf(stderr, "validation markets\n");
    const auto val = report("Validation markets (unseen)", build(201));
    std::printf("\n");
    verdict("the selection markets", sel);
    verdict("the unseen markets", val);
  }
  if (!csv.empty()) {
    std::ifstream f(csv);
    std::stringstream ss;
    ss << f.rdbuf();
    const auto uk = report(("Real data: " + csv).c_str(), {prepare(csv, parseCsv(ss.str()))});
    verdict(csv, uk, 1);
  }
  if (markets == 0)
    std::printf("\nUK part of the ceiling check: %s (the synthetic part is not run here: ceiling_check [markets])\n", go ? "clears" : "fails");
  else
    std::printf("\nCeiling check: %s\n", go ? "clears the bar; gate 1c is worth building" : "fails; no overlay on these portfolios can beat the default");
  return 0;
}
