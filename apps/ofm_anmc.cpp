// ofm_anmc: the adaptive nested Monte Carlo (include/strategy/adaptive_nested.hpp) checked against exact answers, as in
// docs/adaptive-nested-mc.md's validation: GBM with S0 = 100, r = 3%, sigma = 25%, a real-world drift of 8% to the
// horizon t = 10 trading days, risk-neutral from t to T = 0.5 years; the state is log S_t. A short straddle struck at
// 100 for 99% VaR and ES, a risk reversal (long 105 call, short 95 put) for EPE. Exact conditional values are
// Black-Scholes prices; each estimate is compared with the exact statistic on the same evaluation scenarios.
//
//   ofm_anmc [--outer 100000] [--reps 1] [--degrees 1,2,4] [--budget 2000000] [--threads -1] [--check]
//
// --check fails (exit 1) unless refinement keeps every adaptive error within its bound and cuts the degree-1 VaR and
// the ES errors from regression-only: the CI's check that the library still does what the document says.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strategy/adaptive_nested.hpp"

namespace {

constexpr double S0 = 100, R = 0.03, SIGMA = 0.25, MU = 0.08, TH = 10.0 / 252.0, T = 0.5;

double bsCall(double s, double k, double tau) {
  const double d1 = (std::log(s / k) + (R + 0.5 * SIGMA * SIGMA) * tau) / (SIGMA * std::sqrt(tau)), d2 = d1 - SIGMA * std::sqrt(tau);
  return s * anmc::normCdf(d1) - k * std::exp(-R * tau) * anmc::normCdf(d2);
}
double bsPut(double s, double k, double tau) { return bsCall(s, k, tau) - s + k * std::exp(-R * tau); }

struct Book {
  const char* name;
  double (*payoff)(double);
  double (*value)(double, double);  // exact value at spot s with tau to expiry
};
double straddlePay(double s) { return -std::fabs(s - 100); }
double straddleVal(double s, double tau) { return -(bsCall(s, 100, tau) + bsPut(s, 100, tau)); }
double rrPay(double s) { return std::max(s - 105, 0.0) - std::max(95 - s, 0.0); }
double rrVal(double s, double tau) { return bsCall(s, 105, tau) - bsPut(s, 95, tau); }

double gauss(anmc::SplitMix64& g) {  // Box-Muller
  double u1 = g.uniform();
  if (u1 < 1e-300) u1 = 1e-300;
  return std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * g.uniform());
}

// The engine's ordinary run: S_t under the real-world drift, then on to T risk-neutral; the payoff discounted to t.
anmc::OuterScenarios outer(const Book& b, std::size_t n, std::uint64_t seed) {
  anmc::OuterScenarios o;
  o.n = n;
  o.d = 1;
  o.state.resize(n);
  o.payoff.resize(n);
  anmc::SplitMix64 g(anmc::derive(seed, -7, 0));
  const double tau = T - TH;
  for (std::size_t i = 0; i < n; ++i) {
    const double x = std::log(S0) + (MU - 0.5 * SIGMA * SIGMA) * TH + SIGMA * std::sqrt(TH) * gauss(g);
    const double st = std::exp(x + (R - 0.5 * SIGMA * SIGMA) * tau + SIGMA * std::sqrt(tau) * gauss(g));
    o.state[i] = x;
    o.payoff[i] = std::exp(-R * tau) * b.payoff(st);
  }
  return o;
}

// The same engine restarted at a horizon state, as one fused path kernel: the constants once a batch, both normals of
// each Box-Muller draw used (one log and one square root per two paths), the terminal price and the discounted payoff
// in the same pass.
anmc::InnerSimulator inner(const Book& b) {
  return [b](const double* state, std::size_t, double* pay, std::size_t count, std::uint64_t seed) {
    anmc::SplitMix64 g(seed);
    const double tau = T - TH, disc = std::exp(-R * tau), vol = SIGMA * std::sqrt(tau);
    const double s0 = std::exp(state[0] + (R - 0.5 * SIGMA * SIGMA) * tau);
    for (std::size_t k = 0; k < count; k += 2) {
      double u1 = g.uniform();
      if (u1 < 1e-300) u1 = 1e-300;
      const double rad = std::sqrt(-2 * std::log(u1)), ang = 6.283185307179586 * g.uniform();
      pay[k] = disc * b.payoff(s0 * std::exp(vol * rad * std::cos(ang)));
      if (k + 1 < count) pay[k + 1] = disc * b.payoff(s0 * std::exp(vol * rad * std::sin(ang)));
    }
  };
}

// The exact statistic on the evaluation scenarios (the same ones the estimate uses).
double exactStat(const char* stat, const Book& b, const anmc::OuterScenarios& o, const anmc::ScenarioSet& s, double base) {
  const std::size_t n = s.size();
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = b.value(std::exp(o.at(s.original[i])[0]), T - TH);
  if (!std::strcmp(stat, "EPE")) {
    double sum = 0;
    for (double x : v) sum += std::max(x, 0.0);
    return sum / static_cast<double>(n);
  }
  std::vector<double> l(n);
  for (std::size_t i = 0; i < n; ++i) l[i] = base - v[i];
  std::sort(l.begin(), l.end());
  if (!std::strcmp(stat, "VaR")) return l[anmc::ceilCount(static_cast<double>(n) * 0.99) - 1];
  const std::size_t k = anmc::ceilCount(static_cast<double>(n) * 0.01);
  double sum = 0;
  for (std::size_t i = n - k; i < n; ++i) sum += l[i];
  return sum / static_cast<double>(k);
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t nOuter = 100000, reps = 1, budget = 2000000;
  int threads = -1;
  std::vector<int> degrees = {1, 2, 4};
  bool check = false;
  for (int a = 1; a < argc; ++a) {
    const std::string k = argv[a];
    auto next = [&]() -> const char* {
      if (a + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", k.c_str()); std::exit(2); }
      return argv[++a];
    };
    if (k == "--outer") nOuter = std::strtoul(next(), nullptr, 10);
    else if (k == "--reps") reps = std::strtoul(next(), nullptr, 10);
    else if (k == "--budget") budget = std::strtoul(next(), nullptr, 10);
    else if (k == "--check") check = true;
    else if (k == "--threads") threads = std::atoi(next());
    else if (k == "--degrees") {
      degrees.clear();
      for (const char* p = next(); *p;) {
        degrees.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p) ++p;
      }
    } else {
      std::fprintf(stderr, "usage: ofm_anmc [--outer N] [--reps N] [--degrees 1,2,4] [--budget N] [--threads N] [--check]\n");
      return 2;
    }
  }
  const Book straddle{"short straddle 100", straddlePay, straddleVal}, rr{"risk reversal 105/95", rrPay, rrVal};
  struct Case { const char* stat; const Book* book; double bound; };  // bound: the adaptive RMSE allowed, % of exact
  const Case cases[] = {{"VaR", &straddle, 6.0}, {"ES", &straddle, 3.0}, {"EPE", &rr, 3.0}};
  bool ok = true;
  std::printf("statistic,degree,regression-only RMSE %%,adaptive RMSE %%,error cut,inner paths,stop\n");
  for (const auto& c : cases) {
    for (int deg : degrees) {
      double seBase = 0, seAd = 0, paths = 0;
      std::string stop;
      for (std::size_t r = 0; r < reps; ++r) {
        anmc::Options opt;
        opt.hermiteDegree = deg;
        opt.totalInnerBudget = budget;
        opt.maxDegreeOfParallelism = threads;
        opt.seed = 20261122 + r;
        const auto o = outer(*c.book, nOuter, 1000 + r);
        const double base = c.book->value(S0, T);  // today's value
        std::unique_ptr<anmc::RiskStatistic> st;
        if (!std::strcmp(c.stat, "VaR")) st = std::make_unique<anmc::ValueAtRisk>(0.99, base);
        else if (!std::strcmp(c.stat, "ES")) st = std::make_unique<anmc::ExpectedShortfall>(0.99, base);
        else st = std::make_unique<anmc::PositiveExposure>();
        const auto res = anmc::AdaptiveNestedEstimator(opt).estimate(o, inner(*c.book), *st);
        const double exact = exactStat(c.stat, *c.book, o, res.scenarios, base);
        const double eb = (res.baselineEstimate - exact) / std::fabs(exact), ea = (res.estimate - exact) / std::fabs(exact);
        seBase += eb * eb;
        seAd += ea * ea;
        paths += static_cast<double>(res.innerPathsUsed);
        stop = anmc::toString(res.stopReason);
      }
      const double rb = 100 * std::sqrt(seBase / static_cast<double>(reps)), ra = 100 * std::sqrt(seAd / static_cast<double>(reps));
      std::printf("%s 99%% %s,%d,%.2f,%.2f,%.1fx,%.0f,%s\n", c.stat, c.book->name, deg, rb, ra, ra > 0 ? rb / ra : 0.0, paths / static_cast<double>(reps), stop.c_str());
      if (check) {
        if (!(ra <= c.bound)) {
          std::printf("FAIL: %s degree %d: adaptive error %.2f%% above %.1f%%\n", c.stat, deg, ra, c.bound);
          ok = false;
        }
        if (std::strcmp(c.stat, "EPE") && (deg == 1 || !std::strcmp(c.stat, "ES")) && !(ra < rb)) {
          std::printf("FAIL: %s degree %d: refinement did not cut the regression-only error (%.2f%% -> %.2f%%)\n", c.stat, deg, rb, ra);
          ok = false;
        }
      }
    }
  }
  if (check) std::printf(ok ? "adaptive nested Monte Carlo: within bounds; OK\n" : "adaptive nested Monte Carlo: FAILED\n");
  return ok ? 0 : 1;
}
