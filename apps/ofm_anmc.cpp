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
//
//   ofm_anmc --basket 10 [--outer 20000] [--reps 1] [--budget 1000000] [--threads -1] [--check]
//            [--methods hermite2,hermite3,values,differential,trusted,fused] [--stats VaR,ES,EPE] [--epochs N] [--estimates]
//
// The differential network proxy (include/strategy/differential_proxy.hpp) against the Hermite regression on a book of
// d correlated assets whose value at the horizon is exact (see Basket below); each outer path carries its pathwise
// derivatives by the horizon log-spots. --check fails unless the differential network's own VaR and ES errors are below
// half the values-only network's (the derivative labels at work), its adaptive errors stay within their bounds and the
// fused kernel's single precision keeps its errors within 0.25 points.
//
//   ofm_anmc --kernels [--check] [--dump anmc-gpu.json]
//
// The fused kernels (include/strategy/diffnet_gpu.hpp, diffnet_wgsl.hpp) in their f32 reference against the network's
// double precision: the exact weight gradient against central differences, one step's per-sample gradients, a whole fit,
// and the single fused kernel against the two kernels (bit for bit at a batch of 256). --dump writes the WGSL, the
// buffers and the reference's results for the WebGPU test (web/tests/anmcgpu.mjs).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "strategy/adaptive_nested.hpp"
#include "strategy/differential_proxy.hpp"
#include "strategy/diffnet_gpu.hpp"

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

// The statistic of exact values v (base: today's value, for VaR and ES).
double statOf(const char* stat, const std::vector<double>& v, double base) {
  const std::size_t n = v.size();
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

// The exact statistic on the evaluation scenarios (the same ones the estimate uses).
double exactStat(const char* stat, const Book& b, const anmc::OuterScenarios& o, const anmc::ScenarioSet& s, double base) {
  std::vector<double> v(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) v[i] = b.value(std::exp(o.at(s.original[i])[0]), T - TH);
  return statOf(stat, v, base);
}

std::unique_ptr<anmc::RiskStatistic> makeStat(const char* stat, double base) {
  if (!std::strcmp(stat, "VaR")) return std::make_unique<anmc::ValueAtRisk>(0.99, base);
  if (!std::strcmp(stat, "ES")) return std::make_unique<anmc::ExpectedShortfall>(0.99, base);
  return std::make_unique<anmc::PositiveExposure>();
}

// Normals from SplitMix64, both halves of each Box-Muller draw.
struct Normals {
  anmc::SplitMix64 g;
  double spare = 0;
  bool has = false;
  explicit Normals(std::uint64_t seed) : g(seed) {}
  double next() {
    if (has) {
      has = false;
      return spare;
    }
    double u1 = g.uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    const double rad = std::sqrt(-2 * std::log(u1)), ang = 6.283185307179586 * g.uniform();
    spare = rad * std::sin(ang), has = true;
    return rad * std::cos(ang);
  }
};

// d correlated lognormal assets (S0 = 100, volatilities 15% to 35%, correlation 0.3), a real-world drift of 8% to the
// horizon t = 10 trading days, risk-neutral to expiry at 6 months. The book is short 3 straddles struck at 100 on the
// basket's geometric average and holds risk reversals (long the 105 call, short the 95 put; notional +-0.5, alternating)
// on every asset, so its value is not additive across assets; its value at the horizon is exact (Black-Scholes for each
// asset and for the lognormal geometric average). The state is the d log-spots at t; the payoff is the P&L
// dV = V(S_t) - V(S_0) e^{rt}, so the statistics are the P&L's 99% VaR and ES (base 0) and E[max(dV, 0)] (EPE).
struct Basket {
  std::size_t d;
  double s0 = 100, r = 0.03, mu = 0.08, rho = 0.3, horizon = 10.0 / 252.0, expiry = 0.5, strike = 100, straddles = 3;
  std::vector<double> vol, rr;
  double volG2 = 0, meanVar = 0, pnlOffset = 0;

  explicit Basket(std::size_t assets) : d(assets) {
    for (std::size_t k = 0; k < d; ++k) {
      vol.push_back(d == 1 ? 0.25 : 0.15 + 0.20 * static_cast<double>(k) / static_cast<double>(d - 1));
      rr.push_back(k % 2 == 0 ? 0.5 : -0.5);
    }
    for (std::size_t j = 0; j < d; ++j) {
      meanVar += vol[j] * vol[j] / static_cast<double>(d);
      for (std::size_t k = 0; k < d; ++k) volG2 += (j == k ? 1.0 : rho) * vol[j] * vol[k] / static_cast<double>(d * d);
    }
    const std::vector<double> x0(d, std::log(s0));
    pnlOffset = value(x0.data(), expiry) * std::exp(r * horizon);
  }
  static double call(double fwd, double k, double sd, double df) {
    const double d1 = (std::log(fwd / k) + 0.5 * sd * sd) / sd;
    return df * (fwd * anmc::normCdf(d1) - k * anmc::normCdf(d1 - sd));
  }
  // The book's value at log-spots x with tau to expiry.
  double value(const double* x, double tau) const {
    const double df = std::exp(-r * tau);
    double logG = 0;
    for (std::size_t k = 0; k < d; ++k) logG += x[k] / static_cast<double>(d);
    const double fwdG = std::exp(logG + (r - 0.5 * meanVar + 0.5 * volG2) * tau), sdG = std::sqrt(volG2 * tau);
    const double callG = call(fwdG, strike, sdG, df), putG = callG - df * (fwdG - strike);
    double v = -straddles * (callG + putG);
    for (std::size_t k = 0; k < d; ++k) {
      const double fwd = std::exp(x[k] + r * tau), sd = vol[k] * std::sqrt(tau);
      v += rr[k] * (call(fwd, 105, sd, df) - (call(fwd, 95, sd, df) - df * (fwd - 95)));
    }
    return v;
  }
  double pnl(const double* x) const { return value(x, expiry - horizon) - pnlOffset; }
  // One risk-neutral path from log-spots x to expiry: the discounted P&L and (if grad) its pathwise gradient by x.
  double path(const double* x, Normals& z, double* grad) const {
    const double tau = expiry - horizon, df = std::exp(-r * tau), common = z.next();
    double logG = 0;
    std::vector<double> sT(d);
    for (std::size_t k = 0; k < d; ++k) {
      const double eps = std::sqrt(rho) * common + std::sqrt(1 - rho) * z.next();
      const double logS = x[k] + (r - 0.5 * vol[k] * vol[k]) * tau + vol[k] * std::sqrt(tau) * eps;
      sT[k] = std::exp(logS);
      logG += logS / static_cast<double>(d);
    }
    const double g = std::exp(logG);
    double payoff = -straddles * std::fabs(g - strike);
    for (std::size_t k = 0; k < d; ++k) payoff += rr[k] * (std::max(sT[k] - 105, 0.0) - std::max(95 - sT[k], 0.0));
    if (grad)
      for (std::size_t k = 0; k < d; ++k) {
        const double dStraddle = -straddles * (g > strike ? 1.0 : -1.0) * g / static_cast<double>(d);
        const double dRr = rr[k] * ((sT[k] > 105 ? 1.0 : 0.0) + (sT[k] < 95 ? 1.0 : 0.0)) * sT[k];
        grad[k] = df * (dStraddle + dRr);
      }
    return df * payoff - pnlOffset;
  }
  anmc::OuterScenarios outer(std::size_t n, std::uint64_t seed) const {
    anmc::OuterScenarios o;
    o.n = n, o.d = d;
    o.state.resize(n * d), o.payoff.resize(n), o.gradient.resize(n * d);
    Normals z(anmc::derive(seed, -7, 0));
    for (std::size_t i = 0; i < n; ++i) {
      const double common = z.next();
      for (std::size_t k = 0; k < d; ++k) {
        const double eps = std::sqrt(rho) * common + std::sqrt(1 - rho) * z.next();
        o.state[i * d + k] = std::log(s0) + (mu - 0.5 * vol[k] * vol[k]) * horizon + vol[k] * std::sqrt(horizon) * eps;
      }
      o.payoff[i] = path(o.at(i), z, o.gradient.data() + i * d);
    }
    return o;
  }
  anmc::InnerSimulator inner() const {
    return [this](const double* state, std::size_t, double* pay, std::size_t count, std::uint64_t seed) {
      Normals z(seed);
      for (std::size_t k = 0; k < count; ++k) pay[k] = path(state, z, nullptr);
    };
  }
};

// A comma-separated list's members (empty list: everything).
bool listed(const std::string& list, const char* name) {
  if (list.empty()) return true;
  for (std::size_t a = 0; a <= list.size();) {
    const std::size_t b = std::min(list.find(',', a), list.size());
    if (list.compare(a, b - a, name) == 0) return true;
    a = b + 1;
  }
  return false;
}

struct BasketRun {
  std::size_t d = 10, nOuter = 20000, reps = 1, budget = 1000000, epochs = 0;
  int threads = -1;
  bool check = false, estimates = false;
  std::string methods, stats;  // comma lists (empty: all)
};

int basketMain(const BasketRun& run) {
  const std::size_t d = run.d, nOuter = run.nOuter, reps = run.reps, budget = run.budget;
  const int threads = run.threads;
  const bool check = run.check;
  const Basket basket(d);
  struct Method {
    const char* key;
    const char* name;
    anmc::ProxyFactory proxy;
    int degree;
  };
  anmc::DifferentialOptions net;
  if (run.epochs) net.epochs = run.epochs;
  anmc::DifferentialOptions values = net, trusted = net;
  values.useDerivatives = false;
  trusted.estimationVariance = false;
  const Method methods[] = {{"hermite2", "Hermite degree 2", nullptr, 2},
                            {"hermite3", "Hermite degree 3", nullptr, 3},
                            {"values", "network, values only", anmc::differentialProxy(values), 0},
                            {"differential", "differential network", anmc::differentialProxy(net), 0},
                            {"trusted", "differential, no estimation variance", anmc::differentialProxy(trusted), 0},
                            {"fused", "differential, fused kernel (f32)", anmc::fusedDifferentialProxy(net), 0}};
  constexpr std::size_t kMethods = sizeof(methods) / sizeof(methods[0]);
  const std::size_t values_ = 2, diff = 3, fused = 5;
  const char* stats[] = {"VaR", "ES", "EPE"};
  if (check && (!run.methods.empty() || !run.stats.empty() || run.epochs)) {
    std::fprintf(stderr, "--basket --check runs every method and statistic with the default training\n");
    return 2;
  }
  // bounds for --check: the differential network's adaptive RMSE, % of exact (EPE: the noise floor of the outer payoffs
  // the unrefined scenarios keep as their values)
  const double bound[] = {3.0, 3.0, 8.0};
  auto tooBig = [&](std::size_t m) {  // a Hermite basis past 500 functions: minutes a fit, and no better
    return methods[m].degree > 0 && anmc::HermiteBasis(d, methods[m].degree).size() > 500;
  };
  auto skipped = [&](std::size_t m) { return tooBig(m) || !listed(run.methods, methods[m].key); };
  auto chosen = [&](int st) { return listed(run.stats, stats[st]); };
  double proxyErr[kMethods][3] = {}, adErr[kMethods][3] = {}, paths[kMethods][3] = {}, secs[kMethods] = {};
  std::string proxyName[kMethods];
  for (std::size_t r = 0; r < reps; ++r) {
    const anmc::OuterScenarios o = basket.outer(nOuter, 1000 + r);
    for (std::size_t m = 0; m < kMethods; ++m)
      for (int st = 0; st < 3 && !skipped(m); ++st) {
        if (!chosen(st)) continue;
        anmc::Options opt;
        opt.totalInnerBudget = budget;
        opt.maxDegreeOfParallelism = threads;
        opt.seed = 20261122 + r;
        opt.proxy = methods[m].proxy;
        if (methods[m].degree) opt.hermiteDegree = methods[m].degree;
        const auto stat = makeStat(stats[st], 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        const auto res = anmc::AdaptiveNestedEstimator(opt).estimate(o, basket.inner(), *stat);
        secs[m] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::vector<double> v(res.scenarios.size());
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = basket.pnl(o.at(res.scenarios.original[i]));
        const double exact = statOf(stats[st], v, 0.0);
        proxyErr[m][st] += std::pow((res.baselineEstimate - exact) / std::fabs(exact), 2);
        adErr[m][st] += std::pow((res.estimate - exact) / std::fabs(exact), 2);
        paths[m][st] += static_cast<double>(res.innerPathsUsed);
        proxyName[m] = res.proxy->describe();
        if (run.estimates)
          std::printf("estimate,%s,%s,%zu,%.10g,%.10g,%.10g,%zu\n", methods[m].key, stats[st], r, res.estimate, res.baselineEstimate, exact, res.innerPathsUsed);
      }
  }
  const double nr = static_cast<double>(reps);
  std::printf("%zu assets, %zu outer paths (half train the proxy), budget %zu inner paths, %zu repetition(s); RMSE as %% of exact\n", d, nOuter, budget, reps);
  std::printf("proxy,statistic,regression-only RMSE %%,adaptive RMSE %%,inner paths,seconds (3 statistics)\n");
  for (std::size_t m = 0; m < kMethods; ++m)
    for (int st = 0; st < 3; ++st)
      if (!listed(run.methods, methods[m].key) || !chosen(st)) continue;
      else if (tooBig(m)) std::printf("%s,%s,,,,(skipped: %zu functions)\n", methods[m].name, stats[st], anmc::HermiteBasis(d, methods[m].degree).size());
      else
        std::printf("%s,%s,%.2f,%.2f,%.0f,%.1f\n", methods[m].name, stats[st], 100 * std::sqrt(proxyErr[m][st] / nr), 100 * std::sqrt(adErr[m][st] / nr),
                    paths[m][st] / nr, secs[m] / nr);
  for (std::size_t m = 0; m < kMethods; ++m)
    if (!skipped(m)) std::printf("  %s: %s\n", methods[m].name, proxyName[m].c_str());
  if (!check) return 0;
  bool ok = true;
  for (int st = 0; st < 3; ++st) {
    const double pd = 100 * std::sqrt(proxyErr[diff][st] / nr), pv = 100 * std::sqrt(proxyErr[values_][st] / nr);
    const double ad = 100 * std::sqrt(adErr[diff][st] / nr), af = 100 * std::sqrt(adErr[fused][st] / nr);
    if (st < 2 && !(pd < 0.5 * pv)) {  // the derivative labels are what make it work
      std::printf("FAIL: %s: the differential network's own error %.2f%% is not below half the values-only network's %.2f%%\n", stats[st], pd, pv);
      ok = false;
    }
    if (!(std::fabs(af - ad) <= 0.25)) {
      std::printf("FAIL: %s: the fused kernel's single precision changes the error (%.2f%% against %.2f%%)\n", stats[st], af, ad);
      ok = false;
    }
    if (!(ad <= bound[st])) {
      std::printf("FAIL: %s: the differential network's adaptive error %.2f%% above %.1f%%\n", stats[st], ad, bound[st]);
      ok = false;
    }
    if (!(af <= bound[st])) {
      std::printf("FAIL: %s: the fused kernel's adaptive error %.2f%% above %.1f%%\n", stats[st], af, bound[st]);
      ok = false;
    }
  }
  std::printf(ok ? "differential network proxy: within bounds; OK\n" : "differential network proxy: FAILED\n");
  return ok ? 0 : 1;
}

template <class T>
void jsonArray(std::FILE* f, const char* name, const std::vector<T>& v, bool last = false) {
  std::fprintf(f, "\"%s\":[", name);
  for (std::size_t i = 0; i < v.size(); ++i) {
    if constexpr (std::is_floating_point<T>::value) std::fprintf(f, i ? ",%.9g" : "%.9g", static_cast<double>(v[i]));
    else std::fprintf(f, i ? ",%llu" : "%llu", static_cast<unsigned long long>(v[i]));
  }
  std::fprintf(f, last ? "]" : "],");
}
void jsonString(std::FILE* f, const char* name, const char* text) {
  std::fprintf(f, "\"%s\":\"", name);
  for (const char* c = text; *c; ++c) {
    if (*c == '\n') std::fputs("\\n", f);
    else if (*c == '"' || *c == '\\') std::fprintf(f, "\\%c", *c);
    else std::fputc(*c, f);
  }
  std::fputs("\",", f);
}

// A smooth function of d numbers (sines, a product, squares) with noisy values and noisy pathwise-style derivative labels:
// the kernels' check problem. Every seventh row has no derivatives and every fifth weighs 3.
anmc::DifferentialProblem kernelProblem(std::size_t d, std::size_t n, std::size_t epochs, std::size_t batch, std::vector<double>& test,
                                        std::vector<double>& testValue) {
  auto value = [d](const double* v) {
    double s = v[0] * v[1 % d];
    for (std::size_t j = 0; j < d; ++j) s += std::sin(v[j]) / static_cast<double>(j + 1) + (j % 3 == 2 ? 0.25 * v[j] * v[j] : 0.0);
    return s;
  };
  anmc::OuterScenarios o;
  o.n = n, o.d = d;
  o.state.resize(n * d), o.payoff.resize(n), o.gradient.resize(n * d);
  Normals z(11);
  for (std::size_t i = 0; i < n; ++i) {
    double* x = o.state.data() + i * d;
    for (std::size_t j = 0; j < d; ++j) x[j] = z.next();
    o.payoff[i] = value(x) + 0.5 * z.next();
    for (std::size_t j = 0; j < d; ++j) {
      double dv = std::cos(x[j]) / static_cast<double>(j + 1) + (j % 3 == 2 ? 0.5 * x[j] : 0.0);
      if (j == 0) dv += x[1 % d];
      if (j == 1 % d) dv += x[0];
      o.gradient[i * d + j] = dv + 0.5 * z.next();
    }
  }
  test.resize(1000 * d), testValue.resize(1000);
  for (std::size_t i = 0; i < 1000; ++i) {
    for (std::size_t j = 0; j < d; ++j) test[i * d + j] = z.next();
    testValue[i] = value(test.data() + i * d);
  }
  anmc::DifferentialOptions opt;
  opt.epochs = epochs;
  opt.batch = batch;
  std::vector<std::size_t> rows(n);
  for (std::size_t i = 0; i < n; ++i) rows[i] = i;
  anmc::DifferentialProblem p = anmc::DifferentialProxy(opt).problem(o, rows);
  for (std::size_t k = 0; k < n; ++k) {
    if (k % 7 == 0) p.hasGradient[k] = 0;
    if (k % 5 == 0) p.w[k] = 3;
  }
  return p;
}

double relDiff(const std::vector<float>& a, const std::vector<double>& b) {
  double diff = 0, scale = 0;
  for (std::size_t i = 0; i < b.size(); ++i) diff = std::max(diff, std::fabs(double(a[i]) - b[i])), scale = std::max(scale, std::fabs(b[i]));
  return diff / std::max(scale, 1e-300);
}
double relDiff(const std::vector<float>& a, const std::vector<float>& b) { return relDiff(a, std::vector<double>(b.begin(), b.end())); }

int kernelsMain(bool check, const char* dump, int threads) {
  anmc::detail::PoolScope pool(threads);
  bool ok = true;
  auto verdict = [&](const char* what, double got, double bound) {
    const bool pass = got <= bound;
    std::printf("%s %s: %.3g (bound %.0e)\n", pass ? "ok  " : "FAIL", what, got, bound);
    ok = ok && pass;
  };
  verdict("exact weight gradient against central differences", anmc::DifferentialProxy::gradientCheck(3), 1e-5);

  std::vector<double> test, testValue;
  const auto t0 = std::chrono::steady_clock::now();
  const anmc::DifferentialProblem p = kernelProblem(10, 4000, 8, 256, test, testValue);
  const anmc::DiffnetFitPlan fit = anmc::compileDifferentialFit(p);
  std::printf("%zu inputs, %zu rows, %zu weights, %zu steps of %zu\n", p.inputs, p.rows, p.parameters(), fit.steps, fit.batch);
  // one step: the per-sample rows summed against the double-precision gradient of the same rows
  {
    const std::vector<float> rowsG = anmc::runDifferentialGradReference(fit, fit.theta, 0);
    const std::uint32_t start = fit.stepTable[0], m = fit.stepTable[1];
    std::vector<std::uint32_t> rows(fit.order.begin() + start, fit.order.begin() + start + m);
    std::vector<double> g64;
    anmc::DifferentialProxy::lossAndGradient(p, std::vector<double>(fit.theta.begin(), fit.theta.end()), rows, g64);
    std::vector<float> g32(fit.parameters, 0.0f);
    for (std::uint32_t k = 0; k < m; ++k)
      for (std::size_t q = 0; q < fit.parameters; ++q) g32[q] += rowsG[k * (fit.parameters + 1) + q];
    verdict("one step's per-sample gradients (f32) against double precision", relDiff(g32, g64), 1e-4);
  }
  // the whole fit
  std::vector<float> loss32;
  const std::vector<float> theta32 = anmc::runDifferentialFitReference(fit, &loss32);
  const double t32 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const std::vector<double> theta64 = anmc::DifferentialProxy::train(p);
  verdict("the fit's weights (two kernels, f32) against double precision", relDiff(theta32, theta64), 1e-3);
  // the fit's quality on fresh states, from the kernels' prediction
  const anmc::DiffnetPredictPlan predict = anmc::compileDifferentialPredict(p, std::vector<double>(theta32.begin(), theta32.end()), test);
  const std::vector<float> out = anmc::runDifferentialPredictReference(predict);
  double se = 0;
  for (std::size_t i = 0; i < testValue.size(); ++i) se += std::pow(double(out[i * (p.inputs + 1)]) - testValue[i], 2);
  std::printf("     fit: step loss %.4f -> %.4f, test RMS %.4f, f32 reference %.1f s\n", loss32.front() / 256, loss32.back() / 256,
              std::sqrt(se / static_cast<double>(testValue.size())), t32);
  // the single fused kernel: bit for bit the two kernels at a batch of 256, close at 512
  const anmc::DiffnetFusedPlan fused = anmc::compileDifferentialFused(p, test);
  const anmc::DiffnetFusedResult fr = anmc::runDifferentialFusedReference(fused);
  verdict("single fused kernel against the two kernels, weights (batch 256)", relDiff(fr.theta, theta32), 0);
  verdict("single fused kernel against the two kernels, predictions (batch 256)", relDiff(fr.predictions, out), 0);
  {
    std::vector<double> t2, v2;
    const anmc::DifferentialProblem p512 = kernelProblem(10, 4000, 8, 512, t2, v2);
    const std::vector<float> two = anmc::runDifferentialFitReference(anmc::compileDifferentialFit(p512));
    const anmc::DiffnetFusedResult one = anmc::runDifferentialFusedReference(anmc::compileDifferentialFused(p512, {}));
    verdict("single fused kernel against the two kernels, weights (batch 512)", relDiff(one.theta, two), 1e-5);
  }
  if (dump) {
    std::FILE* f = std::fopen(dump, "w");
    if (!f) {
      std::printf("FAIL: cannot write %s\n", dump);
      return 1;
    }
    std::fputc('{', f);
    jsonString(f, "wgsl", anmc::kDiffnetWgsl);
    std::fprintf(f, "\"fit\":{\"parameters\":%zu,\"batch\":%zu,\"steps\":%zu,", fit.parameters, fit.batch, fit.steps);
    jsonArray(f, "header", fit.header), jsonArray(f, "stepTable", fit.stepTable), jsonArray(f, "theta", fit.theta), jsonArray(f, "x", fit.x);
    jsonArray(f, "g", fit.g), jsonArray(f, "rowData", fit.rowData), jsonArray(f, "order", fit.order, true);
    std::fprintf(f, "},\"predict\":{\"rows\":%zu,\"inputs\":%zu,", predict.rows, predict.inputs);
    jsonArray(f, "header", predict.header), jsonArray(f, "x", predict.x, true);
    std::fputs("},\"expected\":{", f);
    jsonArray(f, "theta", theta32), jsonArray(f, "out", out, true);
    std::fputs("},", f);
    jsonString(f, "fusedWgsl", anmc::kDiffnetFusedWgsl);
    std::fprintf(f, "\"fused\":{\"parameters\":%zu,\"steps\":%zu,\"predictRows\":%zu,\"inputs\":%zu,\"workFloats\":%zu,\"outFloats\":%zu,", fused.parameters,
                 fused.steps, fused.predictRows, fused.inputs, fused.workFloats(), fused.outFloats());
    jsonArray(f, "header", fused.header), jsonArray(f, "theta", fused.theta), jsonArray(f, "xg", fused.xg), jsonArray(f, "rowData", fused.rowData);
    jsonArray(f, "datau", fused.datau, true);
    std::fputs("},\"expectedFused\":{", f);
    jsonArray(f, "theta", fr.theta), jsonArray(f, "predictions", fr.predictions), jsonArray(f, "stepLoss", fr.stepLoss, true);
    std::fputs("}}\n", f);
    std::fclose(f);
    std::printf("     wrote %s for web/tests/anmcgpu.mjs\n", dump);
  }
  if (check) std::printf(ok ? "differential kernels: their f32 reference agrees with double precision; OK\n" : "differential kernels: FAILED\n");
  return ok || !check ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t nOuter = 100000, reps = 1, budget = 2000000;
  int threads = -1;
  std::vector<int> degrees = {1, 2, 4};
  bool check = false, outerGiven = false, budgetGiven = false, kernels = false;
  std::size_t basket = 0;
  BasketRun run;
  const char* dump = nullptr;
  for (int a = 1; a < argc; ++a) {
    const std::string k = argv[a];
    auto next = [&]() -> const char* {
      if (a + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", k.c_str()); std::exit(2); }
      return argv[++a];
    };
    if (k == "--outer") nOuter = std::strtoul(next(), nullptr, 10), outerGiven = true;
    else if (k == "--reps") reps = std::strtoul(next(), nullptr, 10);
    else if (k == "--budget") budget = std::strtoul(next(), nullptr, 10), budgetGiven = true;
    else if (k == "--check") check = true;
    else if (k == "--basket") basket = std::strtoul(next(), nullptr, 10);
    else if (k == "--kernels") kernels = true;
    else if (k == "--dump") dump = next();
    else if (k == "--methods") run.methods = next();
    else if (k == "--stats") run.stats = next();
    else if (k == "--epochs") run.epochs = std::strtoul(next(), nullptr, 10);
    else if (k == "--estimates") run.estimates = true;
    else if (k == "--threads") threads = std::atoi(next());
    else if (k == "--degrees") {
      degrees.clear();
      for (const char* p = next(); *p;) {
        degrees.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p) ++p;
      }
    } else {
      std::fprintf(stderr, "ofm_anmc: unknown argument %s\n", k.c_str());
      std::fprintf(stderr, "usage: ofm_anmc [--outer N] [--reps N] [--degrees 1,2,4] [--budget N] [--threads N] [--check]\n"
                           "       ofm_anmc --basket D [--outer N] [--reps N] [--budget N] [--threads N] [--check]\n"
                           "                [--methods hermite2,hermite3,values,differential,trusted,fused] [--stats VaR,ES,EPE] [--epochs N] [--estimates]\n"
                           "       ofm_anmc --kernels [--check] [--dump FILE] [--threads N]\n");
      return 2;
    }
  }
  if (kernels) return kernelsMain(check, dump, threads);
  if (basket) {
    run.d = basket, run.nOuter = outerGiven ? nOuter : 20000, run.reps = reps, run.budget = budgetGiven ? budget : 1000000;
    run.threads = threads, run.check = check;
    return basketMain(run);
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
        const auto st = makeStat(c.stat, base);
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
