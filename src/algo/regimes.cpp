#include "sat/algo/regimes.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "sat/core/stats.hpp"

namespace sat::algo {

namespace {

double density(double x, double m, double s) {
  const double z = (x - m) / s;
  return std::exp(-0.5 * z * z) / (s * 2.5066282746310002);
}

// Scaled forward pass: alpha[t][k] normalised per t, scale c[t]; returns log-likelihood.
double forward(const HiddenMarkovModel& m, const std::vector<double>& x, std::vector<std::vector<double>>& alpha, std::vector<double>& c) {
  const std::size_t T = x.size(), K = m.states();
  alpha.assign(T, std::vector<double>(K));
  c.assign(T, 0.0);
  double ll = 0.0;
  for (std::size_t t = 0; t < T; ++t) {
    double sum = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
      double prior = 0.0;
      if (t == 0) prior = m.initial[k];
      else
        for (std::size_t j = 0; j < K; ++j) prior += alpha[t - 1][j] * m.transition[j][k];
      alpha[t][k] = prior * std::max(density(x[t], m.mean[k], m.sd[k]), 1e-300);
      sum += alpha[t][k];
    }
    c[t] = sum > 0 ? sum : 1e-300;
    for (double& a : alpha[t]) a /= c[t];
    ll += std::log(c[t]);
  }
  return ll;
}

}  // namespace

HiddenMarkovModel fitHmm(const std::vector<double>& x, std::size_t K, std::size_t maxIterations, double tolerance) {
  const std::size_t T = x.size();
  if (K < 2 || K > 5) throw std::invalid_argument("HMM: 2 to 5 states");
  if (T < 20 * K) throw std::invalid_argument("HMM: too few observations for the number of states");
  for (double v : x)
    if (!std::isfinite(v)) throw std::invalid_argument("HMM: returns must be finite");
  HiddenMarkovModel m;
  // Initialise by sorting days on |return| into K groups (calm to volatile).
  std::vector<std::size_t> idx(T);
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return std::fabs(x[a]) < std::fabs(x[b]); });
  const double overallSd = stdev(x);
  for (std::size_t k = 0; k < K; ++k) {
    std::vector<double> g;
    for (std::size_t q = k * T / K; q < (k + 1) * T / K; ++q) g.push_back(x[idx[q]]);
    m.mean.push_back(0.0);
    m.sd.push_back(std::max(1e-4, std::sqrt(mean(std::vector<double>(g.size(), 0.0)) + 0) + overallSd * (0.5 + static_cast<double>(k))));
  }
  m.transition.assign(K, std::vector<double>(K, 0.05 / static_cast<double>(K - 1)));
  for (std::size_t k = 0; k < K; ++k) m.transition[k][k] = 0.95;
  m.initial.assign(K, 1.0 / static_cast<double>(K));
  std::vector<std::vector<double>> alpha, beta(T, std::vector<double>(K));
  std::vector<double> c;
  double prev = -1e300;
  for (std::size_t it = 0; it < maxIterations; ++it) {
    m.logLikelihood = forward(m, x, alpha, c);
    m.iterations = it + 1;
    // Backward pass with the same scales.
    for (std::size_t k = 0; k < K; ++k) beta[T - 1][k] = 1.0;
    for (std::size_t t = T - 1; t-- > 0;)
      for (std::size_t j = 0; j < K; ++j) {
        double s = 0;
        for (std::size_t k = 0; k < K; ++k) s += m.transition[j][k] * density(x[t + 1], m.mean[k], m.sd[k]) * beta[t + 1][k];
        beta[t][j] = s / c[t + 1];
      }
    // Posteriors and re-estimation.
    std::vector<double> gsum(K, 0.0), gx(K, 0.0), gxx(K, 0.0);
    std::vector<std::vector<double>> xi(K, std::vector<double>(K, 0.0));
    std::vector<double> gfrom(K, 0.0);
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t k = 0; k < K; ++k) {
        const double g = alpha[t][k] * beta[t][k];
        gsum[k] += g;
        gx[k] += g * x[t];
        if (t + 1 < T) {
          gfrom[k] += g;
          for (std::size_t j = 0; j < K; ++j)
            xi[k][j] += alpha[t][k] * m.transition[k][j] * density(x[t + 1], m.mean[j], m.sd[j]) * beta[t + 1][j] / c[t + 1];
        }
      }
    for (std::size_t k = 0; k < K; ++k) {
      m.initial[k] = alpha[0][k] * beta[0][k];
      m.mean[k] = gsum[k] > 0 ? gx[k] / gsum[k] : m.mean[k];
    }
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t k = 0; k < K; ++k) gxx[k] += alpha[t][k] * beta[t][k] * (x[t] - m.mean[k]) * (x[t] - m.mean[k]);
    for (std::size_t k = 0; k < K; ++k) {
      m.sd[k] = gsum[k] > 0 ? std::max(1e-5, std::sqrt(gxx[k] / gsum[k])) : m.sd[k];
      double row = 0;
      for (std::size_t j = 0; j < K; ++j) row += xi[k][j];
      for (std::size_t j = 0; j < K; ++j) m.transition[k][j] = row > 0 ? std::max(1e-6, xi[k][j] / row) : m.transition[k][j];
      double norm = 0;
      for (double v : m.transition[k]) norm += v;
      for (double& v : m.transition[k]) v /= norm;
    }
    if (std::fabs(m.logLikelihood - prev) < tolerance * (1.0 + std::fabs(prev))) break;
    prev = m.logLikelihood;
  }
  // Order states by volatility.
  std::vector<std::size_t> order(K);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return m.sd[a] < m.sd[b]; });
  HiddenMarkovModel o = m;
  for (std::size_t a = 0; a < K; ++a) {
    o.mean[a] = m.mean[order[a]];
    o.sd[a] = m.sd[order[a]];
    o.initial[a] = m.initial[order[a]];
    for (std::size_t b = 0; b < K; ++b) o.transition[a][b] = m.transition[order[a]][order[b]];
  }
  return o;
}

std::vector<std::vector<double>> filterHmm(const HiddenMarkovModel& m, const std::vector<double>& x) {
  std::vector<std::vector<double>> alpha;
  std::vector<double> c;
  forward(m, x, alpha, c);
  return alpha;
}

RegimeSwitchResult regimeSwitch(const std::vector<double>& x, const RegimeSwitchSpec& s) {
  const std::size_t T = x.size();
  if (s.window < 60 || s.refitEvery < 1) throw std::invalid_argument("regime switch: window >= 60 and refit >= 1");
  if (T <= s.window + 2) throw std::invalid_argument("regime switch: series shorter than the estimation window");
  RegimeSwitchResult out;
  out.start = s.window;
  HiddenMarkovModel model;
  double prevExposure = 0.0;
  for (std::size_t t = s.window; t < T; ++t) {
    if ((t - s.window) % s.refitEvery == 0)
      model = fitHmm(std::vector<double>(x.begin() + static_cast<std::ptrdiff_t>(t - s.window), x.begin() + static_cast<std::ptrdiff_t>(t)), s.states);
    // Filter the window up to yesterday's close with the current model.
    const std::size_t from = t >= s.window ? t - s.window : 0;
    const auto probs = filterHmm(model, std::vector<double>(x.begin() + static_cast<std::ptrdiff_t>(from), x.begin() + static_cast<std::ptrdiff_t>(t)));
    const double pHigh = probs.back().back();
    const double exposure = pHigh < s.threshold ? 1.0 : s.riskOffExposure;
    out.highVolProbability.push_back(pHigh);
    out.exposure.push_back(exposure);
    out.returns.push_back(exposure * x[t] - s.costBps * 1e-4 * std::fabs(exposure - prevExposure));
    prevExposure = exposure;
  }
  out.lastModel = model;
  return out;
}

}  // namespace sat::algo
