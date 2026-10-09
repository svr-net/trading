#include "sat/adaptive/overlay.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "sat/adaptive/composite.hpp"
#include "sat/adaptive/self_adaptive.hpp"
#include "sat/core/random.hpp"
#include "sat/core/stats.hpp"

namespace sat {

namespace {
constexpr double kBeta1 = 0.9, kBeta2 = 0.999, kEps = 1e-8;
}

OverlayNet::OverlayNet(std::size_t features, const OverlaySpec& spec)
    : F_(features), H_(spec.hidden), K_(spec.maxUnits), spec_(spec) {
  if (F_ < 2 || H_ < 1 || K_ < 1) throw std::invalid_argument("overlay: needs a forecast, a hidden unit and a local unit");
  if (!(spec.unitRadius > 0) || !(spec.learningRate >= 0) || !(spec.sharpeRate > 0 && spec.sharpeRate < 1))
    throw std::invalid_argument("overlay: radius > 0, learning rate >= 0, 0 < Sharpe rate < 1");
  theta_.assign(shared() + K_ * (2 * H_ + 2), 0.0);
  m_.assign(theta_.size(), 0.0);
  v_.assign(theta_.size(), 0.0);
  steps_.assign(K_ + 1, 0);
  Rng rng(spec.seed);
  const double scale = 1.0 / std::sqrt(static_cast<double>(F_));
  for (std::size_t p = 0; p < H_ * F_; ++p) theta_[p] = scale * rng.normal();
}

std::vector<double> OverlayNet::gate(const double state[2], bool allocate) {
  auto dist2 = [&](const std::array<double, 2>& c) { return (state[0] - c[0]) * (state[0] - c[0]) + (state[1] - c[1]) * (state[1] - c[1]); };
  std::vector<double> g(K_, 0.0);
  const double r2 = spec_.unitRadius * spec_.unitRadius;
  auto fill = [&]() {
    double lo = std::numeric_limits<double>::infinity(), sum = 0;
    for (const auto& c : centres_) lo = std::min(lo, dist2(c));
    for (std::size_t k = 0; k < centres_.size(); ++k) sum += g[k] = std::exp(-(dist2(centres_[k]) - lo) / (2 * r2));
    for (std::size_t k = 0; k < centres_.size(); ++k) g[k] /= sum;
  };
  if (allocate && centres_.size() < K_) {
    double lo = std::numeric_limits<double>::infinity();
    for (const auto& c : centres_) lo = std::min(lo, dist2(c));
    if (lo > r2) {
      // The new unit starts as the units' current blend here, so the positions do not jump.
      const std::size_t k = centres_.size(), per = 2 * H_ + 2;
      if (k == 0) {
        Rng rng(spec_.seed + 1);
        for (std::size_t h = 0; h < H_; ++h) theta_[unitBase(0) + h] = 0.01 * rng.normal();
      } else {
        fill();
        for (std::size_t j = 0; j < per; ++j) {
          double x = 0;
          for (std::size_t q = 0; q < k; ++q) x += g[q] * theta_[unitBase(q) + j];
          theta_[unitBase(k) + j] = x;
        }
        std::fill(g.begin(), g.end(), 0.0);
      }
      centres_.push_back({state[0], state[1]});
    }
  }
  if (!centres_.empty()) fill();
  return g;
}

OverlayNet::Step OverlayNet::forward(const std::vector<double>& x, std::size_t N, const std::vector<double>& g,
                                     const std::vector<double>& wPrev, const std::vector<double>& JPrev) const {
  const std::size_t P = theta_.size(), fin = F_ - 1, H = H_;
  const double n = static_cast<double>(N);
  Step s;
  s.w.assign(N, 0.0);
  s.J.assign(N * P, 0.0);
  // The units' blend for this state.
  std::vector<double> cv(H, 0.0), cr(H + 1, 0.0);
  double ce = 0;
  for (std::size_t k = 0; k < K_; ++k) {
    if (g[k] == 0) continue;
    const std::size_t b = unitBase(k);
    for (std::size_t h = 0; h < H; ++h) cv[h] += g[k] * theta_[b + h];
    for (std::size_t h = 0; h <= H; ++h) cr[h] += g[k] * theta_[b + H + h];
    ce += g[k] * theta_[b + 2 * H + 1];
  }
  const double m = std::tanh(ce);
  std::vector<double> hs(N * H), a(N), lam(N);
  for (std::size_t i = 0; i < N; ++i) {
    double ai = 0, qi = cr[H];
    for (std::size_t h = 0; h < H; ++h) {
      const double* W = &theta_[h * F_];
      double pre = theta_[H * F_ + h] + W[fin] * n * wPrev[i];
      for (std::size_t f = 0; f < fin; ++f) pre += W[f] * x[i * fin + f];
      const double y = hs[i * H + h] = std::tanh(pre);
      ai += cv[h] * y, qi += cr[h] * y;
    }
    a[i] = ai;
    lam[i] = 1.0 / (1.0 + std::exp(-qi));
  }
  double abar = 0;
  for (double v : a) abar += v / n;
  std::vector<double> u(N), tz(N);
  for (std::size_t i = 0; i < N; ++i) {
    tz[i] = std::tanh(a[i] - abar);
    u[i] = (m + tz[i]) / n;
    s.w[i] = wPrev[i] + lam[i] * (u[i] - wPrev[i]);
    s.meanRate += lam[i] / n;
    s.exposure += s.w[i];
  }
  // Derivatives of the scores (da) and rate logits (dq) of every asset.
  std::vector<double> da(N * P, 0.0), dq(N * P, 0.0), dabar(P, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    double* dai = &da[i * P];
    double* dqi = &dq[i * P];
    double alphaA = 0, alphaQ = 0;
    for (std::size_t h = 0; h < H; ++h) {
      const double d = 1.0 - hs[i * H + h] * hs[i * H + h], ga = cv[h] * d, gq = cr[h] * d;
      for (std::size_t f = 0; f < fin; ++f) dai[h * F_ + f] = ga * x[i * fin + f], dqi[h * F_ + f] = gq * x[i * fin + f];
      dai[h * F_ + fin] = ga * n * wPrev[i], dqi[h * F_ + fin] = gq * n * wPrev[i];
      dai[H * F_ + h] = ga, dqi[H * F_ + h] = gq;
      alphaA += ga * theta_[h * F_ + fin] * n, alphaQ += gq * theta_[h * F_ + fin] * n;
    }
    // Through the position input, which depends on the parameters.
    const double* Jp = &JPrev[i * P];
    if (alphaA != 0 || alphaQ != 0)
      for (std::size_t p = 0; p < P; ++p) dai[p] += alphaA * Jp[p], dqi[p] += alphaQ * Jp[p];
    for (std::size_t k = 0; k < K_; ++k) {
      if (g[k] == 0) continue;
      const std::size_t b = unitBase(k);
      for (std::size_t h = 0; h < H; ++h) dai[b + h] += g[k] * hs[i * H + h], dqi[b + H + h] += g[k] * hs[i * H + h];
      dqi[b + 2 * H] += g[k];
    }
    for (std::size_t p = 0; p < P; ++p) dabar[p] += dai[p] / n;
  }
  std::vector<double> dm(P, 0.0);
  for (std::size_t k = 0; k < K_; ++k)
    if (g[k] != 0) dm[unitBase(k) + 2 * H + 1] = (1 - m * m) * g[k];
  for (std::size_t i = 0; i < N; ++i) {
    const double c1 = (1 - tz[i] * tz[i]) / n, c2 = lam[i] * (1 - lam[i]) * (u[i] - wPrev[i]);
    const double* Jp = &JPrev[i * P];
    const double* dai = &da[i * P];
    const double* dqi = &dq[i * P];
    double* Ji = &s.J[i * P];
    for (std::size_t p = 0; p < P; ++p)
      Ji[p] = (1 - lam[i]) * Jp[p] + lam[i] * (dm[p] / n + c1 * (dai[p] - dabar[p])) + c2 * dqi[p];
  }
  return s;
}

void OverlayNet::learn(const std::vector<double>& grad, const std::vector<double>& g) {
  const double lr = spec_.learningRate;
  auto adam = [&](std::size_t from, std::size_t to, std::size_t& step, double scale) {
    ++step;
    const double c1 = 1 - std::pow(kBeta1, static_cast<double>(step)), c2 = 1 - std::pow(kBeta2, static_cast<double>(step));
    for (std::size_t p = from; p < to; ++p) {
      m_[p] = kBeta1 * m_[p] + (1 - kBeta1) * grad[p];
      v_[p] = kBeta2 * v_[p] + (1 - kBeta2) * grad[p] * grad[p];
      theta_[p] += lr * scale * (m_[p] / c1) / (std::sqrt(v_[p] / c2) + kEps);
    }
  };
  adam(0, shared(), steps_[K_], 1.0);
  // Each unit learns in proportion to its responsibility for the state the positions were taken in.
  for (std::size_t k = 0; k < centres_.size(); ++k)
    if (g[k] > 0) adam(unitBase(k), unitBase(k) + 2 * H_ + 2, steps_[k], g[k]);
}

OverlayResult overlayTrader(const std::vector<ModelPredictions>& forecasts, const Panel& nextReturns, double costBps, double stampBps,
                            const OverlaySpec& spec) {
  if (forecasts.empty()) throw std::invalid_argument("overlay: needs forecasts");
  if (costBps < 0 || stampBps < 0) throw std::invalid_argument("overlay: costs must be >= 0");
  auto [start, end] = commonRange(forecasts);
  end = std::min(end, nextReturns.dates());
  if (start >= end) throw std::invalid_argument("overlay: no dates to trade");
  const std::size_t N = nextReturns.assets(), nf = forecasts.size();
  for (const auto& f : forecasts)
    if (f.probability.assets() != N) throw std::invalid_argument("overlay: forecasts and returns have different assets");
  OverlayNet net(nf + 1, spec);
  const std::size_t P = net.parameters();
  const double cost = costBps * 1e-4, stamp = stampBps * 1e-4;
  const MarketState ms = marketState(nextReturns);

  OverlayResult out;
  out.start = start;
  out.end = end;
  out.weights = Panel(nextReturns.dates(), N, 0.0);
  std::vector<double> w1(N, 0.0), w2(N, 0.0), J1(N * P, 0.0), J2(N * P, 0.0), g1;  // dates t - 1 and t - 2
  double A = 0, B = 0;                    // moving moments of the net return
  double sn = 0, s1[2] = {0, 0}, s2[2] = {0, 0};  // online standardisation of the market state
  std::vector<double> x(N * nf), grad(P), R(N);

  auto settle = [&](std::size_t t, bool learn) {  // the return of the positions taken on date t
    double gross = 0, turnover = 0, buys = 0;
    for (std::size_t i = 0; i < N; ++i) {
      R[i] = std::isfinite(nextReturns(t, i)) ? nextReturns(t, i) : 0.0;
      const double d = w1[i] - w2[i];
      gross += w1[i] * R[i], turnover += std::fabs(d), buys += std::max(0.0, d);
    }
    const double r = gross - cost * turnover - stamp * buys;
    out.gross.push_back(gross), out.turnover.push_back(turnover), out.buys.push_back(buys), out.net.push_back(r);
    if (!learn) return;
    // Differential Sharpe ratio: dD/dr with the moments before this return.
    const double var = std::max(B - A * A, 1e-8), dDdr = (B - A * r) / (var * std::sqrt(var));
    std::fill(grad.begin(), grad.end(), 0.0);
    for (std::size_t i = 0; i < N; ++i) {
      const double d = w1[i] - w2[i], kappa = cost * (d > 0 ? 1.0 : d < 0 ? -1.0 : 0.0) + stamp * (d > 0 ? 1.0 : 0.0);
      const double c1 = dDdr * (R[i] - kappa), c2 = dDdr * kappa;
      const double* Ja = &J1[i * P];
      const double* Jb = &J2[i * P];
      for (std::size_t p = 0; p < P; ++p) grad[p] += c1 * Ja[p] + c2 * Jb[p];
    }
    net.learn(grad, g1);
    A += spec.sharpeRate * (r - A);
    B += spec.sharpeRate * (r * r - B);
  };

  for (std::size_t t = start; t < end; ++t) {
    if (t > start) settle(t - 1, true);
    // Today's market state, standardised by what has been seen so far.
    double state[2] = {0, 0};
    const double raw[2] = {ms.volatility[t], ms.trend[t]};
    if (std::isfinite(raw[0]) && std::isfinite(raw[1])) {
      sn += 1;
      for (int j = 0; j < 2; ++j) {
        const double d = raw[j] - s1[j];
        s1[j] += d / sn;
        s2[j] += d * (raw[j] - s1[j]);
        const double sd = sn > 1 ? std::sqrt(s2[j] / (sn - 1)) : 0.0;
        state[j] = sd > 0 ? (raw[j] - s1[j]) / sd : 0.0;
      }
    }
    std::vector<double> g = net.gate(state, true);
    // Forecast inputs: each forecast's cross-sectional rank, centred to [-1/2, 1/2].
    for (std::size_t f = 0; f < nf; ++f) {
      std::vector<double> v;
      std::vector<std::size_t> idx;
      for (std::size_t i = 0; i < N; ++i) {
        x[i * nf + f] = 0.0;
        const double p = forecasts[f].probability(t, i);
        if (std::isfinite(p)) v.push_back(p), idx.push_back(i);
      }
      if (v.size() < 2) continue;
      const auto rk = averageRanks(v);
      const double top = static_cast<double>(v.size() - 1);
      for (std::size_t j = 0; j < v.size(); ++j) x[idx[j] * nf + f] = (rk[j] - 1.0) / top - 0.5;
    }
    auto step = net.forward(x, N, g, w1, J1);
    w2.swap(w1);
    J2.swap(J1);
    w1 = std::move(step.w);
    J1 = std::move(step.J);
    g1 = std::move(g);
    std::copy(w1.begin(), w1.end(), out.weights.row(t));
    out.tradeRate.push_back(step.meanRate);
    out.exposure.push_back(step.exposure);
    out.units.push_back(net.units());
  }
  settle(end - 1, false);
  return out;
}

}  // namespace sat
