#include "sat/algo/trend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sat/core/stats.hpp"

namespace sat::algo {

namespace {

std::vector<double> ema(const std::vector<double>& x, double span) {
  const double a = 2.0 / (span + 1.0);
  std::vector<double> out(x.size());
  double m = x.empty() ? 0.0 : x[0];
  for (std::size_t t = 0; t < x.size(); ++t) out[t] = m = a * x[t] + (1 - a) * m;
  return out;
}

}  // namespace

TrendResult trendFollowing(const Panel& close, const TrendSpec& s) {
  const std::size_t T = close.dates(), N = close.assets(), R = s.rules.size();
  if (R == 0) throw std::invalid_argument("trend: at least one rule");
  for (const auto& [f, sl] : s.rules)
    if (f < 1 || sl <= f) throw std::invalid_argument("trend: each rule needs 1 <= fast < slow");
  if (T <= s.warmup + 10) throw std::invalid_argument("trend: series shorter than the warm-up");
  // Per instrument: price volatility (in price units, EWM of daily changes), raw forecasts,
  // and daily returns.
  std::vector<std::vector<std::vector<double>>> raw(R, std::vector<std::vector<double>>(N));  // [rule][instrument][t]
  std::vector<std::vector<double>> pvol(N), ret(N), rvol(N);
  const double av = 2.0 / (s.volSpan + 1.0);
  for (std::size_t i = 0; i < N; ++i) {
    const auto p = close.series(i);
    pvol[i].assign(T, 0.0);
    ret[i].assign(T, 0.0);
    rvol[i].assign(T, 0.0);
    double v = 0, w = 0, vr = 0;
    for (std::size_t t = 1; t < T; ++t) {
      const double d = p[t] - p[t - 1], r = p[t] / p[t - 1] - 1.0;
      ret[i][t] = r;
      v = (1 - av) * v + av * d * d;
      vr = (1 - av) * vr + av * r * r;
      w = (1 - av) * w + av;
      pvol[i][t] = std::sqrt(v / w);
      rvol[i][t] = std::sqrt(vr / w);
    }
    for (std::size_t r = 0; r < R; ++r) {
      const auto f = ema(p, static_cast<double>(s.rules[r].first)), sl = ema(p, static_cast<double>(s.rules[r].second));
      raw[r][i].assign(T, 0.0);
      for (std::size_t t = 1; t < T; ++t) raw[r][i][t] = pvol[i][t] > 0 ? (f[t] - sl[t]) / pvol[i][t] : 0.0;
    }
  }
  TrendResult out;
  out.start = s.warmup;
  out.ruleReturns.assign(R, {});
  out.weights.assign(R, {});
  out.forecastScalars.assign(R, 1.0);
  std::vector<double> weight(R, 1.0 / static_cast<double>(R));
  // Forecast scalars from the absolute raw forecasts seen so far (expanding, no look-ahead).
  std::vector<double> absSum(R, 0.0), absCount(R, 0.0);
  std::vector<double> pos(N, 0.0);
  std::vector<std::vector<double>> rulePos(R, std::vector<double>(N, 0.0));
  const double dailyTarget = s.targetVol / std::sqrt(252.0);
  double idm = 1.0;
  for (std::size_t t = 1; t + 1 < T; ++t) {
    std::vector<std::vector<double>> scaled(R, std::vector<double>(N, 0.0));
    for (std::size_t r = 0; r < R; ++r) {
      for (std::size_t i = 0; i < N; ++i) {
        absSum[r] += std::fabs(raw[r][i][t]);
        absCount[r] += 1;
      }
      const double scalar = absSum[r] > 0 ? 10.0 * absCount[r] / absSum[r] : 1.0;
      out.forecastScalars[r] = scalar;
      for (std::size_t i = 0; i < N; ++i) {
        double f = std::clamp(raw[r][i][t] * scalar, -s.forecastCap, s.forecastCap);
        if (s.longOnly) f = std::max(0.0, f);
        scaled[r][i] = f;
      }
    }
    if (t < s.warmup) continue;
    // Adaptive rule weights from each rule's own recent record.
    if (s.adaptiveWeights && (t - s.warmup) % s.reweightEvery == 0 && out.ruleReturns[0].size() >= 20) {
      std::vector<double> sharpe(R, 0.0);
      double total = 0;
      for (std::size_t r = 0; r < R; ++r) {
        const auto& h = out.ruleReturns[r];
        const std::size_t from = h.size() > s.reweightWindow ? h.size() - s.reweightWindow : 0;
        const std::vector<double> win(h.begin() + static_cast<std::ptrdiff_t>(from), h.end());
        const double sd = stdev(win);
        sharpe[r] = sd > 0 ? std::max(0.0, mean(win) / sd) : 0.0;
        total += sharpe[r];
      }
      for (std::size_t r = 0; r < R; ++r) weight[r] = total > 0 ? sharpe[r] / total : 1.0 / static_cast<double>(R);
    }
    // Instrument diversification multiplier: with equal weights 1/N and average pairwise
    // return correlation rho, w' H w = 1/N + (1 - 1/N) rho; re-estimated with the rule weights.
    if ((t - s.warmup) % s.reweightEvery == 0 && N > 1) {
      const std::size_t from = t > s.reweightWindow ? t - s.reweightWindow + 1 : 1;
      double rho = 0;
      std::size_t pairs = 0;
      for (std::size_t a = 0; a < N; ++a)
        for (std::size_t b = a + 1; b < N; ++b) {
          const std::vector<double> ra(ret[a].begin() + static_cast<std::ptrdiff_t>(from), ret[a].begin() + static_cast<std::ptrdiff_t>(t + 1)),
              rb(ret[b].begin() + static_cast<std::ptrdiff_t>(from), ret[b].begin() + static_cast<std::ptrdiff_t>(t + 1));
          const double c = correlation(ra, rb);
          if (std::isfinite(c)) {
            rho += c;
            ++pairs;
          }
        }
      rho = pairs ? std::max(0.0, rho / static_cast<double>(pairs)) : 0.0;
      const double n = static_cast<double>(N);
      idm = std::min(s.maxIdm, 1.0 / std::sqrt(1.0 / n + (1.0 - 1.0 / n) * rho));
    }
    // Forecast diversification multiplier: 1 / sqrt(w' C w) with C the correlation of rule forecasts
    // across instruments today (bounded in [1, 2.5]).
    double wcw = 0;
    for (std::size_t a = 0; a < R; ++a)
      for (std::size_t b = 0; b < R; ++b) {
        const double c = a == b ? 1.0 : correlation(scaled[a], scaled[b]);
        wcw += weight[a] * weight[b] * (std::isfinite(c) ? std::max(0.0, c) : 0.0);
      }
    const double fdm = std::clamp(wcw > 0 ? 1.0 / std::sqrt(wcw) : 1.0, 1.0, 2.5);
    double turnover = 0, gross = 0, pnl = 0, absForecast = 0;
    for (std::size_t i = 0; i < N; ++i) {
      double f = 0;
      for (std::size_t r = 0; r < R; ++r) f += weight[r] * scaled[r][i];
      f = std::clamp(f * fdm, -s.forecastCap, s.forecastCap);
      absForecast += std::fabs(f) / static_cast<double>(N);
      // Equal volatility budget per instrument: weight = IDM (target / N) / instrument vol.
      const double unit = rvol[i][t] > 0 ? idm * dailyTarget / static_cast<double>(N) / rvol[i][t] : 0.0;
      const double target = unit * f / 10.0;
      if (std::fabs(target - pos[i]) > s.buffer * unit) {
        turnover += std::fabs(target - pos[i]);
        pos[i] = target;
      }
      gross += std::fabs(pos[i]);
      pnl += pos[i] * ret[i][t + 1];
      for (std::size_t r = 0; r < R; ++r) rulePos[r][i] = unit * scaled[r][i] / 10.0;
    }
    out.returns.push_back(pnl - s.costBps * 1e-4 * turnover);
    out.turnover.push_back(turnover);
    out.idm.push_back(idm);
    out.grossLeverage.push_back(gross);
    out.combinedForecastMean.push_back(absForecast);
    for (std::size_t r = 0; r < R; ++r) {
      double rp = 0;
      for (std::size_t i = 0; i < N; ++i) rp += rulePos[r][i] * ret[i][t + 1];
      out.ruleReturns[r].push_back(rp);
      out.weights[r].push_back(weight[r]);
    }
  }
  return out;
}

}  // namespace sat::algo
