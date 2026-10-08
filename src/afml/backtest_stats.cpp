#include "sat/afml/backtest_stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "sat/core/stats.hpp"

namespace sat::afml {

double periodSharpe(const std::vector<double>& r) {
  const double m = mean(r), s = stdev(r);
  return std::isfinite(s) && s > 0 ? m / s : 0.0;
}

double probabilisticSharpe(double sharpe, double benchmark, double n, double skew, double kurt) {
  if (n < 2) return 0.5;
  const double denom = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe * sharpe;
  if (!(denom > 0)) return sharpe > benchmark ? 1.0 : 0.0;
  return normalCdf((sharpe - benchmark) * std::sqrt(n - 1.0) / std::sqrt(denom));
}

double expectedMaxSharpe(double trials, double variance) {
  if (trials <= 1 || !(variance > 0)) return 0.0;
  const double gamma = 0.5772156649015329;  // Euler-Mascheroni
  return std::sqrt(variance) * ((1.0 - gamma) * normalQuantile(1.0 - 1.0 / trials) +
                                gamma * normalQuantile(1.0 - 1.0 / (trials * std::exp(1.0))));
}

double deflatedSharpe(double sharpe, double n, double skew, double kurt, double trials, double trialVariance) {
  return probabilisticSharpe(sharpe, expectedMaxSharpe(trials, trialVariance), n, skew, kurt);
}

PboResult probabilityOfBacktestOverfitting(const Matrix& R, std::size_t S) {
  const std::size_t T = R.rows(), N = R.cols();
  if (S < 2 || S % 2 || S > 20) throw std::invalid_argument("CSCV needs an even number of blocks between 2 and 20");
  if (N < 2 || T < S * 2) throw std::invalid_argument("CSCV needs at least 2 strategies and 2 rows per block");
  // Per block and strategy: sums of returns and squares, so a combination costs O(N S).
  std::vector<std::vector<double>> s1(S, std::vector<double>(N, 0.0)), s2(S, std::vector<double>(N, 0.0));
  std::vector<double> cnt(S, 0.0);
  for (std::size_t b = 0; b < S; ++b) {
    const std::size_t from = b * T / S, to = (b + 1) * T / S;
    cnt[b] = static_cast<double>(to - from);
    for (std::size_t t = from; t < to; ++t)
      for (std::size_t n = 0; n < N; ++n) {
        s1[b][n] += R(t, n);
        s2[b][n] += R(t, n) * R(t, n);
      }
  }
  auto sharpeOf = [&](const std::vector<char>& in, bool wantIn, std::size_t n) {
    double a = 0, q = 0, c = 0;
    for (std::size_t b = 0; b < S; ++b)
      if (static_cast<bool>(in[b]) == wantIn) {
        a += s1[b][n];
        q += s2[b][n];
        c += cnt[b];
      }
    const double m = a / c, v = (q - a * a / c) / (c - 1.0);
    return v > 0 ? m / std::sqrt(v) : 0.0;
  };
  PboResult out;
  std::vector<std::size_t> pick(S / 2);
  std::iota(pick.begin(), pick.end(), std::size_t{0});
  std::vector<double> isS(N), oosS(N);
  double below = 0, losses = 0;
  while (true) {
    std::vector<char> in(S, 0);
    for (std::size_t b : pick) in[b] = 1;
    for (std::size_t n = 0; n < N; ++n) {
      isS[n] = sharpeOf(in, true, n);
      oosS[n] = sharpeOf(in, false, n);
    }
    const std::size_t best = static_cast<std::size_t>(std::max_element(isS.begin(), isS.end()) - isS.begin());
    // Relative rank of the winner out of sample, in (0, 1).
    double rank = 1.0;
    for (std::size_t n = 0; n < N; ++n) rank += oosS[n] < oosS[best] ? 1.0 : (oosS[n] == oosS[best] && n != best ? 0.5 : 0.0);
    const double w = rank / static_cast<double>(N + 1);
    const double logit = std::log(w / (1.0 - w));
    out.logits.push_back(logit);
    out.inSampleSharpe.push_back(isS[best]);
    out.outOfSampleSharpe.push_back(oosS[best]);
    below += logit <= 0 ? 1 : 0;
    losses += oosS[best] < 0 ? 1 : 0;
    std::size_t i = S / 2;
    while (i > 0 && pick[i - 1] == S - S / 2 + i - 1) --i;
    if (i == 0) break;
    ++pick[i - 1];
    for (std::size_t j = i; j < S / 2; ++j) pick[j] = pick[j - 1] + 1;
  }
  out.combinations = out.logits.size();
  out.pbo = below / static_cast<double>(out.combinations);
  out.probabilityOfLoss = losses / static_cast<double>(out.combinations);
  const double mx = mean(out.inSampleSharpe), my = mean(out.outOfSampleSharpe);
  double sxy = 0, sxx = 0;
  for (std::size_t k = 0; k < out.combinations; ++k) {
    sxy += (out.inSampleSharpe[k] - mx) * (out.outOfSampleSharpe[k] - my);
    sxx += (out.inSampleSharpe[k] - mx) * (out.inSampleSharpe[k] - mx);
  }
  out.degradationSlope = sxx > 0 ? sxy / sxx : 0.0;
  return out;
}

DrawdownStats drawdownStats(const std::vector<double>& r) {
  DrawdownStats s;
  double w = 1.0, peak = 1.0, depth = 0.0;
  std::size_t since = 0;
  std::vector<double> depths;
  for (double x : r) {
    w *= 1.0 + x;
    if (w >= peak) {
      if (depth > 0) depths.push_back(depth);
      peak = w;
      depth = 0;
      since = 0;
    } else {
      ++since;
      depth = std::max(depth, 1.0 - w / peak);
      s.longestUnderWater = std::max(s.longestUnderWater, since);
    }
  }
  if (depth > 0) depths.push_back(depth);
  s.episodes = depths.size();
  s.maxDrawdown = depths.empty() ? 0.0 : *std::max_element(depths.begin(), depths.end());
  s.drawdown95 = depths.empty() ? 0.0 : quantile(depths, 0.95);
  return s;
}

double returnConcentration(const std::vector<double>& r, bool positive) {
  std::vector<double> part;
  for (double x : r)
    if (positive ? x > 0 : x < 0) part.push_back(std::fabs(x));
  if (part.size() < 2) return 0.0;
  const double total = std::accumulate(part.begin(), part.end(), 0.0);
  double h = 0;
  for (double x : part) h += (x / total) * (x / total);
  const double n = static_cast<double>(part.size());
  return (h - 1.0 / n) / (1.0 - 1.0 / n);
}

}  // namespace sat::afml
