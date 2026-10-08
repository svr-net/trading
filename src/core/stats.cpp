#include "sat/core/stats.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace sat {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}

double mean(const std::vector<double>& x) {
  double s = 0.0;
  std::size_t n = 0;
  for (double v : x)
    if (finite(v)) {
      s += v;
      ++n;
    }
  return n ? s / static_cast<double>(n) : kNaN;
}

double stdev(const std::vector<double>& x) {
  const double m = mean(x);
  double s = 0.0;
  std::size_t n = 0;
  for (double v : x)
    if (finite(v)) {
      s += (v - m) * (v - m);
      ++n;
    }
  return n > 1 ? std::sqrt(s / static_cast<double>(n - 1)) : kNaN;
}

double correlation(const std::vector<double>& x, const std::vector<double>& y) {
  const std::size_t n = std::min(x.size(), y.size());
  double sx = 0, sy = 0;
  std::size_t m = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (finite(x[i]) && finite(y[i])) {
      sx += x[i];
      sy += y[i];
      ++m;
    }
  if (m < 2) return kNaN;
  const double mx = sx / static_cast<double>(m), my = sy / static_cast<double>(m);
  double sxy = 0, sxx = 0, syy = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (finite(x[i]) && finite(y[i])) {
      sxy += (x[i] - mx) * (y[i] - my);
      sxx += (x[i] - mx) * (x[i] - mx);
      syy += (y[i] - my) * (y[i] - my);
    }
  if (sxx <= 0.0 || syy <= 0.0) return kNaN;
  return sxy / std::sqrt(sxx * syy);
}

std::vector<double> averageRanks(const std::vector<double>& x) {
  std::vector<std::size_t> idx;
  for (std::size_t i = 0; i < x.size(); ++i)
    if (finite(x[i])) idx.push_back(i);
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
  std::vector<double> r(x.size(), kNaN);
  for (std::size_t k = 0; k < idx.size();) {
    std::size_t e = k;
    while (e + 1 < idx.size() && x[idx[e + 1]] == x[idx[k]]) ++e;
    const double avg = 0.5 * static_cast<double>(k + e) + 1.0;
    for (std::size_t j = k; j <= e; ++j) r[idx[j]] = avg;
    k = e + 1;
  }
  return r;
}

double rankCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
  const std::size_t n = std::min(x.size(), y.size());
  std::vector<double> a, b;
  for (std::size_t i = 0; i < n; ++i)
    if (finite(x[i]) && finite(y[i])) {
      a.push_back(x[i]);
      b.push_back(y[i]);
    }
  return correlation(averageRanks(a), averageRanks(b));
}

double quantile(std::vector<double> x, double q) {
  x.erase(std::remove_if(x.begin(), x.end(), [](double v) { return !finite(v); }), x.end());
  if (x.empty()) return kNaN;
  std::sort(x.begin(), x.end());
  const double pos = std::clamp(q, 0.0, 1.0) * static_cast<double>(x.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(pos);
  const std::size_t hi = std::min(lo + 1, x.size() - 1);
  return x[lo] + (pos - static_cast<double>(lo)) * (x[hi] - x[lo]);
}

std::vector<std::size_t> argsortDescending(const double* x, std::size_t n) {
  std::vector<std::size_t> idx(n);
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
    const bool fa = finite(x[a]), fb = finite(x[b]);
    if (fa != fb) return fa;
    return fa && x[a] > x[b];
  });
  return idx;
}

}  // namespace sat
