#include "sat/core/stats.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "sat/core/matrix.hpp"

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

namespace {

std::vector<double> finiteValues(const std::vector<double>& x) {
  std::vector<double> v;
  for (double a : x)
    if (finite(a)) v.push_back(a);
  return v;
}

}  // namespace

double skewness(const std::vector<double>& x) {
  const auto v = finiteValues(x);
  if (v.size() < 3) return kNaN;
  const double m = mean(v);
  double m2 = 0, m3 = 0;
  for (double a : v) {
    m2 += (a - m) * (a - m);
    m3 += (a - m) * (a - m) * (a - m);
  }
  m2 /= static_cast<double>(v.size());
  m3 /= static_cast<double>(v.size());
  return m2 > 0 ? m3 / std::pow(m2, 1.5) : 0.0;
}

double kurtosis(const std::vector<double>& x) {
  const auto v = finiteValues(x);
  if (v.size() < 4) return kNaN;
  const double m = mean(v);
  double m2 = 0, m4 = 0;
  for (double a : v) {
    const double d2 = (a - m) * (a - m);
    m2 += d2;
    m4 += d2 * d2;
  }
  m2 /= static_cast<double>(v.size());
  m4 /= static_cast<double>(v.size());
  return m2 > 0 ? m4 / (m2 * m2) : 3.0;
}

double normalCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double normalQuantile(double p) {
  if (!(p > 0.0 && p < 1.0)) {
    if (p == 0.0) return -INFINITY;
    if (p == 1.0) return INFINITY;
    return kNaN;
  }
  // Acklam's coefficients.
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                             6.680131188771972e+01, -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
  const double lo = 0.02425, hi = 1.0 - lo;
  double x;
  if (p < lo) {
    const double q = std::sqrt(-2.0 * std::log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else if (p > hi) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else {
    const double q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  }
  // One Newton step on the exact distribution function.
  const double e = normalCdf(x) - p;
  const double u = e * std::sqrt(2.0 * 3.141592653589793) * std::exp(0.5 * x * x);
  return x - u / (1.0 + 0.5 * x * u);
}

std::vector<double> ols(const std::vector<std::vector<double>>& X, const std::vector<double>& y, std::vector<double>* se) {
  const std::size_t n = y.size(), k = X.empty() ? 0 : X[0].size();
  if (n <= k || k == 0) throw std::invalid_argument("ols: need more observations than regressors");
  Matrix xtx(k, k, 0.0);
  std::vector<double> xty(k, 0.0);
  for (std::size_t r = 0; r < n; ++r)
    for (std::size_t a = 0; a < k; ++a) {
      xty[a] += X[r][a] * y[r];
      for (std::size_t b = 0; b <= a; ++b) xtx(a, b) += X[r][a] * X[r][b];
    }
  for (std::size_t a = 0; a < k; ++a)
    for (std::size_t b = 0; b < a; ++b) xtx(b, a) = xtx(a, b);
  const auto beta = solveSpd(xtx, xty);
  if (se) {
    double rss = 0;
    for (std::size_t r = 0; r < n; ++r) {
      double fit = 0;
      for (std::size_t a = 0; a < k; ++a) fit += X[r][a] * beta[a];
      rss += (y[r] - fit) * (y[r] - fit);
    }
    const double s2 = rss / static_cast<double>(n - k);
    se->assign(k, 0.0);
    // Diagonal of (X'X)^-1 one column at a time.
    for (std::size_t a = 0; a < k; ++a) {
      std::vector<double> e(k, 0.0);
      e[a] = 1.0;
      const auto col = solveSpd(xtx, e);
      (*se)[a] = std::sqrt(std::max(0.0, s2 * col[a]));
    }
  }
  return beta;
}

}  // namespace sat
