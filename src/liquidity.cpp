#include "ofm/liquidity.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace ofm {

Liquidity::Liquidity(const Market& m, double halfLife) : T(m.T()), N(m.N()), halfSpread(T, N, 0.0), tradedValue(T, N) {
  const double lam = std::exp(-std::log(2.0) / halfLife);
  for (std::size_t i = 0; i < N; ++i) {
    double s2 = 0, tv = kNaN, prevC = kNaN, prevEta = kNaN;
    std::size_t n = 0;
    for (std::size_t t = 0; t < T; ++t) {
      const double o = m.open(t, i), h = m.high(t, i), l = m.low(t, i), c = m.close(t, i), v = m.volume(t, i);
      if (o > 0 && h > l && l > 0 && c > 0 && v > 0) {
        const double eta = 0.5 * (std::log(h) + std::log(l));
        if (std::isfinite(prevC)) {
          const double x = 4 * (prevC - prevEta) * (prevC - eta);
          s2 = n == 0 ? x : lam * s2 + (1 - lam) * x;
          ++n;
        }
        prevC = std::log(c), prevEta = eta;
        const double x = std::log(c * v);
        tv = std::isfinite(tv) ? lam * tv + (1 - lam) * x : x;
      } else {
        prevC = kNaN;
      }
      halfSpread(t, i) = n >= 2 ? 0.5 * std::sqrt(std::max(0.0, s2)) : 0.0;
      tradedValue(t, i) = tv;
    }
  }
}

std::vector<char> Liquidity::universeAt(std::size_t t, std::size_t universe) const {
  std::vector<char> in(N, 1);
  if (universe == 0 || t >= T) return in;
  std::vector<std::pair<double, std::size_t>> v;
  for (std::size_t i = 0; i < N; ++i)
    if (std::isfinite(tradedValue(t, i))) v.emplace_back(tradedValue(t, i), i);
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  std::fill(in.begin(), in.end(), 0);
  for (std::size_t k = 0; k < std::min(universe, v.size()); ++k) in[v[k].second] = 1;
  return in;
}

}  // namespace ofm
