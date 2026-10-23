// The daily trades' kernels (daykernels.hpp): their one source is the WGSL in kernels/day.
#include "daykernels.hpp"

#include <cmath>

#include "day_kernels.hpp"
#include "invoke.hpp"

namespace ofm::day::kernels {

namespace {
using H = gen::DayHeader<double>;

H header(const Net& n, std::size_t nS) {
  H P;
  P.nS = std::uint32_t(nS), P.In = std::uint32_t(n.In), P.Hd = std::uint32_t(n.Hd), P.Out = std::uint32_t(n.Out), P.nP = std::uint32_t(n.params());
  P.lr = n.lr, P.b1 = n.b1, P.b2 = n.b2, P.steps = n.steps;
  return P;
}
}  // namespace

void indicators(const std::vector<double>& bars, std::size_t T, std::size_t N, const std::vector<std::uint32_t>& horizons, std::size_t t0,
                std::size_t cd, std::vector<double>& out) {
  H P;
  P.T = std::uint32_t(T), P.N = std::uint32_t(N), P.nH = std::uint32_t(horizons.size()), P.FI = std::uint32_t(5 * horizons.size());
  P.t0 = std::uint32_t(t0), P.cd = std::uint32_t(cd);
  for (std::size_t k = 0; k < horizons.size() && k < 16; ++k) P.hz[k / 4][k % 4] = horizons[k];
  out.assign(cd * N * P.FI, 0.0);
  gen::Indicators<double> k;
  k.P = P, k.bars = const_cast<double*>(bars.data()), k.ind = out.data();
  detail::dispatch(k, detail::groups(std::uint32_t(N), 64), std::uint32_t(cd));
}

void wilder(const std::vector<double>& bars, std::size_t T, std::size_t N, std::vector<double>& out) {
  H P;
  P.T = std::uint32_t(T), P.N = std::uint32_t(N);
  out.assign(T * N * kWilder, 0.0);
  gen::Wilder<double> k;
  k.P = P, k.bars = const_cast<double*>(bars.data()), k.ind = out.data();
  detail::dispatch(k, detail::groups(std::uint32_t(N), 64));
}

Net::Net(std::size_t in, std::size_t hidden, std::vector<double> quantiles) : In(in), Hd(hidden), Out(quantiles.size()), tau(std::move(quantiles)) {
  const std::size_t sizes[3][2] = {{Hd, In}, {Hd, Hd}, {Out, Hd}};
  std::uint64_t r = 0x9E3779B97F4A7C15ull;
  auto unif = [&]() { r ^= r << 13, r ^= r >> 7, r ^= r << 17; return double(r >> 11) / double(1ull << 53); };
  for (const auto& s : sizes) {  // He initialisation, deterministic
    const double sd = std::sqrt(2.0 / double(s[1]));
    for (std::size_t k = 0; k < s[0] * s[1]; ++k) {
      const double u1 = std::max(1e-12, unif()), u2 = unif();
      w.push_back(sd * std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2));
    }
    w.insert(w.end(), s[0], 0.0);
  }
  m.assign(w.size(), 0.0), v.assign(w.size(), 0.0);
}

std::vector<double> Net::predict(const std::vector<double>& x, std::size_t nS) const {
  std::vector<double> y(nS * Out, 0.0);
  if (nS == 0) return y;
  gen::Forward<double> k;
  k.P = header(*this, nS), k.w = const_cast<double*>(w.data()), k.x = const_cast<double*>(x.data()), k.y = y.data();
  detail::dispatch(k, detail::groups(std::uint32_t(nS), 64));
  return y;
}

void Net::step(const std::vector<double>& x, const std::vector<double>& y, std::size_t nS) {
  if (nS == 0) return;
  std::vector<double> g(nS * params(), 0.0);
  steps += 1;
  gen::Gradient<double> kg;
  kg.P = header(*this, nS), kg.w = w.data(), kg.x = const_cast<double*>(x.data()), kg.y = const_cast<double*>(y.data());
  kg.tau = tau.data(), kg.g = g.data();
  detail::dispatch(kg, detail::groups(std::uint32_t(nS), 64));
  gen::Adam<double> ka;
  ka.P = header(*this, nS), ka.g = g.data(), ka.w = w.data(), ka.m = m.data(), ka.v = v.data();
  detail::dispatch(ka, detail::groups(std::uint32_t(params()), 64));
}

}  // namespace ofm::day::kernels
