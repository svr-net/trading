// The neural network (net.hpp): its one source is the WGSL in kernels/net.
#include "net.hpp"

#include <cmath>
#include <cstdint>

#include "invoke.hpp"
#include "net_kernels.hpp"

namespace ofm::net {

namespace {
using H = gen::NetHeader<double>;

H header(const Net& n, std::size_t nS) {
  H P;
  P.nS = std::uint32_t(nS), P.In = std::uint32_t(n.In), P.Hd = std::uint32_t(n.Hd), P.Out = std::uint32_t(n.Out), P.nP = std::uint32_t(n.params());
  P.lr = n.lr, P.b1 = n.b1, P.b2 = n.b2, P.steps = n.steps;
  return P;
}
}  // namespace

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
  detail::dispatch(k, detail::groups(std::uint32_t(nS), 8));
  return y;
}

void Net::step(const std::vector<double>& x, const std::vector<double>& y, std::size_t nS) {
  if (nS == 0) return;
  // Every entry is written by the kernel: the buffer is kept between steps, not cleared.
  if (grad.size() < nS * params()) grad.resize(nS * params());
  double* g = grad.data();
  steps += 1;
  gen::Gradient<double> kg;
  kg.P = header(*this, nS), kg.w = w.data(), kg.x = const_cast<double*>(x.data()), kg.y = const_cast<double*>(y.data());
  kg.tau = tau.data(), kg.g = g;
  detail::dispatch(kg, detail::groups(std::uint32_t(nS), 8));
  gen::Adam<double> ka;
  ka.P = header(*this, nS), ka.g = g, ka.w = w.data(), ka.m = m.data(), ka.v = v.data();
  detail::dispatch(ka, detail::groups(std::uint32_t(params()), 64));
}

}  // namespace ofm::net
