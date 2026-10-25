// The daily trades' kernels (daykernels.hpp): their one source is the WGSL in kernels/day.
#include "daykernels.hpp"

#include <cmath>
#include <stdexcept>

#include "day_kernels.hpp"
#include "invoke.hpp"
#include "ofm/daybook.hpp"

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

void bollinger(const std::vector<double>& bars, std::size_t T, std::size_t N, const std::vector<std::uint32_t>& horizons, std::vector<double>& out) {
  H P;
  P.T = std::uint32_t(T), P.N = std::uint32_t(N), P.nH = std::uint32_t(horizons.size());
  for (std::size_t k = 0; k < horizons.size() && k < 16; ++k) P.hz[k / 4][k % 4] = horizons[k];
  out.assign(T * N * 3 * horizons.size(), 0.0);
  if (horizons.empty()) return;
  gen::Bollinger<double> k;
  k.P = P, k.bars = const_cast<double*>(bars.data()), k.ind = out.data();
  detail::dispatch(k, detail::groups(std::uint32_t(N), 64));
}

void pairTrades(const std::vector<double>& ticks, const std::vector<double>& cand, const std::vector<std::uint32_t>& entries,
                const std::vector<double>& mult, double buy, double sell, std::vector<double>& out) {
  H P;
  P.nC = std::uint32_t(cand.size() / 3), P.G = std::uint32_t(mult.size()), P.NK = std::uint32_t(entries.size()), P.buy = buy, P.sell = sell;
  for (std::size_t k = 0; k < entries.size() && k < 4; ++k) P.ent[0][k] = entries[k];
  const std::size_t n = std::size_t(P.nC) * P.NK * P.G * P.G;
  out.assign(3 * n, 0.0);
  if (n == 0) return;
  gen::Pairs<double> k;
  k.P = P, k.ticks = const_cast<double*>(ticks.data()), k.cand = const_cast<double*>(cand.data()), k.mult = const_cast<double*>(mult.data());
  k.out = out.data();
  detail::dispatch(k, detail::groups(std::uint32_t(n), 64));
}

}  // namespace ofm::day::kernels

namespace ofm::day {

void AdwinBank::push(const std::vector<std::size_t>& which, const std::vector<double>& x) {
  if (which.empty()) return;
  const std::size_t stride = capacity + 1;
  std::vector<double> list(which.size());
  for (std::size_t k = 0; k < which.size(); ++k) {
    long& s = slot[which[k]];
    if (s < 0) {
      s = long(st.size() / 2);
      p1.resize(p1.size() + stride, 0.0), p2.resize(p2.size() + stride, 0.0), st.resize(st.size() + 2, 0.0);
    }
    if (std::size_t(st[2 * std::size_t(s) + 1]) >= capacity) throw std::runtime_error("AdwinBank: a record is full");
    list[k] = double(s);
  }
  kernels::gen::DayHeader<double> P;
  P.nS = std::uint32_t(which.size()), P.nP = std::uint32_t(stride), P.hzn = horizon;
  kernels::gen::Adwin<double> k;
  k.P = P, k.list = list.data(), k.x = const_cast<double*>(x.data()), k.p1 = p1.data(), k.p2 = p2.data(), k.st = st.data();
  detail::dispatch(k, detail::groups(P.nS, 8));
}

}  // namespace ofm::day

namespace ofm::day::kernels {

void bestPairs(const std::vector<double>& lev, const std::vector<double>& e, const std::vector<double>& ok, const std::vector<double>& lim,
               const std::vector<double>& flo, std::size_t nC, std::size_t S, std::size_t G, std::size_t L, std::size_t V, double maxStop,
               double maxRiskReward, double none, std::vector<double>& out) {
  H P;
  P.nC = std::uint32_t(nC), P.G = std::uint32_t(G), P.NK = std::uint32_t(S), P.u1 = std::uint32_t(L), P.nS = std::uint32_t(V);
  P.buy = maxStop, P.sell = maxRiskReward, P.hzn = none;
  const std::size_t n = V * S * L * nC;
  out.assign(2 * n, 0.0);
  if (n == 0) return;
  gen::Best<double> k;
  k.P = P, k.lev = const_cast<double*>(lev.data()), k.e = const_cast<double*>(e.data()), k.ok = const_cast<double*>(ok.data());
  k.lim = const_cast<double*>(lim.data()), k.flo = const_cast<double*>(flo.data()), k.out = out.data();
  detail::dispatch(k, detail::groups(std::uint32_t(n), 64));
}

Windows::Windows(std::size_t windows, std::size_t capacity, const std::vector<double>& tau0, const std::vector<double>& tau1)
    : nW(windows), W(capacity), G(tau0.size()), has(windows, 0.0), x(windows, 0.0), ring(windows * capacity, 0.0), srt(windows * capacity, 0.0),
      st(2 * windows, 0.0), out(windows * tau0.size(), 0.0) {
  tau = tau0, tau.insert(tau.end(), tau1.begin(), tau1.end());
}

void Windows::run() {
  H P;
  P.nS = std::uint32_t(nW), P.nP = std::uint32_t(W), P.nH = std::uint32_t(G);
  gen::Quantiles<double> k;
  k.P = P, k.has = has.data(), k.x = x.data(), k.tau = tau.data(), k.ring = ring.data(), k.srt = srt.data(), k.st = st.data(), k.out = out.data();
  detail::dispatch(k, detail::groups(P.nS, 64));
  std::fill(has.begin(), has.end(), 0.0);
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

}  // namespace ofm::day::kernels
