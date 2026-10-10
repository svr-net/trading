// The same-day trades' kernels (see daytrade.hpp). Their one source is the WGSL in kernels/daytrade;
// tools/wgsl2cpp.py translates it into C++ at build time, which runs here instantiated in double
// precision (the reference) and in single precision (the emulated GPU, which returns what the GPU
// returns up to the last bit of exp, log, sqrt and division). WebGPU runs the WGSL itself.
#include "ofm/daytrade.hpp"

#include "daytrade_kernels.hpp"

#include <algorithm>

namespace ofm::daytrade {

namespace {

// The uniform and the bindings, as the GPU host (web/gpu.js) sets them.
template <class R>
gen::Header<R> header(const Buffers<R>& b, std::uint32_t t0, std::uint32_t t1) {
  gen::Header<R> h;
  h.T = b.T, h.N = b.N, h.K = b.K, h.s = b.s, h.t0 = t0, h.t1 = t1, h.H = b.H, h.buy = b.buy, h.sell = b.sell;
  for (std::uint32_t k = 0; k < b.H && k < 16; ++k) h.hz[k / 4][k % 4] = b.hz[k];
  return h;
}

template <class R>
void bind(gen::Levels<R>& k, Buffers<R>& b) {
  k.bars = b.bars.data(), k.life = b.life.data(), k.st = b.state.data(), k.obs = b.obs.data(), k.idx = b.idx.data(), k.lev = b.levels.data();
}
template <class R>
void bind(gen::Trades<R>& k, Buffers<R>& b) {
  k.bars = b.bars.data(), k.life = b.life.data(), k.lev = b.levels.data(), k.tra = b.trades.data();
}
template <class R>
void bind(gen::Book<R>& k, Buffers<R>& b) {
  k.bars = b.bars.data(), k.lev = b.levels.data(), k.tra = b.trades.data(), k.ex = b.expected.data(), k.elig = b.elig.data();
  k.days = b.days.data(), k.booked = b.booked.data();
}

}  // namespace

template <class R>
void levels(Buffers<R>& b, std::uint32_t i, std::uint32_t t0, std::uint32_t t1) {
  gen::Levels<R> k;
  k.P = header(b, t0, t1), bind(k, b);
  k.main({i, 0, 0});
}

template <class R>
void trades(Buffers<R>& b, std::uint32_t t, std::uint32_t i) {
  gen::Trades<R> k;
  k.P = header(b, 0, 0), bind(k, b);
  k.main({i, t - b.s, 0});
}

template <class R>
void book(Buffers<R>& b, std::uint32_t t) {
  gen::Book<R> k;
  k.P = header(b, 0, 0), bind(k, b);
  k.main({t - b.s, 0, 0});
}

template <class R>
void run(Buffers<R>& b, std::uint32_t chunk) {
  for (std::uint32_t t0 = 0; t0 < b.T; t0 += chunk)
    for (std::uint32_t i = 0; i < b.N; ++i) levels(b, i, t0, std::min(b.T, t0 + chunk));
  for (std::uint32_t t = b.s; t + 1 < b.T; ++t)
    for (std::uint32_t i = 0; i < b.N; ++i) trades(b, t, i);
  for (std::uint32_t t = b.s; t < b.T; ++t) book(b, t);
}

const std::string& levelsSource() {
  static const std::string s = gen::Levels<float>::wgsl();
  return s;
}
const std::string& tradesSource() {
  static const std::string s = gen::Trades<float>::wgsl();
  return s;
}
const std::string& bookSource() {
  static const std::string s = gen::Book<float>::wgsl();
  return s;
}

template void levels<double>(Buffers<double>&, std::uint32_t, std::uint32_t, std::uint32_t);
template void levels<float>(Buffers<float>&, std::uint32_t, std::uint32_t, std::uint32_t);
template void trades<double>(Buffers<double>&, std::uint32_t, std::uint32_t);
template void trades<float>(Buffers<float>&, std::uint32_t, std::uint32_t);
template void book<double>(Buffers<double>&, std::uint32_t);
template void book<float>(Buffers<float>&, std::uint32_t);
template void run<double>(Buffers<double>&, std::uint32_t);
template void run<float>(Buffers<float>&, std::uint32_t);

}  // namespace ofm::daytrade
