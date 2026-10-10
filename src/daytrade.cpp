// The same-day trades' kernels (see daytrade.hpp). Their one source is the WGSL in kernels/daytrade;
// tools/wgsl2cpp.py translates it into C++ at build time, which runs here instantiated in double
// precision (the reference) and in single precision (the emulated GPU, which returns what the GPU
// returns up to the last bit of exp, log, sqrt and division). WebGPU runs the WGSL itself.
#include "ofm/daytrade.hpp"

#include "daytrade_kernels.hpp"
#include "invoke.hpp"

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
  k.bars = b.bars.data(), k.life = b.life.data(), k.st = b.state.data(), k.obs = b.obs.data(), k.idx = b.idx.data(), k.lev = b.levels.data(),
  k.tra = b.trades.data();
}
template <class R>
void bind(gen::Book<R>& k, Buffers<R>& b) {
  k.bars = b.bars.data(), k.lev = b.levels.data(), k.tra = b.trades.data(), k.ex = b.expected.data(), k.elig = b.elig.data();
  k.days = b.days.data(), k.booked = b.booked.data();
}

}  // namespace

template <class R>
void run(Buffers<R>& b, std::uint32_t chunk) {
  gen::Levels<R> lv;
  bind(lv, b);
  for (std::uint32_t t0 = 0; t0 < b.T; t0 += chunk) {
    lv.P = header(b, t0, std::min(b.T, t0 + chunk));
    detail::dispatch(lv, b.N);
  }
  if (b.s >= b.T) return;
  gen::Book<R> bk;
  bk.P = header(b, 0, 0), bind(bk, b);
  detail::dispatch(bk, detail::groups(b.T - b.s, bk.kWorkgroup[0]));
}

const std::string& levelsSource() {
  static const std::string s = gen::Levels<float>::wgsl();
  return s;
}
const std::string& bookSource() {
  static const std::string s = gen::Book<float>::wgsl();
  return s;
}

template void run<double>(Buffers<double>&, std::uint32_t);
template void run<float>(Buffers<float>&, std::uint32_t);

}  // namespace ofm::daytrade
