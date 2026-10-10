// The model's kernels (see kernels.hpp). Their one source is the WGSL in kernels/model;
// tools/wgsl2cpp.py translates it into C++ at build time, run here once per invocation in the
// GPU's order: float is the emulated GPU (what the GPU returns, up to the last bit of sqrt and
// division, which WGSL leaves to the device), double the reference. WebGPU runs the WGSL itself.
#include "ofm/kernels.hpp"

#include "invoke.hpp"
#include "model_kernels.hpp"

namespace ofm::kernels {

namespace {

template <class F>
gen::Header<F> header(const Plan& p, std::size_t c) {
  const auto h = p.header(c);
  gen::Header<F> P;
  P.T = h[0], P.N = h[1], P.K = h[2], P.H = h[3], P.c0 = h[4], P.cd = h[5], P.nPairs = h[6], P.stride = h[7];
  for (std::size_t k = 0; k < 16; ++k) P.hz[k / 4][k % 4] = h[16 + k];
  return P;
}

}  // namespace

const std::string& zscoreSource() {
  static const std::string s = gen::Zscore<float>::wgsl();
  return s;
}
const std::string& gramSource() {
  static const std::string s = gen::Gram<float>::wgsl();
  return s;
}
const std::string& expectSource() {
  static const std::string s = gen::Expect<float>::wgsl();
  return s;
}

std::uint32_t gramTiles(const Plan& p) {
  const std::uint32_t n = static_cast<std::uint32_t>(p.K + 1 + 7) / 8;
  return n * n;
}

template <class F>
void zscore(const Plan& p, std::size_t c, const F* tables, std::vector<F>& z) {
  const std::uint32_t N = static_cast<std::uint32_t>(p.N), K = static_cast<std::uint32_t>(p.K);
  const std::uint32_t cd = static_cast<std::uint32_t>(p.chunkLength(c));
  std::vector<F> sig(static_cast<std::size_t>(cd) * N * K, F(0));
  z.assign(sig.size(), F(0));
  gen::Zscore<F> k;
  k.P = header<F>(p, c), k.tab = const_cast<F*>(tables), k.sig = sig.data(), k.z = z.data();
  detail::dispatch(k, K, cd);
}

template <class F>
void gram(const Plan& p, std::size_t c, const F* tables, std::vector<F>& z, std::vector<F>& g) {
  const std::uint32_t cd = static_cast<std::uint32_t>(p.chunkLength(c));
  g.assign(cd * p.stride, F(0));
  gen::Gram<F> k;
  k.P = header<F>(p, c), k.tab = const_cast<F*>(tables), k.z = z.data(), k.g = g.data();
  detail::dispatch(k, gramTiles(p), cd);
}

template <class F>
void expect(const Plan& p, std::size_t c, std::vector<F>& z, std::vector<F>& v, std::vector<F>& e) {
  const std::uint32_t cd = static_cast<std::uint32_t>(p.chunkLength(c)), N = static_cast<std::uint32_t>(p.N);
  e.assign(static_cast<std::size_t>(cd) * N, F(0));
  gen::Expect<F> k;
  k.P = header<F>(p, c), k.z = z.data(), k.v = v.data(), k.e = e.data();
  detail::dispatch(k, detail::groups(N, k.kWorkgroup[0]), cd);
}

template void zscore<float>(const Plan&, std::size_t, const float*, std::vector<float>&);
template void zscore<double>(const Plan&, std::size_t, const double*, std::vector<double>&);
template void gram<float>(const Plan&, std::size_t, const float*, std::vector<float>&, std::vector<float>&);
template void gram<double>(const Plan&, std::size_t, const double*, std::vector<double>&, std::vector<double>&);
template void expect<float>(const Plan&, std::size_t, std::vector<float>&, std::vector<float>&, std::vector<float>&);
template void expect<double>(const Plan&, std::size_t, std::vector<double>&, std::vector<double>&, std::vector<double>&);

}  // namespace ofm::kernels
