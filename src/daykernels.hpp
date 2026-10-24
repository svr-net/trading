#pragma once

// The daily trades' fused kernels (kernels/day/*.wgsl, translated to C++ by tools/wgsl2cpp.py and
// run on the CPU in double precision, workgroups spread over the hardware threads):
//  indicators  technical indicators of every stock over a block of days.
//  wilder      Wilder's indicators (ATR, RSI, DMI, ADX, ADXR, Parabolic SAR) of every stock and day.
//  bollinger   Bollinger bands of every stock and day at several horizons.
//  forward     the network's outputs for a batch of samples.
//  gradient    each sample's gradient of the quantile (pinball) loss.
//  adam        one Adam step from a mini-batch's gradients.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ofm::day::kernels {

/// Indicators at the closes t0 .. t0 + cd - 1: out [cd][N][5 horizons.size()]. bars: 5 tables
/// [o, h, l, c, v][T][N], 0 where there is no bar.
void indicators(const std::vector<double>& bars, std::size_t T, std::size_t N, const std::vector<std::uint32_t>& horizons, std::size_t t0,
                std::size_t cd, std::vector<double>& out);

/// Wilder's indicators at every close: out [T][N][kWilder] (kernels/day/wilder.wgsl). bars: as above.
constexpr std::size_t kWilder = 7;
void wilder(const std::vector<double>& bars, std::size_t T, std::size_t N, std::vector<double>& out);

/// Bollinger bands at every close and horizon: out [T][N][3 horizons.size()] (kernels/day/bollinger.wgsl).
void bollinger(const std::vector<double>& bars, std::size_t T, std::size_t N, const std::vector<std::uint32_t>& horizons, std::vector<double>& out);

/// A network of two hidden layers of rectified linear units (at most 64) and linear outputs,
/// trained by Adam on the quantile loss (output o is the quantile tau[o] of its target).
struct Net {
  std::size_t In, Hd, Out;
  std::vector<double> w, m, v, tau;
  double lr = 1e-3, b1 = 0.9, b2 = 0.999, steps = 0;
  Net(std::size_t in, std::size_t hidden, std::vector<double> quantiles);
  std::size_t params() const { return w.size(); }
  /// Outputs [nS][Out] for inputs x [nS][In].
  std::vector<double> predict(const std::vector<double>& x, std::size_t nS) const;
  /// One Adam step on a mini-batch: inputs x [nS][In], targets y [nS][Out].
  void step(const std::vector<double>& x, const std::vector<double>& y, std::size_t nS);
};

}  // namespace ofm::day::kernels
