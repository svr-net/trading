#pragma once

// The neural network's fused kernels (kernels/net/*.wgsl, translated to C++ by tools/wgsl2cpp.py
// and run on the CPU in double precision, workgroups spread over the hardware threads):
//  forward   the network's outputs for a batch of samples.
//  gradient  each sample's gradient of the quantile (pinball) loss.
//  adam      one Adam step from a mini-batch's gradients.

#include <cstddef>
#include <vector>

namespace ofm::net {

/// A network of two hidden layers of rectified linear units (at most 64) and linear outputs,
/// trained by Adam on the quantile loss (output o is the quantile tau[o] of its target).
struct Net {
  std::size_t In, Hd, Out;
  std::vector<double> w, m, v, tau;
  std::vector<double> grad;  ///< the mini-batch's per-sample gradients (scratch)
  double lr = 1e-3, b1 = 0.9, b2 = 0.999, steps = 0;
  Net(std::size_t in, std::size_t hidden, std::vector<double> quantiles);
  std::size_t params() const { return w.size(); }
  /// Outputs [nS][Out] for inputs x [nS][In].
  std::vector<double> predict(const std::vector<double>& x, std::size_t nS) const;
  /// One Adam step on a mini-batch: inputs x [nS][In], targets y [nS][Out].
  void step(const std::vector<double>& x, const std::vector<double>& y, std::size_t nS);
};

}  // namespace ofm::net
