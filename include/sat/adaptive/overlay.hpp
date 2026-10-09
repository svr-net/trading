#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "sat/core/panel.hpp"
#include "sat/ml/walk_forward.hpp"

namespace sat {

/// The overlay trader: a differential forward network with localised adaptive layers that
/// learns the trades themselves from the forecasts, online and net of costs, so how much and
/// how fast to trade is learned rather than set by a trading rule.
///
/// Each date, for each asset, a shared tanh layer reads the forecasts (cross-sectional ranks)
/// and the asset's current position. Local units, each covering one region of the market
/// state (21-day market volatility and 63-day trend, standardised online), read that layer:
/// the units near today's state give each asset a score and a trade rate, and the market
/// exposure. Units are added when the market enters a state no unit covers yet.
///
/// Differential: the network moves each position part of the way to its target,
/// w = w_prev + rate * (target - w_prev), target = exposure / N + tanh(score - mean score) / N,
/// and learns by gradient ascent on the differential Sharpe ratio (Moody and Saffell, 2001) of
/// the daily return net of costs. The gradient of each position with respect to every
/// parameter is carried forward through the positions (forward-mode, real-time recurrent
/// learning), so the cost of a trade is weighed against all the returns the position earns.
/// Localised: a unit's parameters learn in proportion to its responsibility for the current
/// state, each with its own adaptive (Adam) step, so what a unit learned in calm markets is
/// not overwritten in turbulent ones.
///
/// Causal: positions on date t use the forecasts of t and the market up to t's close; the
/// return of the following day (nextReturns(t)) is learned from on date t + 1.
struct OverlaySpec {
  std::size_t hidden = 8;       ///< shared tanh units
  std::size_t maxUnits = 16;    ///< local units
  double unitRadius = 1.0;      ///< new unit when the state is farther than this from every centre (state sd)
  double learningRate = 1e-3;
  double sharpeRate = 0.01;     ///< adaptation rate of the differential Sharpe ratio's moments
  std::uint64_t seed = 7;
};

/// The network itself (exposed for testing): parameters and one date's forward pass with the
/// Jacobian of the positions.
class OverlayNet {
 public:
  OverlayNet(std::size_t features, const OverlaySpec& spec);

  std::size_t features() const { return F_; }
  std::size_t parameters() const { return theta_.size(); }
  std::size_t units() const { return centres_.size(); }
  std::vector<double>& theta() { return theta_; }

  /// Gate of each allocated unit for a (standardised) market state; with `allocate`, first adds
  /// a unit there when none is within the radius.
  std::vector<double> gate(const double state[2], bool allocate);

  struct Step {
    std::vector<double> w, J;  ///< positions (N) and their Jacobian (N x parameters)
    double meanRate = 0, exposure = 0;
  };
  /// Positions for one date. x: N x (features - 1) forecast inputs (the position input is
  /// added here); wPrev, JPrev: the previous positions and their Jacobian.
  Step forward(const std::vector<double>& x, std::size_t N, const std::vector<double>& g, const std::vector<double>& wPrev,
               const std::vector<double>& JPrev) const;

  /// One ascent step on gradient `grad`; `g` is the gate the positions were taken with.
  void learn(const std::vector<double>& grad, const std::vector<double>& g);

 private:
  std::size_t F_, H_, K_;
  OverlaySpec spec_;
  std::vector<double> theta_, m_, v_;
  std::vector<std::size_t> steps_;  // per unit, and the shared layer last
  std::vector<std::array<double, 2>> centres_;
  std::size_t shared() const { return H_ * F_ + H_; }
  std::size_t unitBase(std::size_t k) const { return shared() + k * (2 * H_ + 2); }
};

struct OverlayResult {
  std::size_t start = 0, end = 0;
  Panel weights;  ///< dates x assets; rows start .. end - 1 are set
  std::vector<double> gross, turnover, buys, net;  ///< per date from start
  std::vector<double> tradeRate, exposure;         ///< mean trade rate and net exposure per date
  std::vector<std::size_t> units;                  ///< local units per date
};

/// Runs the overlay trader over the forecasts' common out-of-sample dates. Costs: `costBps` per
/// unit of turnover and `stampBps` more on purchases.
OverlayResult overlayTrader(const std::vector<ModelPredictions>& forecasts, const Panel& nextReturns, double costBps, double stampBps,
                            const OverlaySpec& spec = {});

}  // namespace sat
