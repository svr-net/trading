#pragma once

#include <cstddef>
#include <vector>

#include "sat/core/panel.hpp"

namespace sat::afml {

/// Exponentially weighted standard deviation of simple returns of `close` (span in days),
/// the usual scale for barriers and event filters. The first value is NaN.
std::vector<double> ewmVolatility(const std::vector<double>& close, double span = 50.0);

/// Symmetric CUSUM filter on a log-price series: an event is recorded whenever the
/// cumulative upward or downward drift since the last event exceeds `threshold` (in log
/// units), after which both sums restart. It samples the moments when something happened
/// rather than every bar.
std::vector<std::size_t> cusumFilter(const std::vector<double>& logPrice, double threshold);

/// The triple-barrier method: from the event at t0, the position is closed at the first of
/// an upper barrier (profit taking), a lower barrier (stop loss) or the vertical barrier
/// after `maxHolding` bars. Horizontal barriers sit at multiples of the event's target
/// (a volatility estimate); a multiple of 0 disables that barrier.
struct BarrierSpec {
  double profitTaking = 1.0;
  double stopLoss = 1.0;
  std::size_t maxHolding = 10;
  double minTarget = 0.0;          ///< events whose target is below this are skipped
  bool zeroOnVertical = false;     ///< label 0 when the vertical barrier is hit first (else the sign of the return)
};

struct BarrierEvent {
  std::size_t t0 = 0, t1 = 0;  ///< event and first touch
  double target = 0.0;
  double ret = 0.0;            ///< return from t0 to t1, in the direction of `side`
  int label = 0;               ///< -1, 0, +1 (with a side: 1 if the bet paid, 0 otherwise)
  int barrier = 0;             ///< +1 profit taking, -1 stop loss, 0 vertical
  int side = 1;
};

/// Barrier outcomes for the given events. With `sides` (one per event, +1 or -1) the
/// barriers are those of a position in that direction and `label` is the meta-label:
/// 1 if the position made money, 0 if not. Events too close to the end to reach a barrier
/// or the vertical one are left out.
std::vector<BarrierEvent> tripleBarrier(const std::vector<double>& close, const std::vector<std::size_t>& events,
                                        const std::vector<double>& target, const BarrierSpec& spec,
                                        const std::vector<int>* sides = nullptr);

/// Triple-barrier labels for every date and stock of a panel of closes, as used by the
/// learning pipeline: 1 if the upper barrier is touched first (or the return to the vertical
/// barrier is positive), 0 for the lower barrier (or a negative return); NaN where unknown.
/// `ends` receives the touch date of every label (the end of its span).
Panel tripleBarrierLabels(const Panel& close, const BarrierSpec& spec, double volSpan, Panel* ends = nullptr);

}  // namespace sat::afml
