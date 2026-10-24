#pragma once

// Each stock's liquidity at every close, from its own bars up to that close (no look-ahead):
//  - its bid-ask spread by Abdi and Ranaldo (2017), s^2 = 4 E[(c_t - eta_t)(c_t - eta_t+1)], c the
//    log close and eta the log mid-range, exponentially weighted;
//  - its traded value (log close x volume), weighted the same way, for a point-in-time universe:
//    the stocks then most traded.

#include <cstddef>
#include <vector>

#include "ofm/data.hpp"

namespace ofm {

struct Liquidity {
  std::size_t T = 0, N = 0;
  Panel halfSpread;   ///< half the spread (log) as of each close; 0 until estimable
  Panel tradedValue;  ///< weighted log traded value as of each close (NaN before the first bar)
  /// Weighting half-life in days.
  Liquidity(const Market& m, double halfLife);
  /// Whether stock i is among the `universe` most traded at the close t (always, for 0).
  std::vector<char> universeAt(std::size_t t, std::size_t universe) const;
};

}  // namespace ofm
