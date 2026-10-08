#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "sat/core/panel.hpp"
#include "sat/data/market_data.hpp"

namespace sat {

/// Price-volume inputs of the formulaic alphas, with average daily dollar volumes cached.
class AlphaInputs {
 public:
  explicit AlphaInputs(const MarketData& data);

  const Panel& open() const { return open_; }
  const Panel& high() const { return high_; }
  const Panel& low() const { return low_; }
  const Panel& close() const { return close_; }
  const Panel& volume() const { return volume_; }
  const Panel& vwap() const { return vwap_; }
  const Panel& returns() const { return returns_; }
  /// Average daily dollar volume (close x volume) over the past d days.
  const Panel& adv(std::size_t d) const;

 private:
  Panel open_, high_, low_, close_, volume_, vwap_, returns_;
  mutable std::map<std::size_t, Panel> adv_;
};

/// The 23 alphas of "101 Formulaic Alphas" (Kakushadze, 2016) that the paper uses as
/// features: 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 17, 20, 29, 33, 34, 35, 40, 41, 44, 62, 65, 81.
/// Fractional window lengths are rounded as in the paper's factor table.
const std::vector<int>& paperAlphaIds();

/// The formula of alpha `id` in the operator notation (for display).
std::string alphaFormula(int id);

/// Longest look-back (in days) the alpha needs before its first finite value.
std::size_t alphaLookback(int id);

/// Computes alpha `id` on every date and stock. Throws std::invalid_argument for an id
/// outside paperAlphaIds().
Panel computeAlpha(int id, const AlphaInputs& in);

}  // namespace sat
