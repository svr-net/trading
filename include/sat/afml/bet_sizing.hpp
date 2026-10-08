#pragma once

#include <cstddef>

namespace sat::afml {

/// Bet size in [-1, 1] from the predicted probability of the favoured outcome, testing it
/// against the uniform guess 1/n: z = (p - 1/n) / sqrt(p (1 - p)), size = 2 N(z) - 1.
/// For a binary up/down forecast, p is P(up) and a negative size is a short.
double betSize(double p, std::size_t numClasses = 2);

/// Rounds a bet size to multiples of `step` (to avoid trading on every small change).
double discretizeBet(double size, double step);

}  // namespace sat::afml
