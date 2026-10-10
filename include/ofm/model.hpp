#pragma once

// The orthonormal shrinkage factor model, with every setting derived from the data.
//
// Signals: ten families at dyadic horizons h = 2, 4, 8, ... up to the largest power of two not
// above an eighth of the history (so every factor record covers most of the data), and no more
// horizons than keep the signals fewer than the stocks: return, low
// volatility, nearness to the high, trend quality (return over volatility x sqrt(h)), small
// traded value, and the technical indicators RSI, Bollinger z-score, stochastic %K, money flow and
// intraday range (kernels/model/signal.wgsl).
// Each day: cross-sectional rank z-scores of every signal (ties broken by asset order), made
// exactly orthonormal by symmetric (Loewdin) orthogonalisation S = Z (Z'Z)^(-1/2). Each factor's
// record is its return the day a position decided at that close earns (next open to the open
// after, resolved two days later): f_t = S_t' R_t. A factor's premium is the mean of its record so
// far, shrunk by its own t-statistic (positive-part James-Stein): mean * max(0, 1 - 1/t^2). The
// expected next-day excess return of a stock over the market is E = S @ premia.
//
// Computation (see kernels.hpp): the per-(day, stock, signal) work runs as fused GPU kernels
// (signals + ranks in one pass in workgroup memory; Gram matrix and record projection; expected
// returns), chunk by chunk of days. The small sequential part (Loewdin per day, shrunk premia)
// runs here, between the kernels: Forecaster::absorb.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ofm/data.hpp"

namespace ofm {

constexpr std::size_t kFamilies = 10;
constexpr std::size_t kTables = 10;
constexpr std::size_t kMaxHorizons = 16;  ///< uniform block
const char* familyName(std::size_t f);

/// Packed inputs of the kernels and the chunking of the days.
struct Plan {
  std::size_t T = 0, N = 0, K = 0, H = 0;
  std::vector<std::uint32_t> horizons;
  std::size_t first = 0;      ///< first day any stock can be eligible (the longest horizon has passed)
  std::size_t chunkDays = 0;  ///< days per chunk (z of one chunk fits the GPU buffer budget)
  std::size_t stride = 0;     ///< per day in the Gram read-back: K(K+1)/2 pairs, K projections, eligible count
  /// lc | s1 | s2 | lv | elig | tgt | tmask | lh | ll | tv, each T x N (f32)
  std::vector<float> tables;
  std::size_t numChunks() const { return (T - first + chunkDays - 1) / chunkDays; }
  std::size_t chunkStart(std::size_t c) const { return first + c * chunkDays; }
  std::size_t chunkLength(std::size_t c) const;
  /// Uniform block of chunk c: T, N, K, H, chunk start, chunk days, pairs, stride, 8 unused, 16 horizons.
  std::vector<std::uint32_t> header(std::size_t c) const;
  std::size_t pairs() const { return K * (K + 1) / 2; }
  std::string signalName(std::size_t k) const;
};

Plan compilePlan(const Market& m, std::size_t bufferBudgetBytes = 64u << 20);

/// The sequential part between the kernels, day by day in order.
class Forecaster {
 public:
  explicit Forecaster(const Plan& plan);
  /// Day t's Gram row (stride values). Returns v_t = (Z'Z)^(-1/2) premia_t (K values), so that the
  /// expected returns are E = Z v_t; empty if the day has too few eligible stocks.
  std::vector<double> absorb(std::size_t t, const double* gram);
  /// Factor statistics at the last absorbed day: mean, t-statistic and shrunk premium per factor.
  std::vector<double> mean, tstat, premium;
  std::size_t records() const { return n_; }
  /// Per day, the signal families' returns (empty where unknown); day t is known at t + 2.
  std::vector<std::vector<double>> familyReturns;

 private:
  const Plan& plan_;
  std::size_t n_ = 0;
  std::vector<double> s1_, s2_;
  std::vector<std::vector<double>> pending_;  // records of the last two days, not yet usable
  std::vector<std::size_t> pendingDay_;
};

/// Everything the trading needs from the model.
struct Forecast {
  Panel E;                     ///< expected next-day excess return (NaN: no forecast)
  std::vector<double> mean, tstat, premium;  ///< factor statistics at the last day
  std::size_t records = 0;
  std::vector<std::vector<double>> familyReturns;
  std::vector<std::uint32_t> horizons;
  double kernelMs = 0;
  std::string engine;
};

/// Runs the whole model on the CPU: "reference" (double precision) or "emulated" (the kernels'
/// single-precision arithmetic, executed invocation by invocation).
Forecast runModel(const Market& m, const std::string& engine = "reference");

/// Eigen decomposition of a symmetric matrix (Householder + implicit QL); a is overwritten,
/// eigenvectors are the columns of V.
void eigenSym(std::vector<double>& a, std::size_t n, std::vector<double>& w, std::vector<double>& V);

}  // namespace ofm
