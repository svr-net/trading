#pragma once

// The same-day trades as fused kernels: every number from the bars and the expected returns to the
// booked trades. Two kernels, written once in WGSL (kernels/daytrade): WebGPU runs them, the CPU runs
// their C++ translation in double precision (the reference) and single precision (the emulated GPU).
//  levels  (workgroup 64; grid N; called per chunk of days [t0, t1)): one workgroup per stock walks
//          its days in order. Lane 0 keeps the running state: the open-to-close volatility
//          (exponentially weighted, half-life the forecast's life) and the parabolic SAR / RSI / ADX
//          state. The 64 lanes share the work on the record of its days (how far the low fell and
//          the high rose from the open in its volatility, the close), kept sorted: each day's
//          insertion, and at each decision the stop-loss and take-profit learnt from that record
//          (best return per unit of risk, as parallel scans and reductions; the stop no nearer the
//          open than the median fall or the round trip, both within the plausible moves by
//          Chauvenet's criterion), whether they beat cash, their touch probabilities, and (fused)
//          the trade on the next day's bar: bought at the open, sold at the stop (first), the
//          take-profit or the close. Its state carries over from chunk to chunk.
//  book    (workgroup 64; grid ceil(days/64)): per day, down the ranking by expected return, the first
//          K stocks whose levels beat cash are booked; the day's result, the plain comparison and the
//          evaluation sums. On the last day, the plan for the next open.

#include <cstdint>
#include <string>
#include <vector>

namespace ofm::daytrade {

constexpr std::size_t kState = 24;   // floats of per-stock state carried between chunks
constexpr std::size_t kLevel = 10;   // per (day, stock): a, b, score, trade, pStop, pTake, sigma, maxDrop, maxRise, state
constexpr std::size_t kTrade = 6;    // per (day, stock): valid, ret, exit (0 close, 1 stop, 2 take), ret0, volume ratio, intraday
constexpr std::size_t kDay = 24;     // per day: see book()
constexpr float kNone = 3.0e38f;     // "no level" / infinity in the kernels

/// Everything the kernels read, flat, in the layout of the GPU buffers.
template <class R>
struct Buffers {
  std::uint32_t T = 0, N = 0, K = 0, s = 0, H = 0;
  R buy = 0, sell = 0;
  std::vector<std::uint32_t> hz;     // horizons (H of them)
  std::vector<R> bars;               // [5][T][N]: open, high, low, close, volume (<= 0 or volume < 0: missing)
  std::vector<R> expected;           // [T][N]
  std::vector<std::uint32_t> elig;   // [T][N]: 1 ranked that day
  std::vector<R> life;               // [T]
  std::vector<R> state;              // [N][kState]
  std::vector<R> obs;                // [N][4][T]: lo, hi, close, sigma
  std::vector<std::uint32_t> idx;    // [N][2][T]: obs sorted by lo, by hi
  std::vector<R> levels;             // [T][N][kLevel]
  std::vector<R> trades;             // [T][N][kTrade]
  std::vector<R> days;               // [T][kDay]
  std::vector<std::uint32_t> booked; // [T + 1][K]: booked stocks per day; row T: the plan's ranks
};

/// Runs the kernels on the CPU (double: reference; float: the emulated GPU), in the GPU's order.
template <class R>
void run(Buffers<R>& b, std::uint32_t chunk);

const std::string& levelsSource();
const std::string& bookSource();

}  // namespace ofm::daytrade
