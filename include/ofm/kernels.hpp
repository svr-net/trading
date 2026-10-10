#pragma once

// The model's fused kernels. Their one source is the WGSL in kernels/model: WebGPU runs it, and the
// CPU runs its C++ translation (tools/wgsl2cpp.py) in single precision (F = float: the emulated GPU,
// same inputs, same arithmetic, same outputs) or in double precision (F = double: the reference).
//
// Bindings: 0 uniform header (Plan::header), 1 tables (Plan::tables).
//  signal  (workgroup 64; grid ceil(N/64) x K x days): one stock's signal. 2: sig [day][stock][signal].
//  rank    (workgroup 64; grid ceil(N/64) x K x days): its rank z-score among the day's eligible
//          stocks (ties by stock index). 2: sig (read), 3: z [day][stock][signal] (read_write).
//  gram    (workgroup 64; grid ceil(stride/64) x days): per day, the Gram matrix of the z-scores
//          (upper triangle), each signal's projection on the next-day targets, and the counts of
//          eligible and targeted stocks. 2: z (read), 3: gram [day][stride] (read_write).
//  expect  (workgroup 64; grid ceil(N/64) x days): E = z . v per stock and day.
//          2: z (read), 3: v [day][K] (read), 4: E [day][stock] (read_write).

#include <string>
#include <vector>

#include "ofm/model.hpp"

namespace ofm::kernels {

const std::string& signalSource();
const std::string& rankSource();
const std::string& gramSource();
const std::string& expectSource();

/// The kernels on the CPU for chunk c, in F (float: emulated GPU; double: reference). tables are
/// Plan::tables in F.
template <class F>
void zscore(const Plan& p, std::size_t chunk, const F* tables, std::vector<F>& z);
template <class F>
void gram(const Plan& p, std::size_t chunk, const F* tables, std::vector<F>& z, std::vector<F>& gram);
template <class F>
void expect(const Plan& p, std::size_t chunk, std::vector<F>& z, std::vector<F>& v, std::vector<F>& e);

}  // namespace ofm::kernels
