#pragma once

// The model's fused kernels. Their one source is the WGSL in kernels/model: WebGPU runs it, and the
// CPU runs its C++ translation (tools/wgsl2cpp.py) in single precision (F = float: the emulated GPU,
// same inputs, same arithmetic, same outputs) or in double precision (F = double: the reference).
//
// Bindings: 0 uniform header (Plan::header), 1 tables (Plan::tables).
//  zscore  (workgroup 64; grid K x days): signal k of every stock on the day, then each stock's rank
//          z-score among the day's eligible stocks (ties by stock index), ranked against tiles of 64
//          stocks in workgroup memory. 2: sig [day][stock][signal] (scratch), 3: z (read_write).
//  gram    (workgroup 64; grid gramTiles x days): per day, [Z y]'[Z y] as a tiled matrix product
//          (8 x 8 outputs per workgroup, 32 stocks a step as vec4): the upper triangle of Z'Z, Z'y
//          (projections on the next-day targets) and the counts of eligible and targeted stocks.
//          2: z (read), 3: gram [day][stride] (read_write).
//  expect  (workgroup 64; grid ceil(N/64) x days): E = z . v per stock and day, v staged in
//          workgroup memory, four signals a step. 2: z (read), 3: v [day][K] (read), 4: E (read_write).

#include <string>
#include <vector>

#include "ofm/model.hpp"

namespace ofm::kernels {

const std::string& zscoreSource();
const std::string& gramSource();
const std::string& expectSource();
/// Workgroups per day of the gram kernel.
std::uint32_t gramTiles(const Plan& p);

/// The kernels on the CPU for chunk c, in F (float: emulated GPU; double: reference). tables are
/// Plan::tables in F.
template <class F>
void zscore(const Plan& p, std::size_t chunk, const F* tables, std::vector<F>& z);
template <class F>
void gram(const Plan& p, std::size_t chunk, const F* tables, std::vector<F>& z, std::vector<F>& gram);
template <class F>
void expect(const Plan& p, std::size_t chunk, std::vector<F>& z, std::vector<F>& v, std::vector<F>& e);

}  // namespace ofm::kernels
