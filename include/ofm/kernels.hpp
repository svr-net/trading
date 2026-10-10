#pragma once

// Fused WebGPU kernels of the model (WGSL), and their execution on the CPU in single precision
// (the emulated GPU: same inputs, same arithmetic, same outputs, used where WebGPU is missing).
//
// Bindings: 0 uniform header (Plan::header), 1 tables (Plan::tables).
//  zscore  (workgroup 256 = one day's universe; grid K x days): every invocation computes one
//          stock's signal into workgroup memory, then ranks it against the rest of the day and
//          writes its rank z-score. 2: z [day][stock][signal] (read_write).
//  gram    (workgroup 64; grid ceil(stride/64) x days): per day, the Gram matrix of the z-scores
//          (upper triangle), each signal's projection on the next-day targets, and the counts of
//          eligible and targeted stocks. 2: z (read), 3: gram [day][stride] (read_write).
//  expect  (workgroup 64; grid ceil(N/64) x days): E = z . v per stock and day.
//          2: z (read), 3: v [day][K] (read), 4: E [day][stock] (read_write).

#include <string>
#include <vector>

#include "ofm/model.hpp"

namespace ofm::kernels {

const std::string& zscoreSource();
const std::string& gramSource();
const std::string& expectSource();

void emulateZscore(const Plan& p, std::size_t chunk, std::vector<float>& z);
void emulateGram(const Plan& p, std::size_t chunk, const std::vector<float>& z, std::vector<float>& gram);
void emulateExpect(const Plan& p, std::size_t chunk, const std::vector<float>& z, const std::vector<float>& v, std::vector<float>& e);

}  // namespace ofm::kernels
