#pragma once

// Runs a kernel's invocations on the CPU: f(i) for i in [0, n). Invocations are independent (no
// shared memory, no barriers), so native builds spread them over the hardware threads, as the GPU
// spreads them over its cores; WebAssembly runs them in order.

#include <algorithm>
#include <cstdint>
#include <vector>
#ifndef __EMSCRIPTEN__
#include <thread>
#endif

namespace ofm::detail {

template <class Fn>
void invokeAll(std::uint32_t n, Fn f) {
#ifndef __EMSCRIPTEN__
  const std::uint32_t threads = std::min<std::uint32_t>(n, std::max(1u, std::thread::hardware_concurrency()));
  if (threads > 1) {
    std::vector<std::thread> pool;
    for (std::uint32_t w = 0; w < threads; ++w)
      pool.emplace_back([&, w] {
        for (std::uint32_t i = w; i < n; i += threads) f(i);
      });
    for (auto& t : pool) t.join();
    return;
  }
#endif
  for (std::uint32_t i = 0; i < n; ++i) f(i);
}

}  // namespace ofm::detail
