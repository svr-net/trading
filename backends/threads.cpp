// Every CPU thread: the range split into contiguous chunks, one per thread (std::thread).
#include <algorithm>
#include <thread>
#include <vector>

#include "strategy/backend.hpp"

namespace {

class Threads : public strat::Backend {
 public:
  std::string about() const override { return std::to_string(workers()) + " threads"; }
  void parallelFor(std::size_t n, const std::function<void(std::size_t, std::size_t)>& f) const override {
    const std::size_t w = std::min<std::size_t>(workers(), n);
    if (w <= 1) {
      if (n) f(0, n);
      return;
    }
    std::vector<std::thread> pool;
    const std::size_t step = (n + w - 1) / w;
    for (std::size_t b = 0; b < n; b += step) pool.emplace_back(f, b, std::min(n, b + step));
    for (auto& t : pool) t.join();
  }

 private:
  static std::size_t workers() { return std::max(1u, std::thread::hardware_concurrency()); }
};

}  // namespace

REGISTER_BACKEND("threads", Threads);
