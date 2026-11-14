// One thread: the reference back end, and the WebAssembly build's (no threads there).
#include "strategy/backend.hpp"

namespace {

class Serial : public strat::Backend {
 public:
  std::string about() const override { return "one thread"; }
  void parallelFor(std::size_t n, const std::function<void(std::size_t, std::size_t)>& f) const override {
    if (n) f(0, n);
  }
};

}  // namespace

REGISTER_BACKEND("serial", Serial);
