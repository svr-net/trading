// Injectable compute back ends for apps/ofm_strategy.
//
// The heavy work is a few fused kernels (one pass over the data does what would otherwise be several):
//   prepare  per share, in one sweep over its days: the carried-forward close, the log return, the 60-session
//            volatility, the 20-bar median traded value and the first bar (strat::fusedShare)
//   market   per day: the tradable shares and the day's mean return (strat::fusedDay)
// and the strategies' own per-share loops. A back end decides only how these run: its parallelFor splits the
// index range [0, n) into chunks and runs each. Every back end gives the same numbers: a kernel's chunk
// writes only its own indices.
//
// A back end is one source file in backends/ that defines a class derived from strat::Backend and registers it
// with REGISTER_BACKEND; the build compiles every file there in, and --backend NAME chooses one ("serial": one
// thread, the WebAssembly build's; "threads": every CPU thread, the default natively).
#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace strat {

class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string about() const = 0;
  // Runs f(begin, end) over chunks covering [0, n); returns when all are done.
  virtual void parallelFor(std::size_t n, const std::function<void(std::size_t, std::size_t)>& f) const = 0;
  // A gang: f(w, w + 1) for every w in [0, n), all running at once on their own threads, so a kernel may wait inside
  // for the others (a barrier). False, having run nothing, when the back end cannot promise that (one thread, more
  // members than threads, or busy): the caller then takes its plain path. gangSize(): the most members it can run.
  virtual std::size_t gangSize() const { return 1; }
  virtual bool gang(std::size_t, const std::function<void(std::size_t, std::size_t)>&) const { return false; }
};

using BackendFactory = std::function<std::unique_ptr<Backend>()>;

inline std::map<std::string, BackendFactory>& backends() {
  static std::map<std::string, BackendFactory> r;
  return r;
}

struct RegisterBackend {
  RegisterBackend(const char* name, BackendFactory f) { backends()[name] = std::move(f); }
};

}  // namespace strat

#define REGISTER_BACKEND(NAME, CLASS) \
  static const strat::RegisterBackend strat_backend_##CLASS(NAME, [] { return std::unique_ptr<strat::Backend>(new CLASS()); })
