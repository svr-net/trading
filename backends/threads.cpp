// Every CPU thread, without locks: a fixed set of workers, started once, that spin-wait for work. The caller
// publishes a job (the function, its size, its chunk) under a new generation; every thread, the caller too, claims
// the next chunk by compare-and-exchange on one counter that carries the generation in its high 32 bits and the next
// index in its low 32, so a worker still busy with an earlier job can never claim work of the next one. Each
// finished chunk is counted by compare-and-exchange; the caller spins until the whole range is done. The next job's
// fields are written under a "being written" generation, once no worker is still inside the last one. No mutex, condition variable or join on the hot path; the waits are spins (a pause, then a
// yield after a long one). A call made inside a kernel (nested), or while another thread's job runs, runs in place.
// Results are the same as the serial back end's: a kernel's chunk writes only its own indices.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define OFM_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define OFM_PAUSE() asm volatile("yield")
#else
#define OFM_PAUSE() ((void)0)
#endif

#include "strategy/backend.hpp"

namespace {

using Fn = std::function<void(std::size_t, std::size_t)>;

thread_local bool tInside = false;  // this thread is running a chunk (a nested call runs in place)

class Pool {
 public:
  static Pool& get() {
    static Pool p;
    return p;
  }
  std::size_t size() const { return workers_.size() + 1; }

  // Runs f over [0, n) on every thread; returns when all of it is done. False when it cannot (nested, or another
  // thread's job running): the caller then runs it in place.
  bool run(std::size_t n, const Fn& f) {
    if (tInside || workers_.empty() || n > 0xFFFFFFFFull) return false;
    bool idle = false;
    if (!busy_.compare_exchange_strong(idle, true, std::memory_order_acquire)) return false;
    // The job's fields are written only while no worker can be reading them: the generation first says "being
    // written" (a worker that sees it, or any generation not its own, backs off without reading), then every worker
    // still inside the last job leaves. Sequentially consistent: the caller's store of the generation and its read
    // of the workers inside, and each worker's entry and its read of the generation, are never reordered.
    const std::uint64_t g = ((gen_.load(std::memory_order_seq_cst) & 0xFFFFFFFFull) + 1) & 0xFFFFFFFFull;
    gen_.store(kWriting, std::memory_order_seq_cst);
    spinUntil([&] { return active_.load(std::memory_order_seq_cst) == 0; });
    fn_.store(&f, std::memory_order_relaxed);
    n_.store(n, std::memory_order_relaxed);
    chunk_.store(std::max<std::size_t>(1, n / (size() * 8)), std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    next_.store(g << 32, std::memory_order_relaxed);
    gen_.store(g, std::memory_order_seq_cst);  // published: the workers wake
    work(g);
    spinUntil([&] { return done_.load(std::memory_order_acquire) >= n; });
    busy_.store(false, std::memory_order_release);
    return true;
  }

  ~Pool() {
    stop_.store(true, std::memory_order_release);
    for (auto& t : workers_) t.join();  // at exit only
  }

 private:
  Pool() {
    const std::size_t w = std::max(1u, std::thread::hardware_concurrency());
    for (std::size_t q = 1; q < w; ++q) workers_.emplace_back([this] { loop(); });
  }

  template <class Cond>
  static void spinUntil(Cond c) {
    for (std::uint32_t s = 0; !c(); ++s) {
      if (s < 4096) OFM_PAUSE();
      else std::this_thread::yield();
    }
  }

  static void add(std::atomic<std::size_t>& a, std::size_t x) {  // a += x by compare-and-exchange
    std::size_t cur = a.load(std::memory_order_relaxed);
    while (!a.compare_exchange_weak(cur, cur + x, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    }
  }

  // Claims and runs chunks of job g until none is left (or a newer job replaced it).
  void work(std::uint64_t g) {
    const Fn* f = fn_.load(std::memory_order_acquire);
    const std::size_t n = n_.load(std::memory_order_acquire), chunk = chunk_.load(std::memory_order_acquire);
    std::uint64_t cur = next_.load(std::memory_order_acquire);
    for (;;) {
      if ((cur >> 32) != g) return;  // another job's counter
      const std::size_t a = std::size_t(cur & 0xFFFFFFFFull);
      if (a >= n) return;
      const std::size_t e = std::min(n, a + chunk);
      if (!next_.compare_exchange_weak(cur, (g << 32) | e, std::memory_order_acq_rel, std::memory_order_acquire)) continue;
      tInside = true;
      (*f)(a, e);
      tInside = false;
      add(done_, e - a);
      cur = next_.load(std::memory_order_acquire);
    }
  }

  void loop() {
    std::uint64_t seen = gen_.load(std::memory_order_acquire);
    while (!stop_.load(std::memory_order_acquire)) {
      const std::uint64_t g = gen_.load(std::memory_order_acquire);
      if (g == seen || g == kWriting) {
        for (std::uint32_t s = 0; s < 4096 && gen_.load(std::memory_order_relaxed) == seen && !stop_.load(std::memory_order_relaxed); ++s) OFM_PAUSE();
        if (gen_.load(std::memory_order_relaxed) == seen) std::this_thread::yield();
        continue;
      }
      seen = g;
      std::size_t cur = active_.load(std::memory_order_relaxed);
      while (!active_.compare_exchange_weak(cur, cur + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
      }
      if (gen_.load(std::memory_order_seq_cst) == g) work(g);  // else being written, or newer: not read
      cur = active_.load(std::memory_order_relaxed);
      while (!active_.compare_exchange_weak(cur, cur - 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
      }
    }
  }

  static constexpr std::uint64_t kWriting = 1ull << 40;  // the generation while a job's fields are written
  std::vector<std::thread> workers_;
  std::atomic<bool> busy_{false}, stop_{false};
  std::atomic<std::uint64_t> gen_{0}, next_{0};
  std::atomic<const Fn*> fn_{nullptr};
  std::atomic<std::size_t> n_{0}, chunk_{1}, done_{0}, active_{0};
};

class Threads : public strat::Backend {
 public:
  std::string about() const override { return std::to_string(Pool::get().size()) + " threads (lock-free, spin-waiting)"; }
  void parallelFor(std::size_t n, const Fn& f) const override {
    if (n == 0) return;
    if (n == 1 || !Pool::get().run(n, f)) f(0, n);
  }
};

}  // namespace

REGISTER_BACKEND("threads", Threads);
