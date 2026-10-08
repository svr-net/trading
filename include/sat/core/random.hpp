#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sat {

/// Seeded pseudo-random generator (xoshiro256**, seeded through SplitMix64).
///
/// Every algorithm draws from its own generator with a fixed seed, and the normals come
/// from an explicit Box-Muller transform rather than std::normal_distribution, so a seed
/// gives the same numbers on every compiler and platform, including WebAssembly.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) {
    std::uint64_t x = seed;
    for (auto& s : s_) s = splitmix(x);
  }

  std::uint64_t nextU64() {
    const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  /// Uniform in [0, 1).
  double uniform() { return static_cast<double>(nextU64() >> 11) * 0x1.0p-53; }

  /// Uniform integer in [0, n).
  std::size_t below(std::size_t n) { return n == 0 ? 0 : static_cast<std::size_t>(uniform() * static_cast<double>(n)); }

  /// Standard normal.
  double normal() {
    if (hasSpare_) {
      hasSpare_ = false;
      return spare_;
    }
    double u1 = uniform();
    while (u1 <= 0.0) u1 = uniform();
    const double u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double a = 6.283185307179586 * u2;
    spare_ = r * std::sin(a);
    hasSpare_ = true;
    return r * std::cos(a);
  }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  static std::uint64_t splitmix(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  std::uint64_t s_[4] = {};
  double spare_ = 0.0;
  bool hasSpare_ = false;
};

/// Fisher-Yates shuffle of v with generator rng.
template <class T>
void shuffle(T& v, Rng& rng) {
  for (std::size_t i = v.size(); i > 1; --i) {
    const std::size_t j = rng.below(i);
    auto tmp = v[i - 1];
    v[i - 1] = v[j];
    v[j] = tmp;
  }
}

}  // namespace sat
