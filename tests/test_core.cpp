#include <cmath>
#include <limits>

#include "sat/core/matrix.hpp"
#include "sat/core/panel.hpp"
#include "sat/core/random.hpp"
#include "sat/core/stats.hpp"
#include "test_framework.hpp"

using namespace sat;

TEST(solve_spd_matches_known_solution) {
  Matrix a(3, 3);
  const double v[9] = {4, 1, 0, 1, 3, 1, 0, 1, 2};
  for (int k = 0; k < 9; ++k) a.data()[k] = v[k];
  const auto x = solveSpd(a, {1, 2, 3});
  // A x = b
  for (int r = 0; r < 3; ++r) {
    double s = 0;
    for (int c = 0; c < 3; ++c) s += a(r, c) * x[c];
    CHECK_NEAR(s, r + 1.0, 1e-12);
  }
  Matrix bad(2, 2, 1.0);
  bad(1, 1) = -1.0;
  CHECK_THROWS(solveSpd(bad, {1, 1}));
}

TEST(average_ranks_handle_ties_and_nan) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto r = averageRanks({3.0, 1.0, 3.0, nan, 2.0});
  CHECK_NEAR(r[0], 3.5, 1e-12);
  CHECK_NEAR(r[1], 1.0, 1e-12);
  CHECK_NEAR(r[2], 3.5, 1e-12);
  CHECK(std::isnan(r[3]));
  CHECK_NEAR(r[4], 2.0, 1e-12);
}

TEST(statistics_basics) {
  const std::vector<double> x = {1, 2, 3, 4, 5}, y = {2, 4, 6, 8, 10}, z = {5, 4, 3, 2, 1};
  CHECK_NEAR(mean(x), 3.0, 1e-12);
  CHECK_NEAR(stdev(x), std::sqrt(2.5), 1e-12);
  CHECK_NEAR(correlation(x, y), 1.0, 1e-12);
  CHECK_NEAR(correlation(x, z), -1.0, 1e-12);
  CHECK_NEAR(rankCorrelation(x, {1, 4, 9, 16, 25}), 1.0, 1e-12);
  CHECK_NEAR(quantile(x, 0.5), 3.0, 1e-12);
  CHECK_NEAR(quantile(x, 0.25), 2.0, 1e-12);
}

TEST(argsort_descending_is_stable_and_puts_nan_last) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double p[5] = {0.2, 0.7, nan, 0.7, 0.1};
  const auto o = argsortDescending(p, 5);
  CHECK(o[0] == 1 && o[1] == 3 && o[2] == 0 && o[3] == 4 && o[4] == 2);
}

TEST(rng_is_reproducible_with_standard_normals) {
  Rng a(42), b(42), c(43);
  CHECK(a.nextU64() == b.nextU64());
  CHECK(a.nextU64() != c.nextU64());
  Rng g(7);
  double s = 0, s2 = 0;
  const int n = 200000;
  for (int k = 0; k < n; ++k) {
    const double z = g.normal();
    s += z;
    s2 += z * z;
  }
  CHECK_NEAR(s / n, 0.0, 0.01);
  CHECK_NEAR(s2 / n, 1.0, 0.01);
  for (int k = 0; k < 1000; ++k) {
    const double u = g.uniform();
    CHECK(u >= 0.0 && u < 1.0);
    CHECK(g.below(7) < 7);
  }
}

TEST(panel_layout) {
  Panel p(3, 2, 0.0);
  p(1, 1) = 5;
  CHECK(p.row(1)[1] == 5);
  CHECK(p.series(1)[1] == 5);
  CHECK(p.countFinite() == 6);
  CHECK(std::isnan(p.like()(0, 0)));
}
