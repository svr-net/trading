#include <cmath>

#include "sat/data/synthetic_market.hpp"
#include "sat/factors/alpha101.hpp"
#include "sat/factors/operators.hpp"
#include "sat/features/dataset.hpp"
#include "test_framework.hpp"

using namespace sat;
using namespace sat::ops;

namespace {

Panel column(const std::vector<double>& v) {
  Panel p(v.size(), 1);
  for (std::size_t t = 0; t < v.size(); ++t) p(t, 0) = v[t];
  return p;
}

MarketData smallMarket(std::size_t dates = 220, std::size_t assets = 10) {
  SyntheticMarketSpec s;
  s.numAssets = assets;
  s.numDates = dates;
  s.seed = 3;
  return generateSyntheticMarket(s);
}

}  // namespace

TEST(time_series_operators) {
  const Panel x = column({1, 3, 2, 5, 4});
  CHECK(std::isnan(tsSum(x, 3)(1, 0)));
  CHECK_NEAR(tsSum(x, 3)(2, 0), 6, 1e-12);
  CHECK_NEAR(tsMean(x, 2)(4, 0), 4.5, 1e-12);
  CHECK_NEAR(tsMin(x, 3)(4, 0), 2, 1e-12);
  CHECK_NEAR(tsMax(x, 3)(4, 0), 5, 1e-12);
  CHECK_NEAR(tsArgMax(x, 3)(4, 0), 2, 1e-12);  // 5 sits in the middle of (2, 5, 4)
  CHECK_NEAR(tsRank(x, 3)(3, 0), 1.0, 1e-12);   // 5 is the largest of (3, 2, 5)
  CHECK_NEAR(tsRank(x, 3)(2, 0), 2.0 / 3.0, 1e-12);
  CHECK_NEAR(tsProduct(x, 2)(1, 0), 3, 1e-12);
  CHECK_NEAR(tsStddev(x, 5)(4, 0), std::sqrt(2.5), 1e-12);
  CHECK_NEAR(delay(x, 2)(4, 0), 2, 1e-12);
  CHECK_NEAR(delta(x, 1)(3, 0), 3, 1e-12);
  const Panel y = column({2, 6, 4, 10, 8});
  CHECK_NEAR(tsCorr(x, y, 4)(4, 0), 1.0, 1e-12);
  CHECK_NEAR(tsCov(x, y, 5)(4, 0), 5.0, 1e-12);
  CHECK(std::isnan(tsCorr(x, column({1, 1, 1, 1, 1}), 3)(4, 0)));  // constant window
}

TEST(cross_sectional_operators) {
  Panel x(1, 4);
  x(0, 0) = 3;
  x(0, 1) = -1;
  x(0, 2) = 3;
  x(0, 3) = 7;
  const Panel r = rank(x);
  CHECK_NEAR(r(0, 1), 0.25, 1e-12);
  CHECK_NEAR(r(0, 0), 0.625, 1e-12);
  CHECK_NEAR(r(0, 3), 1.0, 1e-12);
  const Panel s = scale(x);
  CHECK_NEAR(std::fabs(s(0, 0)) + std::fabs(s(0, 1)) + std::fabs(s(0, 2)) + std::fabs(s(0, 3)), 1.0, 1e-12);
  CHECK_NEAR(signedPower(x, 2)(0, 1), -1, 1e-12);
  CHECK(std::isnan(log(x)(0, 1)));
  CHECK_NEAR(where(lessThan(x, 2.0), x, 0.0)(0, 1), -1, 1e-12);
  CHECK_NEAR(where(lessThan(x, 2.0), x, 0.0)(0, 3), 0, 1e-12);
}

TEST(paper_alphas_are_defined_and_finite) {
  const MarketData d = smallMarket();
  const AlphaInputs in(d);
  CHECK(paperAlphaIds().size() == 23);
  for (int id : paperAlphaIds()) {
    const Panel a = computeAlpha(id, in);
    CHECK(a.dates() == d.numDates() && a.assets() == d.numAssets());
    // Finite once the look-back has passed, except where a correlation's window is constant:
    // alphas 3 and 81 correlate cross-sectional ranks of price levels, which rarely move over
    // a few days, so they are often undefined (NaN; features map it to the cross-sectional centre).
    std::size_t finiteCount = 0, total = 0;
    for (std::size_t t = alphaLookback(id) + 2; t < d.numDates(); ++t)
      for (std::size_t i = 0; i < d.numAssets(); ++i) {
        ++total;
        finiteCount += std::isfinite(a(t, i)) ? 1 : 0;
      }
    const double need = id == 3 || id == 81 ? 0.02 : 0.9;
    if (!(finiteCount > total * need)) ::sattest::fail("alpha " + std::to_string(id) + " is mostly undefined", __FILE__, __LINE__);
    CHECK(!alphaFormula(id).empty());
  }
  CHECK_THROWS(computeAlpha(8, in));
}

TEST(alphas_never_look_ahead) {
  // An alpha on date t must not change when later dates are removed.
  const MarketData full = smallMarket(200, 8);
  const MarketData cut = full.slice(0, 150);
  const AlphaInputs a(full), b(cut);
  for (int id : paperAlphaIds()) {
    const Panel x = computeAlpha(id, a), y = computeAlpha(id, b);
    for (std::size_t t = 0; t < 150; ++t)
      for (std::size_t i = 0; i < 8; ++i) {
        const bool same = (std::isnan(x(t, i)) && std::isnan(y(t, i))) || std::fabs(x(t, i) - y(t, i)) < 1e-9;
        if (!same) ::sattest::fail("alpha " + std::to_string(id) + " looks ahead at t = " + std::to_string(t), __FILE__, __LINE__);
      }
  }
}

TEST(features_are_normalised_and_complete) {
  const MarketData d = smallMarket();
  const FeatureSet f = buildFeatures(d, paperAlphaIds(), Normalisation::Rank);
  CHECK(f.size() == 23 && f.warmup == 81);
  for (const Panel& p : f.panels)
    for (std::size_t t = 0; t < p.dates(); ++t)
      for (std::size_t i = 0; i < p.assets(); ++i) CHECK(std::isfinite(p(t, i)) && std::fabs(p(t, i)) <= 0.5);
  const FeatureSet z = buildFeatures(d, {12, 41}, Normalisation::ZScore);
  CHECK(z.size() == 2 && z.names[1] == "alpha41");
  CHECK_THROWS(buildFeatures(d, {}, Normalisation::Rank));
  CHECK(parseNormalisation("zscore") == Normalisation::ZScore);
  CHECK_THROWS(parseNormalisation("nope"));
}

TEST(labels_direction_excess_and_minmax) {
  const MarketData d = smallMarket(120, 6);
  const Panel up = makeLabels(d, {LabelKind::Direction, 2, 10});
  const Panel fwd = d.forwardReturns(2);
  CHECK(up(10, 3) == (fwd(10, 3) > 0 ? 1.0 : 0.0));
  CHECK(std::isnan(up(119, 0)) && std::isnan(up(118, 0)));
  const Panel ex = makeLabels(d, {LabelKind::ExcessDirection, 1, 10});
  double ones = 0;
  for (std::size_t i = 0; i < 6; ++i) ones += ex(50, i);
  CHECK(ones == 3);  // half of the stocks beat the median
  const Panel mm = makeLabels(d, {LabelKind::MinMax, 1, 10});
  for (std::size_t t = 0; t < 120; ++t)
    for (std::size_t i = 0; i < 6; ++i) {
      if (std::isnan(mm(t, i))) continue;
      CHECK(t >= 5 && t + 5 < 120);
      for (std::size_t k = t - 5; k <= t + 5; ++k)
        if (k != t) CHECK(mm(t, i) == 1.0 ? d.close(k, i) > d.close(t, i) : d.close(k, i) < d.close(t, i));
    }
  CHECK((LabelSpec{LabelKind::MinMax, 1, 10}.lookahead() == 5));
}

TEST(assemble_builds_lagged_rows) {
  const MarketData d = smallMarket(120, 5);
  const FeatureSet f = buildFeatures(d, {12, 33}, Normalisation::Rank);
  const Panel y = makeLabels(d, {});
  const Dataset ds = assemble(f, y, 100, 110, 3, false);
  CHECK(ds.X.rows() == 50 && ds.X.cols() == 6 && ds.lags == 3);
  // Row for (date 105, stock 2): oldest lag first.
  std::size_t r = 0;
  while (!(ds.date[r] == 105 && ds.asset[r] == 2)) ++r;
  CHECK(ds.X(r, 0) == f.panels[0](103, 2) && ds.X(r, 5) == f.panels[1](105, 2));
  const Dataset labelled = assemble(f, y, 110, 120, 1, true);
  CHECK(labelled.X.rows() == 9 * 5);  // the last date has no label
}
