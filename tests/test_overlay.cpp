#include <cmath>

#include "sat/adaptive/overlay.hpp"
#include "sat/adaptive/self_adaptive.hpp"
#include "sat/core/random.hpp"
#include "test_framework.hpp"

using namespace sat;

namespace {

// N assets over T dates; one forecast that knows the sign of the next return with noise, the
// signal persisting for `hold` dates, and one forecast that is pure noise.
struct SignalWorld {
  Panel next;
  std::vector<ModelPredictions> forecasts;
  SignalWorld(std::size_t T, std::size_t N, std::size_t hold, std::uint64_t seed) : next(T, N, 0.0) {
    Rng rng(seed);
    ModelPredictions good, noise;
    good.name = "good", noise.name = "noise";
    good.probability = Panel(T, N, 0.5), noise.probability = Panel(T, N, 0.5);
    std::vector<double> z(N);
    for (std::size_t t = 0; t < T; ++t) {
      if (t % hold == 0)
        for (auto& v : z) v = rng.normal();
      for (std::size_t i = 0; i < N; ++i) {
        next(t, i) = 0.002 * z[i] + 0.01 * rng.normal();
        good.probability(t, i) = 1.0 / (1.0 + std::exp(-z[i] - 0.5 * rng.normal()));
        noise.probability(t, i) = rng.uniform();
      }
    }
    good.start = noise.start = 5;
    good.end = noise.end = T - 1;
    forecasts = {good, noise};
  }
};

}  // namespace

TEST(overlay_forward_jacobian_matches_finite_differences) {
  OverlaySpec spec;
  spec.hidden = 4;
  spec.maxUnits = 3;
  OverlayNet net(3, spec);
  const double s0[2] = {0, 0}, s1[2] = {2, 0.5};
  net.gate(s0, true);
  net.gate(s1, true);
  CHECK(net.units() == 2);
  Rng rng(3);
  for (double& p : net.theta()) p = 0.7 * rng.normal();
  const std::size_t N = 5, P = net.parameters();
  std::vector<double> x1(N * 2), x2(N * 2), w0(N);
  for (auto& v : x1) v = rng.uniform() - 0.5;
  for (auto& v : x2) v = rng.uniform() - 0.5;
  for (auto& v : w0) v = 0.1 * rng.normal();
  const double sa[2] = {0.4, 0.1}, sb[2] = {1.5, 0.3};
  const auto ga = net.gate(sa, false), gb = net.gate(sb, false);
  auto twoSteps = [&]() {
    const auto a = net.forward(x1, N, ga, w0, std::vector<double>(N * P, 0.0));
    return net.forward(x2, N, gb, a.w, a.J);
  };
  const auto base = twoSteps();
  double worst = 0;
  for (std::size_t p = 0; p < P; ++p) {
    const double keep = net.theta()[p], h = 1e-6;
    net.theta()[p] = keep + h;
    const auto up = twoSteps();
    net.theta()[p] = keep - h;
    const auto down = twoSteps();
    net.theta()[p] = keep;
    for (std::size_t i = 0; i < N; ++i) worst = std::max(worst, std::fabs((up.w[i] - down.w[i]) / (2 * h) - base.J[i * P + p]));
  }
  CHECK(worst < 1e-7);
}

TEST(overlay_is_causal) {
  SignalWorld a(400, 8, 5, 11), b(400, 8, 5, 11);
  for (std::size_t t = 300; t < 400; ++t)
    for (std::size_t i = 0; i < 8; ++i) b.next(t, i) = -b.next(t, i) + 0.01;
  const auto ra = overlayTrader(a.forecasts, a.next, 10, 0), rb = overlayTrader(b.forecasts, b.next, 10, 0);
  // Positions up to date 300 use returns up to 299 only (nextReturns(299) is learned on 300).
  for (std::size_t t = ra.start; t <= 300; ++t)
    for (std::size_t i = 0; i < 8; ++i) CHECK(ra.weights(t, i) == rb.weights(t, i));
  bool differs = false;
  for (std::size_t i = 0; i < 8; ++i) differs = differs || ra.weights(310, i) != rb.weights(310, i);
  CHECK(differs);
}

TEST(overlay_learns_to_trade_the_informative_forecast) {
  SignalWorld w(2000, 20, 1, 5);
  const auto r = overlayTrader(w.forecasts, w.next, 1, 0);
  CHECK(r.net.size() == r.end - r.start && r.units.back() >= 1);
  double late = 0, corr = 0;
  for (std::size_t d = r.net.size() / 2; d < r.net.size(); ++d) late += r.net[d];
  // Positions line up with the informative forecast, not the noise.
  for (std::size_t t = 1500; t < r.end; ++t)
    for (std::size_t i = 0; i < 20; ++i) corr += r.weights(t, i) * (w.forecasts[0].probability(t, i) - 0.5);
  CHECK(late > 0);
  CHECK(corr > 0);
}

TEST(overlay_trades_less_when_trading_costs_more) {
  SignalWorld w(2000, 20, 10, 9);
  const auto cheap = overlayTrader(w.forecasts, w.next, 0, 0), dear = overlayTrader(w.forecasts, w.next, 30, 50);
  auto lateTurnover = [](const OverlayResult& r) {
    double s = 0;
    for (std::size_t d = r.turnover.size() / 2; d < r.turnover.size(); ++d) s += r.turnover[d];
    return s;
  };
  CHECK(lateTurnover(dear) < lateTurnover(cheap));
  double buys = 0, net = 0;
  for (std::size_t d = 0; d < dear.net.size(); ++d) buys += dear.buys[d], net += dear.gross[d] - 30e-4 * dear.turnover[d] - 50e-4 * dear.buys[d] - dear.net[d];
  CHECK(buys > 0 && std::fabs(net) < 1e-12);
}

TEST(book_charges_stamp_duty_on_purchases_and_takes_external_candidates) {
  SignalWorld w(300, 6, 3, 2);
  const std::vector<StrategySpec> rules = {{StrategyKind::LongTopK, 2, 1}, {StrategyKind::LongShort, 1, 3}};
  const CandidateBook plain(w.forecasts, rules, w.next, 10), stamped(w.forecasts, rules, w.next, 10, 50);
  std::vector<double> wt(6), prev(6, 0.0);
  for (std::size_t d = 0; d < stamped.days(); ++d) {
    stamped.weights(1, d, wt.data());
    double buys = 0;
    for (std::size_t i = 0; i < 6; ++i) buys += std::max(0.0, wt[i] - prev[i]);
    prev = wt;
    CHECK_NEAR(stamped.buys(1, d), buys, 1e-12);
    CHECK_NEAR(stamped.net(1, d), plain.net(1, d) - 50e-4 * buys, 1e-12);
  }
  // An external candidate with candidate 0's positions has candidate 0's record.
  Panel pos(300, 6, 0.0);
  for (std::size_t d = 0; d < stamped.days(); ++d) stamped.weights(0, d, pos.row(stamped.start() + d));
  CandidateBook book = stamped;
  book.addCandidate("copy", pos);
  CHECK(book.size() == stamped.size() + 1 && book.candidates().back().label == "copy");
  for (std::size_t d = 0; d < book.days(); ++d) CHECK_NEAR(book.net(book.size() - 1, d), book.net(0, d), 1e-12);
  const auto a = runSelector(book, SelectorSpec{});
  CHECK(a.net.size() == book.days() - SelectorSpec{}.lookback);
}
