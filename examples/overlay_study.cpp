// Gate 1b of the overlay trader: does a network that learns the trades beat the fixed rules?
//
//   overlay_study [markets per set=6] [days=2520] [--csv data.csv]
//   overlay_study 0 --csv data.csv     (real data only)
//
// Each market: the default forecast pipeline (two models, four market-state specialists, the
// self-adaptive forecast). Then, over the same days and at two cost levels (10 bp per unit of
// turnover; UK: 10 bp plus 50 bp stamp duty on purchases):
//  - default: the self-adaptive selector over the five fixed rules on the self-adaptive forecast,
//  - network: the overlay trader alone (inputs: all seven forecasts), fully invested, at each of
//    three learning rates (1e-3, 1e-2, 1e-1),
//  - overlay: the selector over the five rules and the three networks as candidates, so the
//    selector, not a hand-set value, chooses the learning rate.
// Nothing is tuned: the other settings are OverlaySpec's defaults, fixed before any result.
// (Gate 1, with a network free to scale its book down and a single learning rate of 1e-3,
// failed: the network held almost nothing.)
// Gate (fixed in advance): on the selection markets and on the unseen markets, at both cost
// levels, the overlay's mean Sharpe ratio is above the default's and it beats the default on at
// least 4 of 6 markets; on UK data, at UK costs, it is at least as good as the default.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "sat/sat.hpp"

using namespace sat;

namespace {

struct Costs {
  const char* name;
  double cost, stamp;
};
const Costs kCosts[] = {{"10 bp", 10, 0}, {"UK: 10 bp + 50 bp stamp duty on purchases", 10, 50}};
const double kRates[] = {1e-3, 1e-2, 1e-1};
constexpr std::size_t kRows = 2 + std::size(kRates);  // default, networks, overlay

struct Outcome {
  PerformanceMetrics metrics;
  double turnoverYear = 0, networkShare = 0, rate = 0, exposure = 0;
  std::size_t units = 0;
};

struct Market {
  std::string name;
  PredictionSet p;
  std::size_t first = 0;
  // [cost][default, network per learning rate, overlay]
  std::vector<std::vector<Outcome>> out;
};

ExperimentSpec defaultSpec() {
  ExperimentSpec e;
  e.stateSpecialists = true;
  e.composite.method = CompositeMethod::Adaptive;
  e.composite.window = ScoringWindow::MarketAdwin;
  e.composite.decision = ScoringDecision::Evidence;
  return e;
}

Market prepare(const std::string& name, const MarketData& data) {
  Market m{name, runPredictions(data, defaultSpec()), 0, {}};
  for (const auto& mp : m.p.models) m.first = std::max(m.first, mp.start);
  m.first += SelectorSpec{}.lookback;
  const std::vector<ModelPredictions> forecast = {m.p.models.back()};
  for (const auto& c : kCosts) {
    std::vector<Outcome> row(kRows);
    const CandidateBook base(forecast, ExperimentSpec::defaultStrategies(), m.p.nextReturns, c.cost, c.stamp);
    const std::size_t from = m.first - base.start();
    auto yearly = [](const PerformanceMetrics& pm) { return 252.0 * pm.averageTurnover; };
    const auto a = runSelector(base, SelectorSpec{}, from);
    row[0].metrics = a.metrics, row[0].turnoverYear = yearly(a.metrics);

    CandidateBook book = base;
    for (std::size_t q = 0; q < std::size(kRates); ++q) {
      OverlaySpec spec;
      spec.learningRate = kRates[q];
      const auto net = overlayTrader(m.p.models, m.p.nextReturns, c.cost, c.stamp, spec);
      if (net.start != base.start()) throw std::logic_error("overlay and book start on different dates");
      const std::vector<double> n(net.net.begin() + static_cast<std::ptrdiff_t>(from), net.net.end()),
          t(net.turnover.begin() + static_cast<std::ptrdiff_t>(from), net.turnover.end());
      Outcome& o = row[1 + q];
      o.metrics = evaluatePerformance(n, t), o.turnoverYear = yearly(o.metrics);
      for (std::size_t d = from; d < net.tradeRate.size(); ++d) {
        o.rate += net.tradeRate[d] / static_cast<double>(net.tradeRate.size() - from);
        o.exposure += net.exposure[d] / static_cast<double>(net.tradeRate.size() - from);
      }
      o.units = net.units.back();
      book.addCandidate("overlay network", net.weights);
    }
    const auto o = runSelector(book, SelectorSpec{}, from);
    Outcome& ov = row[kRows - 1];
    ov.metrics = o.metrics, ov.turnoverYear = yearly(o.metrics);
    for (std::size_t q = 0; q < std::size(kRates); ++q) ov.networkShare += o.share[base.size() + q];
    m.out.push_back(row);
  }
  std::fprintf(stderr, "  %s ready\n", name.c_str());
  return m;
}

// Prints the comparison; returns whether the overlay passes this set's part of the gate per cost.
std::vector<bool> report(const char* title, const std::vector<Market>& set) {
  const double n = static_cast<double>(set.size());
  std::vector<std::string> names = {"default (selector, 5 rules)"};
  for (double r : kRates) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "network alone, learning rate %g", r);
    names.push_back(buf);
  }
  names.push_back("overlay (selector, 5 rules + 3 networks)");
  std::vector<bool> pass;
  for (std::size_t c = 0; c < std::size(kCosts); ++c) {
    std::printf("\n%s, %s\n  %-42s %8s %8s %8s %8s %8s %6s %6s %6s\n", title, kCosts[c].name, "", "ann.ret", "Sharpe", "worst", "max DD",
                "turn/yr", "beats", "rate", "expo");
    std::vector<double> meanSharpe(kRows, 0.0);
    std::vector<int> beats(kRows, 0);
    for (std::size_t k = 0; k < kRows; ++k) {
      double ret = 0, dd = 0, worst = 1e9, turn = 0, rate = 0, expo = 0;
      for (const auto& m : set) {
        const auto& o = m.out[c][k];
        ret += o.metrics.annualReturn / n, meanSharpe[k] += o.metrics.sharpe / n, dd += o.metrics.maxDrawdown / n, turn += o.turnoverYear / n;
        rate += o.rate / n, expo += o.exposure / n;
        worst = std::min(worst, o.metrics.sharpe);
        if (k > 0 && o.metrics.sharpe > m.out[c][0].metrics.sharpe + 1e-9) ++beats[k];
      }
      std::printf("  %-42s %7.1f%% %8.2f %8.2f %7.1f%% %7.0fx", names[k].c_str(), 100 * ret, meanSharpe[k], worst, 100 * dd, turn);
      if (k > 0) std::printf(" %3d/%-2zu", beats[k], set.size());
      if (k > 0 && k + 1 < kRows) std::printf(" %6.2f %+6.2f", rate, expo);
      std::printf("\n");
    }
    double share = 0;
    for (const auto& m : set) share += m.out[c][kRows - 1].networkShare / n;
    std::printf("  the overlay held a network %.0f%% of days\n", 100 * share);
    pass.push_back(set.size() == 1 ? meanSharpe[kRows - 1] >= meanSharpe[0] : meanSharpe[kRows - 1] > meanSharpe[0] && beats[kRows - 1] >= 4);
  }
  return pass;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::size_t markets = 6, days = 2520;
  std::string csv;
  std::vector<std::string> pos;
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv" && k + 1 < argc) csv = argv[++k];
    else pos.push_back(a);
  }
  if (pos.size() > 0) markets = std::strtoul(pos[0].c_str(), nullptr, 10);
  if (pos.size() > 1) days = std::strtoul(pos[1].c_str(), nullptr, 10);
  bool gate = true;
  auto verdict = [&](const char* set, const std::vector<bool>& pass) {
    for (std::size_t c = 0; c < pass.size(); ++c) {
      std::printf("Gate 1b on %s, %s: %s\n", set, kCosts[c].name, pass[c] ? "pass" : "FAIL");
      gate = gate && pass[c];
    }
  };
  if (markets > 0) {
    auto build = [&](std::uint64_t base) {
      std::vector<Market> set;
      for (std::size_t k = 0; k < markets; ++k) {
        SyntheticMarketSpec ms;
        ms.seed = base + k;
        ms.numDates = days;
        set.push_back(prepare("seed " + std::to_string(ms.seed), generateSyntheticMarket(ms)));
      }
      return set;
    };
    std::fprintf(stderr, "selection markets\n");
    const auto selection = report("Selection markets", build(101));
    std::fprintf(stderr, "validation markets\n");
    const auto validation = report("Validation markets (unseen)", build(201));
    std::printf("\n");
    verdict("the selection markets", selection);
    verdict("the unseen markets", validation);
  }
  if (!csv.empty()) {
    std::ifstream f(csv);
    std::stringstream ss;
    ss << f.rdbuf();
    const auto uk = report(("Real data: " + csv).c_str(), {prepare(csv, parseCsv(ss.str()))});
    // The UK part of the gate is at UK costs only.
    std::printf("Gate 1b on %s, %s: %s\n", csv.c_str(), kCosts[1].name, uk[1] ? "pass" : "FAIL");
    gate = gate && uk[1];
  } else if (markets == 0) {
    std::fprintf(stderr, "overlay_study 0 needs --csv\n");
    return 2;
  }
  if (markets == 0)
    std::printf("\nUK part of gate 1b %s (the synthetic markets' part is not run here: overlay_study [markets])\n", gate ? "passed" : "failed");
  else
    std::printf("\nGate 1b %s\n", gate ? "passed" : "failed");
  return 0;
}
