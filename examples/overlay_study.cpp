// Gate 1 of the overlay trader: does a network that learns the trades beat the fixed rules?
//
//   overlay_study [markets per set=6] [days=2520] [--csv data.csv]
//   overlay_study 0 --csv data.csv     (real data only)
//
// Each market: the default forecast pipeline (two models, four market-state specialists, the
// self-adaptive forecast). Then, over the same days and at two cost levels (10 bp per unit of
// turnover; UK: 10 bp plus 50 bp stamp duty on purchases):
//  - default: the self-adaptive selector over the five fixed rules on the self-adaptive forecast,
//  - network: the overlay trader alone (inputs: all seven forecasts),
//  - overlay: the selector over the five rules and the network as one more candidate.
// Nothing is tuned: the network's settings are OverlaySpec's defaults, fixed before any result.
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

struct Outcome {
  PerformanceMetrics metrics;
  double turnoverYear = 0, networkShare = 0, rate = 0, exposure = 0;
  std::size_t units = 0;
};

struct Market {
  std::string name;
  PredictionSet p;
  std::size_t first = 0;
  // [cost][default, network, overlay]
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
    std::vector<Outcome> row(3);
    const CandidateBook base(forecast, ExperimentSpec::defaultStrategies(), m.p.nextReturns, c.cost, c.stamp);
    const std::size_t from = m.first - base.start();
    auto yearly = [](const PerformanceMetrics& pm) { return 252.0 * pm.averageTurnover; };
    const auto a = runSelector(base, SelectorSpec{}, from);
    row[0].metrics = a.metrics, row[0].turnoverYear = yearly(a.metrics);

    const auto net = overlayTrader(m.p.models, m.p.nextReturns, c.cost, c.stamp);
    if (net.start != base.start()) throw std::logic_error("overlay and book start on different dates");
    const std::vector<double> n(net.net.begin() + static_cast<std::ptrdiff_t>(from), net.net.end()),
        t(net.turnover.begin() + static_cast<std::ptrdiff_t>(from), net.turnover.end());
    row[1].metrics = evaluatePerformance(n, t), row[1].turnoverYear = yearly(row[1].metrics);
    for (std::size_t d = from; d < net.tradeRate.size(); ++d) {
      row[1].rate += net.tradeRate[d] / static_cast<double>(net.tradeRate.size() - from);
      row[1].exposure += net.exposure[d] / static_cast<double>(net.tradeRate.size() - from);
    }
    row[1].units = net.units.back();

    CandidateBook book = base;
    book.addCandidate("overlay network", net.weights);
    const auto o = runSelector(book, SelectorSpec{}, from);
    row[2].metrics = o.metrics, row[2].turnoverYear = yearly(o.metrics), row[2].networkShare = o.share[book.size() - 1];
    m.out.push_back(row);
  }
  std::fprintf(stderr, "  %s ready\n", name.c_str());
  return m;
}

// Prints the comparison; returns whether the overlay passes this set's part of the gate per cost.
std::vector<bool> report(const char* title, const std::vector<Market>& set) {
  const double n = static_cast<double>(set.size());
  const char* names[] = {"default (selector, 5 rules)", "network alone", "overlay (selector, 5 rules + network)"};
  std::vector<bool> pass;
  for (std::size_t c = 0; c < std::size(kCosts); ++c) {
    std::printf("\n%s, %s\n  %-40s %8s %8s %8s %8s %8s %6s\n", title, kCosts[c].name, "", "ann.ret", "Sharpe", "worst", "max DD", "turn/yr",
                "beats");
    double meanSharpe[3] = {0, 0, 0};
    int beats[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k) {
      double ret = 0, dd = 0, worst = 1e9, turn = 0;
      for (const auto& m : set) {
        const auto& o = m.out[c][static_cast<std::size_t>(k)];
        ret += o.metrics.annualReturn / n, meanSharpe[k] += o.metrics.sharpe / n, dd += o.metrics.maxDrawdown / n, turn += o.turnoverYear / n;
        worst = std::min(worst, o.metrics.sharpe);
        if (k > 0 && o.metrics.sharpe > m.out[c][0].metrics.sharpe + 1e-9) ++beats[k];
      }
      std::printf("  %-40s %7.1f%% %8.2f %8.2f %7.1f%% %7.0fx", names[k], 100 * ret, meanSharpe[k], worst, 100 * dd, turn);
      if (k > 0) std::printf(" %3d/%-2zu", beats[k], set.size());
      std::printf("\n");
    }
    double units = 0, rate = 0, expo = 0, share = 0;
    for (const auto& m : set)
      units += static_cast<double>(m.out[c][1].units) / n, rate += m.out[c][1].rate / n, expo += m.out[c][1].exposure / n,
          share += m.out[c][2].networkShare / n;
    std::printf("  network: %.1f local units, mean trade rate %.2f, mean net exposure %+.2f; the overlay held it %.0f%% of days\n", units, rate,
                expo, 100 * share);
    pass.push_back(set.size() == 1 ? meanSharpe[2] >= meanSharpe[0] : meanSharpe[2] > meanSharpe[0] && beats[2] >= 4);
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
      std::printf("Gate on %s, %s: %s\n", set, kCosts[c].name, pass[c] ? "pass" : "FAIL");
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
    std::printf("Gate on %s, %s: %s\n", csv.c_str(), kCosts[1].name, uk[1] ? "pass" : "FAIL");
    gate = gate && uk[1];
  } else if (markets == 0) {
    std::fprintf(stderr, "overlay_study 0 needs --csv\n");
    return 2;
  }
  std::printf("\nGate 1 %s\n", gate ? "passed" : "failed");
  return 0;
}
