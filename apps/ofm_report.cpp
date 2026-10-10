// Runs the model and the backtest and prints the report (results only, no prices).
//
//   ofm_report [--csv stocks.csv] [--series futures.csv] [--engine reference|emulated]
//              [--cost-pct 0.10] [--stamp-pct 0] [--futures-bps 1] [--json out.json] [--universe-index-hedge]
//
// --cost-pct: transaction cost per side in percent of the value traded (commission and half the
// spread; market standard 0.10). --stamp-pct: stamp duty on purchases in percent (0: CFDs, spread
// bets and AIM shares; 0.50 for UK shares bought directly). --buy-bps / --sell-bps set the two sides directly instead.
//
// --universe-index-hedge adds the stocks' own equal-weight index as a hedge series, a stand-in for
// an index future on these stocks (FTSE 250 futures for a UK universe).
//
// Without --csv it runs on the synthetic sample market.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "ofm/report.hpp"
#include "ofm/topk.hpp"

namespace {
std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string csv, series, engine = "reference", json;
    bool universeHedge = false;
    double costPct = ofm::kDealingPct, stampPct = ofm::kStampPct, buyBps = -1, sellBps = -1, futBps = 1;
    for (int k = 1; k < argc; ++k) {
      const std::string a = argv[k];
      auto next = [&]() -> std::string {
        if (k + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++k];
      };
      if (a == "--csv") csv = next();
      else if (a == "--series") series = next();
      else if (a == "--engine") engine = next();
      else if (a == "--json") json = next();
      else if (a == "--cost-pct") costPct = std::stod(next());
      else if (a == "--stamp-pct") stampPct = std::stod(next());
      else if (a == "--buy-bps") buyBps = std::stod(next());
      else if (a == "--sell-bps") sellBps = std::stod(next());
      else if (a == "--futures-bps") futBps = std::stod(next());
      else if (a == "--universe-index-hedge") universeHedge = true;
      else throw std::invalid_argument("unknown option " + a);
    }
    ofm::Costs costs = ofm::Costs::fromPercent(costPct, stampPct, futBps);
    if (buyBps >= 0) costs.buyBps = buyBps;
    if (sellBps >= 0) costs.sellBps = sellBps;
    ofm::Market m;
    std::map<std::string, std::vector<double>> s;
    if (csv.empty()) {
      auto syn = ofm::syntheticMarket();
      m = std::move(syn.market), s = std::move(syn.series);
    } else {
      m = ofm::parseMarketCsv(readFile(csv));
      if (!series.empty()) s = ofm::parseSeriesCsv(readFile(series), m.dates);
    }
    if (universeHedge) s[ofm::kUniverseIndex] = ofm::equalWeightIndex(m);
    const ofm::Forecast f = ofm::runModel(m, engine);
    const ofm::MarketStructure ms = ofm::marketStructure(m, f.horizons, f.familyReturns);
    const ofm::Backtest bt = ofm::backtest(m, f, costs, s, &ms);
    const auto top = ofm::topKBacktests(m, f, costs, bt, 10);
    std::cout << ofm::reportText(m, f, bt, costs, &ms) << ofm::topKText(m, top, bt);
    if (!json.empty()) {
      std::string j = ofm::reportJson(m, f, bt, costs, &ms);
      j.pop_back();  // add the rolling top-10 portfolios to the report object
      std::ofstream(json) << j << ",\"top10\":" << ofm::topKJson(m, top, bt) << "}";
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
