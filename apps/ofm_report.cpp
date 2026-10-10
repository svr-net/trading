// Runs the model and the backtest and prints the report (results only, no prices).
//
//   ofm_report [--csv stocks.csv] [--series futures.csv] [--engine reference|emulated]
//              [--buy-bps 60] [--sell-bps 10] [--futures-bps 1] [--json out.json]
//
// Without --csv it runs on the synthetic sample market.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "ofm/report.hpp"

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
    ofm::Costs costs;
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
      else if (a == "--buy-bps") costs.buyBps = std::stod(next());
      else if (a == "--sell-bps") costs.sellBps = std::stod(next());
      else if (a == "--futures-bps") costs.futuresBps = std::stod(next());
      else throw std::invalid_argument("unknown option " + a);
    }
    ofm::Market m;
    std::map<std::string, std::vector<double>> s;
    if (csv.empty()) {
      auto syn = ofm::syntheticMarket();
      m = std::move(syn.market), s = std::move(syn.series);
    } else {
      m = ofm::parseMarketCsv(readFile(csv));
      if (!series.empty()) s = ofm::parseSeriesCsv(readFile(series), m.dates);
    }
    const ofm::Forecast f = ofm::runModel(m, engine);
    const ofm::MarketStructure ms = ofm::marketStructure(m, f.horizons, f.familyReturns);
    const ofm::Backtest bt = ofm::backtest(m, f, costs, s, &ms);
    std::cout << ofm::reportText(m, f, bt, costs, &ms);
    if (!json.empty()) std::ofstream(json) << ofm::reportJson(m, f, bt, costs, &ms);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
