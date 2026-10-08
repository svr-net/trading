#include <cmath>

#include "sat/data/market_data.hpp"
#include "sat/data/synthetic_market.hpp"
#include "test_framework.hpp"

using namespace sat;

TEST(synthetic_market_is_consistent) {
  SyntheticMarketSpec s;
  s.numAssets = 12;
  s.numDates = 400;
  const MarketData d = generateSyntheticMarket(s);
  CHECK(d.numDates() == 400 && d.numAssets() == 12);
  CHECK(d.dates.size() == 400 && d.dates[0] == "2018-01-02" && d.dates[1] == "2018-01-03");
  CHECK(d.tickers[0] == "0001.HK");
  for (std::size_t t = 0; t < d.numDates(); ++t)
    for (std::size_t i = 0; i < d.numAssets(); ++i) {
      CHECK(d.low(t, i) <= std::min(d.open(t, i), d.close(t, i)) + 1e-9);
      CHECK(d.high(t, i) >= std::max(d.open(t, i), d.close(t, i)) - 1e-9);
      CHECK(d.vwap(t, i) >= d.low(t, i) - 1e-9 && d.vwap(t, i) <= d.high(t, i) + 1e-9);
      CHECK(d.volume(t, i) >= 0);
    }
  // Regimes are persistent (mean duration about 1 / (1 - 0.985) days).
  std::size_t changes = 0;
  for (std::size_t t = 1; t < d.regime.size(); ++t) changes += d.regime[t] != d.regime[t - 1];
  CHECK(changes > 0 && changes < 30);
  // Same seed, same market.
  const MarketData e = generateSyntheticMarket(s);
  CHECK(e.close(399, 11) == d.close(399, 11));
}

TEST(synthetic_market_rejects_bad_transitions) {
  SyntheticMarketSpec s;
  s.transition = {{0.5, 0.5, 0.1}, {0, 1, 0}, {0, 0, 1}};
  CHECK_THROWS(generateSyntheticMarket(s));
}

TEST(returns_and_forward_returns) {
  SyntheticMarketSpec s;
  s.numAssets = 3;
  s.numDates = 10;
  const MarketData d = generateSyntheticMarket(s);
  const Panel r = d.returns(), f = d.forwardReturns(2);
  CHECK(std::isnan(r(0, 0)));
  CHECK_NEAR(r(5, 1), d.close(5, 1) / d.close(4, 1) - 1, 1e-14);
  CHECK_NEAR(f(3, 2), d.close(5, 2) / d.close(3, 2) - 1, 1e-14);
  CHECK(std::isnan(f(8, 0)));
}

TEST(csv_round_trip_and_gaps) {
  SyntheticMarketSpec s;
  s.numAssets = 4;
  s.numDates = 30;
  const MarketData d = generateSyntheticMarket(s);
  const MarketData e = parseCsv(toCsv(d));
  CHECK(e.numDates() == 30 && e.numAssets() == 4);
  CHECK_NEAR(e.close(17, 2), d.close(17, 2), 1e-6 * d.close(17, 2));
  CHECK_NEAR(e.vwap(3, 1), d.vwap(3, 1), 1e-6 * d.vwap(3, 1));

  // Gaps are filled forward with zero volume; columns in any order, no vwap.
  const std::string csv =
      "Ticker,Date,Close,Open,High,Low,Volume\n"
      "A,2024-01-02,10,9,11,8,100\nB,2024-01-02,20,20,21,19,50\n"
      "A,2024-01-03,11,10,12,10,120\n"
      "A,2024-01-04,12,11,12,11,90\nB,2024-01-04,22,21,23,20,60\n";
  const MarketData g = parseCsv(csv);
  CHECK(g.numDates() == 3 && g.numAssets() == 2 && g.tickers[1] == "B");
  CHECK(g.close(1, 1) == 20 && g.volume(1, 1) == 0);
  CHECK_NEAR(g.vwap(0, 0), (11.0 + 8.0 + 10.0) / 3.0, 1e-12);
  CHECK_THROWS(parseCsv("date,ticker,open,high,low\n"));
  CHECK_THROWS(parseCsv("date,ticker,open,high,low,close,volume\nx,A,1,1,1,oops,1\n"));
  CHECK_THROWS(parseCsv("date,ticker,open,high,low,close,volume\nx,A,1,1,1,-1,1\ny,A,1,1,1,1,1\n"));
}

TEST(slice_keeps_alignment) {
  SyntheticMarketSpec s;
  s.numAssets = 3;
  s.numDates = 50;
  const MarketData d = generateSyntheticMarket(s);
  const MarketData q = d.slice(10, 20);
  CHECK(q.numDates() == 10 && q.dates[0] == d.dates[10] && q.close(4, 2) == d.close(14, 2) && q.regime[0] == d.regime[10]);
}
