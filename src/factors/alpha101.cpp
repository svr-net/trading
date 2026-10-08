#include "sat/factors/alpha101.hpp"

#include <algorithm>
#include <stdexcept>

#include "sat/factors/operators.hpp"

namespace sat {

using namespace ops;

AlphaInputs::AlphaInputs(const MarketData& data)
    : open_(data.open), high_(data.high), low_(data.low), close_(data.close), volume_(data.volume), vwap_(data.vwap),
      returns_(data.returns()) {}

const Panel& AlphaInputs::adv(std::size_t d) const {
  auto it = adv_.find(d);
  if (it == adv_.end()) it = adv_.emplace(d, tsMean(close_ * volume_, d)).first;
  return it->second;
}

const std::vector<int>& paperAlphaIds() {
  static const std::vector<int> ids = {1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 17, 20, 29, 33, 34, 35, 40, 41, 44, 62, 65, 81};
  return ids;
}

std::string alphaFormula(int id) {
  switch (id) {
    case 1: return "rank(ts_argmax(signedpower(returns < 0 ? stddev(returns, 20) : close, 2), 5)) - 0.5";
    case 2: return "-corr(rank(delta(log(volume), 2)), rank((close - open) / open), 6)";
    case 3: return "-corr(rank(open), rank(volume), 10)";
    case 4: return "-ts_rank(rank(low), 9)";
    case 5: return "rank(open - sum(vwap, 10) / 10) * -abs(rank(close - vwap))";
    case 6: return "-corr(open, volume, 10)";
    case 7: return "adv20 < volume ? -ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7)) : -1";
    case 9: return "0 < ts_min(delta(close, 1), 5) ? delta(close, 1) : (ts_max(delta(close, 1), 5) < 0 ? delta(close, 1) : -delta(close, 1))";
    case 12: return "sign(delta(volume, 1)) * -delta(close, 1)";
    case 13: return "-rank(covariance(rank(close), rank(volume), 5))";
    case 14: return "-rank(delta(returns, 3)) * corr(open, volume, 10)";
    case 17: return "-rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5))";
    case 20: return "-rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))";
    case 29: return "ts_min(rank(rank(scale(log(ts_min(rank(rank(-rank(delta(close - 1, 5)))), 2))))), 5) + ts_rank(delay(-returns, 6), 5)";
    case 33: return "rank(-(1 - open / close))";
    case 34: return "rank((1 - rank(stddev(returns, 2) / stddev(returns, 5))) + (1 - rank(delta(close, 1))))";
    case 35: return "ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))";
    case 40: return "-rank(stddev(high, 10)) * corr(high, volume, 10)";
    case 41: return "(high * low)^0.5 - vwap";
    case 44: return "-corr(high, rank(volume), 5)";
    case 62: return "-(rank(corr(vwap, sum(adv20, 22), 10)) < rank((rank(open) + rank(open)) < (rank((high + low) / 2) + rank(high))))";
    case 65: return "-(rank(corr(open * 0.00817205 + vwap * (1 - 0.00817205), sum(adv60, 9), 6)) < rank(open - ts_min(open, 14)))";
    case 81: return "-(rank(log(product(rank(rank(corr(vwap, sum(adv10, 50), 8))^4), 15))) < rank(corr(rank(vwap), rank(volume), 5)))";
    default: throw std::invalid_argument("alpha " + std::to_string(id) + " is not one of the paper's factors");
  }
}

std::size_t alphaLookback(int id) {
  switch (id) {
    case 1: return 24;
    case 2: return 8;
    case 3: return 10;
    case 4: return 9;
    case 5: return 10;
    case 6: return 10;
    case 7: return 67;
    case 9: return 6;
    case 12: return 2;
    case 13: return 5;
    case 14: return 10;
    case 17: return 25;
    case 20: return 2;
    case 29: return 12;
    case 33: return 1;
    case 34: return 6;
    case 35: return 33;
    case 40: return 10;
    case 41: return 1;
    case 44: return 5;
    case 62: return 51;
    case 65: return 74;
    case 81: return 81;
    default: throw std::invalid_argument("alpha " + std::to_string(id) + " is not one of the paper's factors");
  }
}

Panel computeAlpha(int id, const AlphaInputs& in) {
  const Panel& open = in.open();
  const Panel& high = in.high();
  const Panel& low = in.low();
  const Panel& close = in.close();
  const Panel& volume = in.volume();
  const Panel& vwap = in.vwap();
  const Panel& returns = in.returns();
  switch (id) {
    case 1:
      return rank(tsArgMax(signedPower(where(lessThan(returns, 0.0), tsStddev(returns, 20), close), 2.0), 5)) - 0.5;
    case 2:
      return -tsCorr(rank(delta(log(volume), 2)), rank((close - open) / open), 6);
    case 3:
      return -tsCorr(rank(open), rank(volume), 10);
    case 4:
      return -tsRank(rank(low), 9);
    case 5:
      return rank(open - tsSum(vwap, 10) / 10.0) * -abs(rank(close - vwap));
    case 6:
      return -tsCorr(open, volume, 10);
    case 7: {
      const Panel dc = delta(close, 7);
      return where(lessThan(in.adv(20), volume), -tsRank(abs(dc), 60) * sign(dc), -1.0);
    }
    case 9: {
      const Panel d1 = delta(close, 1);
      return where(lessThan(-tsMin(d1, 5), 0.0) /* 0 < ts_min */, d1, where(lessThan(tsMax(d1, 5), 0.0), d1, -d1));
    }
    case 12:
      return sign(delta(volume, 1)) * -delta(close, 1);
    case 13:
      return -rank(tsCov(rank(close), rank(volume), 5));
    case 14:
      return -rank(delta(returns, 3)) * tsCorr(open, volume, 10);
    case 17:
      return -rank(tsRank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(tsRank(volume / in.adv(20), 5));
    case 20:
      return -rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1));
    case 29: {
      // product(x, 1) and sum(x, 1) are the identity; min(x, 5) is the time-series minimum.
      const Panel inner = tsMin(rank(rank(-rank(delta(close - 1.0, 5)))), 2);
      return tsMin(rank(rank(scale(log(inner)))), 5) + tsRank(delay(-returns, 6), 5);
    }
    case 33:
      return rank(-(1.0 - open / close));
    case 34:
      return rank((1.0 - rank(tsStddev(returns, 2) / tsStddev(returns, 5))) + (1.0 - rank(delta(close, 1))));
    case 35:
      return tsRank(volume, 32) * (1.0 - tsRank(close + high - low, 16)) * (1.0 - tsRank(returns, 32));
    case 40:
      return -rank(tsStddev(high, 10)) * tsCorr(high, volume, 10);
    case 41:
      return power(high * low, 0.5) - vwap;
    case 44:
      return -tsCorr(high, rank(volume), 5);
    case 62: {
      const Panel lhs = rank(tsCorr(vwap, tsSum(in.adv(20), 22), 10));
      const Panel rhs = rank(lessThan(rank(open) + rank(open), rank((high + low) / 2.0) + rank(high)));
      return -lessThan(lhs, rhs);
    }
    case 65: {
      const double w = 0.00817205;
      const Panel lhs = rank(tsCorr(open * w + vwap * (1.0 - w), tsSum(in.adv(60), 9), 6));
      return -lessThan(lhs, rank(open - tsMin(open, 14)));
    }
    case 81: {
      const Panel lhs = rank(log(tsProduct(rank(power(rank(tsCorr(vwap, tsSum(in.adv(10), 50), 8)), 4.0)), 15)));
      return -lessThan(lhs, rank(tsCorr(rank(vwap), rank(volume), 5)));
    }
    default:
      throw std::invalid_argument("alpha " + std::to_string(id) + " is not one of the paper's factors");
  }
}

}  // namespace sat
