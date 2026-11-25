// ofm_day: day trades only (bought at the open, sold at the close, nothing held overnight), backtested on a
// market's daily bars, before and after the broker's costs. Research: which intraday signal, if any, beats the
// costs of a round trip every day.
//
//   ofm_day --csv data/stocks.csv --market US [--k 5,10] [--min-value 1000000] [--stamp-free aim.txt]
//
// Each day t, the k shares a signal ranks best are bought at t's open and sold at t's close: the return is
// C/O - 1 less the round trip (commission both ways, stamp duty on buying in the UK; the auctions at one price,
// a gap signal's buy at half the spread over the open). The
// signal sees only what is known before the trade: the bars up to t-1, and for the gap signals t's open (an order
// placed just after the open, at the open plus half the spread).
// Signals (the literature):
//   rev1   yesterday's biggest losers (Jegadeesh 1990, Lehmann 1990: short-term reversal)
//   rev5   the last week's biggest losers
//   gapdn  the biggest gaps down at the open (Berkman, Koch, Tuttle and Zhang 2012: opening gaps reverse)
//   gapup  the biggest gaps up (the gap's continuation)
//   imom   the best intraday returns (C/O) over the last 20 sessions (Lou, Polk and Skouras 2019: intraday
//          returns continue, overnight returns reverse)
//   all    every liquid share, equally (the market's open-to-close; the baseline)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& m) {
  std::fprintf(stderr, "ofm_day: %s\n", m.c_str());
  std::exit(2);
}

struct Panel {
  std::vector<std::string> days, tickers;
  std::size_t T = 0, N = 0;
  std::vector<double> O, H, L, C, V;  // [t * N + i], NaN: no bar
};

Panel load(const std::string& path) {
  std::ifstream in(path);
  if (!in) die("cannot read " + path);
  std::string line;
  std::getline(in, line);
  struct Row {
    std::string d, k;
    double o, h, l, c, v;
  };
  std::vector<Row> rows;
  std::set<std::string> ds, ks;
  while (std::getline(in, line)) {
    std::stringstream s(line);
    Row r;
    std::string f[7];
    for (auto& x : f) std::getline(s, x, ',');
    if (f[0].empty() || f[1].empty()) continue;
    r.d = f[0], r.k = f[1];
    r.o = std::atof(f[2].c_str()), r.h = std::atof(f[3].c_str()), r.l = std::atof(f[4].c_str()), r.c = std::atof(f[5].c_str()), r.v = std::atof(f[6].c_str());
    if (!(r.o > 0 && r.c > 0 && r.h > 0 && r.l > 0)) continue;
    ds.insert(r.d), ks.insert(r.k);
    rows.push_back(r);
  }
  Panel p;
  p.days.assign(ds.begin(), ds.end()), p.tickers.assign(ks.begin(), ks.end());
  p.T = p.days.size(), p.N = p.tickers.size();
  std::map<std::string, std::size_t> di, ki;
  for (std::size_t i = 0; i < p.T; ++i) di[p.days[i]] = i;
  for (std::size_t i = 0; i < p.N; ++i) ki[p.tickers[i]] = i;
  const double nan = NAN;
  for (auto* v : {&p.O, &p.H, &p.L, &p.C, &p.V}) v->assign(p.T * p.N, nan);
  for (const auto& r : rows) {
    const std::size_t q = di[r.d] * p.N + ki[r.k];
    p.O[q] = r.o, p.H[q] = r.h, p.L[q] = r.l, p.C[q] = r.c, p.V[q] = r.v;
  }
  return p;
}

// The round trip's costs as a share of the value (IBKR, about 5,000 a trade), the spread both ways apart.
double roundTrip(const std::string& m, const std::string& ticker, double price, double value, const std::set<std::string>& stampFree) {
  if (m == "US") {
    const double sh = std::max(1.0, std::floor(value / price));
    const double c = std::max(0.35, 0.0035 * sh) + 0.0005 * sh;  // tiered, with the exchange's and clearing fees
    return (2 * c + 0.0000278 * value + std::min(8.30, 0.000166 * sh)) / value;
  }
  if (m == "UK") {
    const double c = std::max(1.0, 0.0005 * value) + std::max(0.10, 0.000045 * value) + 0.06;
    return 2 * c / value + (stampFree.count(ticker) ? 0.0 : 0.005);
  }
  if (m == "APAC") return 2 * std::max(6.0, 0.0008 * value) / value;
  return 2 * (std::max(1.25, 0.0005 * value) + 0.0002 * value) / value;  // the euro area's exchanges, tiered (a day trade pays no FTT)
}

struct Result {
  std::vector<double> gross, net;  // each day's book return
  std::size_t trades = 0;
};

double mean(const std::vector<double>& x, std::size_t a, std::size_t b) {
  if (b <= a) return 0;
  return std::accumulate(x.begin() + long(a), x.begin() + long(b), 0.0) / double(b - a);
}

void report(const std::string& name, std::size_t k, const Result& r) {
  const auto stats = [](const std::vector<double>& x, std::size_t a, std::size_t b, double& ann, double& sh, double& dd, double& hit) {
    const double m = mean(x, a, b);
    double v = 0, eq = 0, peak = 0;
    std::size_t up = 0;
    dd = 0;
    for (std::size_t i = a; i < b; ++i) {
      v += (x[i] - m) * (x[i] - m), up += x[i] > 0;
      eq += std::log1p(x[i]), peak = std::max(peak, eq), dd = std::max(dd, 1 - std::exp(eq - peak));
    }
    const double sd = b > a + 1 ? std::sqrt(v / double(b - a - 1)) : 0;
    ann = std::expm1(252 * std::log1p(m)), sh = sd > 0 ? m / sd * std::sqrt(252.0) : 0, hit = b > a ? double(up) / double(b - a) : 0;
  };
  const std::size_t n = r.net.size(), h = n / 2;
  double a, s, d, w, ga, gs, gd, gw, a1, s1, d1, w1, a2, s2, d2, w2;
  stats(r.gross, 0, n, ga, gs, gd, gw);
  stats(r.net, 0, n, a, s, d, w);
  stats(r.net, 0, h, a1, s1, d1, w1);
  stats(r.net, h, n, a2, s2, d2, w2);
  std::printf("day: %-6s k=%-2zu %5zu days: before costs %+7.2f bp a day (Sharpe %5.2f) | after costs %+7.2f bp a day, %+6.1f%% a year, Sharpe %5.2f, max drawdown %5.1f%%, %2.0f%% of days up | halves: Sharpe %5.2f, %5.2f\n",
              name.c_str(), k, n, 1e4 * mean(r.gross, 0, n), gs, 1e4 * mean(r.net, 0, n), 100 * a, s, 100 * d, 100 * w, s1, s2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string csv, market, stampPath;
  std::vector<std::size_t> ks{5, 10};
  double minValue = 1e6, value = 5000, spreadBp = -1;
  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    const auto next = [&]() -> std::string { if (a + 1 >= argc) die(s + ": a value"); return argv[++a]; };
    if (s == "--csv") csv = next();
    else if (s == "--market") market = next();
    else if (s == "--stamp-free") stampPath = next();
    else if (s == "--min-value") minValue = std::atof(next().c_str());
    else if (s == "--value") value = std::atof(next().c_str());
    else if (s == "--spread-bp") spreadBp = std::atof(next().c_str());
    else if (s == "--k") {
      ks.clear();
      std::stringstream ss(next());
      std::string x;
      while (std::getline(ss, x, ',')) ks.push_back(std::size_t(std::atoi(x.c_str())));
    } else die("unknown argument " + s);
  }
  if (csv.empty() || market.empty()) die("usage: ofm_day --csv stocks.csv --market US [--k 5,10] [--min-value 1e6] [--stamp-free aim.txt] [--spread-bp N]");
  std::set<std::string> stampFree;
  if (!stampPath.empty()) {
    std::ifstream f(stampPath);
    std::string l;
    while (std::getline(f, l)) if (!l.empty() && l[0] != '#') stampFree.insert(l.substr(0, l.find_first_of(" ,\t")));
  }
  const Panel p = load(csv);
  const std::size_t T = p.T, N = p.N;
  std::printf("day: %s, %zu shares, %zu days (%s .. %s), at least %.0f traded a day (the 20-day median), %.0f a trade\n", market.c_str(), N, T,
              T ? p.days.front().c_str() : "-", T ? p.days.back().c_str() : "-", minValue, value);
  // The spread a side: the Corwin and Schultz (2012) estimate from the share's last 20 highs and lows (at least
  // the market's floor: 2 bp in the US, 5 bp elsewhere), or --spread-bp for every share.
  const double floorBp = market == "US" ? 2 : 5;
  std::vector<double> halfSpread(T * N, NAN), liq(T * N, 0);
  for (std::size_t i = 0; i < N; ++i) {
    std::vector<double> vals, cs;
    for (std::size_t t = 1; t < T; ++t) {
      const std::size_t q = t * N + i, q0 = (t - 1) * N + i;
      if (std::isfinite(p.C[q]) && std::isfinite(p.V[q])) vals.push_back(p.C[q] * p.V[q]);
      if (vals.size() > 20) vals.erase(vals.begin());
      if (std::isfinite(p.H[q]) && std::isfinite(p.H[q0])) {
        const double b = std::pow(std::log(p.H[q] / p.L[q]), 2) + std::pow(std::log(p.H[q0] / p.L[q0]), 2);
        const double g = std::pow(std::log(std::max(p.H[q], p.H[q0]) / std::min(p.L[q], p.L[q0])), 2);
        const double al = (std::sqrt(2 * b) - std::sqrt(b)) / (3 - 2 * std::sqrt(2.0)) - std::sqrt(g / (3 - 2 * std::sqrt(2.0)));
        const double sp = 2 * (std::exp(al) - 1) / (1 + std::exp(al));
        cs.push_back(std::max(0.0, sp));
        if (cs.size() > 20) cs.erase(cs.begin());
      }
      // known after t's close, used for t+1's trade
      if (t + 1 < T) {
        std::vector<double> sv = vals;
        if (sv.size() >= 10) {
          std::nth_element(sv.begin(), sv.begin() + long(sv.size() / 2), sv.end());
          liq[(t + 1) * N + i] = sv[sv.size() / 2];
        }
        const double est = cs.empty() ? 0 : std::accumulate(cs.begin(), cs.end(), 0.0) / double(cs.size()) / 2;
        halfSpread[(t + 1) * N + i] = spreadBp >= 0 ? spreadBp / 1e4 : std::max(floorBp / 1e4, std::min(0.02, est));
      }
    }
  }
  const std::vector<std::string> sigs{"all", "rev1", "rev5", "gapdn", "gapup", "imom"};
  for (const std::string& sig : sigs)
    for (std::size_t k : ks) {
      if (sig == "all" && k != ks.front()) continue;
      Result r;
      for (std::size_t t = 25; t < T; ++t) {
        std::vector<std::pair<double, std::size_t>> c;
        for (std::size_t i = 0; i < N; ++i) {
          const std::size_t q = t * N + i, q1 = (t - 1) * N + i, q2 = (t - 2) * N + i;
          if (!(std::isfinite(p.O[q]) && std::isfinite(p.C[q]) && std::isfinite(p.C[q1]) && std::isfinite(p.C[q2]))) continue;
          if (liq[q] < minValue || p.C[q1] < 1) continue;
          const double gap = p.O[q] / p.C[q1] - 1;
          if (std::fabs(gap) > 0.3) continue;  // a split or a bad bar
          double score = NAN;
          if (sig == "all") score = 0;
          else if (sig == "rev1") score = -(p.C[q1] / p.C[q2] - 1);
          else if (sig == "rev5") {
            const std::size_t q6 = (t - 6) * N + i;
            if (std::isfinite(p.C[q6])) score = -(p.C[q1] / p.C[q6] - 1);
          } else if (sig == "gapdn") score = -gap;
          else if (sig == "gapup") score = gap;
          else if (sig == "imom") {
            double s = 0;
            std::size_t n = 0;
            for (std::size_t u = t - 20; u < t; ++u) {
              const std::size_t qu = u * N + i;
              if (std::isfinite(p.O[qu]) && std::isfinite(p.C[qu])) s += std::log(p.C[qu] / p.O[qu]), ++n;
            }
            if (n >= 15) score = s / double(n);
          }
          if (std::isfinite(score)) c.emplace_back(score, i);
        }
        if (c.empty()) {
          r.gross.push_back(0), r.net.push_back(0);
          continue;
        }
        std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        const std::size_t n = sig == "all" ? c.size() : std::min(k, c.size());
        double g = 0, nt = 0;
        for (std::size_t j = 0; j < n; ++j) {
          const std::size_t i = c[j].second, q = t * N + i;
          // the opening and closing auctions trade at one price: no spread for a signal known before the open (a
          // market-on-open buy, a market-on-close sale); a gap signal buys just after the open, at half the spread
          // over it, and sells in the closing auction
          const bool gapSig = sig == "gapdn" || sig == "gapup";
          const double hs = gapSig ? halfSpread[q] : 0.0;
          const double gr = p.C[q] / p.O[q] - 1;
          const double buy = p.O[q] * (1 + hs), sell = p.C[q];
          g += gr;
          nt += sell / buy - 1 - roundTrip(market, p.tickers[i], p.O[q], value, stampFree);
        }
        r.gross.push_back(g / double(n)), r.net.push_back(nt / double(n)), r.trades += n;
      }
      report(sig, sig == "all" ? 0 : k, r);
    }
  return 0;
}
