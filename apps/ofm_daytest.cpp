// Daily top-10 test against the real bars, compared with the model portfolio (research only).
//
//   ofm_daytest --csv stocks.csv [--series futures.csv] [--cost-pct 0.10] [--stamp-pct 0]
//
// Each close t, the 10 stocks with the best expected return (the model's forecast, known at the
// close) that have a valid bar on day t+1 are bought at its open, one tenth of the capital each,
// and sold the same day: at the stop-loss, the take-profit or the close, read from that day's
// high, low and close. Every trade pays the purchase and the sale cost; the book compounds daily.
//
// Stops and take-profits are multiples of the stock's own open-to-close volatility (all its days
// before t). The multiples are learnt walk-forward: each day the pair whose record (Sharpe ratio
// of the 10-trade book over all earlier days, worst-case fills) is best is used; "none" is one of
// the choices. Daily bars cannot tell whether the stop or the take-profit was touched first when
// both were; the main result assumes the stop (worst case), the optimistic one the take-profit.
//
// Neural levels: a small network (shared by all stocks, fed each stock's own features at the
// close) predicts the net log return (the compounding objective) of each of the 36 stop / take
// pairs for that stock the next day; each stock trades with the pair it rates best. It learns
// online, walk-forward: after each day, every eligible stock's 36 actual outcomes on that day's bar
// (worst-case fills, after costs) are training examples. It never sees a day before trading it.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ofm/data.hpp"
#include "ofm/model.hpp"
#include "ofm/report.hpp"
#include "ofm/structure.hpp"
#include "ofm/trade.hpp"

namespace {

std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

constexpr double kInf = std::numeric_limits<double>::infinity();
const std::vector<double> kGrid = {0.5, 1.0, 1.5, 2.0, 3.0, kInf};  // multiples of the volatility

struct Bar { double o, h, l, c, v; };

// One trade's net return: bought at the open, sold at the stop, the take-profit or the close.
// worstCase: when both levels were touched, the stop is taken to come first.
double tradeReturn(const Bar& b, double sig, double ks, double kt, double buy, double sell, bool worstCase, int* exit) {
  const double stop = std::isfinite(ks) ? b.o * std::exp(-ks * sig) : 0.0;
  const double take = std::isfinite(kt) ? b.o * std::exp(kt * sig) : kInf;
  const bool hitStop = b.l <= stop, hitTake = b.h >= take;
  double px = b.c;
  *exit = 0;
  if (hitStop && (worstCase || !hitTake)) px = stop, *exit = 1;
  else if (hitTake) px = take, *exit = 2;
  return (px / b.o) * (1 - sell) / (1 + buy) - 1;
}

struct Stats {
  double s = 0, s2 = 0, n = 0;
  void add(double x) { s += x, s2 += x * x, n += 1; }
  double sharpe() const {
    if (n < 2) return -kInf;
    const double m = s / n, v = (s2 - n * m * m) / (n - 1);
    return v > 0 ? m / std::sqrt(v) : (m > 0 ? kInf : -kInf);
  }
};


// A two-hidden-layer perceptron (tanh), trained online by Adam on squared error.
struct Mlp {
  std::size_t nIn, nH, nOut;
  std::vector<double> w1, b1, w2, b2, w3, b3;
  std::vector<std::vector<double>*> params;
  std::vector<std::vector<double>> m, v;
  double step = 0;
  Mlp(std::size_t in, std::size_t h, std::size_t out) : nIn(in), nH(h), nOut(out) {
    std::mt19937 rng(7);
    auto init = [&](std::vector<double>& w, std::size_t fanIn, std::size_t n) {
      std::normal_distribution<double> g(0.0, 1.0 / std::sqrt(double(fanIn)));
      w.resize(n);
      for (double& x : w) x = g(rng);
    };
    init(w1, in, in * h), b1.assign(h, 0.0), init(w2, h, h * h), b2.assign(h, 0.0), init(w3, h, h * out), b3.assign(out, 0.0);
    params = {&w1, &b1, &w2, &b2, &w3, &b3};
    for (auto* p : params) m.emplace_back(p->size(), 0.0), v.emplace_back(p->size(), 0.0);
  }
  void forward(const std::vector<double>& x, std::vector<double>& h1, std::vector<double>& h2, std::vector<double>& y) const {
    h1.assign(nH, 0.0), h2.assign(nH, 0.0), y.assign(nOut, 0.0);
    for (std::size_t j = 0; j < nH; ++j) {
      double a = b1[j];
      for (std::size_t i = 0; i < nIn; ++i) a += w1[j * nIn + i] * x[i];
      h1[j] = std::tanh(a);
    }
    for (std::size_t j = 0; j < nH; ++j) {
      double a = b2[j];
      for (std::size_t i = 0; i < nH; ++i) a += w2[j * nH + i] * h1[i];
      h2[j] = std::tanh(a);
    }
    for (std::size_t k = 0; k < nOut; ++k) {
      double a = b3[k];
      for (std::size_t i = 0; i < nH; ++i) a += w3[k * nH + i] * h2[i];
      y[k] = a;
    }
  }
  std::vector<double> predict(const std::vector<double>& x) const {
    std::vector<double> h1, h2, y;
    forward(x, h1, h2, y);
    return y;
  }
  // One Adam step on a batch (mean squared error over all outputs).
  void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y) {
    if (X.empty()) return;
    std::vector<std::vector<double>> g;
    for (auto* p : params) g.emplace_back(p->size(), 0.0);
    std::vector<double> h1, h2, y, d3(nOut), d2(nH), d1(nH);
    for (std::size_t n = 0; n < X.size(); ++n) {
      forward(X[n], h1, h2, y);
      for (std::size_t k = 0; k < nOut; ++k) d3[k] = (y[k] - Y[n][k]) / double(nOut);
      for (std::size_t k = 0; k < nOut; ++k) {
        g[5][k] += d3[k];
        for (std::size_t i = 0; i < nH; ++i) g[4][k * nH + i] += d3[k] * h2[i];
      }
      for (std::size_t i = 0; i < nH; ++i) {
        double a = 0;
        for (std::size_t k = 0; k < nOut; ++k) a += w3[k * nH + i] * d3[k];
        d2[i] = a * (1 - h2[i] * h2[i]);
      }
      for (std::size_t j = 0; j < nH; ++j) {
        g[3][j] += d2[j];
        for (std::size_t i = 0; i < nH; ++i) g[2][j * nH + i] += d2[j] * h1[i];
      }
      for (std::size_t i = 0; i < nH; ++i) {
        double a = 0;
        for (std::size_t j = 0; j < nH; ++j) a += w2[j * nH + i] * d2[j];
        d1[i] = a * (1 - h1[i] * h1[i]);
      }
      for (std::size_t j = 0; j < nH; ++j) {
        g[1][j] += d1[j];
        for (std::size_t i = 0; i < nIn; ++i) g[0][j * nIn + i] += d1[j] * X[n][i];
      }
    }
    const double lr = 1e-3, b1m = 0.9, b2m = 0.999, eps = 1e-8;
    step += 1;
    for (std::size_t q = 0; q < params.size(); ++q)
      for (std::size_t i = 0; i < params[q]->size(); ++i) {
        const double gi = g[q][i] / double(X.size());
        m[q][i] = b1m * m[q][i] + (1 - b1m) * gi, v[q][i] = b2m * v[q][i] + (1 - b2m) * gi * gi;
        const double mh = m[q][i] / (1 - std::pow(b1m, step)), vh = v[q][i] / (1 - std::pow(b2m, step));
        (*params[q])[i] -= lr * mh / (std::sqrt(vh) + eps);
      }
  }
};

// Running standardisation of the features (mean and variance of everything seen so far).
struct Scaler {
  std::vector<Stats> s;
  explicit Scaler(std::size_t n) : s(n) {}
  void add(const std::vector<double>& x) { for (std::size_t i = 0; i < x.size(); ++i) s[i].add(x[i]); }
  std::vector<double> apply(std::vector<double> x) const {
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (s[i].n < 2) { x[i] = 0; continue; }
      const double mu = s[i].s / s[i].n, sd = std::sqrt(std::max(1e-18, s[i].s2 / s[i].n - mu * mu));
      x[i] = std::clamp((x[i] - mu) / sd, -5.0, 5.0);
    }
    return x;
  }
};

void row(const char* name, const std::vector<double>& r, std::size_t lo, std::size_t hi) {
  const ofm::Metrics m = ofm::metrics(std::vector<double>(r.begin() + lo, r.begin() + hi));
  double w = 1;
  for (std::size_t k = lo; k < hi; ++k) w *= 1 + r[k];
  std::printf("  %-44s %7.1f%% %7.1f%% %7.2f %7.1f%% %9.2fx\n", name, 100 * m.annualReturn, 100 * m.volatility, m.sharpe, 100 * m.maxDrawdown, w);
}

// The mean daily difference a - b, its t-statistic, and the share of days a was ahead.
void compare(const char* name, const std::vector<double>& a, const std::vector<double>& b) {
  Stats d;
  double ahead = 0;
  for (std::size_t k = 0; k < a.size(); ++k) d.add(a[k] - b[k]), ahead += a[k] > b[k];
  const double m = d.s / d.n, v = (d.s2 - d.n * m * m) / (d.n - 1);
  std::printf("  %-44s %+8.3f%% a day, t %+5.2f, ahead on %4.1f%% of days\n", name, 100 * m, m / std::sqrt(v / d.n), 100 * ahead / d.n);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string csv, series;
    double costPct = ofm::kDealingPct, stampPct = ofm::kStampPct;
    for (int k = 1; k < argc; ++k) {
      const std::string a = argv[k];
      auto next = [&]() -> std::string {
        if (k + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++k];
      };
      if (a == "--csv") csv = next();
      else if (a == "--series") series = next();
      else if (a == "--cost-pct") costPct = std::stod(next());
      else if (a == "--stamp-pct") stampPct = std::stod(next());
      else throw std::invalid_argument("unknown option " + a);
    }
    const ofm::Costs costs = ofm::Costs::fromPercent(costPct, stampPct, 1.0);
    const double buy = costs.buyBps * 1e-4, sell = costs.sellBps * 1e-4;
    ofm::Market m;
    std::map<std::string, std::vector<double>> s;
    if (csv.empty()) {
      auto syn = ofm::syntheticMarket();
      m = std::move(syn.market), s = std::move(syn.series);
    } else {
      m = ofm::parseMarketCsv(readFile(csv));
      if (!series.empty()) s = ofm::parseSeriesCsv(readFile(series), m.dates);
    }
    const ofm::Forecast f = ofm::runModel(m, "reference");
    const ofm::MarketStructure ms = ofm::marketStructure(m, f.horizons, f.familyReturns);
    const ofm::Backtest bt = ofm::backtest(m, f, costs, s, &ms);
    const std::size_t T = m.T(), N = m.N(), K = 10, G = kGrid.size();
    if (bt.start == 0 || bt.model.empty()) throw std::runtime_error("no backtest");

    auto barAt = [&](std::size_t d, std::size_t i, Bar* b) {
      *b = {m.open(d, i), m.high(d, i), m.low(d, i), m.close(d, i), m.volume(d, i)};
      return b->o > 0 && b->h > 0 && b->l > 0 && b->c > 0 && b->l <= std::min(b->o, b->c) && b->h >= std::max(b->o, b->c) &&
             std::isfinite(b->v) && b->v > 0;
    };
    // Each stock's open-to-close log-return moments up to the current day.
    std::vector<Stats> vol(N);
    auto addVol = [&](std::size_t t) {
      Bar b;
      for (std::size_t i = 0; i < N; ++i)
        if (barAt(t, i, &b)) vol[i].add(std::log(b.c / b.o));
    };
    for (std::size_t t = 0; t + 1 < bt.start; ++t) addVol(t);

    // Strategies: no stops; learnt pair (worst-case fills); the same pair with optimistic fills.
    std::vector<double> plain, learnt, optimistic, gross, model, market, neural, neuralOpt;
    std::vector<Stats> pairRecord(G * G);
    // Neural levels: features of stock i at the close t (empty if its recent bars are missing).
    constexpr std::size_t F = 10;
    std::vector<Stats> logVolume(N);
    auto sigmaOf = [&](std::size_t i) {
      const double mu = vol[i].s / vol[i].n;
      return std::sqrt(std::max(0.0, vol[i].s2 / vol[i].n - mu * mu));
    };
    auto features = [&](std::size_t t, std::size_t i, double medSig, double mkt) {
      Bar b0, b1, b5;
      const double sig = sigmaOf(i);
      if (t < 5 || !(sig > 0) || !barAt(t, i, &b0) || !barAt(t - 1, i, &b1) || !barAt(t - 5, i, &b5) || logVolume[i].n < 2) return std::vector<double>{};
      const double lvMu = logVolume[i].s / logVolume[i].n, lvSd = std::sqrt(std::max(1e-12, logVolume[i].s2 / logVolume[i].n - lvMu * lvMu));
      return std::vector<double>{f.E(t, i) / sig, std::log(sig / medSig), std::log(b0.c / b0.o) / sig, std::log(b0.c / b1.c) / sig,
                                 std::log(b0.o / b1.c) / sig, b0.h > b0.l ? (b0.c - b0.l) / (b0.h - b0.l) : 0.5, std::log(b0.h / b0.l) / sig,
                                 std::log(b0.c / b5.c) / sig, (std::log(b0.v) - lvMu) / lvSd, mkt / medSig};
    };
    Mlp net(F, 16, G * G);
    Scaler scaler(F);
    std::map<std::string, int> neuralUse;
    std::vector<std::map<std::size_t, int>> perStock(N);  // pairs each stock was given
    std::size_t nnStops = 0, nnTakes = 0, nnTrades = 0;
    Stats nnGain;  // realised log-return gain of the chosen pair over none/none, per trade
    std::size_t days = 0, trades = 0, both = 0, lowIsOpen = 0, stops = 0, takes = 0, full = 0;
    std::map<std::string, int> pairUse;
    for (std::size_t k = 0; k < bt.model.size(); ++k) {
      const std::size_t d = bt.start + k, t = d - 1;
      if (t > 0) addVol(t);  // everything known at the close t
      // The day's 10: best expected return with a valid bar on day d.
      std::vector<std::size_t> order;
      Bar b;
      for (std::size_t i = 0; i < N; ++i)
        if (std::isfinite(f.E(t, i)) && vol[i].n >= 2 && barAt(d, i, &b)) order.push_back(i);
      const std::vector<std::size_t> candidates = order;
      // The day's median volatility and the market's close-to-close move at t (for the features).
      std::vector<double> sigs;
      for (std::size_t i : candidates) sigs.push_back(sigmaOf(i));
      std::sort(sigs.begin(), sigs.end());
      const double medSig = sigs.empty() ? 1.0 : std::max(1e-9, sigs[sigs.size() / 2]);
      double mkt = 0, mktN = 0;
      for (std::size_t i = 0; i < N; ++i) {
        Bar b0, b1;
        if (t > 0 && barAt(t, i, &b0) && barAt(t - 1, i, &b1)) mkt += std::log(b0.c / b1.c), mktN += 1;
      }
      mkt = mktN > 0 ? mkt / mktN : 0.0;
      std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return f.E(t, x) > f.E(t, y); });
      if (order.size() > K) order.resize(K);
      full += order.size() == K;
      // Today's pair: the best record so far (none/none before any record).
      std::size_t best = G * G - 1;
      double bestSharpe = -kInf;
      for (std::size_t p = 0; p < G * G; ++p)
        if (pairRecord[p].sharpe() > bestSharpe + 1e-12) bestSharpe = pairRecord[p].sharpe(), best = p;
      const double ks = kGrid[best / G], kt = kGrid[best % G];
      auto name = [](double x) { return std::isfinite(x) ? std::to_string(x).substr(0, 3) : std::string("none"); };
      ++pairUse[name(ks) + " / " + name(kt)];
      // The trades, against day d's bar.
      double rPlain = 0, rLearnt = 0, rOpt = 0, rGross = 0, rNeural = 0, rNeuralOpt = 0;
      std::vector<double> rPair(G * G, 0.0);
      for (std::size_t i : order) {
        barAt(d, i, &b);
        const double sig = std::sqrt(std::max(0.0, vol[i].s2 / vol[i].n - (vol[i].s / vol[i].n) * (vol[i].s / vol[i].n)));
        int e;
        rPlain += tradeReturn(b, sig, kInf, kInf, buy, sell, true, &e);
        rGross += b.c / b.o - 1;
        rLearnt += tradeReturn(b, sig, ks, kt, buy, sell, true, &e);
        stops += e == 1, takes += e == 2;
        rOpt += tradeReturn(b, sig, ks, kt, buy, sell, false, &e);
        if (std::isfinite(ks) && std::isfinite(kt) && b.l <= b.o * std::exp(-ks * sig) && b.h >= b.o * std::exp(kt * sig)) ++both;
        lowIsOpen += b.l >= b.o;
        for (std::size_t p = 0; p < G * G; ++p) rPair[p] += tradeReturn(b, sig, kGrid[p / G], kGrid[p % G], buy, sell, true, &e);
        ++trades;
        // The network's pair for this stock (none/none until it has features and training).
        std::size_t pick = G * G - 1;
        const auto x = features(t, i, medSig, mkt);
        if (!x.empty() && net.step > 0) {
          const auto y = net.predict(scaler.apply(x));
          pick = std::size_t(std::max_element(y.begin(), y.end()) - y.begin());
        }
        const double nks = kGrid[pick / G], nkt = kGrid[pick % G];
        const double rn = tradeReturn(b, sig, nks, nkt, buy, sell, true, &e);
        rNeural += rn, nnStops += e == 1, nnTakes += e == 2, ++nnTrades;
        rNeuralOpt += tradeReturn(b, sig, nks, nkt, buy, sell, false, &e);
        nnGain.add(std::log1p(rn) - std::log1p(tradeReturn(b, sig, kInf, kInf, buy, sell, true, &e)));
        ++neuralUse[name(nks) + " / " + name(nkt)], ++perStock[i][pick];
      }
      const double n = static_cast<double>(K);  // empty slots stay in cash
      plain.push_back(rPlain / n), learnt.push_back(rLearnt / n), optimistic.push_back(rOpt / n), gross.push_back(rGross / n);
      neural.push_back(rNeural / n), neuralOpt.push_back(rNeuralOpt / n);
      // The network learns from today: every eligible stock's 36 outcomes on day d's bar, in its
      // volatility (log returns, worst-case fills, after costs).
      std::vector<std::vector<double>> X, Y;
      for (std::size_t i : candidates) {
        auto x = features(t, i, medSig, mkt);
        if (x.empty()) continue;
        barAt(d, i, &b);
        const double sig = sigmaOf(i);
        std::vector<double> y(G * G);
        int e;
        for (std::size_t p = 0; p < G * G; ++p) y[p] = std::log1p(tradeReturn(b, sig, kGrid[p / G], kGrid[p % G], buy, sell, true, &e)) / sig;
        scaler.add(x);
        X.push_back(std::move(x)), Y.push_back(std::move(y));
      }
      for (auto& x : X) x = scaler.apply(x);
      for (std::size_t q = 0; q < X.size(); q += 32) {
        const std::size_t r = std::min(X.size(), q + 32);
        net.train({X.begin() + q, X.begin() + r}, {Y.begin() + q, Y.begin() + r});
      }
      // Volume history for the features, after the day (known at its close).
      for (std::size_t i = 0; i < N; ++i)
        if (barAt(d, i, &b)) logVolume[i].add(std::log(b.v));
      for (std::size_t p = 0; p < G * G; ++p) pairRecord[p].add(rPair[p] / n);
      model.push_back(bt.model[k]), market.push_back(bt.market[k]);
      ++days;
    }

    const std::size_t n = days, h = n / 2;
    std::printf("Daily top 10 against the real bars: %zu stocks, %s .. %s; costs %.2f%% on purchases, %.2f%% on sales\n", N,
                m.dates[bt.start].c_str(), m.dates[bt.start + n - 1].c_str(), 100 * buy, 100 * sell);
    std::printf("%zu days, %zu trades (%.2f a day; 10 on %.1f%% of days)\n\n", n, trades, double(trades) / n, 100.0 * full / n);
    const char* head = "  strategy (compounded daily)                    return/yr      vol  Sharpe   max DD    growth\n";
    const std::pair<const char*, std::pair<std::size_t, std::size_t>> parts[] = {{"whole period", {0, n}}, {"first half", {0, h}}, {"second half", {h, n}}};
    for (const auto& [pname, range] : parts) {
      std::printf("%s: %s .. %s\n%s", pname, m.dates[bt.start + range.first].c_str(), m.dates[bt.start + range.second - 1].c_str(), head);
      row("top 10, open to close, before costs", gross, range.first, range.second);
      row("top 10, no stops, after costs", plain, range.first, range.second);
      row("top 10, learnt stop/take, worst-case fills", learnt, range.first, range.second);
      row("top 10, learnt stop/take, optimistic fills", optimistic, range.first, range.second);
      row("top 10, neural per-stock levels, worst-case", neural, range.first, range.second);
      row("top 10, neural per-stock levels, optimistic", neuralOpt, range.first, range.second);
      row("model portfolio (multi-day), after costs", model, range.first, range.second);
      row("market (bought and held)", market, range.first, range.second);
      std::printf("\n");
    }
    std::printf("Daily difference against the model portfolio (paired, same days):\n");
    compare("top 10 before costs", gross, model);
    compare("top 10, no stops, after costs", plain, model);
    compare("top 10, learnt stop/take, worst-case", learnt, model);
    compare("top 10, neural levels, worst-case", neural, model);
    compare("neural levels vs no stops (both after costs)", neural, plain);
    compare("neural levels vs shared learnt levels", neural, learnt);
    std::printf("\nExits (learnt levels, worst case): %zu at the stop, %zu at the take-profit, %zu at the close\n", stops, takes,
                trades - stops - takes);
    std::printf("Trades that touched both levels (order unknown on daily bars): %zu (%.1f%%); the low was the open: %.1f%%\n", both,
                100.0 * both / std::max<std::size_t>(1, trades), 100.0 * lowIsOpen / std::max<std::size_t>(1, trades));
    std::printf("Costs paid by the daily book: %.2f%% a day (every trade pays %.2f%% on a tenth of the capital)\n",
                100 * (buy + sell) * double(trades) / n / K, 100 * (buy + sell));
    std::printf("Learnt pairs used (stop / take, multiples of the volatility; days):");
    for (const auto& [pname, c] : pairUse) std::printf(" %s %d;", pname.c_str(), c);
    std::printf("\n");
    std::printf("\nNeural per-stock levels: %zu trades; %zu at the stop, %zu at the take-profit, %zu at the close\n", nnTrades, nnStops, nnTakes,
                nnTrades - nnStops - nnTakes);
    std::printf("Realised gain of the network's pair over no stop / no take: %+.4f%% a trade (t %+.2f over %.0f trades)\n",
                100 * nnGain.s / std::max(1.0, nnGain.n),
                nnGain.n > 1 ? (nnGain.s / nnGain.n) / std::sqrt(std::max(1e-30, (nnGain.s2 - nnGain.s * nnGain.s / nnGain.n) / (nnGain.n - 1)) / nnGain.n) : 0.0,
                nnGain.n);
    std::printf("Pairs the network gave (stop / take; trades):");
    std::vector<std::pair<int, std::string>> top;
    for (const auto& [pname, c] : neuralUse) top.push_back({c, pname});
    std::sort(top.rbegin(), top.rend());
    for (std::size_t q = 0; q < top.size() && q < 12; ++q) std::printf(" %s %d;", top[q].second.c_str(), top[q].first);
    std::printf("\n");
    std::size_t traded = 0, varied = 0;
    for (const auto& ps : perStock)
      if (!ps.empty()) ++traded, varied += ps.size() > 1;
    std::printf("Stocks traded: %zu; given more than one pair over time: %zu\n", traded, varied);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
