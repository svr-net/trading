#include "ofm/trade.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ofm {

Metrics metrics(const std::vector<double>& r) {
  Metrics m;
  m.days = static_cast<double>(r.size());
  if (r.size() < 2) return m;
  double w = 1, peak = 1, s = 0, s2 = 0;
  for (double x : r) {
    w *= 1 + x, peak = std::max(peak, w), m.maxDrawdown = std::max(m.maxDrawdown, 1 - w / peak);
    s += x, s2 += x * x;
  }
  const double n = m.days, mean = s / n, var = std::max(0.0, (s2 - n * mean * mean) / (n - 1));
  m.annualReturn = std::pow(w, 252.0 / n) - 1;
  m.volatility = std::sqrt(var * 252.0);
  m.sharpe = var > 0 ? mean / std::sqrt(var) * std::sqrt(252.0) : 0.0;
  return m;
}

namespace {

double rankCorrelation(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = a.size();
  if (n < 3) return kNaN;
  auto ranks = [n](const std::vector<double>& x) {
    std::vector<std::size_t> o(n);
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](std::size_t p, std::size_t q) { return x[p] < x[q]; });
    std::vector<double> r(n);
    for (std::size_t k = 0; k < n; ++k) r[o[k]] = static_cast<double>(k);
    return r;
  };
  const auto ra = ranks(a), rb = ranks(b);
  const double mu = (static_cast<double>(n) - 1) / 2;
  double sab = 0, saa = 0;
  for (std::size_t k = 0; k < n; ++k) sab += (ra[k] - mu) * (rb[k] - mu), saa += (ra[k] - mu) * (ra[k] - mu);
  return saa > 0 ? sab / saa : kNaN;
}

// Upper tail probability of |Z| for a standard normal.
double twoSided(double z) { return std::erfc(std::fabs(z) / std::sqrt(2.0)); }

}  // namespace

std::vector<std::string> rankHedgeSeries(const Market& m, const std::map<std::string, std::vector<double>>& series, std::size_t before) {
  const std::size_t N = m.N();
  std::vector<double> mkt(m.T(), kNaN);
  for (std::size_t t = 1; t < m.T(); ++t) {
    double s = 0, n = 0;
    for (std::size_t i = 0; i < N; ++i) {
      const double a = m.close(t - 1, i), b = m.close(t, i);
      if (a > 0 && b > 0) s += b / a - 1, n += 1;
    }
    if (n > 0) mkt[t] = s / n;
  }
  std::vector<std::pair<double, std::string>> ranked;
  for (const auto& [name, px] : series) {
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, n = 0;
    for (std::size_t t = 1; t < std::min(before, px.size()); ++t) {
      if (!(px[t] > 0 && px[t - 1] > 0) || !std::isfinite(mkt[t])) continue;
      const double x = px[t] / px[t - 1] - 1, y = mkt[t];
      sx += x, sy += y, sxx += x * x, syy += y * y, sxy += x * y, n += 1;
    }
    if (n < 3) continue;
    const double c = (sxy - sx * sy / n) / std::sqrt(std::max((sxx - sx * sx / n) * (syy - sy * sy / n), 1e-300));
    if (c > 0) ranked.push_back({c, name});
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<std::string> out;
  for (const auto& r : ranked) out.push_back(r.second);
  return out;
}

std::string pickHedgeSeries(const Market& m, const std::map<std::string, std::vector<double>>& series, std::size_t before) {
  const auto r = rankHedgeSeries(m, series, before);
  return r.empty() ? std::string() : r.front();
}

Backtest backtest(const Market& m, const Forecast& f, const Costs& costs, const std::map<std::string, std::vector<double>>& series,
                  const MarketStructure* ms) {
  const std::size_t T = m.T(), N = m.N();
  const double buy = costs.buyBps * 1e-4, sell = costs.sellBps * 1e-4, hurdle = buy + sell, fut = costs.futuresBps * 1e-4;
  Backtest bt;
  // Closes carried forward for valuation on days a stock has no bar.
  Panel cf = m.close;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t t = 1; t < T; ++t)
      if (!std::isfinite(cf(t, i))) cf(t, i) = cf(t - 1, i);
  auto hasE = [&](std::size_t t, std::size_t i) { return std::isfinite(f.E(t, i)); };
  std::size_t s = T;
  for (std::size_t t = 0; t < T && s == T; ++t)
    for (std::size_t i = 0; i < N; ++i)
      if (hasE(t, i)) {
        s = t;
        break;
      }
  if (s + 2 >= T) return bt;
  bt.start = s + 1;

  // Market: every stock with a forecast at the first decision, equal weight, bought once.
  std::vector<double> mw(N, 0.0);
  double cnt = 0;
  for (std::size_t i = 0; i < N; ++i) cnt += hasE(s, i) ? 1 : 0;
  for (std::size_t i = 0; i < N; ++i) mw[i] = hasE(s, i) ? 1.0 / cnt : 0.0;

  std::vector<double> h(N, 0.0), entryValue(N, 0.0);
  std::vector<std::size_t> entryDay(N, 0);
  double cash = 1;
  std::vector<int> buyOrder(N, 0), sellOrder(N, 0);
  bool initial = true;
  double rhoSum = 0, rhoN = 0;
  std::vector<double> prevE;  // forecasts of the previous close (NaN where none)

  auto lifeAt = [&]() {
    if (rhoN < 1) return 1.0;
    const double rho = std::min(rhoSum / rhoN, 1.0 - 1.0 / (rhoN + 1.0));
    return 1.0 / (1.0 - std::max(rho, 0.0));
  };
  auto decide = [&](std::size_t t) {
    std::vector<double> a, b;
    std::vector<double> cur(N, kNaN);
    for (std::size_t i = 0; i < N; ++i) {
      cur[i] = f.E(t, i);
      if (!prevE.empty() && std::isfinite(cur[i]) && std::isfinite(prevE[i])) a.push_back(cur[i]), b.push_back(prevE[i]);
    }
    const double r = rankCorrelation(a, b);
    if (std::isfinite(r)) rhoSum += r, rhoN += 1;
    prevE = cur;
    const double life = lifeAt();
    for (std::size_t i = 0; i < N; ++i) {
      buyOrder[i] = sellOrder[i] = 0;
      if (!std::isfinite(cur[i])) continue;
      if (h[i] > 0 && -cur[i] * life > hurdle) sellOrder[i] = 1;
      if (h[i] <= 0 && cur[i] * life > hurdle) buyOrder[i] = 1;
    }
    return life;
  };

  // Hedge signal from the chosen series.
  // Hedge series: ranked by correlation with the market before trading starts. The first drives
  // the sign forecast; the hedge trades the first one with bars on both days, so it carries on
  // when a continuation series ends.
  const std::vector<std::string> ranked = rankHedgeSeries(m, series, s);
  bt.hedgeSeries = ranked.empty() ? std::string() : ranked.front();
  std::vector<double> idx;
  if (!bt.hedgeSeries.empty()) idx = series.at(bt.hedgeSeries);
  // Volatility management (Moreira and Muir 2017): exposure cut to (long-run variance / predicted
  // variance) of the book when that is below one. The prediction is the exponentially weighted
  // variance whose half-life (among the model's horizons) has forecast next-day variance best so
  // far (QLIKE loss).
  std::vector<double> ewVar(f.horizons.size(), 0.0), qlike(f.horizons.size(), 0.0);
  double varSum = 0, varN = 0, hVol = 0;
  // Hedge features: the index future's trends at the model's horizons, and the market's
  // topology and geometry (tree length, effective dimension, geodesic distance) at each half-life.
  const std::vector<std::uint32_t> hz = f.horizons;
  const std::size_t HZ = hz.size();
  const std::size_t HS = ms ? ms->halfLives.size() : 0;
  const std::size_t G = HZ + (HS ? 3 * HS - 1 : 0);
  std::vector<double> mean(G, 0.0), cov(G * G, 0.0), recS(G, 0.0), recS2(G, 0.0);
  double featN = 0, recN = 0, yS = 0, yS2 = 0;
  std::vector<std::vector<double>> pendingRec;  // x_t of the last close, waiting for its next return
  // Persistence of the index forecast: lag-one correlation of its daily values so far.
  double eIdxPrev = kNaN, ea = 0, eb = 0, eaa = 0, ebb = 0, eab = 0, en = 0, hedgeOn = 0;
  // Book beta to the future and the future's beta to the stock market (for rolls), expanding.
  double bx = 0, by = 0, bxx = 0, bxy = 0, bn = 0;
  double mx = 0, my = 0, mxx = 0, mxy = 0, mn = 0;
  std::vector<double> resid;  // the future's moves net of the market's, for Chauvenet

  for (std::size_t t = s; t + 1 < T; ++t) {
    // ---- the decision at close t ----
    if (t == s) {
      prevE.clear();
      decide(t);  // records persistence; the first portfolio is the market regardless
    } else {
      bt.life.push_back(decide(t));
    }
    // Hedge decision at close t.
    double hedgeWant = 0;
    hVol = 0;
    if (varN >= 2 && !ewVar.empty() && ewVar[0] > 0) {
      std::size_t best = 0;
      for (std::size_t j = 1; j < ewVar.size(); ++j)
        if (qlike[j] < qlike[best]) best = j;
      const double predicted = ewVar[best], longRun = varSum / varN;
      if (predicted > 0) hVol = std::max(0.0, std::min(1.0, 1.0 - longRun / predicted));
    }
    if (!idx.empty() && std::isfinite(idx[t])) {
      std::vector<double> g(G, kNaN);
      bool ok = true;
      for (std::size_t j = 0; j < HZ; ++j) {
        if (t < hz[j] || !(idx[t - hz[j]] > 0)) ok = false;
        else g[j] = std::log(idx[t] / idx[t - hz[j]]);
      }
      for (std::size_t j = 0, q = HZ; j < HS; ++j) {
        g[q++] = ms->treeLength(t, j);
        g[q++] = ms->dimension(t, j);
        if (j + 1 < HS) g[q++] = ms->geodesic(t, j);
      }
      for (double x : g) ok = ok && std::isfinite(x);
      if (ok) {
        featN += 1;
        for (std::size_t j = 0; j < G; ++j) {
          const double dd = g[j] - mean[j];
          mean[j] += dd / featN;
          for (std::size_t k = 0; k < G; ++k) cov[j * G + k] += dd * (g[k] - mean[k]);
        }
        if (featN > static_cast<double>(G) + 1) {
          // Whiten by the expanding covariance (symmetric inverse square root).
          std::vector<double> C(cov), V, w;
          for (double& c : C) c /= (featN - 1);
          eigenSym(C, G, w, V);
          double top = 0;
          for (double x : w) top = std::max(top, x);
          std::vector<double> x(G, 0.0);
          for (std::size_t k = 0; k < G; ++k) {
            if (!(w[k] > top * 1e-12)) continue;
            double proj = 0;
            for (std::size_t j = 0; j < G; ++j) proj += V[j * G + k] * (g[j] - mean[j]);
            proj /= std::sqrt(w[k]);
            for (std::size_t j = 0; j < G; ++j) x[j] += V[j * G + k] * proj;
          }
          pendingRec.clear();
          pendingRec.push_back(x);
          // Expected next-day return of the future: its average return (shrunk by its own
          // t-statistic) plus the features' conditional part. The features' premia are shrunk
          // jointly (positive-part James-Stein over the G of them): unless together they carry
          // more evidence than G - 2 chance t-statistics would, the conditional part is zero.
          double base = 0, cond = 0;
          if (recN >= 3) {
            const double ybar = yS / recN, vy = std::max(0.0, (yS2 - recN * ybar * ybar) / (recN - 1));
            if (vy > 0) {
              const double ty = ybar / std::sqrt(vy / recN);
              base = ybar * std::max(0.0, 1 - 1 / (ty * ty));
            }
            std::vector<double> mu(G, 0.0), tz(G, 0.0);
            double sumT2 = 0;
            for (std::size_t j = 0; j < G; ++j) {
              mu[j] = recS[j] / recN;
              const double var = std::max(0.0, (recS2[j] - recN * mu[j] * mu[j]) / (recN - 1));
              if (var > 0) tz[j] = mu[j] / std::sqrt(var / recN), sumT2 += tz[j] * tz[j];
            }
            const double shrink = G > 2 ? (sumT2 > 0 ? std::max(0.0, 1 - (static_cast<double>(G) - 2) / sumT2) : 0.0) : 1.0;
            for (std::size_t j = 0; j < G; ++j) {
              const double own = tz[j] != 0 ? std::max(0.0, 1 - 1 / (tz[j] * tz[j])) : 0.0;
              cond += x[j] * mu[j] * (G > 2 ? shrink : own);
            }
          }
          const double e = base + cond;
          if (std::isfinite(eIdxPrev)) ea += cond, eb += eIdxPrev, eaa += cond * cond, ebb += eIdxPrev * eIdxPrev, eab += cond * eIdxPrev, en += 1;
          eIdxPrev = cond;
          double rho = 0;
          if (en > 2) {
            const double cv = eab - ea * eb / en, va = eaa - ea * ea / en, vb = ebb - eb * eb / en;
            if (va > 0 && vb > 0) rho = std::max(0.0, std::min(cv / std::sqrt(va * vb), 1.0 - 1.0 / (en + 1.0)));
          }
          // Expected fall over the forecast's life beats the futures round trip.
          if (e / (1.0 - rho) < -2 * fut) hedgeWant = 1;
        }
      }
    }

    // ---- day t+1 ----
    const std::size_t d = t + 1;
    double eqPrev = cash;
    for (std::size_t i = 0; i < N; ++i) eqPrev += h[i];
    // Overnight to the open.
    for (std::size_t i = 0; i < N; ++i)
      if (h[i] > 0) {
        const double o = m.open(d, i), c = cf(t, i);
        if (o > 0 && c > 0) h[i] *= o / c;
      }
    double eqOpen = cash;
    for (std::size_t i = 0; i < N; ++i) eqOpen += h[i];
    double traded = 0, cost = 0;
    auto tradable = [&](std::size_t i) { return m.open(d, i) > 0; };
    if (initial) {
      for (std::size_t i = 0; i < N; ++i)
        if (mw[i] > 0) {
          const double v = eqOpen * mw[i] / (1 + buy);
          h[i] = v, cost += v * buy, traded += v, entryValue[i] = v * (1 + buy), entryDay[i] = d;
        }
      cash = eqOpen - std::accumulate(h.begin(), h.end(), 0.0) - cost;
      initial = false;
    } else {
      // Sales.
      for (std::size_t i = 0; i < N; ++i)
        if (sellOrder[i] && h[i] > 0 && tradable(i)) {
          const double v = h[i];
          cash += v * (1 - sell), cost += v * sell, traded += v;
          bt.trades.push_back({i, entryDay[i], d, v * (1 - sell) / entryValue[i] - 1});
          h[i] = 0;
        }
      // Purchases: each new stock an equal share of the portfolio.
      std::vector<std::size_t> buys;
      for (std::size_t i = 0; i < N; ++i)
        if (buyOrder[i] && h[i] <= 0 && tradable(i)) buys.push_back(i);
      std::size_t heldCount = 0;
      for (std::size_t i = 0; i < N; ++i) heldCount += h[i] > 0 ? 1 : 0;
      const double eq = cash + std::accumulate(h.begin(), h.end(), 0.0);
      if (!buys.empty()) {
        const double share = eq / static_cast<double>(heldCount + buys.size());
        double need = share * static_cast<double>(buys.size());
        if (need > cash) {
          // Fund the rest pro rata from the current holdings.
          const double fromHoldings = (need - cash) / (1 - sell);
          const double held = eq - cash;
          for (std::size_t i = 0; i < N; ++i)
            if (h[i] > 0 && held > 0) {
              const double v = h[i] * fromHoldings / held;
              h[i] -= v, cash += v * (1 - sell), cost += v * sell, traded += v;
            }
        }
        need = std::min(need, cash);
        for (std::size_t i : buys) {
          const double v = need / static_cast<double>(buys.size()) / (1 + buy);
          h[i] = v, cash -= v * (1 + buy), cost += v * buy, traded += v, entryValue[i] = v * (1 + buy), entryDay[i] = d;
        }
      }
      if (cash > 1e-12) {
        const double held = std::accumulate(h.begin(), h.end(), 0.0);
        if (held > 0) {
          const double spend = cash / (1 + buy);
          for (std::size_t i = 0; i < N; ++i)
            if (h[i] > 0) {
              const double v = spend * h[i] / held;
              h[i] += v, cost += v * buy, traded += v, entryValue[i] += v * (1 + buy);
            }
          cash = 0;
        }
      }
    }
    // Open to close.
    for (std::size_t i = 0; i < N; ++i)
      if (h[i] > 0) {
        const double o = m.open(d, i), c = cf(d, i), prev = cf(t, i);
        if (o > 0 && c > 0) h[i] *= c / o;
        else if (c > 0 && prev > 0) h[i] *= c / prev;
      }
    double eq = cash;
    std::size_t heldCount = 0;
    for (std::size_t i = 0; i < N; ++i) eq += h[i], heldCount += h[i] > 0 ? 1 : 0;
    const double r = eq / eqPrev - 1;
    bt.model.push_back(r);
    bt.holdings.push_back(static_cast<double>(heldCount));
    bt.turnover.push_back(traded / eqPrev);

    // Market (bought at the first open, then held).
    double mb = 0, ma = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (mw[i] > 0) {
        mb += mw[i];
        const double c1 = cf(d, i), c0 = d == s + 1 ? m.open(d, i) : cf(t, i);
        mw[i] *= (c1 > 0 && c0 > 0) ? c1 / c0 : 1.0;
        ma += mw[i];
      }
    bt.market.push_back(ma / mb - 1 - (d == s + 1 ? buy : 0.0));

    // Hedge over day d: decided at close t, sized by the book's beta to the future so far.
    double rf = kNaN, rm = 0;
    const std::vector<double>* used = nullptr;
    for (const auto& name : ranked) {
      const auto& px = series.at(name);
      if (d < px.size() && px[t] > 0 && px[d] > 0) {
        used = &px;
        break;
      }
    }
    if (used) {
      rf = (*used)[d] / (*used)[t] - 1;
      double sm = 0, nm = 0;
      for (std::size_t i = 0; i < N; ++i)
        if (cf(t, i) > 0 && cf(d, i) > 0) sm += cf(d, i) / cf(t, i) - 1, nm += 1;
      rm = nm > 0 ? sm / nm : 0;
      // A roll jump: the future's move against the market's is an outlier (Chauvenet).
      const double bm = mn > 2 ? (mxy - mx * my / mn) / std::max(mxx - mx * mx / mn, 1e-300) : 1.0;
      const double e = rf - bm * rm;
      bool roll = false;
      if (resid.size() > 20) {
        std::vector<double> sorted(resid);
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2), sorted.end());
        const double med = sorted[sorted.size() / 2];
        for (double& x : sorted) x = std::fabs(x - med);
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2), sorted.end());
        const double mad = 1.4826 * sorted[sorted.size() / 2];
        if (mad > 0 && twoSided((e - med) / mad) * static_cast<double>(resid.size() + 1) < 0.5) roll = true;
      }
      if (roll) rf = bm * rm;
      else resid.push_back(e);
      mx += rm, my += rf, mxx += rm * rm, mxy += rm * rf, mn += 1;
      // Record of the index forecast's features against this return (primary series only).
      if (!pendingRec.empty() && used == &series.at(bt.hedgeSeries)) {
        const auto& x = pendingRec.back();
        for (std::size_t j = 0; j < G; ++j) recS[j] += x[j] * rf, recS2[j] += x[j] * rf * x[j] * rf;
        yS += rf, yS2 += rf * rf;
        recN += 1;
        pendingRec.clear();
      }
    }
    const double beta = bn > 2 ? (bxy - bx * by / bn) / std::max(bxx - bx * bx / bn, 1e-300) : 1.0;
    const double want = std::isfinite(rf) ? std::max(hedgeWant, hVol) * std::max(0.0, beta) : 0.0;
    double hcost = std::fabs(want - hedgeOn) * fut;
    hedgeOn = want;
    bt.hedged.push_back(r - (std::isfinite(rf) ? hedgeOn * rf : 0.0) - hcost);
    bt.hedge.push_back(hedgeOn);
    if (std::isfinite(rf)) bx += rf, by += r, bxx += rf * rf, bxy += rf * r, bn += 1;
    // The book's variance forecasts, scored on today's return before they are updated.
    for (std::size_t j = 0; j < ewVar.size(); ++j) {
      if (ewVar[j] > 0) qlike[j] += std::log(ewVar[j]) + r * r / ewVar[j];
      const double lam = std::pow(2.0, -1.0 / static_cast<double>(f.horizons[j]));
      ewVar[j] = ewVar[j] > 0 ? lam * ewVar[j] + (1 - lam) * r * r : r * r;
    }
    varSum += r * r, varN += 1;
  }
  // The last close's state: what is held and what would be traded at the next open.
  bt.held.assign(N, 0), bt.orders.assign(N, 0);
  const double life = decide(T - 1);
  bt.life.push_back(life);
  for (std::size_t i = 0; i < N; ++i) bt.held[i] = h[i] > 0 ? 1 : 0, bt.orders[i] = buyOrder[i] ? 1 : sellOrder[i] ? -1 : 0;
  return bt;
}

}  // namespace ofm
