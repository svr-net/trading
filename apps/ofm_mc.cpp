// A strategy's backtest put through Monte Carlo (research; results only): how much of its result could be
// luck, the range of a year it could have, and what is left when fewer of its orders fill.
//
// From ofm_research's trade log (X_TRADELOG: one line per order; local, never published) and the
// account's day by day beside it (X_TRADELOG.days: date,ret, the backtest's daily results):
//   luck      the probabilistic Sharpe ratio (the chance the true Sharpe is above 0, given the days'
//             skew and fat tails) and the deflated one (Bailey and Lopez de Prado 2014: above the
//             best Sharpe that --trials strategies with no edge at all would show by chance)
//   a year    the days resampled in blocks (the stationary bootstrap of Politis and Romano 1994, mean
//             block --block days, so runs of good and bad days stay together), --paths years of 252
//             days: the year's return (5%, median, 95%), the chance of a losing year, the worst
//             drawdown (median, 95%)
//   fills     the same with each filled order kept with chance k (--keep; 6 Nov: IBKR filled 8 of the
//             UK backtest's 18 fills, 5 of the EU's 5): a dropped fill takes its share of the day
//             (its return over the day's slots, as the account counts it) out of the day
//
//   ofm_mc --days log.csv.days --trades log.csv [--slots 10] [--trials 30] [--paths 10000] [--block 10] [--keep 1,0.75,0.5]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::string> cells(const std::string& line) {
  std::vector<std::string> c;
  std::stringstream ss(line);
  for (std::string x; std::getline(ss, x, ',');) c.push_back(x);
  return c;
}
double phi(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
double phiInv(double p) {  // Acklam's rational approximation of the normal quantile
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
  if (p < 0.02425) {
    const double q = std::sqrt(-2 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - 0.02425) return -phiInv(1 - p);
  const double q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
struct Rng {  // xorshift64*
  std::uint64_t s;
  double u() { s ^= s >> 12, s ^= s << 25, s ^= s >> 27; return double((s * 2685821657736338717ull) >> 11) / double(1ull << 53); }
};
double quantile(std::vector<double> v, double f) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[std::min(v.size() - 1, std::size_t(f * double(v.size())))];
}
}  // namespace

int main(int argc, char** argv) {
  std::string daysFile, tradesFile, keepList = "1,0.75,0.5";
  double slots = 10, trials = 30, block = 10;
  std::size_t paths = 10000;
  for (int k = 1; k + 1 < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--days") daysFile = argv[++k];
    else if (a == "--trades") tradesFile = argv[++k];
    else if (a == "--slots") slots = std::atof(argv[++k]);
    else if (a == "--trials") trials = std::max(1.0, std::atof(argv[++k]));
    else if (a == "--paths") paths = std::size_t(std::atol(argv[++k]));
    else if (a == "--block") block = std::max(1.0, std::atof(argv[++k]));
    else if (a == "--keep") keepList = argv[++k];
  }
  if (daysFile.empty()) {
    std::fprintf(stderr, "usage: ofm_mc --days log.csv.days [--trades log.csv] [--slots 10] [--trials 30] [--paths 10000] [--block 10] [--keep 1,0.75,0.5]\n");
    return 2;
  }
  // The account's days, and each day's fills (their share of the day).
  std::vector<std::string> dates;
  std::vector<double> ret;
  {
    std::ifstream in(daysFile);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto c = cells(line);
      if (c.size() >= 2) dates.push_back(c[0]), ret.push_back(std::atof(c[1].c_str()));
    }
  }
  const std::size_t T = ret.size();
  if (T < 60) {
    std::printf("mc: only %zu days in %s\n", T, daysFile.c_str());
    return 1;
  }
  std::map<std::string, std::size_t> at;
  for (std::size_t t = 0; t < T; ++t) at[dates[t]] = t;
  std::vector<std::vector<double>> fills(T);
  double nFills = 0;
  if (!tradesFile.empty()) {
    std::ifstream in(tradesFile);
    std::string line;
    std::getline(in, line);
    const auto head = cells(line);
    const auto col = [&](const char* n) { return std::size_t(std::find(head.begin(), head.end(), n) - head.begin()); };
    const std::size_t cd = col("date"), cr = col("ret"), cf = col("filled");
    while (std::getline(in, line)) {
      const auto c = cells(line);
      if (c.size() <= std::max({cd, cr, cf})) continue;
      const auto it = at.find(c[cd]);
      if (it != at.end() && std::atof(c[cf].c_str()) > 0) fills[it->second].push_back(std::atof(c[cr].c_str())), nFills += 1;
    }
    for (auto& f : fills) {  // each fill's share of the day: its return over the day's slots (or its fills, when more)
      const double div = std::max(slots, double(f.size()));
      for (double& x : f) x /= div;
    }
  }
  // The backtest itself.
  double m = 0, v = 0, s3 = 0, s4 = 0, w = 0, peak = 0, dd = 0;
  for (double r : ret) m += r / double(T);
  for (double r : ret) {
    const double z = r - m;
    v += z * z / double(T), s3 += z * z * z / double(T), s4 += z * z * z * z / double(T);
    w += std::log1p(r), peak = std::max(peak, w), dd = std::max(dd, 1 - std::exp(w - peak));
  }
  const double sd = std::sqrt(v), skew = sd > 0 ? s3 / (sd * sd * sd) : 0, kurt = sd > 0 ? s4 / (v * v) : 3, sr = sd > 0 ? m / sd : 0;
  const double years = double(T) / 252.0;
  std::printf("mc: %zu days (%s to %s), %.0f fills; the backtest: %+.1f%% a year, Sharpe %.2f, worst drawdown %.1f%%; daily skew %+.2f, kurtosis %.1f\n", T, dates.front().c_str(),
              dates.back().c_str(), nFills, 100 * std::expm1(w / years), sr * std::sqrt(252.0), 100 * dd, skew, kurt);
  // Luck: the probabilistic and the deflated Sharpe ratios (daily units).
  const double var = std::max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr), se = std::sqrt(var / double(T - 1));
  const double psr = phi(sr / se);
  const double g = 0.5772156649, sr0 = trials > 1 ? se * ((1 - g) * phiInv(1 - 1 / trials) + g * phiInv(1 - 1 / (trials * std::exp(1.0)))) : 0.0;
  const double dsr = phi((sr - sr0) / se);
  std::printf("mc: luck: the chance the true Sharpe is above 0: %.1f%%; above the best of %.0f strategies with no edge (Sharpe %.2f by chance): %.1f%% (the deflated Sharpe ratio)\n",
              100 * psr, trials, sr0 * std::sqrt(252.0), 100 * dsr);
  // A year, resampled in blocks; with fewer fills.
  std::vector<double> keeps;
  {
    std::stringstream ss(keepList);
    for (std::string x; std::getline(ss, x, ',');) keeps.push_back(std::atof(x.c_str()));
  }
  for (double keep : keeps) {
    if (keep < 1 && tradesFile.empty()) continue;
    Rng rng{0x9E3779B97F4A7C15ull};
    std::vector<double> yr(paths), wd(paths);
    for (std::size_t q = 0; q < paths; ++q) {
      std::size_t t = std::size_t(rng.u() * double(T));
      double lw = 0, pk = 0, md = 0;
      for (std::size_t d = 0; d < 252; ++d) {
        if (rng.u() < 1 / block) t = std::size_t(rng.u() * double(T));  // a new block
        else t = (t + 1) % T;
        double r = ret[t];
        if (keep < 1)
          for (double c : fills[t])
            if (rng.u() >= keep) r -= c;
        lw += std::log1p(std::max(r, -0.99)), pk = std::max(pk, lw), md = std::max(md, 1 - std::exp(lw - pk));
      }
      yr[q] = std::expm1(lw), wd[q] = md;
    }
    double lose = 0;
    for (double y : yr) lose += y < 0;
    std::printf("mc: a year (%zu resampled, blocks of %.0f days), fills kept %3.0f%%: return 5%% %+.1f%%, median %+.1f%%, 95%% %+.1f%%; a losing year %.1f%%; worst drawdown median %.1f%%, 95%% %.1f%%\n",
                paths, block, 100 * keep, 100 * quantile(yr, 0.05), 100 * quantile(yr, 0.5), 100 * quantile(yr, 0.95), 100 * lose / double(paths), 100 * quantile(wd, 0.5),
                100 * quantile(wd, 0.95));
  }
  return 0;
}
