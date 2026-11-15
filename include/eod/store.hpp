// The store's daily bars: one JSON file per symbol, <root>/store/<group>/<code>.json, {"YYYY-MM-DD": {"o", "h", "l",
// "c", "v"}} in date order (the format the earlier JavaScript wrote, numbers the shortest way). The root is the clone
// of svr-net/eod (EOD_STORE, or data/eoddata where the scan keeps it).
#pragma once

#include <map>
#include <regex>
#include <string>

#include "eod/common.hpp"

namespace eod::store {

struct Bar {
  double o = 0, h = 0, l = 0, c = 0, v = 0;
};
using Bars = std::map<std::string, Bar>;  // date -> bar, in date order

inline fs::path root() {
  const char* r = std::getenv("EOD_STORE");
  return r && *r ? fs::path(r) : fs::path("data") / "eoddata";
}
inline std::string safe(const std::string& s) {
  std::string o = s;
  for (auto& ch : o)
    if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '.' || ch == '-')) ch = '_';
  return o;
}
inline fs::path fileOf(const std::string& group, const std::string& code) { return root() / "store" / safe(group) / (safe(code) + ".json"); }

inline bool has(const std::string& group, const std::string& code) { return fs::exists(fileOf(group, code)); }

inline Bars load(const std::string& group, const std::string& code) {
  Bars out;
  std::string text;
  if (!tryRead(fileOf(group, code), text) || text.empty()) return out;
  const json j = json::parse(text);
  for (auto it = j.begin(); it != j.end(); ++it) {
    const json& b = it.value();
    out[it.key()] = Bar{num(field(b, {"o"})), num(field(b, {"h"})), num(field(b, {"l"})), num(field(b, {"c"})), num(field(b, {"v"}))};
  }
  return out;
}

inline std::string dump(const Bars& bars) {
  std::string s = "{";
  bool first = true;
  for (const auto& [d, b] : bars) {
    if (!first) s += ',';
    first = false;
    s += "\"" + d + "\":{\"o\":" + shortest(b.o) + ",\"h\":" + shortest(b.h) + ",\"l\":" + shortest(b.l) + ",\"c\":" + shortest(b.c) + ",\"v\":" + shortest(b.v) + "}";
  }
  return s + "}";
}

inline void save(const std::string& group, const std::string& code, const Bars& bars) { writeFile(fileOf(group, code), dump(bars)); }

inline std::string lastDate(const Bars& b) { return b.empty() ? "" : b.rbegin()->first; }

// A bar from an API record: its day and bar, or false.
inline bool bar(const json& rec, std::string& day, Bar& b) {
  day = str(field(rec, {"dateStamp", "date"})).substr(0, 10);
  const double c = num(field(rec, {"close"}));
  if (!isDay(day) || !(c > 0)) return false;
  const double o = num(field(rec, {"open"})), h = num(field(rec, {"high"})), l = num(field(rec, {"low"})), v = num(field(rec, {"volume"}));
  b = Bar{o > 0 ? o : c, h > 0 ? h : c, l > 0 ? l : c, c, std::isfinite(v) ? v : 0};
  return true;
}

// API records added to the bars; returns how many days were new.
inline int merge(Bars& bars, const json& list) {
  int n = 0;
  for (const auto& rec : list) {
    std::string d;
    Bar b;
    if (!bar(rec, d, b)) continue;
    n += bars.count(d) == 0;
    bars[d] = b;
  }
  return n;
}

// All but the newest `keep` files of a folder whose names start with `prefix` deleted.
inline void prune(const fs::path& dir, const std::string& prefix, std::size_t keep = 1) {
  if (!fs::exists(dir)) return;
  std::vector<std::string> files;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().filename().string().rfind(prefix, 0) == 0) files.push_back(e.path().filename().string());
  std::sort(files.begin(), files.end());
  for (std::size_t k = 0; k + keep < files.size(); ++k) fs::remove(dir / files[k]);
}

}  // namespace eod::store
