// Shared pieces of the C++ loaders and scan (apps/ofm_eod, apps/ofm_scan): JSON (nlohmann), the API's field names
// matched case-insensitively, dates, files, and numbers written the shortest way (as JavaScript did, so the store's
// files keep their format).
#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace eod {

using json = nlohmann::json;
namespace fs = std::filesystem;

// An option's value (--name value), or the fallback.
inline std::string opt(const std::vector<std::string>& args, const std::string& name, const std::string& fallback = "") {
  for (std::size_t k = 0; k + 1 < args.size(); ++k)
    if (args[k] == "--" + name) return args[k + 1];
  return fallback;
}
inline bool flag(const std::vector<std::string>& args, const std::string& name) {
  return std::find(args.begin(), args.end(), "--" + name) != args.end();
}

inline std::string lower(std::string s) {
  for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// The first of the names present in the object, matched case-insensitively (the API's names may change case).
inline const json* field(const json& o, std::initializer_list<const char*> names) {
  if (!o.is_object()) return nullptr;
  for (const char* n : names)
    for (auto it = o.begin(); it != o.end(); ++it)
      if (lower(it.key()) == lower(n)) return &it.value();
  return nullptr;
}
inline std::string str(const json* v) {
  if (!v || v->is_null()) return "";
  if (v->is_string()) return v->get<std::string>();
  return v->dump();
}
inline double num(const json* v) {
  if (!v || v->is_null()) return NAN;
  if (v->is_number()) return v->get<double>();
  if (v->is_string()) {
    const std::string s = v->get<std::string>();
    char* e = nullptr;
    const double x = std::strtod(s.c_str(), &e);
    return e && e != s.c_str() ? x : NAN;
  }
  return NAN;
}
// A list from an API answer: the array itself, or one of its usual wrappers.
inline json asList(const json& body, std::initializer_list<const char*> names = {"data", "items", "quotes", "symbols"}) {
  if (body.is_array()) return body;
  const json* v = field(body, names);
  return v && v->is_array() ? *v : json::array();
}

// A number the shortest way that reads back exactly (integers without a decimal point), as JavaScript writes it.
inline std::string shortest(double x) {
  if (!std::isfinite(x)) return "0";
  if (x == std::floor(x) && std::fabs(x) < 1e15) {
    char b[32];
    std::snprintf(b, sizeof b, "%.0f", x);
    return b;
  }
  char b[64];
  const auto r = std::to_chars(b, b + sizeof b, x);
  return std::string(b, r.ptr);
}

// Dates as YYYY-MM-DD (UTC).
inline std::string ymd(std::time_t t) {
  char b[16];
  std::tm tm{};
  gmtime_r(&t, &tm);
  std::strftime(b, sizeof b, "%Y-%m-%d", &tm);
  return b;
}
inline std::string today() { return ymd(std::time(nullptr)); }
inline std::time_t parseDay(const std::string& d) {
  std::tm tm{};
  tm.tm_year = std::atoi(d.substr(0, 4).c_str()) - 1900, tm.tm_mon = std::atoi(d.substr(5, 2).c_str()) - 1, tm.tm_mday = std::atoi(d.substr(8, 2).c_str()), tm.tm_hour = 12;
  return timegm(&tm);
}
inline std::string addDays(const std::string& d, int n) { return ymd(parseDay(d) + std::time_t(n) * 86400); }
inline std::string previousWeekday(const std::string& d) {
  std::time_t t = parseDay(d);
  for (;;) {
    t -= 86400;
    std::tm tm{};
    gmtime_r(&t, &tm);
    if (tm.tm_wday != 0 && tm.tm_wday != 6) return ymd(t);
  }
}
inline bool isDay(const std::string& d) {
  return d.size() == 10 && std::isdigit(static_cast<unsigned char>(d[0])) && d[4] == '-' && d[7] == '-';
}
inline int daysBetween(const std::string& a, const std::string& b) { return int(std::llround(double(parseDay(b) - parseDay(a)) / 86400.0)); }

inline std::string readFile(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p.string());
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}
inline bool tryRead(const fs::path& p, std::string& out) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return false;
  std::ostringstream s;
  s << f.rdbuf();
  out = s.str();
  return true;
}
inline void writeFile(const fs::path& p, const std::string& data) {
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + p.string());
  f << data;
}
inline json readJson(const fs::path& p) { return json::parse(readFile(p)); }

inline std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) out.push_back(cur), cur.clear();
    else cur += c;
  }
  out.push_back(cur);
  return out;
}

}  // namespace eod
