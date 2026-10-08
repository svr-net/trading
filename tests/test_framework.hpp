#pragma once

// Minimal self-registering test framework (no external dependencies).

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sattest {

struct TestCase {
  std::string name;
  std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline void fail(const std::string& message, const char* file, int line) {
  std::ostringstream os;
  os << file << ":" << line << ": " << message;
  throw Failure(os.str());
}

}  // namespace sattest

#define SAT_CONCAT_INNER(a, b) a##b
#define SAT_CONCAT(a, b) SAT_CONCAT_INNER(a, b)

#define TEST(name)                                                                             \
  static void SAT_CONCAT(test_, name)();                                                       \
  static ::sattest::Registrar SAT_CONCAT(registrar_, name)(#name, &SAT_CONCAT(test_, name));   \
  static void SAT_CONCAT(test_, name)()

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) ::sattest::fail("CHECK(" #condition ") failed", __FILE__, __LINE__); \
  } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                                   \
  do {                                                                                            \
    const double sat_a = (actual);                                                                \
    const double sat_e = (expected);                                                              \
    if (!(std::fabs(sat_a - sat_e) <= (tolerance))) {                                             \
      std::ostringstream sat_os;                                                                  \
      sat_os.precision(12);                                                                       \
      sat_os << "CHECK_NEAR(" #actual ", " #expected ") failed: " << sat_a << " vs " << sat_e      \
             << " (tolerance " << (tolerance) << ")";                                             \
      ::sattest::fail(sat_os.str(), __FILE__, __LINE__);                                          \
    }                                                                                             \
  } while (0)

#define CHECK_THROWS(statement)                                                       \
  do {                                                                                \
    bool sat_threw = false;                                                           \
    try {                                                                             \
      statement;                                                                      \
    } catch (...) {                                                                   \
      sat_threw = true;                                                               \
    }                                                                                 \
    if (!sat_threw) ::sattest::fail("expected exception: " #statement, __FILE__, __LINE__); \
  } while (0)
