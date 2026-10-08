#include <chrono>
#include <cstring>
#include <iostream>

#include "test_framework.hpp"

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int passed = 0, failed = 0;
  for (const auto& test : sattest::registry()) {
    if (filter && test.name.find(filter) == std::string::npos) continue;
    const auto start = std::chrono::steady_clock::now();
    try {
      test.body();
      ++passed;
      const auto ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
      std::cout << "[  OK  ] " << test.name << " (" << ms << " ms)\n";
    } catch (const std::exception& e) {
      ++failed;
      std::cout << "[ FAIL ] " << test.name << "\n         " << e.what() << "\n";
    }
  }
  std::cout << "\n" << passed << " passed, " << failed << " failed\n";
  return failed == 0 ? 0 : 1;
}
