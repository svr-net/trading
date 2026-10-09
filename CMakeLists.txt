cmake_minimum_required(VERSION 3.16)

project(trading_sat
  VERSION 0.3.3
  DESCRIPTION "Predicting stock prices with machine learning to build a self-adaptive trading strategy"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SAT_BUILD_TESTS "Build the unit tests" ON)
option(SAT_BUILD_EXAMPLES "Build the example programs" ON)

add_library(sat
  src/core/matrix.cpp
  src/core/panel.cpp
  src/core/stats.cpp
  src/data/market_data.cpp
  src/data/synthetic_market.cpp
  src/factors/operators.cpp
  src/factors/alpha101.cpp
  src/features/dataset.cpp
  src/ml/classifier.cpp
  src/ml/linear.cpp
  src/ml/trees.cpp
  src/ml/neural.cpp
  src/ml/metrics.cpp
  src/ml/walk_forward.cpp
  src/strategy/performance.cpp
  src/strategy/strategy.cpp
  src/adaptive/self_adaptive.cpp
  src/adaptive/experiment.cpp
  src/afml/bars.cpp
  src/afml/fracdiff.cpp
  src/afml/labeling.cpp
  src/afml/sampling.cpp
  src/afml/importance.cpp
  src/afml/bet_sizing.cpp
  src/afml/backtest_stats.cpp
  src/afml/portfolio.cpp
  src/afml/microstructure.cpp
  src/afml/features.cpp
  src/afml/overfitting.cpp
  src/hedge/hedging.cpp
  src/hedge/options.cpp
  src/algo/pairs.cpp
  src/algo/trend.cpp
  src/algo/regimes.cpp
  src/algo/execution.cpp
  src/algo/ensemble.cpp
  src/algo/tournament.cpp
  src/gpu/fused_backtest.cpp
  src/gpu/fused_kernels.cpp
  src/gpu/fused_reference.cpp
)
add_library(sat::sat ALIAS sat)

target_include_directories(sat PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(sat PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
elseif(MSVC)
  target_compile_options(sat PRIVATE /W4)
endif()

if(EMSCRIPTEN)
  # WebAssembly build of the library with the Embind wrapper used by the web front end:
  #   emcmake cmake -S . -B build-wasm && cmake --build build-wasm
  # produces web/wasm/sat.js + sat.wasm.
  set(SAT_WASM_FLAGS -fwasm-exceptions -msimd128)
  target_compile_options(sat PRIVATE ${SAT_WASM_FLAGS})
  add_executable(sat_wasm wasm/bindings.cpp)
  target_compile_options(sat_wasm PRIVATE ${SAT_WASM_FLAGS})
  target_compile_definitions(sat_wasm PRIVATE SAT_VERSION="${PROJECT_VERSION}")
  target_link_libraries(sat_wasm PRIVATE sat)
  target_link_options(sat_wasm PRIVATE
    -lembind -fwasm-exceptions
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createSatModule
    -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=web,worker,node)
  set_target_properties(sat_wasm PROPERTIES
    OUTPUT_NAME sat SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/web/wasm)
  set(SAT_BUILD_TESTS OFF)
  set(SAT_BUILD_EXAMPLES OFF)
endif()

if(SAT_BUILD_TESTS)
  enable_testing()
  add_executable(sat_tests
    tests/test_main.cpp
    tests/test_core.cpp
    tests/test_data.cpp
    tests/test_factors.cpp
    tests/test_ml.cpp
    tests/test_strategy.cpp
    tests/test_adaptive.cpp
    tests/test_gpu.cpp
    tests/test_afml.cpp
    tests/test_hedge_algo.cpp
  )
  target_link_libraries(sat_tests PRIVATE sat)
  add_test(NAME sat_tests COMMAND sat_tests)
endif()

if(SAT_BUILD_EXAMPLES)
  add_executable(self_adaptive_trading_demo examples/self_adaptive_trading_demo.cpp)
  target_link_libraries(self_adaptive_trading_demo PRIVATE sat)
  add_executable(strategy_tournament examples/strategy_tournament.cpp)
  target_link_libraries(strategy_tournament PRIVATE sat)
  add_executable(uk_portfolio examples/uk_portfolio.cpp)
  target_link_libraries(uk_portfolio PRIVATE sat)
endif()

include(GNUInstallDirs)
install(TARGETS sat EXPORT satTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/sat DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT satTargets NAMESPACE sat:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sat)
