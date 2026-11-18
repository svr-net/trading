cmake_minimum_required(VERSION 3.16)
project(ofm VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The daily scan's engine (apps/ofm_strategy.cpp) with every injectable strategy (strategies/*.cpp) and compute
# back end (backends/*.cpp): a new file in either folder is built in and chosen by name, nothing else changes.
file(GLOB OFM_STRATEGIES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/strategies/*.cpp)
file(GLOB OFM_BACKENDS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/backends/*.cpp)
find_package(Threads REQUIRED)
add_executable(ofm_strategy apps/ofm_strategy.cpp ${OFM_STRATEGIES} ${OFM_BACKENDS})
target_include_directories(ofm_strategy PRIVATE include)
target_link_libraries(ofm_strategy PRIVATE Threads::Threads)
if(NOT MSVC)
  target_compile_options(ofm_strategy PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)
endif()
# The engine's kernels for the CPUs that run it: on x86-64, x86-64-v3 (AVX2, FMA; every GitHub runner) by default
# (OFM_ARCH= for the baseline). No fused multiply-add contraction and no reordered sums: the same numbers as the
# serial back end and the WebAssembly Lab, bit for bit.
set(OFM_ARCH "x86-64-v3" CACHE STRING "the engine's -march on x86-64 (empty: the compiler's default)")
if(NOT MSVC AND NOT EMSCRIPTEN)
  target_compile_options(ofm_strategy PRIVATE -ffp-contract=off)
  if(OFM_ARCH AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_options(ofm_strategy PRIVATE -march=${OFM_ARCH})
  endif()
endif()

# The loaders and the scan (apps/ofm_eod, apps/ofm_scan): the store svr-net/eod through libgit2, EODData's API and
# the report's mail through libcurl, JSON through nlohmann-json (Ubuntu: libgit2-dev libcurl4-openssl-dev
# nlohmann-json3-dev zlib1g-dev). Not built where they are missing (the WebAssembly Lab needs only the engine).
find_package(PkgConfig QUIET)
find_package(nlohmann_json QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(LIBGIT2 IMPORTED_TARGET libgit2)
  pkg_check_modules(LIBCURL IMPORTED_TARGET libcurl)
  pkg_check_modules(ZLIB IMPORTED_TARGET zlib)
endif()
if(LIBGIT2_FOUND AND LIBCURL_FOUND AND ZLIB_FOUND AND nlohmann_json_FOUND)
  add_executable(ofm_eod apps/ofm_eod.cpp)
  target_include_directories(ofm_eod PRIVATE include)
  target_link_libraries(ofm_eod PRIVATE PkgConfig::LIBGIT2 PkgConfig::LIBCURL PkgConfig::ZLIB nlohmann_json::nlohmann_json)
  if(EXISTS ${CMAKE_SOURCE_DIR}/apps/ofm_scan.cpp)
    add_executable(ofm_scan apps/ofm_scan.cpp)
    target_include_directories(ofm_scan PRIVATE include)
    target_link_libraries(ofm_scan PRIVATE PkgConfig::LIBCURL nlohmann_json::nlohmann_json Threads::Threads)
  endif()
  foreach(t ofm_eod ofm_scan)
    if(TARGET ${t} AND NOT MSVC)
      target_compile_options(${t} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)
    endif()
  endforeach()
else()
  message(STATUS "ofm_eod and ofm_scan not built: libgit2, libcurl, zlib and nlohmann-json are needed")
endif()
