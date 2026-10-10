cmake_minimum_required(VERSION 3.16)
project(ofm VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(ofm
  src/data.cpp
  src/model.cpp
  src/kernels.cpp
  src/trade.cpp
  src/structure.cpp
  src/report.cpp
  src/topk.cpp)
target_include_directories(ofm PUBLIC include)
if(NOT MSVC)
  target_compile_options(ofm PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)
endif()

if(EMSCRIPTEN)
  # emcmake cmake -S . -B build-wasm && cmake --build build-wasm  ->  web/wasm/ofm.js + ofm.wasm
  target_compile_options(ofm PRIVATE -fwasm-exceptions)
  add_executable(ofm_wasm wasm/bindings.cpp)
  target_compile_options(ofm_wasm PRIVATE -fwasm-exceptions)
  target_compile_definitions(ofm_wasm PRIVATE OFM_VERSION="${PROJECT_VERSION}")
  target_link_libraries(ofm_wasm PRIVATE ofm)
  target_link_options(ofm_wasm PRIVATE -fwasm-exceptions -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createOfm
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sENVIRONMENT=web,worker,node -sFILESYSTEM=0 -O3)
  set_target_properties(ofm_wasm PROPERTIES OUTPUT_NAME ofm RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/web/wasm)
else()
  add_executable(ofm_report apps/ofm_report.cpp)
  target_link_libraries(ofm_report PRIVATE ofm)
  enable_testing()
  add_executable(ofm_tests tests/tests.cpp)
  target_link_libraries(ofm_tests PRIVATE ofm)
  add_test(NAME ofm_tests COMMAND ofm_tests)
endif()
