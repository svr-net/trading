cmake_minimum_required(VERSION 3.16)
project(ofm VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The library: market data and the neural network; the model's kernels are WGSL (kernels/v2).
add_library(ofm
  src/data.cpp
  src/net.cpp)
target_include_directories(ofm PUBLIC include)
# The kernels' one source is WGSL (kernels/<group>/*.wgsl); the CPU compiles its C++ translation.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(OFM_GENERATED ${CMAKE_BINARY_DIR}/generated)
set(ns_net ofm::net::gen)
set(ns_v2 ofm::v2::gen)
foreach(group net v2)
  file(GLOB wgsl CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/kernels/${group}/*.wgsl)
  add_custom_command(OUTPUT ${OFM_GENERATED}/${group}_kernels.hpp
    COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/wgsl2cpp.py ${CMAKE_SOURCE_DIR}/kernels/${group} ${ns_${group}}
            ${OFM_GENERATED}/${group}_kernels.hpp
    DEPENDS ${wgsl} ${CMAKE_SOURCE_DIR}/tools/wgsl2cpp.py
    COMMENT "Translating kernels/${group}/*.wgsl to C++")
  list(APPEND ofm_generated ${OFM_GENERATED}/${group}_kernels.hpp)
endforeach()
add_custom_target(ofm_kernels DEPENDS ${ofm_generated})
add_dependencies(ofm ofm_kernels)
target_include_directories(ofm PRIVATE ${OFM_GENERATED})
find_package(Threads REQUIRED)
target_link_libraries(ofm PUBLIC Threads::Threads)
if(NOT MSVC)
  target_compile_options(ofm PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)
endif()

add_executable(ofm_research apps/ofm_research.cpp)
target_link_libraries(ofm_research PRIVATE ofm)
target_include_directories(ofm_research PRIVATE ${OFM_GENERATED} ${CMAKE_SOURCE_DIR}/src)
enable_testing()
add_executable(ofm_paths apps/ofm_paths.cpp)

add_executable(ofm_tests tests/tests.cpp)
target_link_libraries(ofm_tests PRIVATE ofm)
target_include_directories(ofm_tests PRIVATE ${OFM_GENERATED} ${CMAKE_SOURCE_DIR}/src)
add_test(NAME ofm_tests COMMAND ofm_tests)
