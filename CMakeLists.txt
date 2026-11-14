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
