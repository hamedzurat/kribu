# Engine sources will be added here
add_library(kribu_engine INTERFACE)
target_include_directories(kribu_engine INTERFACE include)

# Setup CPM packages
cpmaddpackage(
  NAME
  eve
  GITHUB_REPOSITORY
  jfalcou/eve
  VERSION
  2023.02.15
  OPTIONS
  "EVE_BUILD_TEST OFF"
  "EVE_BUILD_DOCUMENTATION OFF"
  GIT_SHALLOW
  1)

target_link_libraries(kribu_engine INTERFACE eve::eve)

# Setup Catch2 tests using CPM
cpmaddpackage(
  NAME
  Catch2
  GITHUB_REPOSITORY
  catchorg/Catch2
  VERSION
  3.4.0
  GIT_SHALLOW
  1)

# Engine tests will be added here
add_executable(
  kribu_engine_tests
  tests/test_board.cpp tests/test_minimax.cpp tests/test_mcts.cpp
  tests/test_fast_rng.cpp tests/test_benchmark.cpp)
target_link_libraries(kribu_engine_tests PRIVATE kribu_engine
                                                 Catch2::Catch2WithMain)
target_include_directories(kribu_engine_tests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmark)

# Include Catch2 test discovery so ctest can find them
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(kribu_engine_tests)

add_subdirectory(benchmark)
