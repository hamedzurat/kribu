/**
 * @file test_benchmark.cpp
 * @brief Unit tests for the benchmark / simulation engine.
 */

#include <catch2/catch_test_macros.hpp>
#include <kribu/player/random.hpp>
#include <vector>

#include "../benchmark/benchmark.hpp"

using namespace kribu::benchmark;

/**
 * @brief Test case to verify benchmark game play run simulation.
 */
TEST_CASE("Benchmark - Game Run Simulation", "[benchmark]") {
  Player player1{.name = "P1", .select = kribu::player::select_random};
  Player player2{.name = "P2", .select = kribu::player::select_random};

  GamePerf perf;
  std::vector<TurnRecord> history;

  const GameOutcome outcome = play_single_game(player1, player2, perf, 100, history);

  REQUIRE(!history.empty());
  REQUIRE(outcome.winMargin >= 0);
}
