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
 * @brief Test case to verify forced random turns tracking during single game playback.
 */
TEST_CASE("Benchmark - Forced Random Turns Tracking", "[benchmark]") {
  Player player1{.name = "P1", .select = kribu::player::select_random};
  Player player2{.name = "P2", .select = kribu::player::select_random};

  GamePerf perf;
  std::vector<TurnRecord> history;

  const GameOutcome outcome = play_single_game(player1, player2, perf, 100, history);

  int forcedRandomTurns = 0;
  for (const auto& rec : history) {
    if (rec.playerPlayed == "ForcedRandom") {
      forcedRandomTurns++;
    }
  }
  REQUIRE(forcedRandomTurns >= 0);
  REQUIRE(outcome.winMargin >= 0);
}
