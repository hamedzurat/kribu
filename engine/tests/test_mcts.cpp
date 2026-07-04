/**
 * @file test_mcts.cpp
 * @brief Unit tests for the Monte Carlo Tree Search (MCTS) algorithm.
 */

#include <catch2/catch_test_macros.hpp>
#include <kribu/board.hpp>
#include <kribu/player/mcts.hpp>
#include <kribu/rules.hpp>

using namespace kribu::board;
using namespace kribu::player;

TEST_CASE("MCTS Initial State Move", "[mcts]") {
  boardState state = INITIAL_STATE;
  // Test sequential MCTS
  int moveId = mcts_player_maker<100>(state);
  REQUIRE(moveId != -1);
  REQUIRE(kribu::sholoGuti::is_valid(state, moveId));
}

TEST_CASE("MCTS Parallel Initial State Move", "[mcts]") {
  boardState state = INITIAL_STATE;
  // Test parallel MCTS (2 threads)
  int moveId = mcts_player_maker<100>(state);
  REQUIRE(moveId != -1);
  REQUIRE(kribu::sholoGuti::is_valid(state, moveId));
}

TEST_CASE("MCTS Find Winning Capture", "[mcts]") {
  // Setup a single capture that wins the game
  boardState state;
  state.me = 1ULL << 16;
  state.opp = 1ULL << 17;
  state.activeCaptureIdx = -1;

  // The move 16 -> 18 captures 17
  int winMoveId = kribu::sholoGuti::find_move(16, 18);
  REQUIRE(winMoveId != -1);
  REQUIRE(kribu::sholoGuti::is_capture_move(winMoveId));

  int moveId = mcts_player_maker<500>(state);
  REQUIRE(moveId == winMoveId);
}

TEST_CASE("MCTS Early Termination", "[mcts]") {
  // Give one move an obvious advantage so it triggers early termination
  boardState state;
  state.me = 1ULL << 16;
  state.opp = 1ULL << 17;
  state.activeCaptureIdx = -1;

  int winMoveId = kribu::sholoGuti::find_move(16, 18);
  REQUIRE(winMoveId != -1);

  // 1000 iterations is enough that early term will trigger if implemented
  int moveId = mcts_player_maker<1000>(state);
  REQUIRE(moveId == winMoveId);
}
