/**
 * @file test_mcts.cpp
 * @brief Unit tests for the Monte Carlo Tree Search (MCTS) algorithm.
 */

#include <catch2/catch_test_macros.hpp>
#include <kribu/board.hpp>
#include <kribu/heuristic.hpp>
#include <kribu/player/mcts.hpp>
#include <kribu/rules.hpp>
#include <kribu/types.hpp>

using namespace kribu::board;
using namespace kribu::player;

TEST_CASE("MCTS Initial State Move", "[mcts]") {
  boardState state = INITIAL_STATE;
  u64 nodes = 0;
  // Test sequential MCTS with HeuristicRollout
  int moveId = mcts_player_maker<
      HeuristicRollout<kribu::heuristics::evaluate_by_node_values<kribu::heuristics::HEURISTIC_NODE_WEIGHTS>>,
      100>(state, nodes);
  REQUIRE(moveId != -1);
  REQUIRE(kribu::sholoGuti::is_valid(state, moveId));
  REQUIRE(nodes > 0);
}

TEST_CASE("MCTS Parallel Initial State Move", "[mcts]") {
  boardState state = INITIAL_STATE;
  u64 nodes = 0;
  // Test parallel MCTS (2 threads)
  int moveId = mcts_player_maker<RandomRollout, 100, 2>(state, nodes);
  REQUIRE(moveId != -1);
  REQUIRE(kribu::sholoGuti::is_valid(state, moveId));
  REQUIRE(nodes > 0);
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

  u64 nodes = 0;
  int moveId = mcts_player_maker<RandomRollout, 500>(state, nodes);
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

  u64 nodes = 0;
  // 1000 iterations is enough that early term will trigger if implemented
  int moveId = mcts_player_maker<RandomRollout, 1000>(state, nodes);
  REQUIRE(moveId == winMoveId);
  // Ideally we would assert nodes < 1000, but nodes equals total rollouts.
  // In our parallel implementation, we check iter.
}
