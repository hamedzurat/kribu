/**
 * @file test_minimax.cpp
 * @brief Unit tests for the minimax search algorithm.
 */

#include <catch2/catch_test_macros.hpp>
#include <kribu/board.hpp>
#include <kribu/heuristic.hpp>
#include <kribu/player/minimax.hpp>
#include <kribu/rules.hpp>

using namespace kribu::board;
using namespace kribu::player;

TEST_CASE("Minimax Initial State Evaluation", "[minimax]") {  // NOLINT(readability-function-cognitive-complexity)
  boardState state = INITIAL_STATE;
  REQUIRE(kribu::heuristics::evaluate_by_node_values<kribu::heuristics::HEURISTIC_NODE_WEIGHTS>(state) == -1135);
}

TEST_CASE("Minimax Terminal State Evaluation", "[minimax]") {  // NOLINT(readability-function-cognitive-complexity)
  boardState winState;
  winState.me = 1ULL << 0;
  winState.opp = 0;
  MinimaxResult resWin = minimax(winState, 2, -INFINITY_VAL, INFINITY_VAL);
  REQUIRE(resWin.score >= INFINITY_VAL);
  REQUIRE(resWin.moveId == -1);

  boardState loseState;
  loseState.me = 0;
  loseState.opp = 1ULL << 0;
  MinimaxResult resLose = minimax(loseState, 2, -INFINITY_VAL, INFINITY_VAL);
  REQUIRE(resLose.score <= -INFINITY_VAL);
  REQUIRE(resLose.moveId == -1);
}

TEST_CASE("Minimax Choose Winning Move", "[minimax]") {  // NOLINT(readability-function-cognitive-complexity)
  // Setup a single capture that wins the game
  boardState state;
  state.me = 1ULL << 16;
  state.opp = 1ULL << 17;
  state.activeCaptureIdx = -1;

  // The move 16 -> 18 captures 17
  int winMoveId = find_move(16, 18);
  REQUIRE(winMoveId != -1);
  REQUIRE(is_capture_move(winMoveId));

  MinimaxResult res = minimax(state, 1, -INFINITY_VAL, INFINITY_VAL);
  REQUIRE(res.moveId == winMoveId);
  REQUIRE(res.score >= INFINITY_VAL);
}
