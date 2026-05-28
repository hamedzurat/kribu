/**
 * @file test_minimax.cpp
 * @brief Unit tests for the minimax search algorithm.
 */

#include <catch2/catch_test_macros.hpp>
#include <kribu/board.hpp>
#include <kribu/heuristic.hpp>
#include <kribu/player/minimax.hpp>
#include <kribu/rules.hpp>
#include <kribu/transposition_table.hpp>
#include <kribu/types.hpp>

using namespace kribu::board;
using namespace kribu::player;

TEST_CASE("Minimax Initial State Evaluation", "[minimax]") {
  boardState state = INITIAL_STATE;
  REQUIRE(kribu::heuristics::evaluate_by_node_values<kribu::heuristics::HEURISTIC_NODE_WEIGHTS>(state) == -1135);
}

TEST_CASE("Minimax Terminal State Evaluation", "[minimax]") {
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

TEST_CASE("Minimax Choose Winning Move", "[minimax]") {
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

TEST_CASE("Iterative Deepening Finds Winning Move", "[minimax]") {
  boardState state;
  state.me = 1ULL << 16;
  state.opp = 1ULL << 17;
  state.activeCaptureIdx = -1;

  int winMoveId = find_move(16, 18);
  REQUIRE(winMoveId != -1);

  // Iterative deepening at depth 3 should still find the winning capture
  kribu::TranspositionTable transpositionTable(1048576);
  MinimaxResult res = minimax<kribu::heuristics::evaluate_by_node_values<kribu::heuristics::HEURISTIC_NODE_WEIGHTS>>(
      state, 3, -INFINITY_VAL, INFINITY_VAL, &transpositionTable);
  REQUIRE(res.moveId == winMoveId);
  REQUIRE(res.score >= INFINITY_VAL);
}

TEST_CASE("Player Maker Interface", "[minimax]") {
  boardState state = INITIAL_STATE;
  int moveId =
      minimax_player_maker<kribu::heuristics::evaluate_by_node_values<kribu::heuristics::HEURISTIC_NODE_WEIGHTS>, 2>(
          state);
  REQUIRE(moveId != -1);
  REQUIRE(kribu::sholoGuti::is_valid(state, moveId));
}

TEST_CASE("Killer Table Store And Query", "[minimax]") {  // NOLINT(readability-function-cognitive-complexity)
  KillerTable killerTable;
  killerTable.store(5, 42);
  REQUIRE(killerTable.is_killer(5, 42));
  REQUIRE_FALSE(killerTable.is_killer(5, 99));
  REQUIRE_FALSE(killerTable.is_killer(3, 42));

  // Second killer replaces slot 1, slot 0 keeps the first
  killerTable.store(5, 77);
  REQUIRE(killerTable.is_killer(5, 77));
  REQUIRE(killerTable.is_killer(5, 42));

  killerTable.clear();
  REQUIRE_FALSE(killerTable.is_killer(5, 42));
  REQUIRE_FALSE(killerTable.is_killer(5, 77));
}
