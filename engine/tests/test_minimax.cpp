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

using namespace kribu::board;
using namespace kribu::player;
using namespace kribu::sholoGuti;

namespace {

/**
 * @brief Builds state10 from the back-and-forth repetition cycle in test_board.cpp.
 */
boardState build_state10_from_repetition_cycle() {
  const int mForward = find_move(21, 16);
  const int mBackward = find_move(16, 21);

  boardState state0 = INITIAL_STATE;
  boardState state1 = flip_board(apply_move(state0, mForward));
  boardState state2Turn1 = flip_board(apply_move(state1, mForward));

  boardState state3 = flip_board(apply_move(state2Turn1, mBackward));
  boardState state4 = flip_board(apply_move(state3, mBackward));

  boardState state5 = flip_board(apply_move(state4, mForward));
  boardState state6 = flip_board(apply_move(state5, mForward));

  boardState state7 = flip_board(apply_move(state6, mBackward));
  boardState state8 = flip_board(apply_move(state7, mBackward));

  boardState state9 = flip_board(apply_move(state8, mForward));
  return flip_board(apply_move(state9, mForward));
}

}  // namespace

TEST_CASE("Minimax Initial State Evaluation", "[minimax]") {
  boardState state = INITIAL_STATE;
  REQUIRE(kribu::heuristics::evaluate(state, 2) == 0);
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
  MinimaxResult res = minimax(state, 3, -INFINITY_VAL, INFINITY_VAL, &transpositionTable);
  REQUIRE(res.moveId == winMoveId);
  REQUIRE(res.score >= INFINITY_VAL);
}

TEST_CASE("Player Maker Interface", "[minimax]") {
  boardState state = INITIAL_STATE;
  int moveId = minimax_player_maker<2>(state);
  REQUIRE(moveId != -1);
  REQUIRE(is_valid(state, moveId));
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

TEST_CASE("Minimax ignores invalid TT move at root", "[minimax]") {
  boardState state = INITIAL_STATE;
  kribu::TranspositionTable transpositionTable(1024);

  const int invalidMoveId = 9999;
  transpositionTable.store(state.hash, 2, 100, invalidMoveId, kribu::TTFlag::EXACT);

  MinimaxResult res = minimax(state, 2, -INFINITY_VAL, INFINITY_VAL, &transpositionTable);

  REQUIRE(res.moveId != invalidMoveId);
  REQUIRE(res.moveId != -1);
  REQUIRE(is_valid(state, res.moveId));
}

TEST_CASE("Minimax rejects stale TT move when repetition history differs", "[minimax]") {
  const int mBackward = find_move(16, 21);
  REQUIRE(mBackward != -1);

  boardState state10 = build_state10_from_repetition_cycle();
  REQUIRE_FALSE(is_valid(state10, mBackward));

  boardState state10Padded = state10;
  for (int k = 0; k < 8; ++k) {
    state10Padded.history[k] = 0x9999ULL + static_cast<u64>(k);
  }
  state10Padded.historyCount = 8;
  REQUIRE(state10.hash == state10Padded.hash);
  REQUIRE(is_valid(state10Padded, mBackward));

  kribu::TranspositionTable transpositionTable(1024);
  transpositionTable.store(state10.hash, 8, 0, mBackward, kribu::TTFlag::EXACT);

  MinimaxResult res = minimax(state10, 4, -INFINITY_VAL, INFINITY_VAL, &transpositionTable);
  REQUIRE(res.moveId != mBackward);
  REQUIRE(is_valid(state10, res.moveId));
}

TEST_CASE("Minimax player maker stays legal through repetition cycle", "[minimax]") {
  boardState state = INITIAL_STATE;
  bool isP1Turn = true;

  for (int turn = 0; turn < 12; ++turn) {
    const int moveId = minimax_player_maker<4>(state);
    REQUIRE(moveId != -1);
    REQUIRE(is_valid(state, moveId));

    const boardState nextState = apply_move(state, moveId);
    if (nextState.activeCaptureIdx == -1) {
      state = flip_board(nextState);
      isP1Turn = !isP1Turn;
    } else {
      state = nextState;
    }
    (void) isP1Turn;
  }
}
