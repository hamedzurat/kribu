/**
 * @file greedy.hpp
 * @brief Greedy player move selector for Sholo Guti.
 */

#pragma once

#include <limits>

#include "kribu/board.hpp"
#include "kribu/fast_rng.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/rules.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

using namespace kribu::board;
using namespace kribu::sholoGuti;

/**
 * @brief Selects a move using a template greedy strategy with a custom heuristic.
 * @details If multiple moves share the same highest heuristic evaluation score,
 *          one is selected randomly (50% chance to replace on tie).
 *  * @param state Current board state.
 * @return The selected move ID.
 * @throws std::runtime_error if no moves are available.
 */
[[nodiscard]] inline int greedy_player_maker(const boardState& state) {
  const MoveList moves = all_possible_moves(state);
  if (moves.empty()) {
    return -1;
  }

  int bestMove = -1;
  i32 bestValue = std::numeric_limits<i32>::min();
  int tieCount = 0;

  for (int moveIndex = 0; moveIndex < moves.size(); ++moveIndex) {
    const int moveId = moves.moves[moveIndex];
    const boardState nextState = apply_move(state, moveId);

    const i32 moveValue = nextState.activeCaptureIdx == -1 ? -heuristics::evaluate(flip_board(nextState), 3)
                                                           : heuristics::evaluate(nextState, 3);

    if (moveValue > bestValue) {
      bestValue = moveValue;
      bestMove = moveId;
      tieCount = 1;
    } else if (moveValue == bestValue) {
      ++tieCount;
      if ((rng() % static_cast<u32>(tieCount)) == 0U) {
        bestMove = moveId;
      }
    }
  }

  return bestMove;
}

}  // namespace kribu::player
