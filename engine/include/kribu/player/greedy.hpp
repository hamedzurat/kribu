/**
 * @file greedy.hpp
 * @brief Greedy player move selector for Sholo Guti.
 */

#pragma once

#include <stdexcept>

#include "kribu/board.hpp"
#include "kribu/fast_rng.hpp"
#include "kribu/rules.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

using namespace kribu::board;
using namespace kribu::sholoGuti;

/**
 * @brief Selects a move using a template greedy strategy with a custom heuristic.
 * @details If multiple moves share the same highest heuristic evaluation score,
 *          one is selected uniformly at random using reservoir sampling.
 * @tparam EvalFunc Heuristic function evaluating the board state.
 * @param state Current board state.
 * @param nodes Out-parameter tracking explored states.
 * @return The selected move ID.
 * @throws std::runtime_error if no moves are available.
 */
template <auto EvalFunc>
[[nodiscard]] constexpr int greedy_player_maker(const boardState& state, u64& nodes) {
  const MoveList moves = all_possible_moves(state);
  if (moves.empty()) {
    throw std::runtime_error("Greedy player: no moves available!");
  }
  int bestMove = -1;
  i32 bestVal = -999999;
  int bestCount = 0;
  for (int idx = 0; idx < moves.size(); ++idx) {
    nodes++;
    const int moveId = moves.moves[idx];
    const boardState next = apply_move(state, moveId);
    i32 val = 0;
    if (next.activeCaptureIdx == -1) {
      val = -EvalFunc(flip_board(next));
    } else {
      val = EvalFunc(next);
    }
    if (val > bestVal) {
      bestVal = val;
      bestMove = moveId;
      bestCount = 1;
    } else if (val == bestVal) {
      bestCount++;
      if ((rng() % bestCount) == 0) {
        bestMove = moveId;
      }
    }
  }
  return bestMove;
}

}  // namespace kribu::player
