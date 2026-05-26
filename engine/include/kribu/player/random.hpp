/**
 * @file random.hpp
 * @brief Random player move selector for Sholo Guti.
 */

#pragma once

#include <random>

#include "kribu/board.hpp"
#include "kribu/rules.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

using namespace kribu::board;
using namespace kribu::sholoGuti;

/**
 * @brief Selects a random move from the valid move list.
 * @param state Current board state.
 * @param nodes Out-parameter tracking explored states (set to 1).
 * @return The selected move ID, or -1 if no moves are available.
 */
inline int select_random(const boardState& state, u64& nodes) {
  nodes = 1;
  const MoveList moves = all_possible_moves(state);
  if (moves.empty()) {
    return -1;
  }
  std::uniform_int_distribution<int> dist(0, moves.size() - 1);
  return moves.moves[dist(rng)];
}

}  // namespace kribu::player
