/**
 * @file player.hpp
 * @brief Centralized player configurations and factory instantiations for Sholo Guti.
 */

#pragma once

#include "kribu/board.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/player/greedy.hpp"
#include "kribu/player/minimax.hpp"
#include "kribu/player/random.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

/**
 * @struct PieceCountEval
 * @brief Evaluation functor wrapping the piece-count heuristic.
 */
struct PieceCountEval {
  /**
   * @brief Evaluates the board state using piece count.
   * @param state The current board state.
   * @return The evaluation score.
   */
  constexpr i32 operator()(const boardState& state) const noexcept { return heuristics::evaluate_piece_count(state); }
};

/**
 * @struct NodeWeightsEval
 * @brief Evaluation functor wrapping the node-weight position heuristic.
 */
struct NodeWeightsEval {
  /**
   * @brief Evaluates the board state using node weights.
   * @param state The current board state.
   * @return The evaluation score.
   */
  constexpr i32 operator()(const boardState& state) const noexcept {
    return heuristics::evaluate_by_node_values<heuristics::HEURISTIC_NODE_WEIGHTS>(state);
  }
};

/**
 * @brief Selects a random move.
 * @param state The current board state.
 * @param nodes Count of visited nodes.
 * @return Chosen move ID, or -1 if none.
 */
inline int select_random_player(const boardState& state, u64& nodes) {
  return select_random(state, nodes);
}

/**
 * @brief Selects a move using the greedy strategy with node weight evaluations.
 * @param state The current board state.
 * @param nodes Count of visited nodes.
 * @return Chosen move ID, or -1 if none.
 */
inline int select_greedy(const boardState& state, u64& nodes) {
  return greedy_player_maker<NodeWeightsEval>(state, nodes);
}

/**
 * @brief Minimax player with piece count heuristic at depth 4.
 * @param state The current board state.
 * @param nodes Count of visited nodes.
 * @return Chosen move ID, or -1 if none.
 */
inline int select_minimax_piece_d4(const boardState& state, u64& nodes) {
  return minimax_player_maker<PieceCountEval, 4>(state, nodes);
}

/**
 * @brief Minimax player with position-based heuristic at depth 4.
 * @param state The current board state.
 * @param nodes Count of visited nodes.
 * @return Chosen move ID, or -1 if none.
 */
inline int select_minimax_position_d4(const boardState& state, u64& nodes) {
  return minimax_player_maker<NodeWeightsEval, 4>(state, nodes);
}

}  // namespace kribu::player
