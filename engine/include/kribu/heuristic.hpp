/**
 * @file heuristic.hpp
 * @brief Board evaluation heuristics for the Sholo Guti engine.
 */

#pragma once

#include <array>
#include <bit>

#include "board.hpp"
#include "rules.hpp"
#include "types.hpp"

namespace kribu::heuristics {

using namespace kribu::board;

/**
 * @brief Default values for each of the 37 nodes on the board.
 * @details Values are initialized to 100 to replicate the baseline piece-count heuristic.
 */
constexpr std::array<i32, NUM_NODES> HEURISTIC_NODE_WEIGHTS = {
    // 120────────────160────────────120
    //   ╲             │             ╱
    //     ╲           │           ╱
    //       ╲         │         ╱
    //         110────150────110
    //           ╲      │      ╱
    //             ╲    │    ╱
    //               ╲  │  ╱
    // 100────110─────140─────110────100
    // │ ╲     │     ╱ │ ╲     │     ╱ │
    // │   ╲   │   ╱   │   ╲   │   ╱   │
    // │     ╲ │ ╱     │     ╲ │ ╱     │
    // 80──────90─────120──────90──────80
    // │     ╱ │ ╲     │     ╱ │ ╲     │
    // │   ╱   │   ╲   │   ╱   │   ╲   │
    // │ ╱     │     ╲ │ ╱     │     ╲ │
    // 65──────80─────100──────80──────65
    // │ ╲     │     ╱ │ ╲     │     ╱ │
    // │   ╲   │   ╱   │   ╲   │   ╱   │
    // │     ╲ │ ╱     │     ╲ │ ╱     │
    // 50──────70──────90──────70──────50
    // │     ╱ │ ╲     │     ╱ │ ╲     │
    // │   ╱   │   ╲   │   ╱   │   ╲   │
    // │ ╱     │     ╲ │ ╱     │     ╲ │
    // 30──────40──────65──────40──────30
    //               ╱  │  ╲
    //             ╱    │    ╲
    //           ╱      │      ╲
    //         20──────40──────20
    //       ╱          │         ╲
    //     ╱            │           ╲
    //   ╱              │             ╲
    // 10──────────────20──────────────10

    // Top triangle / Crown
    120,
    160,
    120,
    110,
    150,
    110,

    // Main grid rows
    100,
    110,
    140,
    110,
    100,
    80,
    90,
    120,
    90,
    80,
    65,
    80,
    100,
    80,
    65,
    50,
    70,
    90,
    70,
    50,
    30,
    40,
    65,
    40,
    30,

    // Bottom triangle
    20,
    40,
    20,
    10,
    20,
    10};

/**
 * @brief Evaluates a board state by summing specified values for each player's pieces.
 * @param state The board state to evaluate.
 * @param nodeValues Array of weights for each node index.
 * @return The static evaluation score. Positive values favor the active player.
 */
template <const std::array<i32, NUM_NODES>& Weights>
[[nodiscard]] constexpr i32 evaluate_by_node_values(const boardState& state) noexcept {
  i32 score = 0;
  for (u64 bits = state.me; bits != 0ULL; bits &= bits - 1ULL) {
    score += Weights[std::countr_zero(bits)];
  }
  for (u64 bits = state.opp; bits != 0ULL; bits &= bits - 1ULL) {
    score -= Weights[std::countr_zero(bits)];
  }
  return score;
}

/**
 * @brief Simple piece-count evaluation multiplied by 100.
 * @param state The board state to evaluate.
 * @return The static evaluation score.
 */
[[nodiscard]] constexpr i32 evaluate_piece_count(const boardState& state) noexcept {
  return 100 * (static_cast<i32>(std::popcount(state.me)) - static_cast<i32>(std::popcount(state.opp)));
}

/**
 * @brief Evaluates a board state based on the mobility (number of legal moves) difference.
 * @param state The board state to evaluate.
 * @return The mobility difference score.
 */
[[nodiscard]] inline i32 evaluate_mobility(const boardState& state) noexcept {
  int activeMoves = sholoGuti::all_possible_moves(state).size();
  int oppMoves = sholoGuti::all_possible_moves(sholoGuti::flip_board(state)).size();
  return 10 * (activeMoves - oppMoves);
}

/**
 * @brief Combines piece count, mobility, and positional node weights.
 * @param state The board state to evaluate.
 * @return The combined evaluation score.
 */
[[nodiscard]] inline i32 evaluate_mixed(const boardState& state) noexcept {
  i32 pieces = evaluate_piece_count(state);
  i32 mobility = evaluate_mobility(state);
  i32 position = evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>(state);
  return pieces + mobility + (position / 4);
}

}  // namespace kribu::heuristics
