/**
 * @file heuristic.hpp
 * @brief Board evaluation heuristics for the Sholo Guti engine.
 */

#pragma once

#include <array>
#include <bit>

#include "board.hpp"
#include "types.hpp"

namespace kribu::heuristics {

using namespace kribu::board;

/**
 * @brief Default values for each of the 37 nodes on the board.
 * @details Values are initialized to 100 to replicate the baseline piece-count heuristic.
 */
constexpr std::array<i32, NUM_NODES> HEURISTIC_NODE_WEIGHTS = []() constexpr {
  std::array<i32, NUM_NODES> values{};

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

  values[0] = 120;
  values[1] = 160;
  values[2] = 120;
  values[3] = 110;
  values[4] = 150;
  values[5] = 110;
  values[6] = 100;
  values[7] = 110;
  values[8] = 140;
  values[9] = 110;
  values[10] = 100;
  values[11] = 80;
  values[12] = 90;
  values[13] = 120;
  values[14] = 90;
  values[15] = 80;
  values[16] = 65;
  values[17] = 80;
  values[18] = 100;
  values[19] = 80;
  values[20] = 65;
  values[21] = 50;
  values[22] = 70;
  values[23] = 90;
  values[24] = 70;
  values[25] = 50;
  values[26] = 30;
  values[27] = 40;
  values[28] = 65;
  values[29] = 40;
  values[30] = 30;
  values[31] = 20;
  values[32] = 40;
  values[33] = 20;
  values[34] = 10;
  values[35] = 20;
  values[36] = 10;

  return values;
}();

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

}  // namespace kribu::heuristics
