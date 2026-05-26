/**
 * @file minimax.hpp
 * @brief Minimax search algorithm with alpha-beta pruning for Sholo Guti.
 */

#pragma once

// Note: Alpha-beta minimax search is optimized by ordering capture moves first.

#include <algorithm>
#include <array>
#include <future>
#include <tuple>
#include <type_traits>

#include "kribu/board.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/rules.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

using namespace kribu::board;
using namespace kribu::sholoGuti;

/**
 * @brief Infinity value used as bounds in alpha-beta pruning.
 */
constexpr i32 INFINITY_VAL = 1000000000;

/**
 * @struct MinimaxResult
 * @brief Represents the output of a minimax search, containing the best score and the best move ID.
 */
struct MinimaxResult {
  /**
   * @brief The evaluation score of the best move found.
   */
  i32 score = 0;

  /**
   * @brief The move ID of the best move found.
   */
  int moveId = -1;
};

/**
 * @struct OrderedMoveList
 * @brief Stores an ordered array of move IDs and the count of elements.
 */
struct OrderedMoveList {
  /**
   * @brief Array buffer storing the ordered move IDs.
   */
  std::array<i16, MAX_MOVES_PER_STATE> moves{};

  /**
   * @brief The number of elements currently stored in the list.
   */
  int count = 0;
};

/**
 * @brief Orders a MoveList to prioritize capture moves for optimal alpha-beta cutoffs.
 * @param moves The original MoveList to order.
 * @return OrderedMoveList containing capture moves first, then simple/end-chain moves.
 */
[[nodiscard]] constexpr OrderedMoveList order_moves(const MoveList& moves) noexcept {
  OrderedMoveList ordered;
  for (int i = 0; i < moves.count; ++i) {
    if (is_capture_move(moves.moves[i])) {
      ordered.moves[ordered.count++] = moves.moves[i];
    }
  }
  for (int i = 0; i < moves.count; ++i) {
    if (!is_capture_move(moves.moves[i])) {
      ordered.moves[ordered.count++] = moves.moves[i];
    }
  }
  return ordered;
}

template <typename EvalFunc>
[[nodiscard]] constexpr MinimaxResult minimax(
    const boardState& state, int depth, i32 alpha, i32 beta, u64& nodeCounter, EvalFunc&& eval) noexcept;

/**
 * @brief Performs sequential alpha-beta minimax search recursively.
 */
template <typename EvalFunc>
[[nodiscard]] constexpr MinimaxResult minimax_seq_helper(
    const boardState& state, int depth, i32 alpha, i32 beta, u64& nodeCounter, EvalFunc&& eval) noexcept {
  nodeCounter++;

  if (piece_count(state.opp) == 0) {
    return MinimaxResult{.score = INFINITY_VAL + depth, .moveId = -1};
  }
  if (piece_count(state.me) == 0) {
    return MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
  }

  if (depth <= 0) {
    return MinimaxResult{.score = eval(state), .moveId = -1};
  }

  const MoveList moves = all_possible_moves(state);
  if (moves.empty()) {
    return MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
  }

  const OrderedMoveList ordered = order_moves(moves);
  int bestMoveId = -1;
  i32 bestScore = -INFINITY_VAL - 10000;

  for (int i = 0; i < ordered.count; ++i) {
    const int moveId = ordered.moves[i];
    const boardState nextState = apply_move(state, moveId);

    i32 score = 0;
    if (nextState.activeCaptureIdx == -1) {
      const boardState flippedState = flip_board(nextState);
      const MinimaxResult res = minimax_seq_helper(flippedState, depth - 1, -beta, -alpha, nodeCounter, eval);
      score = -res.score;
    } else {
      const MinimaxResult res = minimax_seq_helper(nextState, depth - 1, alpha, beta, nodeCounter, eval);
      score = res.score;
    }

    if (score > bestScore) {
      bestScore = score;
      bestMoveId = moveId;
    }

    alpha = std::max(alpha, bestScore);
    if (alpha >= beta) {
      break;
    }
  }

  return MinimaxResult{.score = bestScore, .moveId = bestMoveId};
}

/**
 * @brief Performs root-level parallel minimax search.
 */
template <typename EvalFunc>
// TODO: if possible make this constexpr
[[nodiscard]] inline MinimaxResult minimax_parallel_root(const boardState& state,
                                                         int depth,
                                                         i32 alpha,
                                                         i32 beta,
                                                         u64& nodeCounter,
                                                         EvalFunc&& eval,
                                                         const OrderedMoveList& ordered) noexcept {
  std::array<std::future<std::tuple<int, MinimaxResult, u64>>, MAX_MOVES_PER_STATE> futures;

  for (int i = 0; i < ordered.count; ++i) {
    const int moveId = ordered.moves[i];
    const boardState nextState = apply_move(state, moveId);

    futures[i] = std::async(std::launch::async, [nextState, depth, alpha, beta, eval, moveId]() {
      u64 localNodes = 0;
      MinimaxResult res;
      if (nextState.activeCaptureIdx == -1) {
        const boardState flippedState = flip_board(nextState);
        res = minimax_seq_helper(flippedState, depth - 1, -beta, -alpha, localNodes, eval);
        res.score = -res.score;
      } else {
        res = minimax_seq_helper(nextState, depth - 1, alpha, beta, localNodes, eval);
      }
      return std::make_tuple(moveId, res, localNodes);
    });
  }

  int bestMoveId = -1;
  i32 bestScore = -INFINITY_VAL - 10000;

  for (int i = 0; i < ordered.count; ++i) {
    auto [moveId, res, localNodes] = futures[i].get();
    nodeCounter += localNodes;
    if (res.score > bestScore) {
      bestScore = res.score;
      bestMoveId = moveId;
    }
  }

  return MinimaxResult{.score = bestScore, .moveId = bestMoveId};
}

/**
 * @brief Executes a minimax search with alpha-beta pruning.
 * @param state The current board state to search from.
 * @param depth The maximum search depth remaining.
 * @param alpha The lower bound score of the search window.
 * @param beta The upper bound score of the search window.
 * @param nodeCounter Counter tracking evaluated / visited nodes.
 * @param eval The evaluation function.
 * @return A MinimaxResult containing the best score and best move ID.
 */
template <typename EvalFunc>
// TODO: name it minmax_player_maker
[[nodiscard]] constexpr MinimaxResult minimax(
    const boardState& state, int depth, i32 alpha, i32 beta, u64& nodeCounter, EvalFunc&& eval) noexcept {
  nodeCounter++;

  if (piece_count(state.opp) == 0) {
    return MinimaxResult{.score = INFINITY_VAL + depth, .moveId = -1};
  }
  if (piece_count(state.me) == 0) {
    return MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
  }

  if (depth <= 0) {
    return MinimaxResult{.score = eval(state), .moveId = -1};
  }

  const MoveList moves = all_possible_moves(state);
  if (moves.empty()) {
    return MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
  }

  const OrderedMoveList ordered = order_moves(moves);

  if (!std::is_constant_evaluated() && depth > 1 && ordered.count > 1) {
    return minimax_parallel_root(state, depth, alpha, beta, nodeCounter, eval, ordered);
  }

  return minimax_seq_helper(state, depth, alpha, beta, nodeCounter, eval);
}

/**
 * @brief Compatibility wrapper for minimax search.
 */
[[nodiscard]] constexpr MinimaxResult minimax(const boardState& state, int depth, i32 alpha, i32 beta) noexcept {
  u64 dummy = 0;
  return minimax(state, depth, alpha, beta, dummy, [](const boardState& board) noexcept {
    return heuristics::evaluate_by_node_values<heuristics::HEURISTIC_NODE_WEIGHTS>(board);
  });
}

/**
 * @brief Player maker utilizing minimax search.
 * @tparam EvalFunc Functor type evaluating the board state.
 * @tparam Depth The search depth.
 * @param state The current board state.
 * @param nodes Out-parameter tracking explored states.
 * @return The selected move ID.
 */
template <typename EvalFunc, int Depth>
[[nodiscard]] constexpr int minimax_player_maker(const boardState& state, u64& nodes) {
  MinimaxResult res = minimax(state, Depth, -INFINITY_VAL, INFINITY_VAL, nodes, EvalFunc{});
  return res.moveId;
}

}  // namespace kribu::player
