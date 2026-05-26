/**
 * @file minimax.hpp
 * @brief Minimax search algorithm with alpha-beta pruning for Sholo Guti.
 */

#pragma once

#include <algorithm>
#include <array>
#include <future>
#include <random>
#include <tuple>
#include <type_traits>

#include "kribu/board.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/player/random.hpp"
#include "kribu/rules.hpp"
#include "kribu/transposition_table.hpp"
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
[[nodiscard]] constexpr OrderedMoveList order_moves(const MoveList& moves, int ttMoveId = -1) noexcept {
  OrderedMoveList ordered;
  if (ttMoveId != -1) {
    for (int i = 0; i < moves.count; ++i) {
      if (moves.moves[i] == ttMoveId) {
        ordered.moves[ordered.count++] = moves.moves[i];
        break;
      }
    }
  }
  for (int i = 0; i < moves.count; ++i) {
    if (moves.moves[i] == ttMoveId) {
      continue;
    }
    if (is_capture_move(moves.moves[i])) {
      ordered.moves[ordered.count++] = moves.moves[i];
    }
  }
  for (int i = 0; i < moves.count; ++i) {
    if (moves.moves[i] == ttMoveId) {
      continue;
    }
    if (!is_capture_move(moves.moves[i])) {
      ordered.moves[ordered.count++] = moves.moves[i];
    }
  }
  return ordered;
}

template <auto EvalFunc>
[[nodiscard]] constexpr MinimaxResult minimax(const boardState& state,
                                              int depth,
                                              i32 alpha,
                                              i32 beta,
                                              u64& nodeCounter,
                                              TranspositionTable* transTable = nullptr) noexcept;

/**
 * @brief Performs quiescence search to evaluate captures recursively.
 */
template <auto EvalFunc>
[[nodiscard]] constexpr MinimaxResult quiescence_search(const boardState& state,
                                                        i32 alpha,
                                                        i32 beta,
                                                        u64& nodeCounter) noexcept {
  nodeCounter++;

  i32 standPat = EvalFunc(state);
  if (standPat >= beta) {
    return MinimaxResult{.score = beta, .moveId = -1};
  }
  alpha = std::max(alpha, standPat);

  const MoveList moves = all_possible_moves(state);
  if (moves.empty()) {
    return MinimaxResult{.score = standPat, .moveId = -1};
  }

  OrderedMoveList ordered;
  for (int i = 0; i < moves.count; ++i) {
    if (is_capture_move(moves.moves[i]) || (state.activeCaptureIdx != -1 && moves.moves[i] == END_CHAIN_MOVE)) {
      ordered.moves[ordered.count++] = moves.moves[i];
    }
  }

  if (ordered.count == 0) {
    return MinimaxResult{.score = standPat, .moveId = -1};
  }

  int bestMoveId = -1;
  i32 bestScore = standPat;

  for (int i = 0; i < ordered.count; ++i) {
    const int moveId = ordered.moves[i];
    const boardState nextState = apply_move(state, moveId);

    i32 score = 0;
    if (nextState.activeCaptureIdx == -1) {
      const boardState flippedState = flip_board(nextState);
      const MinimaxResult res = quiescence_search<EvalFunc>(flippedState, -beta, -alpha, nodeCounter);
      score = -res.score;
    } else {
      const MinimaxResult res = quiescence_search<EvalFunc>(nextState, alpha, beta, nodeCounter);
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
 * @brief Performs sequential alpha-beta minimax search recursively.
 */
template <auto EvalFunc>
[[nodiscard]] constexpr MinimaxResult minimax_seq_helper(const boardState& state,
                                                         int depth,
                                                         i32 alpha,
                                                         i32 beta,
                                                         u64& nodeCounter,
                                                         TranspositionTable* transTable = nullptr) noexcept {
  nodeCounter++;

  if (!std::is_constant_evaluated() && maxRepetitions > 0 && !currentGameHistory.empty()) {
    int repetitions = 0;
    for (u64 prevHash : currentGameHistory) {
      if (prevHash == state.hash) {
        repetitions++;
        if (repetitions >= maxRepetitions - 1) {
          return MinimaxResult{.score = 0, .moveId = -1};
        }
      }
    }
  }

  if (piece_count(state.opp) == 0) {
    return MinimaxResult{.score = INFINITY_VAL + depth, .moveId = -1};
  }
  if (piece_count(state.me) == 0) {
    return MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
  }

  if (depth <= 0) {
    return quiescence_search<EvalFunc>(state, alpha, beta, nodeCounter);
  }

  i32 ttScore = 0;
  int ttMoveId = -1;
  if (transTable && transTable->probe(state.hash, depth, alpha, beta, ttScore, ttMoveId)) {
    return MinimaxResult{.score = ttScore, .moveId = ttMoveId};
  }

  const MoveList moves = all_possible_moves(state);
  if (moves.empty()) {
    return MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
  }

  const OrderedMoveList ordered = order_moves(moves, ttMoveId);
  int bestMoveId = -1;
  i32 bestScore = -INFINITY_VAL - 10000;
  i32 originalAlpha = alpha;

  for (int i = 0; i < ordered.count; ++i) {
    const int moveId = ordered.moves[i];
    const boardState nextState = apply_move(state, moveId);

    i32 score = 0;
    if (nextState.activeCaptureIdx == -1) {
      const boardState flippedState = flip_board(nextState);
      const MinimaxResult res =
          minimax_seq_helper<EvalFunc>(flippedState, depth - 1, -beta, -alpha, nodeCounter, transTable);
      score = -res.score;
    } else {
      const MinimaxResult res =
          minimax_seq_helper<EvalFunc>(nextState, depth - 1, alpha, beta, nodeCounter, transTable);
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

  if (transTable) {
    TTFlag flag = TTFlag::EXACT;
    if (bestScore <= originalAlpha) {
      flag = TTFlag::ALPHA;
    } else if (bestScore >= beta) {
      flag = TTFlag::BETA;
    }
    transTable->store(state.hash, depth, bestScore, bestMoveId, flag);
  }

  return MinimaxResult{.score = bestScore, .moveId = bestMoveId};
}

/**
 * @brief Performs root-level parallel minimax search.
 */
template <auto EvalFunc>
[[nodiscard]] inline MinimaxResult minimax_parallel_root(const boardState& state,
                                                         int depth,
                                                         i32 alpha,
                                                         i32 beta,
                                                         u64& nodeCounter,
                                                         const OrderedMoveList& ordered) noexcept {
  std::array<std::future<std::tuple<int, MinimaxResult, u64>>, MAX_MOVES_PER_STATE> futures;

  for (int i = 0; i < ordered.count; ++i) {
    const int moveId = ordered.moves[i];
    const boardState nextState = apply_move(state, moveId);

    futures[i] = std::async(std::launch::async, [nextState, depth, alpha, beta, moveId]() {
      u64 localNodes = 0;
      MinimaxResult res;
      thread_local TranspositionTable threadLocalTT(1048576);
      if (nextState.activeCaptureIdx == -1) {
        const boardState flippedState = flip_board(nextState);
        res = minimax_seq_helper<EvalFunc>(flippedState, depth - 1, -beta, -alpha, localNodes, &threadLocalTT);
        res.score = -res.score;
      } else {
        res = minimax_seq_helper<EvalFunc>(nextState, depth - 1, alpha, beta, localNodes, &threadLocalTT);
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
 * @param transTable Optional pointer to transposition table.
 * @return A MinimaxResult containing the best score and best move ID.
 */
template <auto EvalFunc>
[[nodiscard]] constexpr MinimaxResult minimax(const boardState& state,
                                              int depth,
                                              i32 alpha,
                                              i32 beta,
                                              u64& nodeCounter,
                                              TranspositionTable* transTable) noexcept {
  nodeCounter++;

  if (piece_count(state.opp) == 0) {
    return MinimaxResult{.score = INFINITY_VAL + depth, .moveId = -1};
  }
  if (piece_count(state.me) == 0) {
    return MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
  }

  if (depth <= 0) {
    return quiescence_search<EvalFunc>(state, alpha, beta, nodeCounter);
  }

  i32 ttScore = 0;
  int ttMoveId = -1;
  if (transTable && transTable->probe(state.hash, depth, alpha, beta, ttScore, ttMoveId)) {
    return MinimaxResult{.score = ttScore, .moveId = ttMoveId};
  }

  const MoveList moves = all_possible_moves(state);
  if (moves.empty()) {
    return MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
  }

  const OrderedMoveList ordered = order_moves(moves, ttMoveId);

  if (!std::is_constant_evaluated() && depth > 1 && ordered.count > 1) {
    return minimax_parallel_root<EvalFunc>(state, depth, alpha, beta, nodeCounter, ordered);
  }

  return minimax_seq_helper<EvalFunc>(state, depth, alpha, beta, nodeCounter, transTable);
}

/**
 * @brief Compatibility wrapper for minimax search.
 */
[[nodiscard]] constexpr MinimaxResult minimax(const boardState& state, int depth, i32 alpha, i32 beta) noexcept {
  u64 dummy = 0;
  return minimax<heuristics::evaluate_by_node_values<heuristics::HEURISTIC_NODE_WEIGHTS>>(
      state, depth, alpha, beta, dummy, nullptr);
}

/**
 * @brief Player maker utilizing minimax search.
 * @tparam EvalFunc Heuristic function evaluating the board state.
 * @tparam Depth The search depth.
 * @param state The current board state.
 * @param nodes Out-parameter tracking explored states.
 * @param madness Percentage chance to play a random move (0-100).
 * @return The selected move ID.
 */
template <auto EvalFunc, int Depth>
[[nodiscard]] inline int minimax_player_maker(const boardState& state, u64& nodes, int madness = 0) {
  if (madness > 0) {
    std::uniform_int_distribution<int> dist(0, 99);
    if (dist(rng) < madness) {
      return select_random(state, nodes);
    }
  }
  thread_local TranspositionTable localTT(1048576);
  MinimaxResult res = minimax<EvalFunc>(state, Depth, -INFINITY_VAL, INFINITY_VAL, nodes, &localTT);
  return res.moveId;
}

}  // namespace kribu::player
