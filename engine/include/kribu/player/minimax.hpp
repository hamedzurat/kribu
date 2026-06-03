/**
 * @file minimax.hpp
 * @brief Minimax search algorithm with alpha-beta pruning for Sholo Guti.
 *
 * @details Implements iterative deepening, Principal Variation Search (PVS),
 */

#pragma once

#include <algorithm>
#include <array>

#include "kribu/board.hpp"
#include "kribu/heuristic.hpp"
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
constexpr i32 DRAW_CONTEMPT_CAP = 250;
constexpr i32 DRAW_CONTEMPT_DIVISOR = 32;

/**
 * @brief Default null-move reduction depth.
 */
constexpr int NULL_MOVE_R = 2;

/**
 * @brief Minimum depth at which null-move pruning is applied.
 */
constexpr int NULL_MOVE_MIN_DEPTH = 3;

/**
 * @brief Minimum number of pieces required for null-move pruning.
 * @details Avoids null-move in zugzwang-prone endgames.
 */
constexpr i32 NULL_MOVE_MIN_PIECES = 6;

/**
 * @brief Default aspiration window half-width.
 */
constexpr i32 ASPIRATION_DELTA = 50;

/**
 * @brief Number of killer move slots tracked per search depth.
 */
constexpr int NUM_KILLER_SLOTS = 2;

/**
 * @brief Maximum depth supported for killer move tracking.
 */
constexpr int MAX_KILLER_DEPTH = 64;

/**
 * @brief Move index threshold after which Late Move Reductions kick in.
 */
constexpr int LMR_MOVE_THRESHOLD = 3;

/**
 * @brief Minimum depth at which Late Move Reductions are applied.
 */
constexpr int LMR_MIN_DEPTH = 3;

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
 * @brief Return a finite draw score with mild contempt based on static evaluation.
 * @details Draws should not be treated as perfectly neutral when the current side
 *          is clearly better or worse. A small contempt term helps the stronger
 *          side keep pressing while still allowing the weaker side to seek draws.
 */
[[nodiscard]] inline i32 draw_score(const boardState& state) noexcept {
  const i32 staticEval = heuristics::evaluate(state, 1);
  const i32 contempt = std::clamp(staticEval / DRAW_CONTEMPT_DIVISOR, -DRAW_CONTEMPT_CAP, DRAW_CONTEMPT_CAP);
  return -contempt;
}

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
 * @struct KillerTable
 * @brief Tracks "killer moves" per depth — quiet moves that caused beta cutoffs.
 * @details Used for move ordering: killers are tried after TT move and captures.
 */
struct KillerTable {
  /**
   * @brief Two-dimensional array storing killer move IDs for each depth.
   */
  std::array<std::array<i16, NUM_KILLER_SLOTS>, MAX_KILLER_DEPTH> slots{};

  /**
   * @brief Constructs the killer table, initializing all slots to -1.
   */
  constexpr KillerTable() noexcept { clear(); }

  /**
   * @brief Stores a killer move at the given depth.
   * @param depth The current search depth.
   * @param moveId The ID of the killer move.
   */
  constexpr void store(int depth, i16 moveId) noexcept {
    if (depth < 0 || depth >= MAX_KILLER_DEPTH) {
      return;
    }

    if (moveId == END_CHAIN_MOVE || is_capture_move(moveId)) {
      return;
    }

    if (slots[depth][0] == moveId) {
      return;
    }

    slots[depth][1] = slots[depth][0];
    slots[depth][0] = moveId;
  }

  /**
   * @brief Checks if a move is a killer at the given depth.
   * @param depth The search depth.
   * @param moveId The move ID to check.
   * @return True if the move is a killer at this depth.
   */
  [[nodiscard]] constexpr bool is_killer(int depth, i16 moveId) const noexcept {
    if (depth < 0 || depth >= MAX_KILLER_DEPTH) {
      return false;
    }

    return slots[depth][0] == moveId || slots[depth][1] == moveId;
  }

  /**
   * @brief Clears the killer table.
   */
  constexpr void clear() noexcept {
    for (auto& depthSlots : slots) {
      depthSlots.fill(static_cast<i16>(-1));
    }
  }
};

/**
 * @struct SearchContext
 * @brief Bundles mutable search state passed through the recursive search.
 * @details Keeps the function signatures clean and avoids passing many parameters.
 */
struct SearchContext {
  /**
   * @brief Pointer to the transposition table used during search.
   */
  TranspositionTable* transTable = nullptr;

  /**
   * @brief Killer move table for move ordering.
   */
  KillerTable killers;

  /**
   * @brief Pre-allocated stack-based array tracking Zobrist hashes along the active search path.
   * @details Using std::array to avoid any heap allocations during the search.
   */
  std::array<u64, 128> searchPath{};

  /**
   * @brief The current size/depth of the search path.
   */
  int pathSize = 0;
};

/**
 * @struct SearchPathGuard
 * @brief RAII guard to safely push and pop board hashes on the SearchContext search path.
 */
struct SearchPathGuard {
  /**
   * @brief Reference to the search context.
   */
  SearchContext& ctx;

  /**
   * @brief Hash value of the board state.
   */
  u64 hash;

  /**
   * @brief Constructor that pushes the hash onto the search path if capacity permits.
   * @param context Reference to the search context.
   * @param stateHash The board state hash.
   */
  SearchPathGuard(SearchContext& context, u64 stateHash) noexcept : ctx(context), hash(stateHash) {
    if (ctx.pathSize < 128) {
      ctx.searchPath[ctx.pathSize++] = hash;
    }
  }

  /**
   * @brief Destructor that pops the hash from the search path.
   */
  ~SearchPathGuard() noexcept {
    if (ctx.pathSize > 0 && ctx.searchPath[ctx.pathSize - 1] == hash) {
      ctx.pathSize--;
    }
  }

  SearchPathGuard(const SearchPathGuard&) = delete;
  SearchPathGuard& operator=(const SearchPathGuard&) = delete;
  SearchPathGuard(SearchPathGuard&&) = delete;
  SearchPathGuard& operator=(SearchPathGuard&&) = delete;
};

/**
 * @brief Appends a move to the ordered list, avoiding duplicates with excludeId.
 * @param ordered The ordered move list to append to.
 * @param moveId The move ID to append.
 * @param excludeId Move ID to skip (typically the TT move already placed first).
 */
constexpr void push_if_not_excluded(OrderedMoveList& ordered, i16 moveId, int excludeId) noexcept {
  if (moveId != excludeId) {
    ordered.moves[ordered.count++] = moveId;
  }
}

/**
 * @brief Orders a MoveList to prioritize moves for optimal alpha-beta cutoffs.
 * @details Order: TT move → captures → killer moves → remaining quiet moves.
 * @param moves The original MoveList to order.
 * @param ttMoveId Best move from the transposition table (-1 if none).
 * @param killers Pointer to the killer table (may be null).
 * @param depth Current search depth (for killer lookup).
 * @return OrderedMoveList with moves prioritized for best cutoff rates.
 */

/**
 * @brief Appends the transposition table move (if present in moves) to the ordered list.
 * @param ordered The ordered move list to append to.
 * @param moves The original MoveList containing all possible moves.
 * @param ttMoveId Best move from the transposition table (-1 if none).
 */
constexpr void push_tt_move(OrderedMoveList& ordered, const MoveList& moves, i32 ttMoveId) noexcept {
  if (ttMoveId != -1) {
    for (i32 moveIdx = 0; moveIdx < moves.count; ++moveIdx) {
      if (moves.moves[moveIdx] == ttMoveId) {
        ordered.moves[ordered.count++] = moves.moves[moveIdx];
        break;
      }
    }
  }
}

/**
 * @brief Appends capture moves to the ordered list, excluding the transposition table move.
 * @param ordered The ordered move list to append to.
 * @param moves The original MoveList containing all possible moves.
 * @param ttMoveId Best move from the transposition table (-1 if none).
 */
constexpr void push_capture_moves(OrderedMoveList& ordered, const MoveList& moves, i32 ttMoveId) noexcept {
  for (i32 moveIdx = 0; moveIdx < moves.count; ++moveIdx) {
    if (is_capture_move(moves.moves[moveIdx])) {
      push_if_not_excluded(ordered, moves.moves[moveIdx], ttMoveId);
    }
  }
}

/**
 * @brief Appends killer moves to the ordered list, excluding the transposition table move.
 * @param ordered The ordered move list to append to.
 * @param moves The original MoveList containing all possible moves.
 * @param ttMoveId Best move from the transposition table (-1 if none).
 * @param killers Pointer to the killer table (may be null).
 * @param depth Current search depth.
 */
constexpr void push_killer_moves(
    OrderedMoveList& ordered, const MoveList& moves, i32 ttMoveId, const KillerTable* killers, i32 depth) noexcept {
  if (killers != nullptr) {
    for (i32 moveIdx = 0; moveIdx < moves.count; ++moveIdx) {
      if (!is_capture_move(moves.moves[moveIdx]) && killers->is_killer(depth, moves.moves[moveIdx])) {
        push_if_not_excluded(ordered, moves.moves[moveIdx], ttMoveId);
      }
    }
  }
}

/**
 * @brief Appends the remaining quiet moves to the ordered list, excluding the transposition table move.
 * @param ordered The ordered move list to append to.
 * @param moves The original MoveList containing all possible moves.
 * @param ttMoveId Best move from the transposition table (-1 if none).
 * @param killers Pointer to the killer table (may be null).
 * @param depth Current search depth.
 */
constexpr void push_remaining_quiet_moves(
    OrderedMoveList& ordered, const MoveList& moves, i32 ttMoveId, const KillerTable* killers, i32 depth) noexcept {
  for (i32 moveIdx = 0; moveIdx < moves.count; ++moveIdx) {
    const bool isKiller = (killers != nullptr && killers->is_killer(depth, moves.moves[moveIdx]));
    if (!is_capture_move(moves.moves[moveIdx]) && !isKiller) {
      push_if_not_excluded(ordered, moves.moves[moveIdx], ttMoveId);
    }
  }
}

/**
 * @brief Orders a MoveList to prioritize moves for optimal alpha-beta cutoffs.
 * @details Order: TT move → captures → killer moves → remaining quiet moves.
 * @param moves The original MoveList to order.
 * @param ttMoveId Best move from the transposition table (-1 if none).
 * @param killers Pointer to the killer table (may be null).
 * @param depth Current search depth (for killer lookup).
 * @return OrderedMoveList with moves prioritized for best cutoff rates.
 */
[[nodiscard]] inline OrderedMoveList order_moves(const MoveList& moves,
                                                 i32 ttMoveId = -1,
                                                 const KillerTable* killers = nullptr,
                                                 i32 depth = 0) noexcept {
  OrderedMoveList ordered;

  push_tt_move(ordered, moves, ttMoveId);
  push_capture_moves(ordered, moves, ttMoveId);
  push_killer_moves(ordered, moves, ttMoveId, killers, depth);
  push_remaining_quiet_moves(ordered, moves, ttMoveId, killers, depth);

  return ordered;
}

/**
 * @brief Checks if the opponent has no legal moves (stalemate or loss).
 * @param state Current board state.
 * @return True if the opponent has no legal moves, false otherwise.
 */
[[nodiscard]] inline bool opponent_has_no_moves(const boardState& state) noexcept {
  const boardState flippedState = flip_board(state);
  return all_possible_moves(flippedState).empty();
}

// Forward declaration
[[nodiscard]] inline MinimaxResult alpha_beta(
    const boardState& state, int depth, i32 alpha, i32 beta, SearchContext& ctx, bool isRoot = false) noexcept;

[[nodiscard]] inline MinimaxResult quiescence_search(const boardState& state,
                                                     i32 alpha,
                                                     i32 beta,
                                                     SearchContext& ctx) noexcept;

/**
 * @brief Helper for quiescence search during an active capture chain.
 * @param state The current board state.
 * @param alpha Lower bound of the search window.
 * @param beta Upper bound of the search window.
 * @param moves List of all possible moves in the state.
 * @param ctx Search context.
 * @return MinimaxResult with the score and move ID.
 */
[[nodiscard]] inline MinimaxResult quiescence_search_chain(
    const boardState& state, i32 alpha, i32 beta, const MoveList& moves, SearchContext& ctx) noexcept {
  int bestMoveId = -1;
  i32 bestScore = -INFINITY_VAL;

  for (int i = 0; i < moves.count; ++i) {
    const int moveId = moves.moves[i];

    // In a chain, quiescence only cares about captures and END_CHAIN_MOVE.
    if (!is_capture_move(moveId) && moveId != END_CHAIN_MOVE) {
      continue;
    }

    const boardState nextState = apply_move(state, moveId);
    i32 score = 0;

    if (nextState.activeCaptureIdx == -1) {
      // Chain ended, turn passes to opponent.
      const boardState flippedState = flip_board(nextState);
      const MinimaxResult res = quiescence_search(flippedState, -beta, -alpha, ctx);
      score = -res.score;
    } else {
      // Same player continues the chain.
      const MinimaxResult res = quiescence_search(nextState, alpha, beta, ctx);
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

  // Defensive fallback. This should usually not happen if active-chain move
  // generation always includes captures and/or END_CHAIN_MOVE.
  if (bestMoveId == -1) {
    return MinimaxResult{.score = heuristics::evaluate(state, 1), .moveId = -1};
  }

  return MinimaxResult{.score = bestScore, .moveId = bestMoveId};
}

/**
 * @brief Helper for quiescence search during normal non-chain play.
 * @param state The current board state.
 * @param alpha Lower bound of the search window.
 * @param beta Upper bound of the search window.
 * @param moves List of all possible moves in the state.
 * @param ctx Search context.
 * @return MinimaxResult with the score and move ID.
 */
[[nodiscard]] inline MinimaxResult quiescence_search_normal(
    const boardState& state, i32 alpha, i32 beta, const MoveList& moves, SearchContext& ctx) noexcept {
  const i32 standPat = heuristics::evaluate(state, 1);

  if (standPat >= beta) {
    return MinimaxResult{.score = beta, .moveId = -1};
  }

  alpha = std::max(alpha, standPat);

  int bestMoveId = -1;
  i32 bestScore = standPat;

  for (int i = 0; i < moves.count; ++i) {
    const int moveId = moves.moves[i];

    // Outside chains, quiescence only searches captures.
    if (!is_capture_move(moveId)) {
      continue;
    }

    const boardState nextState = apply_move(state, moveId);
    i32 score = 0;

    if (nextState.activeCaptureIdx == -1) {
      // Turn passes to opponent.
      const boardState flippedState = flip_board(nextState);
      const MinimaxResult res = quiescence_search(flippedState, -beta, -alpha, ctx);
      score = -res.score;
    } else {
      // Same player continues capture chain.
      const MinimaxResult res = quiescence_search(nextState, alpha, beta, ctx);
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
 * @brief Performs quiescence search to evaluate captures recursively.
 * @details Only explores capture moves (and END_CHAIN_MOVE during chains) to
 * avoid the horizon effect. Uses stand-pat evaluation as the baseline.
 * @param state The current board state.
 * @param alpha Lower bound of the search window.
 * @param beta Upper bound of the search window.
 * @param ctx Search context containing mutable state.
 * @return MinimaxResult with the quiescence score.
 */
[[nodiscard]] inline MinimaxResult quiescence_search(const boardState& state,
                                                     i32 alpha,
                                                     i32 beta,
                                                     SearchContext& ctx) noexcept {
  if (state.historyCount >= MAX_HISTORY_LIMIT) {
    return MinimaxResult{.score = draw_score(state), .moveId = -1};
  }

  // Terminal: current player has captured all opponent pieces.
  if (piece_count(state.opp) == 0) {
    return MinimaxResult{.score = INFINITY_VAL, .moveId = -1};
  }

  // Terminal: current player has no pieces.
  if (piece_count(state.me) == 0) {
    return MinimaxResult{.score = -INFINITY_VAL, .moveId = -1};
  }

  // Important:
  // Only check opponent stalemate when NOT inside an active capture chain.
  // During a chain, it is still the same player's turn.
  if (state.activeCaptureIdx == -1 && opponent_has_no_moves(state)) {
    return MinimaxResult{.score = INFINITY_VAL, .moveId = -1};
  }

  const MoveList moves = all_possible_moves(state);

  // Current player has no legal moves.
  if (moves.empty()) {
    return MinimaxResult{.score = -INFINITY_VAL, .moveId = -1};
  }

  // Delegate based on active capture chain status to keep cognitive complexity low.
  if (state.activeCaptureIdx != -1) {
    return quiescence_search_chain(state, alpha, beta, moves, ctx);
  }

  return quiescence_search_normal(state, alpha, beta, moves, ctx);
}

/**
 * @brief Evaluates terminal conditions (win/loss/stalemate).
 * @param state The current board state.
 * @param depth Current remaining search depth.
 * @param result Output parameter set if the position is terminal.
 * @return True if the position is terminal and result was set, false otherwise.
 */
[[nodiscard]] inline bool check_terminal(const boardState& state, int depth, MinimaxResult& result) noexcept {
  if (state.historyCount >= MAX_HISTORY_LIMIT) {
    result = MinimaxResult{.score = draw_score(state), .moveId = -1};
    return true;
  }

  if (piece_count(state.opp) == 0) {
    result = MinimaxResult{.score = INFINITY_VAL + depth, .moveId = -1};
    return true;
  }

  if (piece_count(state.me) == 0) {
    result = MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
    return true;
  }

  if (state.activeCaptureIdx == -1 && opponent_has_no_moves(state)) {
    result = MinimaxResult{.score = INFINITY_VAL + depth, .moveId = -1};
    return true;
  }
  return false;
}

/**
 * @brief Attempts null-move pruning to skip searching this node.
 * @details Passes the turn to the opponent (null move) and searches at reduced depth.
 * If the opponent still can't beat beta, the node is pruned. Disabled during
 * capture chains and endgames with few pieces.
 * @param state The current board state.
 * @param depth Current remaining search depth.
 * @param beta Upper bound of the search window.
 * @param ctx Search context.
 * @param cutoffScore Output: the score if pruning succeeds.
 * @return True if null-move pruning produced a cutoff, false otherwise.
 */
[[nodiscard]] inline bool try_null_move_pruning(
    const boardState& state, int depth, i32 beta, SearchContext& ctx, i32& cutoffScore) noexcept {
  // Don't null-move during capture chains (passing is meaningless mid-chain)
  if (state.activeCaptureIdx != -1) {
    return false;
  }
  // Don't null-move at shallow depths
  if (depth < NULL_MOVE_MIN_DEPTH) {
    return false;
  }
  // Don't null-move in endgames (zugzwang risk)
  if (piece_count(state.me) < NULL_MOVE_MIN_PIECES) {
    return false;
  }

  // "Pass" the turn by flipping the board without making a move
  const boardState nullState = flip_board(state);
  const int reducedDepth = depth - 1 - NULL_MOVE_R;
  const MinimaxResult nullRes = alpha_beta(nullState, reducedDepth, -beta, -beta + 1, ctx);
  const i32 nullScore = -nullRes.score;

  if (nullScore >= beta) {
    cutoffScore = beta;
    return true;
  }
  return false;
}

/**
 * @brief Searches a child move using PVS (Principal Variation Search) strategy.
 * @details The first move is searched with a full window. Subsequent moves are
 * searched with a null (zero) window first; if they fail high, a full re-search is done.
 * @param nextState The board state after applying the move.
 * @param depth Current remaining search depth.
 * @param alpha Current alpha bound.
 * @param beta Current beta bound.
 * @param moveIndex Index of this move in the ordered list (0 = first/PV move).
 * @param ctx Search context.
 * @return The negamax score for this child move.
 */
[[nodiscard]] inline i32 search_child_pvs(
    const boardState& nextState, int depth, i32 alpha, i32 beta, int moveIndex, SearchContext& ctx) noexcept {
  // Determine if the turn flips (non-chain move)
  const bool turnFlips = (nextState.activeCaptureIdx == -1);
  const boardState searchState = turnFlips ? flip_board(nextState) : nextState;
  const int childDepth = turnFlips ? depth - 1 : depth;

  if (moveIndex == 0) {
    // First move: full window search
    const MinimaxResult res = turnFlips ? alpha_beta(searchState, childDepth, -beta, -alpha, ctx)
                                        : alpha_beta(searchState, childDepth, alpha, beta, ctx);
    return turnFlips ? -res.score : res.score;
  }

  // PVS: null-window scout search first
  MinimaxResult res = turnFlips ? alpha_beta(searchState, childDepth, -alpha - 1, -alpha, ctx)
                                : alpha_beta(searchState, childDepth, alpha, alpha + 1, ctx);
  i32 score = turnFlips ? -res.score : res.score;

  // Re-search with full window if the scout found a better move
  if (score > alpha && score < beta) {
    res = turnFlips ? alpha_beta(searchState, childDepth, -beta, -alpha, ctx)
                    : alpha_beta(searchState, childDepth, alpha, beta, ctx);
    score = turnFlips ? -res.score : res.score;
  }

  return score;
}

/**
 * @brief Determines the LMR reduction amount for a given move.
 * @param depth Current search depth.
 * @param moveIndex The move's position in the ordered list.
 * @param moveId The move ID being considered.
 * @return The depth reduction to apply (0 if no reduction).
 */
[[nodiscard]] constexpr int lmr_reduction(int depth, int moveIndex, int moveId) noexcept {
  if (depth < LMR_MIN_DEPTH) {
    return 0;
  }
  if (moveIndex < LMR_MOVE_THRESHOLD) {
    return 0;
  }
  if (is_capture_move(moveId) || moveId == END_CHAIN_MOVE) {
    return 0;
  }
  // Logarithmic reduction scaled by depth and move index
  // Approximation: 1 for moderate, clamp to avoid over-reducing
  int reduction = 1;
  if (depth >= 6 && moveIndex >= 6) {
    reduction = 2;
  }
  return std::min(reduction, depth - 1);
}

/**
 * @brief Searches a child move with Late Move Reduction applied.
 * @details Late-ordered quiet moves are searched at reduced depth first.
 * If the reduced search beats alpha, a full-depth re-search is performed.
 * @param nextState Board state after applying the move.
 * @param moveId The move ID being searched.
 * @param depth Current search depth.
 * @param alpha Current alpha bound.
 * @param beta Current beta bound.
 * @param moveIndex Index in the ordered move list.
 * @param ctx Search context.
 * @return The negamax score for this child.
 */
[[nodiscard]] inline i32 search_child_lmr(const boardState& nextState,
                                          int moveId,
                                          int depth,
                                          i32 alpha,
                                          i32 beta,
                                          int moveIndex,
                                          SearchContext& ctx) noexcept {
  int reduction = lmr_reduction(depth, moveIndex, moveId);

  if (reduction > 0) {
    // Reduced-depth search
    i32 score = search_child_pvs(nextState, depth - reduction, alpha, beta, moveIndex, ctx);
    // If reduced search fails high, re-search at full depth
    if (score <= alpha) {
      return score;
    }
  }

  return search_child_pvs(nextState, depth, alpha, beta, moveIndex, ctx);
}

/**
 * @brief Stores search results in the transposition table with proper flag determination.
 * @param ctx Search context with TT pointer.
 * @param hash Zobrist hash of the position.
 * @param depth Search depth.
 * @param bestScore Best score found.
 * @param bestMoveId Best move found.
 * @param originalAlpha The alpha value at the start of the node.
 * @param beta The beta value at the node.
 */
inline void store_tt_result(
    SearchContext& ctx, u64 hash, int depth, i32 bestScore, int bestMoveId, i32 originalAlpha, i32 beta) noexcept {
  if (ctx.transTable == nullptr) {
    return;
  }
  TTFlag flag = TTFlag::EXACT;
  if (bestScore <= originalAlpha) {
    flag = TTFlag::ALPHA;
  } else if (bestScore >= beta) {
    flag = TTFlag::BETA;
  }
  ctx.transTable->store(hash, depth, bestScore, bestMoveId, flag);
}

/**
 * @brief Evaluates all ordered children and finds the best move.
 * @param state The current board state.
 * @param depth The current search depth.
 * @param alpha The lower bound score of the search window.
 * @param beta The upper bound score of the search window.
 * @param ordered The ordered list of moves to evaluate.
 * @param ctx Search context containing mutable state.
 * @return A MinimaxResult containing the best score and best move ID.
 */
[[nodiscard]] inline MinimaxResult evaluate_children(const boardState& state,
                                                     int depth,
                                                     i32 alpha,
                                                     i32 beta,
                                                     const OrderedMoveList& ordered,
                                                     SearchContext& ctx) noexcept {
  int bestMoveId = ordered.count > 0 ? ordered.moves[0] : -1;
  i32 bestScore = -INFINITY_VAL - 10000;

  for (int i = 0; i < ordered.count; ++i) {
    const int moveId = ordered.moves[i];
    const boardState nextState = apply_move(state, moveId);

    i32 score = search_child_lmr(nextState, moveId, depth, alpha, beta, i, ctx);

    if (score > bestScore) {
      bestScore = score;
      bestMoveId = moveId;
    }

    alpha = std::max(alpha, bestScore);
    if (alpha >= beta) {
      // Record killer if this is a quiet move causing a cutoff
      if (!is_capture_move(moveId) && moveId != END_CHAIN_MOVE) {
        ctx.killers.store(depth, static_cast<i16>(moveId));
      }
      break;
    }
  }

  return MinimaxResult{.score = bestScore, .moveId = bestMoveId};
}

/**
 * @brief Returns true if a transposition-table entry may be used for this state.
 * @details Position hash ignores repetition history, so any non-empty repetition
 *          window can make a cached score or move stale. In those cases we skip
 *          TT reuse entirely.
 */
[[nodiscard]] inline bool tt_entry_is_usable(const boardState& state, int moveId) noexcept {
  if (state.historyCount != 0) {
    return false;
  }
  return moveId == -1 || sholoGuti::is_valid(state, moveId);
}

/**
 * @brief Core alpha-beta search with PVS, null-move pruning, LMR, and killer heuristic.
 * @param state The current board state.
 * @param depth The maximum search depth remaining.
 * @param alpha The lower bound score of the search window.
 * @param beta The upper bound score of the search window.
 * @param ctx Search context containing mutable state.
 * @param isRoot True if this is the root node of the search.
 * @return A MinimaxResult containing the best score and best move ID.
 */
[[nodiscard]] inline MinimaxResult alpha_beta(
    const boardState& state, int depth, i32 alpha, i32 beta, SearchContext& ctx, bool isRoot) noexcept {
  // Repetition/cycle detection on the active search path
  if (!isRoot) {
    for (int i = 0; i < ctx.pathSize; ++i) {
      if (ctx.searchPath[i] == state.hash) {
        return MinimaxResult{.score = draw_score(state), .moveId = -1};
      }
    }
  }

  // Terminal checks
  MinimaxResult termResult;
  if (check_terminal(state, depth, termResult)) {
    return termResult;
  }

  // Leaf node: drop to quiescence search
  if (depth <= 0) {
    return quiescence_search(state, alpha, beta, ctx);
  }

  // Push current state to the stack-allocated search path
  SearchPathGuard guard(ctx, state.hash);

  // Transposition table probe
  i32 ttScore = 0;
  int ttMoveId = -1;
  if (ctx.transTable != nullptr && ctx.transTable->probe(state.hash, depth, alpha, beta, ttScore, ttMoveId)) {
    if (!tt_entry_is_usable(state, ttMoveId)) {
      ttMoveId = -1;
    } else if (!isRoot || ttMoveId != -1) {
      return MinimaxResult{.score = ttScore, .moveId = ttMoveId};
    }
  }

  // Null-move pruning
  i32 cutoff = 0;
  if (try_null_move_pruning(state, depth, beta, ctx, cutoff)) {
    return MinimaxResult{.score = cutoff, .moveId = -1};
  }

  // Generate and order moves
  const MoveList moves = all_possible_moves(state);
  if (moves.empty()) {
    return MinimaxResult{.score = -INFINITY_VAL - depth, .moveId = -1};
  }

  const OrderedMoveList ordered = order_moves(moves, ttMoveId, &ctx.killers, depth);
  const i32 originalAlpha = alpha;

  MinimaxResult bestResult = evaluate_children(state, depth, alpha, beta, ordered, ctx);

  if (state.historyCount == 0) {
    store_tt_result(ctx, state.hash, depth, bestResult.score, bestResult.moveId, originalAlpha, beta);
  }

  return bestResult;
}

/**
 * @brief Performs one iteration of iterative deepening search at a specific depth.
 * @param state The root board state.
 * @param depth The depth to search.
 * @param alpha Lower bound of the aspiration window.
 * @param beta Upper bound of the aspiration window.
 * @param ctx Search context.
 * @return MinimaxResult with the best score and move at this depth.
 */
[[nodiscard]] inline MinimaxResult search_at_depth(
    const boardState& state, int depth, i32 alpha, i32 beta, SearchContext& ctx) noexcept {
  MinimaxResult res = alpha_beta(state, depth, alpha, beta, ctx, true);

  // Aspiration window fail-low: re-search with full window
  if (res.score <= alpha) {
    res = alpha_beta(state, depth, -INFINITY_VAL, beta, ctx, true);
  }
  // Aspiration window fail-high: re-search with full window
  if (res.score >= beta) {
    res = alpha_beta(state, depth, alpha, INFINITY_VAL, ctx, true);
  }

  return res;
}

/**
 * @brief Iterative deepening search with aspiration windows.
 * @details Searches from depth 1 up to the target depth, using the previous
 * iteration's score to set narrow aspiration windows for the next iteration.
 * Each shallower iteration populates the TT, greatly improving move ordering.
 * @param state The root board state.
 * @param targetDepth The maximum depth to search.
 * @param ctx Search context containing TT and killers.
 * @return MinimaxResult with the best score and move from the deepest completed search.
 */
[[nodiscard]] inline MinimaxResult iterative_deepening(const boardState& state,
                                                       int targetDepth,
                                                       SearchContext& ctx) noexcept {
  MinimaxResult best{};
  i32 prevScore = 0;

  for (int depth = 1; depth <= targetDepth; ++depth) {
    ctx.killers.clear();
    ctx.pathSize = 0;

    i32 alpha = -INFINITY_VAL;
    i32 beta = INFINITY_VAL;

    // Apply aspiration windows from depth 3 onward
    if (depth >= 3) {
      alpha = prevScore - ASPIRATION_DELTA;
      beta = prevScore + ASPIRATION_DELTA;
    }

    MinimaxResult res = search_at_depth(state, depth, alpha, beta, ctx);
    best = res;
    prevScore = res.score;
  }

  return best;
}

/**
 * @brief Executes a minimax search with alpha-beta pruning.
 * @details Entry point wrapping iterative deepening. Kept for backward compatibility.
 * @param state The current board state to search from.
 * @param depth The maximum search depth remaining.
 * @param alpha The lower bound score of the search window.
 * @param beta The upper bound score of the search window.
 * @param transTable Optional pointer to transposition table.
 * @return A MinimaxResult containing the best score and best move ID.
 */
[[nodiscard]] inline MinimaxResult minimax(
    const boardState& state, int depth, i32 alpha, i32 beta, TranspositionTable* transTable = nullptr) noexcept {
  // alpha/beta kept for interface compatibility; iterative deepening manages its own windows
  (void) alpha;
  (void) beta;

  SearchContext ctx;
  ctx.transTable = transTable;

  MinimaxResult res = iterative_deepening(state, depth, ctx);
  return res;
}

/**
 * @brief Player maker utilizing minimax search with all optimizations.
 * @details Uses iterative deepening, PVS, null-move pruning, LMR, aspiration
 *          windows, killer moves, and Lazy SMP parallelism.
 * @tparam Depth The search depth.
 * @param state The current board state.
 * @return The selected move ID.
 */
template <int Depth>
[[nodiscard]] inline int minimax_player_maker(const boardState& state) {
  static_assert(Depth >= 2, "Depth must be at least 2");

  thread_local TranspositionTable localTT(1048576);
  SearchContext ctx;
  ctx.transTable = &localTT;

  if (state == INITIAL_STATE) {
    localTT.clear();
  }

  MinimaxResult res = iterative_deepening(state, Depth, ctx);
  if (res.moveId != -1 && sholoGuti::is_valid(state, res.moveId)) {
    return res.moveId;
  }

  const MoveList moves = all_possible_moves(state);
  return moves.empty() ? -1 : static_cast<int>(moves.moves[0]);
}

}  // namespace kribu::player
