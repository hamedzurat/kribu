/**
 * @file rollout.hpp
 * @brief Strong Heuristic Rollout Policy for MCTS.
 */

#pragma once

#include <algorithm>
#include <array>
#include <limits>

#include "kribu/board.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/rules.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

using namespace kribu::board;
using namespace kribu::sholoGuti;

/**
 * @struct Rollout
 * @brief Deterministic rollout policy using static evaluation plus opponent-reply checking.
 *
 * @details
 * This policy is intended for stronger MCTS simulations and self-play data generation.
 *
 * It does not use randomness. Variation should come from the player-level randomness
 * parameter, not from rollout policy internals.
 *
 * Selection logic:
 * - Generate all legal moves.
 * - Score every move using heuristics::evaluate().
 * - Keep the top STATIC_CANDIDATES moves.
 * - For those candidates only, check the opponent's best immediate reply.
 * - Pick the move with the best reply-aware score.
 *
 * This avoids the main weakness of simple greedy rollout:
 * choosing a move that looks good immediately but allows a devastating reply.
 */
struct Rollout {
  /**
   * @brief Heuristic depth used for fast first-pass move scoring.
   *
   * depth 3 includes:
   * - material
   * - structure
   * - advancement
   * - mobility
   * - immediate capture pressure
   * - best capture chain
   * - vulnerability
   */
  static constexpr int STATIC_EVAL_DEPTH = 3;

  /**
   * @brief Heuristic depth used after opponent reply.
   *
   * Keeping this at 2 is usually a good tradeoff.
   * depth 3 here is stronger but can be noticeably slower.
   */
  static constexpr int REPLY_EVAL_DEPTH = 2;

  /**
   * @brief Number of top static moves to examine with opponent-reply checking.
   *
   * Increasing this improves tactical caution but costs more.
   * Good values: 3, 4, 5.
   */
  static constexpr int STATIC_CANDIDATES = 5;

  /**
   * @brief Large score used for terminal win/loss situations.
   */
  static constexpr i32 WIN_SCORE = heuristics::EVALUATION_WIN_SCORE;

  /**
   * @brief Selects the best rollout move deterministically.
   * @param state Current board state.
   * @return Selected move ID, or -1 if no legal move exists.
   */
  static int select_move(const boardState& state) {
    const MoveList moves = all_possible_moves(state);

    if (moves.empty()) {
      return -1;
    }

    return pick_best_move(state, moves);
  }

  /**
   * @brief Evaluates rollout cutoff state.
   * @param state Board state to evaluate.
   * @return Heuristic score from active player's perspective.
   */
  static f64 evaluate(const boardState& state) noexcept {
    return static_cast<f64>(heuristics::evaluate(state, STATIC_EVAL_DEPTH));
  }

 private:
  struct ScoredMove {
    int moveId = -1;
    i32 staticScore = std::numeric_limits<i32>::min();
    i32 finalScore = std::numeric_limits<i32>::min();
  };

  /**
   * @brief Checks whether a state has no pieces for either side.
   */
  [[nodiscard]] static bool is_piece_terminal(const boardState& state) noexcept {
    return piece_count(state.me) == 0 || piece_count(state.opp) == 0;
  }

  /**
   * @brief Evaluates a state from the current side's perspective.
   *
   * @param state Board state, already oriented so `state.me` is the side being evaluated.
   * @param depth Heuristic evaluation depth.
   * @return Positive score favors `state.me`.
   */
  [[nodiscard]] static i32 eval_current_player(const boardState& state, int depth) noexcept {
    return heuristics::evaluate(state, depth);
  }

  /**
   * @brief Scores the position after current player makes moveId.
   *
   * @details
   * The returned score is always from the original current player's perspective.
   *
   * If the move continues a capture chain, the board is still from the same
   * player's perspective, so evaluation is direct.
   *
   * If the turn changes, flip_board() makes the opponent become `me`, so the
   * evaluation must be negated.
   */
  [[nodiscard]] static i32 score_after_own_move(const boardState& state, int moveId, int depth) {
    boardState next = apply_move(state, moveId);

    if (piece_count(next.opp) == 0) {
      return WIN_SCORE;
    }

    if (piece_count(next.me) == 0) {
      return -WIN_SCORE;
    }

    // Same player continues during capture chain.
    if (next.activeCaptureIdx != -1) {
      return eval_current_player(next, depth);
    }

    // Turn changed. After flipping, `me` is the opponent.
    next = flip_board(next);
    return -eval_current_player(next, depth);
  }

  /**
   * @brief Scores the position after the opponent makes a reply.
   *
   * @details
   * `opponentState` is already oriented so the opponent is `me`.
   *
   * The returned score is from our original player's perspective.
   */
  [[nodiscard]] static i32 score_after_opponent_reply(const boardState& opponentState, int replyMoveId) {
    boardState replyNext = apply_move(opponentState, replyMoveId);

    if (piece_count(replyNext.opp) == 0) {
      // Opponent captured all of our pieces.
      return -WIN_SCORE;
    }

    if (piece_count(replyNext.me) == 0) {
      // Opponent somehow has no pieces.
      return WIN_SCORE;
    }

    if (replyNext.activeCaptureIdx != -1) {
      // Opponent continues moving. Evaluation is good for opponent, so negate.
      return -eval_current_player(replyNext, REPLY_EVAL_DEPTH);
    }

    // Turn comes back to us. After flip, `me` is us again.
    replyNext = flip_board(replyNext);
    return eval_current_player(replyNext, REPLY_EVAL_DEPTH);
  }

  /**
   * @brief Evaluates a candidate move with opponent best-reply checking.
   *
   * @details
   * This prevents rollout from choosing moves that look good immediately but
   * allow a strong opponent capture or tactical reply.
   *
   * Returned score is from the current player's perspective.
   */
  [[nodiscard]] static i32 score_with_best_reply(const boardState& state, int moveId) {
    boardState next = apply_move(state, moveId);

    if (piece_count(next.opp) == 0) {
      return WIN_SCORE;
    }

    if (piece_count(next.me) == 0) {
      return -WIN_SCORE;
    }

    // If the move continues our capture chain, do not check opponent reply yet.
    // The same player still has the move.
    if (next.activeCaptureIdx != -1) {
      return eval_current_player(next, STATIC_EVAL_DEPTH);
    }

    // Turn changed. Orient board for opponent.
    boardState opponentState = flip_board(next);

    const MoveList replies = all_possible_moves(opponentState);

    if (replies.empty()) {
      // Opponent has no legal move, good for us.
      return WIN_SCORE;
    }

    i32 worstScoreForUs = std::numeric_limits<i32>::max();

    for (int i = 0; i < replies.count; ++i) {
      const int replyMoveId = replies.moves[i];
      const i32 scoreForUs = score_after_opponent_reply(opponentState, replyMoveId);

      // Opponent chooses the reply that is worst for us.
      worstScoreForUs = std::min(worstScoreForUs, scoreForUs);
    }

    return worstScoreForUs;
  }

  /**
   * @brief Deterministic tie-breaker.
   *
   * @details
   * Prefer higher score.
   * If equal, prefer captures.
   * If still equal, prefer lower move ID for stable deterministic behavior.
   */
  [[nodiscard]] static bool better_than(const ScoredMove& lhs, const ScoredMove& rhs) noexcept {
    if (lhs.finalScore != rhs.finalScore) {
      return lhs.finalScore > rhs.finalScore;
    }

    const bool lhsCapture = is_capture_move(lhs.moveId);
    const bool rhsCapture = is_capture_move(rhs.moveId);

    if (lhsCapture != rhsCapture) {
      return lhsCapture;
    }

    return lhs.moveId < rhs.moveId;
  }

  /**
   * @brief Picks the best move using static scoring plus reply-aware rescoring.
   */
  static int pick_best_move(const boardState& state, const MoveList& moves) {
    std::array<ScoredMove, MAX_MOVES_PER_STATE> scored{};
    int scoredCount = 0;

    for (int i = 0; i < moves.count; ++i) {
      const int moveId = moves.moves[i];

      const i32 staticScore = score_after_own_move(state, moveId, STATIC_EVAL_DEPTH);

      scored[static_cast<usize>(scoredCount)] = ScoredMove{
          .moveId = moveId,
          .staticScore = staticScore,
          .finalScore = staticScore,
      };

      ++scoredCount;
    }

    // First pass: sort by cheap static score.
    std::sort(scored.begin(), scored.begin() + scoredCount, [](const ScoredMove& lhs, const ScoredMove& rhs) {
      if (lhs.staticScore != rhs.staticScore) {
        return lhs.staticScore > rhs.staticScore;
      }

      const bool lhsCapture = is_capture_move(lhs.moveId);
      const bool rhsCapture = is_capture_move(rhs.moveId);

      if (lhsCapture != rhsCapture) {
        return lhsCapture;
      }

      return lhs.moveId < rhs.moveId;
    });

    // Second pass: only top few moves get opponent-reply checking.
    const int candidateCount = std::min(STATIC_CANDIDATES, scoredCount);

    for (int i = 0; i < candidateCount; ++i) {
      scored[static_cast<usize>(i)].finalScore = score_with_best_reply(state, scored[static_cast<usize>(i)].moveId);
    }

    // Pick best final candidate.
    int bestIndex = 0;

    for (int i = 1; i < candidateCount; ++i) {
      if (better_than(scored[static_cast<usize>(i)], scored[static_cast<usize>(bestIndex)])) {
        bestIndex = i;
      }
    }

    return scored[static_cast<usize>(bestIndex)].moveId;
  }
};

}  // namespace kribu::player
