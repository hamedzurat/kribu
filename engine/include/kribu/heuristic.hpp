/**
 * @file heuristic.hpp
 * @brief Static board evaluation heuristics for the Sholo Guti engine.
 *
 * @details
 * This file evaluates a board from the perspective of `boardState::me`, the active player.
 *
 * The board orientation assumed by this evaluator is:
 *
 * @code
 *   0───────────────1───────────────2
 *     ╲             │             ╱
 *       ╲           │           ╱
 *         ╲         │         ╱
 *           3───────4───────5
 *            ╲      │      ╱
 *              ╲    │    ╱
 *                ╲  │  ╱
 *   6───────7───────8───────9──────10
 *   │ ╲     │     ╱ │ ╲     │     ╱ │
 *   │   ╲   │   ╱   │   ╲   │   ╱   │
 *   │     ╲ │ ╱     │     ╲ │ ╱     │
 *  11──────12──────13──────14──────15
 *   │     ╱ │ ╲     │     ╱ │ ╲     │
 *   │   ╱   │   ╲   │   ╱   │   ╲   │
 *   │ ╱     │     ╲ │ ╱     │     ╲ │
 *  16──────17──────18──────19──────20
 *   │ ╲     │     ╱ │ ╲     │     ╱ │
 *   │   ╲   │   ╱   │   ╲   │   ╱   │
 *   │     ╲ │ ╱     │     ╲ │ ╱     │
 *  21──────22──────23──────24──────25
 *   │     ╱ │ ╲     │     ╱ │ ╲     │
 *   │   ╱   │   ╲   │   ╱   │   ╲   │
 *   │ ╱     │     ╲ │ ╱     │     ╲ │
 *  26──────27──────28──────29──────30
 *                ╱  │  ╲
 *              ╱    │    ╲
 *            ╱      │      ╲
 *          31──────32──────33
 *        ╱          │         ╲
 *      ╱            │           ╲
 *    ╱              │             ╲
 *  34──────────────35──────────────36
 * @endcode
 *
 * In the standard initial state, `me` starts near the bottom on nodes 21..36 and
 * `opp` starts near the top on nodes 0..15. Therefore, advancement for `me`
 * means moving toward smaller node indexes. Directional advancement tables must
 * give high values to top-side nodes.
 *
 * This evaluator does not call `all_possible_moves()`. It uses direct bitboard
 * counting and `BOARD_METADATA` to avoid constructing move lists in the minimax
 * leaf hot path.
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>

#include "board.hpp"
#include "rules.hpp"
#include "types.hpp"

namespace kribu::heuristics {

using namespace kribu::board;

/**
 * @brief Very large terminal score used when a player has no pieces or no legal moves.
 */
constexpr i32 EVALUATION_WIN_SCORE = 1'000'000;

/**
 * @brief Material value of one piece.
 * @details
 * Keep this much larger than positional values. In Sholo Guti, losing one piece
 * is usually more important than improving several nodes.
 */
constexpr i32 EVALUATION_MATERIAL_WEIGHT = 1000;

/**
 * @brief Weight for graph-control node values.
 * @details
 * This rewards structurally strong board locations such as central hubs and
 * capture-rich junctions.
 */
constexpr i32 EVALUATION_STRUCTURE_WEIGHT = 2;

/**
 * @brief Weight for directional advancement.
 * @details
 * This rewards `me` moving upward toward nodes 0..2 and punishes the opponent
 * moving downward from their own perspective.
 */
constexpr i32 EVALUATION_ADVANCEMENT_WEIGHT = 1;

/**
 * @brief Weight for raw legal movement options.
 * @details
 * Mobility is useful, but should not dominate material or capture pressure.
 */
constexpr i32 EVALUATION_MOBILITY_WEIGHT = 4;

/**
 * @brief Weight for immediate one-jump capture opportunities.
 */
constexpr i32 EVALUATION_IMMEDIATE_CAPTURE_WEIGHT = 160;

/**
 * @brief Weight for the best available multi-capture chain length.
 */
constexpr i32 EVALUATION_CHAIN_CAPTURE_WEIGHT = 260;

/**
 * @brief Weight for pieces that can be captured by the opponent immediately.
 */
constexpr i32 EVALUATION_VULNERABILITY_WEIGHT = 90;

/**
 * @brief Minimum evaluation depth needed to include mobility and immediate captures.
 */
constexpr int EVALUATION_MOBILITY_DEPTH = 1;

/**
 * @brief Minimum evaluation depth needed to include best multi-capture chain analysis.
 */
constexpr int EVALUATION_CHAIN_DEPTH = 2;

/**
 * @brief Minimum evaluation depth needed to include vulnerability analysis.
 */
constexpr int EVALUATION_VULNERABILITY_DEPTH = 3;

/**
 * @brief Directional advancement table from `me`'s physical starting orientation.
 *
 * @details
 * Higher values are closer to the opponent side for the active player in the
 * standard initial position.
 *
 * Since `me` starts around nodes 21..36 and advances upward, nodes 0..2 are
 * high value and nodes 34..36 are low value.
 *
 * For opponent pieces, this table must be read through `flip_node()` so that
 * the opponent is evaluated from its own forward direction.
 */
constexpr std::array<i32, NUM_NODES> ADVANCEMENT_NODE_WEIGHTS = {
    // Top triangle: most advanced for `me`
    120,
    160,
    120,
    110,
    150,
    110,

    // Main grid, top to bottom
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

    // Bottom triangle: least advanced for `me`
    20,
    40,
    20,
    10,
    20,
    10,
};

/**
 * @brief Symmetric graph-control table.
 *
 * @details
 * Unlike `ADVANCEMENT_NODE_WEIGHTS`, this table is not directional.
 * It values nodes based on board structure: centrality, number of routes,
 * and tactical connectivity.
 *
 * This table must satisfy:
 *
 * @code
 * STRUCTURE_NODE_WEIGHTS[node] == STRUCTURE_NODE_WEIGHTS[flip_node(node)]
 * @endcode
 *
 * Important hubs such as 8, 18, and 28 receive high values.
 */
constexpr std::array<i32, NUM_NODES> STRUCTURE_NODE_WEIGHTS = {
    // Top triangle
    85,
    80,
    85,
    89,
    84,
    89,

    // Main grid
    98,
    89,
    160,
    89,
    98,
    89,
    124,
    110,
    124,
    89,
    123,
    101,
    160,
    101,
    123,
    89,
    124,
    110,
    124,
    89,
    98,
    89,
    160,
    89,
    98,

    // Bottom triangle
    89,
    84,
    89,
    85,
    80,
    85,
};

/**
 * @brief Returns the 180-degree rotated node index.
 *
 * @param nodeIndex Physical node index in the range [0, 36].
 * @return Mirrored node index.
 *
 * @details
 * With your board numbering, the board is indexed in row-major order from
 * top to bottom. Therefore the 180-degree flip is simply:
 *
 * @code
 * flipped = 36 - nodeIndex
 * @endcode
 */
[[nodiscard]] constexpr int flip_node(int nodeIndex) noexcept {
  return NUM_NODES - 1 - nodeIndex;
}

static_assert(flip_node(0) == 36);
static_assert(flip_node(1) == 35);
static_assert(flip_node(18) == 18);
static_assert(flip_node(35) == 1);
static_assert(flip_node(36) == 0);

/**
 * @brief Creates a bit mask for one board node.
 *
 * @param nodeIndex Physical node index.
 * @return Bit mask with only `nodeIndex` set.
 */
[[nodiscard]] constexpr u64 node_mask(int nodeIndex) noexcept {
  return 1ULL << nodeIndex;
}

/**
 * @brief Counts pieces in a bitboard.
 *
 * @param pieceMask Bitboard to count.
 * @return Number of set bits.
 *
 * @details
 * Uses the existing rule-layer piece counter, which already chooses a constexpr
 * path during constant evaluation and an EVE-backed path at runtime.
 */
[[nodiscard]] constexpr i32 count_pieces(u64 pieceMask) noexcept {
  return kribu::sholoGuti::piece_count(pieceMask);
}

/**
 * @brief Scores a non-directional node table.
 *
 * @tparam NodeWeights Symmetric node-weight table.
 * @param activePieces Bitboard containing active-player pieces.
 * @param opponentPieces Bitboard containing opponent pieces.
 * @return Positive score if active player owns better nodes.
 */
template <const std::array<i32, NUM_NODES>& NodeWeights>
[[nodiscard]] constexpr i32 evaluate_symmetric_nodes(u64 activePieces, u64 opponentPieces) noexcept {
  i32 scoreValue = 0;

  for (u64 activeBits = activePieces; activeBits != 0ULL; activeBits &= activeBits - 1ULL) {
    const int nodeIndex = std::countr_zero(activeBits);
    scoreValue += NodeWeights[nodeIndex];
  }

  for (u64 opponentBits = opponentPieces; opponentBits != 0ULL; opponentBits &= opponentBits - 1ULL) {
    const int nodeIndex = std::countr_zero(opponentBits);
    scoreValue -= NodeWeights[nodeIndex];
  }

  return scoreValue;
}

/**
 * @brief Scores a directional node table.
 *
 * @tparam NodeWeights Directional node-weight table for `me`.
 * @param activePieces Bitboard containing active-player pieces.
 * @param opponentPieces Bitboard containing opponent pieces.
 * @return Positive score if active player is more advanced.
 *
 * @details
 * Active pieces are scored directly because the table is written from `me`'s
 * physical starting orientation. Opponent pieces are scored through `flip_node()`
 * so the opponent is evaluated from its own forward direction.
 */
template <const std::array<i32, NUM_NODES>& NodeWeights>
[[nodiscard]] constexpr i32 evaluate_directional_nodes(u64 activePieces, u64 opponentPieces) noexcept {
  i32 scoreValue = 0;

  for (u64 activeBits = activePieces; activeBits != 0ULL; activeBits &= activeBits - 1ULL) {
    const int nodeIndex = std::countr_zero(activeBits);
    scoreValue += NodeWeights[nodeIndex];
  }

  for (u64 opponentBits = opponentPieces; opponentBits != 0ULL; opponentBits &= opponentBits - 1ULL) {
    const int nodeIndex = std::countr_zero(opponentBits);
    scoreValue -= NodeWeights[flip_node(nodeIndex)];
  }

  return scoreValue;
}

/**
 * @brief Counts simple non-capturing moves for one side.
 *
 * @param activePieces Pieces belonging to the side being counted.
 * @param opponentPieces Pieces belonging to the other side.
 * @return Number of currently legal simple moves.
 *
 * @details
 * This does not call `all_possible_moves()`. It scans only occupied active nodes
 * and checks precomputed neighbor lists from `BOARD_METADATA`.
 */
[[nodiscard]] constexpr int count_simple_moves_side(u64 activePieces, u64 opponentPieces) noexcept {
  const u64 occupiedPieces = activePieces | opponentPieces;
  int moveCount = 0;

  for (u64 activeBits = activePieces; activeBits != 0ULL; activeBits &= activeBits - 1ULL) {
    const int sourceNode = std::countr_zero(activeBits);
    const int neighborCount = static_cast<int>(static_cast<u8>(BOARD_METADATA.counts[sourceNode]));

    for (int neighborIndex = 0; neighborIndex < neighborCount; ++neighborIndex) {
      const int targetNode = static_cast<int>(static_cast<u8>(BOARD_METADATA.neighbors[sourceNode][neighborIndex]));
      moveCount += static_cast<int>((occupiedPieces & node_mask(targetNode)) == 0ULL);
    }
  }

  return moveCount;
}

/**
 * @brief Counts legal capture moves from one source node.
 *
 * @param activePieces Pieces belonging to the side being counted.
 * @param opponentPieces Pieces belonging to the other side.
 * @param sourceNode Source node of the capturing piece.
 * @return Number of immediate capture moves from `sourceNode`.
 */
[[nodiscard]] constexpr int count_captures_from(u64 activePieces, u64 opponentPieces, int sourceNode) noexcept {
  const u64 occupiedPieces = activePieces | opponentPieces;
  int captureCount = 0;

  const int captureMoveCount = static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[sourceNode]));

  for (int captureIndex = 0; captureIndex < captureMoveCount; ++captureIndex) {
    const move& currentMove = MOVE_TABLE[BOARD_METADATA.captureMoveIdxByNode[sourceNode][captureIndex]];

    const bool targetEmpty = (occupiedPieces & node_mask(currentMove.to)) == 0ULL;
    const bool victimExists = (opponentPieces & node_mask(currentMove.captured)) != 0ULL;

    captureCount += static_cast<int>(targetEmpty && victimExists);
  }

  return captureCount;
}

/**
 * @brief Counts legal immediate captures for one side.
 *
 * @param activePieces Pieces belonging to the side being counted.
 * @param opponentPieces Pieces belonging to the other side.
 * @return Number of legal immediate capture moves.
 */
[[nodiscard]] constexpr int count_captures_side(u64 activePieces, u64 opponentPieces) noexcept {
  int captureCount = 0;

  for (u64 activeBits = activePieces; activeBits != 0ULL; activeBits &= activeBits - 1ULL) {
    const int sourceNode = std::countr_zero(activeBits);
    captureCount += count_captures_from(activePieces, opponentPieces, sourceNode);
  }

  return captureCount;
}

/**
 * @brief Counts simple moves plus capture moves for one side.
 *
 * @param activePieces Pieces belonging to the side being counted.
 * @param opponentPieces Pieces belonging to the other side.
 * @return Number of one-step legal move choices, ignoring capture-chain state.
 *
 * @details
 * Normal captures are not mandatory in your rules, so simple moves and captures
 * are both counted when no chain is active.
 */
[[nodiscard]] constexpr int count_moves_side(u64 activePieces, u64 opponentPieces) noexcept {
  return count_simple_moves_side(activePieces, opponentPieces) + count_captures_side(activePieces, opponentPieces);
}

/**
 * @brief Finds the longest capture chain starting from one source node.
 *
 * @param activePieces Pieces belonging to the side being counted.
 * @param opponentPieces Pieces belonging to the other side.
 * @param sourceNode Source node of the capturing piece.
 * @return Maximum number of pieces capturable in one chain.
 *
 * @details
 * This is recursive, but bounded: each recursive step removes exactly one
 * opponent piece. With at most 16 opposing pieces, the recursion cannot grow
 * beyond the number of capturable opponent pieces.
 */
[[nodiscard]] constexpr int best_capture_chain_from(u64 activePieces, u64 opponentPieces, int sourceNode) noexcept {
  const u64 occupiedPieces = activePieces | opponentPieces;
  int bestCaptureCount = 0;

  const int captureMoveCount = static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[sourceNode]));

  for (int captureIndex = 0; captureIndex < captureMoveCount; ++captureIndex) {
    const move& currentMove = MOVE_TABLE[BOARD_METADATA.captureMoveIdxByNode[sourceNode][captureIndex]];

    const u64 sourceMask = node_mask(sourceNode);
    const u64 targetMask = node_mask(currentMove.to);
    const u64 victimMask = node_mask(currentMove.captured);

    if ((occupiedPieces & targetMask) != 0ULL) {
      continue;
    }

    if ((opponentPieces & victimMask) == 0ULL) {
      continue;
    }

    const u64 nextActivePieces = (activePieces & ~sourceMask) | targetMask;
    const u64 nextOpponentPieces = opponentPieces & ~victimMask;

    const int chainCaptureCount = 1 + best_capture_chain_from(nextActivePieces, nextOpponentPieces, currentMove.to);

    bestCaptureCount = std::max(bestCaptureCount, chainCaptureCount);
  }

  return bestCaptureCount;
}

/**
 * @brief Finds the longest available capture chain for one side.
 *
 * @param activePieces Pieces belonging to the side being counted.
 * @param opponentPieces Pieces belonging to the other side.
 * @return Maximum number of pieces capturable by any one active piece.
 */
[[nodiscard]] constexpr int best_capture_chain_side(u64 activePieces, u64 opponentPieces) noexcept {
  int bestCaptureCount = 0;

  for (u64 activeBits = activePieces; activeBits != 0ULL; activeBits &= activeBits - 1ULL) {
    const int sourceNode = std::countr_zero(activeBits);
    bestCaptureCount = std::max(bestCaptureCount, best_capture_chain_from(activePieces, opponentPieces, sourceNode));
  }

  return bestCaptureCount;
}

/**
 * @brief Counts pieces that can be captured immediately by the opponent.
 *
 * @param activePieces Pieces that might be vulnerable.
 * @param opponentPieces Pieces that might capture them.
 * @return Number of active pieces that are currently capturable.
 *
 * @details
 * A piece is counted once even if multiple opponent pieces can capture it.
 * This is intentionally more expensive and is only enabled at higher evaluation
 * depths.
 */
[[nodiscard]] constexpr int count_vulnerable_pieces(u64 activePieces, u64 opponentPieces) noexcept {
  const u64 occupiedPieces = activePieces | opponentPieces;
  int vulnerableCount = 0;

  for (u64 activeBits = activePieces; activeBits != 0ULL; activeBits &= activeBits - 1ULL) {
    const int victimNode = std::countr_zero(activeBits);
    bool pieceVulnerable = false;

    for (u64 opponentBits = opponentPieces; opponentBits != 0ULL; opponentBits &= opponentBits - 1ULL) {
      const int sourceNode = std::countr_zero(opponentBits);
      const int captureMoveCount = static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[sourceNode]));

      for (int captureIndex = 0; captureIndex < captureMoveCount; ++captureIndex) {
        const move& currentMove = MOVE_TABLE[BOARD_METADATA.captureMoveIdxByNode[sourceNode][captureIndex]];

        if (currentMove.captured != victimNode) {
          continue;
        }

        const bool targetEmpty = (occupiedPieces & node_mask(currentMove.to)) == 0ULL;

        if (targetEmpty) {
          pieceVulnerable = true;
          break;
        }
      }

      if (pieceVulnerable) {
        break;
      }
    }

    vulnerableCount += static_cast<int>(pieceVulnerable);
  }

  return vulnerableCount;
}

/**
 * @brief Evaluates a board state from the active player's perspective.
 *
 * @param state Current board state.
 * @param depth Evaluation budget.
 * @return Positive score favors `state.me`; negative score favors `state.opp`.
 *
 * @details
 * This function is designed for minimax or negamax engines that store every
 * position from the active player's perspective.
 *
 * The `depth` parameter controls how much heuristic work is allowed:
 *
 * - `depth == 0`: material + structure + advancement.
 * - `depth >= 1`: also includes mobility and immediate capture pressure.
 * - `depth >= 2`: also includes best multi-capture chain potential.
 * - `depth >= 3`: also includes vulnerability analysis.
 *
 * This parameter is an evaluation-budget depth, not necessarily the same as
 * remaining minimax depth. If your minimax leaf always has remaining depth 0,
 * call this with a configured evaluation budget instead:
 *
 * @code
 * const i32 score = kribu::heuristics::evaluate(state, 2);
 * @endcode
 *
 * During an active capture chain, `activeCaptureIdx != -1`, the same player is
 * still moving. In that case, this evaluator scores only the locked piece's
 * continuation captures and does not pretend that it is the opponent's turn.
 */
[[nodiscard]] constexpr i32 evaluate(const boardState& state, int depth) noexcept {
  const i32 activePieceCount = count_pieces(state.me);
  const i32 opponentPieceCount = count_pieces(state.opp);

  if (opponentPieceCount == 0) {
    return EVALUATION_WIN_SCORE;
  }

  if (activePieceCount == 0) {
    return -EVALUATION_WIN_SCORE;
  }

  const i32 materialScore = EVALUATION_MATERIAL_WEIGHT * (activePieceCount - opponentPieceCount);

  const i32 structureScore =
      EVALUATION_STRUCTURE_WEIGHT * evaluate_symmetric_nodes<STRUCTURE_NODE_WEIGHTS>(state.me, state.opp);

  const i32 advancementScore =
      EVALUATION_ADVANCEMENT_WEIGHT * evaluate_directional_nodes<ADVANCEMENT_NODE_WEIGHTS>(state.me, state.opp);

  i32 totalScore = materialScore + structureScore + advancementScore;

  if (depth < EVALUATION_MOBILITY_DEPTH) {
    return totalScore;
  }

  if (state.activeCaptureIdx != -1) {
    const int lockedSourceNode = static_cast<int>(static_cast<u8>(state.activeCaptureIdx));

    const int activeCaptureCount = count_captures_from(state.me, state.opp, lockedSourceNode);

    totalScore += EVALUATION_IMMEDIATE_CAPTURE_WEIGHT * activeCaptureCount;

    // In an active capture chain, END_CHAIN_MOVE is legal, plus every valid
    // continuation capture from the locked piece.
    totalScore += EVALUATION_MOBILITY_WEIGHT * (1 + activeCaptureCount);

    if (depth >= EVALUATION_CHAIN_DEPTH) {
      const int activeBestChain = best_capture_chain_from(state.me, state.opp, lockedSourceNode);

      totalScore += EVALUATION_CHAIN_CAPTURE_WEIGHT * activeBestChain;
    }

    return totalScore;
  }

  const int activeCaptureCount = count_captures_side(state.me, state.opp);
  const int opponentCaptureCount = count_captures_side(state.opp, state.me);

  totalScore += EVALUATION_IMMEDIATE_CAPTURE_WEIGHT * (activeCaptureCount - opponentCaptureCount);

  const int activeMoveCount = count_simple_moves_side(state.me, state.opp) + activeCaptureCount;

  const int opponentMoveCount = count_simple_moves_side(state.opp, state.me) + opponentCaptureCount;

  if (activeMoveCount == 0) {
    return -EVALUATION_WIN_SCORE;
  }

  if (opponentMoveCount == 0) {
    return EVALUATION_WIN_SCORE;
  }

  totalScore += EVALUATION_MOBILITY_WEIGHT * (activeMoveCount - opponentMoveCount);

  if (depth >= EVALUATION_CHAIN_DEPTH) {
    const int activeBestChain = best_capture_chain_side(state.me, state.opp);
    const int opponentBestChain = best_capture_chain_side(state.opp, state.me);

    totalScore += EVALUATION_CHAIN_CAPTURE_WEIGHT * (activeBestChain - opponentBestChain);
  }

  if (depth >= EVALUATION_VULNERABILITY_DEPTH) {
    const int activeVulnerableCount = count_vulnerable_pieces(state.me, state.opp);
    const int opponentVulnerableCount = count_vulnerable_pieces(state.opp, state.me);

    totalScore += EVALUATION_VULNERABILITY_WEIGHT * (opponentVulnerableCount - activeVulnerableCount);
  }

  return totalScore;
}

}  // namespace kribu::heuristics