/**
 * @file board.hpp
 * @brief Board representation and topology definitions for the Sholo Guti engine.
 * @details This file defines the `boardState` and `move` structures, coordinate-free move tables,
 *          and compile-time board metadata calculations (adjacency list, move mappings, maximum potential moves).
 */

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "kribu/zobrist.hpp"
#include "types.hpp"

/**
 * @namespace kribu::board
 * @brief Namespace containing the data structures, types, and constants representing the board.
 */
namespace kribu::board {

/**
 * @struct boardState
 * @brief Represents the physical layout of pieces on the 37-node Sholo Guti board.
 * @details Uses two 64-bit integer bitmasks to represent player pieces, and a signed 8-bit
 *          integer to keep track of a capturing piece during a multi-capture chain.
 */
struct boardState {
  /**
   * @brief Bitmask representing the active player's pieces.
   * @details Up to 16 pieces, corresponding to set bits in the range [0, 36].
   */
  u64 me = 0;

  /**
   * @brief Bitmask representing the opponent player's pieces.
   * @details Up to 16 pieces, corresponding to set bits in the range [0, 36].
   */
  u64 opp = 0;

  /**
   * @brief Node index of the capturing piece locked in a multi-capture chain.
   * @details Equals -1 if no capture chain is active. When locked, only capture moves starting at
   *          this node index (or the END_CHAIN_MOVE sentinel) are considered legal.
   */
  i8 activeCaptureIdx = -1;

  /**
   * @brief Incremental Zobrist hash key representing the board state.
   */
  u64 hash = 0;

  /**
   * @brief Compares two board states for exact equality of piece placements and capture states.
   * @param other The board state to compare against.
   * @return True if both player masks and active capture indices are identical, false otherwise.
   */
  bool operator==(const boardState& other) const {
    return hash == other.hash && me == other.me && opp == other.opp && activeCaptureIdx == other.activeCaptureIdx;
  }
};

}  // namespace kribu::board

namespace kribu::board {

/**
 * @brief The standard initial layout of a Sholo Guti board.
 * @details Contains 16 pieces for each player:
 *          - Opponent pieces occupy nodes 0 to 15 (lower rows, bits 0-15 set).
 *          - Active player pieces occupy nodes 21 to 36 (upper rows, bits 21-36 set).
 *          - The center row (nodes 16 to 20, bits 16-20) starts completely empty.
 *          - No active capture chain is in progress (activeCaptureIdx is -1).
 */
constexpr boardState INITIAL_STATE{
    .me = 0x0000'001F'FFE0'0000ULL,   // Bits 21-36
    .opp = 0x0000'0000'0000'FFFFULL,  // Bits 0-15
    .activeCaptureIdx = -1,
    .hash = kribu::zobrist::compute_hash(
        {.activePlayer = 0x0000'001F'FFE0'0000ULL, .opponentPlayer = 0x0000'0000'0000'FFFFULL, .activeCaptureIdx = -1}),
};

/**
 * @brief The total number of playable nodes (vertices) on a standard Sholo Guti board.
 */
constexpr int NUM_NODES = 37;

/**
 * @brief Special sentinel move ID signifying that a player chooses to terminate an active capture chain.
 * @details When a player completes a capture, they may continue capturing if further captures are structurally
 *          legal. If they choose to pass/stop or no further capture moves are available, they must play the
 *          END_CHAIN_MOVE to conclude their turn.
 */
constexpr int END_CHAIN_MOVE = 0;

/**
 * @struct move
 * @brief Represents the physical details of a single board move or transition.
 * @details Encodes the source node, destination node, and the jumped-over opponent node (for captures).
 */
struct move {
  /**
   * @brief The starting node index of the moving piece.
   */
  i8 from;

  /**
   * @brief The target/destination node index of the moving piece.
   */
  i8 to;

  /**
   * @brief The node index containing the opponent's piece being captured.
   */
  i8 captured;

  /**
   * @brief Three-way structural comparison operator for sorting and ordering moves.
   * @details Compares moves lexicographically by from, to, and captured fields.
   */
  constexpr auto operator<=>(const move&) const = default;
};

/**
 * @brief The complete static list of all potential moves based solely on the board's graph topology.
 * @details Contains the end-chain sentinel, all simple sliding moves, and all jump-capturing moves.
 *          These represent structural opportunities and must be verified against dynamic board masks during play.
 */
constexpr std::array MOVES = std::to_array<move>({
    {.from = -1, .to = -1, .captured = -1},
    // 0 (4)
    {.from = 0, .to = 1, .captured = -1},
    {.from = 0, .to = 2, .captured = 1},
    {.from = 0, .to = 3, .captured = -1},
    {.from = 0, .to = 8, .captured = 3},
    // 1 (4)
    {.from = 1, .to = 0, .captured = -1},
    {.from = 1, .to = 2, .captured = -1},
    {.from = 1, .to = 4, .captured = -1},
    {.from = 1, .to = 8, .captured = 4},
    // 2 (4)
    {.from = 2, .to = 0, .captured = 1},
    {.from = 2, .to = 1, .captured = -1},
    {.from = 2, .to = 5, .captured = -1},
    {.from = 2, .to = 8, .captured = 5},
    // 3 (5)
    {.from = 3, .to = 0, .captured = -1},
    {.from = 3, .to = 4, .captured = -1},
    {.from = 3, .to = 5, .captured = 4},
    {.from = 3, .to = 8, .captured = -1},
    {.from = 3, .to = 14, .captured = 8},
    // 4 (5)
    {.from = 4, .to = 1, .captured = -1},
    {.from = 4, .to = 3, .captured = -1},
    {.from = 4, .to = 5, .captured = -1},
    {.from = 4, .to = 8, .captured = -1},
    {.from = 4, .to = 13, .captured = 8},
    // 5 (5)
    {.from = 5, .to = 2, .captured = -1},
    {.from = 5, .to = 3, .captured = 4},
    {.from = 5, .to = 4, .captured = -1},
    {.from = 5, .to = 8, .captured = -1},
    {.from = 5, .to = 12, .captured = 8},
    // 6 (6)
    {.from = 6, .to = 7, .captured = -1},
    {.from = 6, .to = 8, .captured = 7},
    {.from = 6, .to = 11, .captured = -1},
    {.from = 6, .to = 12, .captured = -1},
    {.from = 6, .to = 16, .captured = 11},
    {.from = 6, .to = 18, .captured = 12},
    // 7 (5)
    {.from = 7, .to = 6, .captured = -1},
    {.from = 7, .to = 8, .captured = -1},
    {.from = 7, .to = 9, .captured = 8},
    {.from = 7, .to = 12, .captured = -1},
    {.from = 7, .to = 17, .captured = 12},
    // 8 (16)
    {.from = 8, .to = 0, .captured = 3},
    {.from = 8, .to = 1, .captured = 4},
    {.from = 8, .to = 2, .captured = 5},
    {.from = 8, .to = 3, .captured = -1},
    {.from = 8, .to = 4, .captured = -1},
    {.from = 8, .to = 5, .captured = -1},
    {.from = 8, .to = 6, .captured = 7},
    {.from = 8, .to = 7, .captured = -1},
    {.from = 8, .to = 9, .captured = -1},
    {.from = 8, .to = 10, .captured = 9},
    {.from = 8, .to = 12, .captured = -1},
    {.from = 8, .to = 13, .captured = -1},
    {.from = 8, .to = 14, .captured = -1},
    {.from = 8, .to = 16, .captured = 12},
    {.from = 8, .to = 18, .captured = 13},
    {.from = 8, .to = 20, .captured = 14},
    // 9 (5)
    {.from = 9, .to = 7, .captured = 8},
    {.from = 9, .to = 8, .captured = -1},
    {.from = 9, .to = 10, .captured = -1},
    {.from = 9, .to = 14, .captured = -1},
    {.from = 9, .to = 19, .captured = 14},
    // 10 (6)
    {.from = 10, .to = 8, .captured = 9},
    {.from = 10, .to = 9, .captured = -1},
    {.from = 10, .to = 14, .captured = -1},
    {.from = 10, .to = 15, .captured = -1},
    {.from = 10, .to = 18, .captured = 14},
    {.from = 10, .to = 20, .captured = 15},
    // 11 (5)
    {.from = 11, .to = 6, .captured = -1},
    {.from = 11, .to = 12, .captured = -1},
    {.from = 11, .to = 13, .captured = 12},
    {.from = 11, .to = 16, .captured = -1},
    {.from = 11, .to = 21, .captured = 16},
    // 12 (12)
    {.from = 12, .to = 5, .captured = 8},
    {.from = 12, .to = 6, .captured = -1},
    {.from = 12, .to = 7, .captured = -1},
    {.from = 12, .to = 8, .captured = -1},
    {.from = 12, .to = 11, .captured = -1},
    {.from = 12, .to = 13, .captured = -1},
    {.from = 12, .to = 14, .captured = 13},
    {.from = 12, .to = 16, .captured = -1},
    {.from = 12, .to = 17, .captured = -1},
    {.from = 12, .to = 18, .captured = -1},
    {.from = 12, .to = 22, .captured = 17},
    {.from = 12, .to = 24, .captured = 18},
    // 13 (8)
    {.from = 13, .to = 4, .captured = 8},
    {.from = 13, .to = 8, .captured = -1},
    {.from = 13, .to = 11, .captured = 12},
    {.from = 13, .to = 12, .captured = -1},
    {.from = 13, .to = 14, .captured = -1},
    {.from = 13, .to = 15, .captured = 14},
    {.from = 13, .to = 18, .captured = -1},
    {.from = 13, .to = 23, .captured = 18},
    // 14 (12)
    {.from = 14, .to = 3, .captured = 8},
    {.from = 14, .to = 8, .captured = -1},
    {.from = 14, .to = 9, .captured = -1},
    {.from = 14, .to = 10, .captured = -1},
    {.from = 14, .to = 12, .captured = 13},
    {.from = 14, .to = 13, .captured = -1},
    {.from = 14, .to = 15, .captured = -1},
    {.from = 14, .to = 18, .captured = -1},
    {.from = 14, .to = 19, .captured = -1},
    {.from = 14, .to = 20, .captured = -1},
    {.from = 14, .to = 22, .captured = 18},
    {.from = 14, .to = 24, .captured = 19},
    // 15 (5)
    {.from = 15, .to = 10, .captured = -1},
    {.from = 15, .to = 13, .captured = 14},
    {.from = 15, .to = 14, .captured = -1},
    {.from = 15, .to = 20, .captured = -1},
    {.from = 15, .to = 25, .captured = 20},
    // 16 (10)
    {.from = 16, .to = 6, .captured = 11},
    {.from = 16, .to = 8, .captured = 12},
    {.from = 16, .to = 11, .captured = -1},
    {.from = 16, .to = 12, .captured = -1},
    {.from = 16, .to = 17, .captured = -1},
    {.from = 16, .to = 18, .captured = 17},
    {.from = 16, .to = 21, .captured = -1},
    {.from = 16, .to = 22, .captured = -1},
    {.from = 16, .to = 26, .captured = 21},
    {.from = 16, .to = 28, .captured = 22},
    // 17 (7)
    {.from = 17, .to = 7, .captured = 12},
    {.from = 17, .to = 12, .captured = -1},
    {.from = 17, .to = 16, .captured = -1},
    {.from = 17, .to = 18, .captured = -1},
    {.from = 17, .to = 19, .captured = 18},
    {.from = 17, .to = 22, .captured = -1},
    {.from = 17, .to = 27, .captured = 22},
    // 18 (16)
    {.from = 18, .to = 6, .captured = 12},
    {.from = 18, .to = 8, .captured = 13},
    {.from = 18, .to = 10, .captured = 14},
    {.from = 18, .to = 12, .captured = -1},
    {.from = 18, .to = 13, .captured = -1},
    {.from = 18, .to = 14, .captured = -1},
    {.from = 18, .to = 16, .captured = 17},
    {.from = 18, .to = 17, .captured = -1},
    {.from = 18, .to = 19, .captured = -1},
    {.from = 18, .to = 20, .captured = 19},
    {.from = 18, .to = 22, .captured = -1},
    {.from = 18, .to = 23, .captured = -1},
    {.from = 18, .to = 24, .captured = -1},
    {.from = 18, .to = 26, .captured = 22},
    {.from = 18, .to = 28, .captured = 23},
    {.from = 18, .to = 30, .captured = 24},
    // 19 (7)
    {.from = 19, .to = 9, .captured = 14},
    {.from = 19, .to = 14, .captured = -1},
    {.from = 19, .to = 17, .captured = 18},
    {.from = 19, .to = 18, .captured = -1},
    {.from = 19, .to = 20, .captured = -1},
    {.from = 19, .to = 24, .captured = -1},
    {.from = 19, .to = 29, .captured = 24},
    // 20 (10)
    {.from = 20, .to = 8, .captured = 14},
    {.from = 20, .to = 10, .captured = 15},
    {.from = 20, .to = 14, .captured = -1},
    {.from = 20, .to = 15, .captured = -1},
    {.from = 20, .to = 18, .captured = 19},
    {.from = 20, .to = 19, .captured = -1},
    {.from = 20, .to = 24, .captured = -1},
    {.from = 20, .to = 25, .captured = -1},
    {.from = 20, .to = 28, .captured = 24},
    {.from = 20, .to = 30, .captured = 25},
    // 21 (5)
    {.from = 21, .to = 11, .captured = 16},
    {.from = 21, .to = 16, .captured = -1},
    {.from = 21, .to = 22, .captured = -1},
    {.from = 21, .to = 23, .captured = 22},
    {.from = 21, .to = 26, .captured = -1},
    // 22 (12)
    {.from = 22, .to = 12, .captured = 17},
    {.from = 22, .to = 14, .captured = 18},
    {.from = 22, .to = 16, .captured = -1},
    {.from = 22, .to = 17, .captured = -1},
    {.from = 22, .to = 18, .captured = -1},
    {.from = 22, .to = 21, .captured = -1},
    {.from = 22, .to = 23, .captured = -1},
    {.from = 22, .to = 24, .captured = 23},
    {.from = 22, .to = 26, .captured = -1},
    {.from = 22, .to = 27, .captured = -1},
    {.from = 22, .to = 28, .captured = -1},
    {.from = 22, .to = 33, .captured = 28},
    // 23 (8)
    {.from = 23, .to = 13, .captured = 18},
    {.from = 23, .to = 18, .captured = -1},
    {.from = 23, .to = 21, .captured = 22},
    {.from = 23, .to = 22, .captured = -1},
    {.from = 23, .to = 24, .captured = -1},
    {.from = 23, .to = 25, .captured = 24},
    {.from = 23, .to = 28, .captured = -1},
    {.from = 23, .to = 32, .captured = 28},
    // 24 (12)
    {.from = 24, .to = 12, .captured = 18},
    {.from = 24, .to = 14, .captured = 19},
    {.from = 24, .to = 18, .captured = -1},
    {.from = 24, .to = 19, .captured = -1},
    {.from = 24, .to = 20, .captured = -1},
    {.from = 24, .to = 22, .captured = 23},
    {.from = 24, .to = 23, .captured = -1},
    {.from = 24, .to = 25, .captured = -1},
    {.from = 24, .to = 28, .captured = -1},
    {.from = 24, .to = 29, .captured = -1},
    {.from = 24, .to = 30, .captured = -1},
    {.from = 24, .to = 31, .captured = 28},
    // 25 (5)
    {.from = 25, .to = 15, .captured = 20},
    {.from = 25, .to = 20, .captured = -1},
    {.from = 25, .to = 23, .captured = 24},
    {.from = 25, .to = 24, .captured = -1},
    {.from = 25, .to = 30, .captured = -1},
    // 26 (6)
    {.from = 26, .to = 16, .captured = 21},
    {.from = 26, .to = 18, .captured = 22},
    {.from = 26, .to = 21, .captured = -1},
    {.from = 26, .to = 22, .captured = -1},
    {.from = 26, .to = 27, .captured = -1},
    {.from = 26, .to = 28, .captured = 27},
    // 27 (5)
    {.from = 27, .to = 17, .captured = 22},
    {.from = 27, .to = 22, .captured = -1},
    {.from = 27, .to = 26, .captured = -1},
    {.from = 27, .to = 28, .captured = -1},
    {.from = 27, .to = 29, .captured = 28},
    // 28 (16)
    {.from = 28, .to = 16, .captured = 22},
    {.from = 28, .to = 18, .captured = 23},
    {.from = 28, .to = 20, .captured = 24},
    {.from = 28, .to = 22, .captured = -1},
    {.from = 28, .to = 23, .captured = -1},
    {.from = 28, .to = 24, .captured = -1},
    {.from = 28, .to = 26, .captured = 27},
    {.from = 28, .to = 27, .captured = -1},
    {.from = 28, .to = 29, .captured = -1},
    {.from = 28, .to = 30, .captured = 29},
    {.from = 28, .to = 31, .captured = -1},
    {.from = 28, .to = 32, .captured = -1},
    {.from = 28, .to = 33, .captured = -1},
    {.from = 28, .to = 34, .captured = 31},
    {.from = 28, .to = 35, .captured = 32},
    {.from = 28, .to = 36, .captured = 33},
    // 29 (5)
    {.from = 29, .to = 19, .captured = 24},
    {.from = 29, .to = 24, .captured = -1},
    {.from = 29, .to = 27, .captured = 28},
    {.from = 29, .to = 28, .captured = -1},
    {.from = 29, .to = 30, .captured = -1},
    // 30 (6)
    {.from = 30, .to = 18, .captured = 24},
    {.from = 30, .to = 20, .captured = 25},
    {.from = 30, .to = 24, .captured = -1},
    {.from = 30, .to = 25, .captured = -1},
    {.from = 30, .to = 28, .captured = 29},
    {.from = 30, .to = 29, .captured = -1},
    // 31 (5)
    {.from = 31, .to = 24, .captured = 28},
    {.from = 31, .to = 28, .captured = -1},
    {.from = 31, .to = 32, .captured = -1},
    {.from = 31, .to = 33, .captured = 32},
    {.from = 31, .to = 34, .captured = -1},
    // 32 (5)
    {.from = 32, .to = 23, .captured = 28},
    {.from = 32, .to = 28, .captured = -1},
    {.from = 32, .to = 31, .captured = -1},
    {.from = 32, .to = 33, .captured = -1},
    {.from = 32, .to = 35, .captured = -1},
    // 33 (5)
    {.from = 33, .to = 22, .captured = 28},
    {.from = 33, .to = 28, .captured = -1},
    {.from = 33, .to = 31, .captured = 32},
    {.from = 33, .to = 32, .captured = -1},
    {.from = 33, .to = 36, .captured = -1},
    // 34 (4)
    {.from = 34, .to = 28, .captured = 31},
    {.from = 34, .to = 31, .captured = -1},
    {.from = 34, .to = 35, .captured = -1},
    {.from = 34, .to = 36, .captured = 35},
    // 35 (4)
    {.from = 35, .to = 28, .captured = 32},
    {.from = 35, .to = 32, .captured = -1},
    {.from = 35, .to = 34, .captured = -1},
    {.from = 35, .to = 36, .captured = -1},
    // 36 (4)
    {.from = 36, .to = 28, .captured = 33},
    {.from = 36, .to = 33, .captured = -1},
    {.from = 36, .to = 34, .captured = 35},
    {.from = 36, .to = 35, .captured = -1},
});

/**
 * @brief Validates and generates a sorted array of unique legal moves at compile time.
 *
 * @details Copies the global `MOVES` list, sorts it according to specific move priority
 * rules, and verifies that the resulting array contains no duplicate entries. If duplicates
 * are detected, a compile-time static assertion will fail.
 *
 * The sort hierarchy applied is:
 * 1. End-chain sentinel moves (where `from == -1`) take absolute priority.
 * 2. Simple moves (where `captured == -1`) precede capture moves.
 * 3. Tied categories are resolved lexicographically using the `move` structural comparison operator<.
 *
 * @return A sorted, unique `std::array` of moves.
 */
constexpr auto generate_validated_move_table() {
  auto sorted = MOVES;

  std::ranges::sort(sorted, [](const move& lhs, const move& rhs) {
    return std::forward_as_tuple(lhs.from != -1, lhs.captured != -1, lhs)
           < std::forward_as_tuple(rhs.from != -1, rhs.captured != -1, rhs);
  });

  // Verify uniqueness post-sort
  const bool is_unique = std::ranges::adjacent_find(sorted) == sorted.end();
  if (!is_unique) {
    throw std::logic_error("MOVE_TABLE array contains duplicate entries!");
  }

  return sorted;
}

/**
 * @brief Compile-time validated, sorted mapping of all valid moves.
 * @details Order guarantees end-chain sentinels first, followed by simple moves, then captures.
 */
constexpr auto MOVE_TABLE = generate_validated_move_table();

namespace detail {

/**
 * @struct boardMetadata
 * @brief Stores structural properties of the board computed at compile time.
 */
struct boardMetadata {
  std::array<std::array<i8, 8>, NUM_NODES> neighbors{};
  std::array<i8, NUM_NODES> counts{};
  std::array<std::array<i16, 8>, NUM_NODES> captureMoveIdxByNode{};
  std::array<i8, NUM_NODES> captureMoveCountByNode{};
  std::array<std::array<i16, NUM_NODES>, NUM_NODES> moveIdMap{};

  int numSimpleMoves = 0;
  int numCaptureMoves = 0;
  int maxMovesPerState = 0;
};

/**
 * @brief Constructs the boardMetadata at compile time from MOVE_TABLE.
 * @return Computed boardMetadata.
 */
constexpr boardMetadata build_metadata() {
  boardMetadata meta{};

  for (auto& row : meta.moveIdMap) {
    row.fill(-1);
  }

  for (int idx = 0; std::cmp_less(idx, MOVE_TABLE.size()); ++idx) {
    const auto& mvEntry = MOVE_TABLE[idx];
    if (mvEntry.from < 0) {
      [[unlikely]] continue;
    }

    meta.moveIdMap[mvEntry.from][mvEntry.to] = static_cast<i16>(idx);

    if (mvEntry.captured == -1) {
      meta.neighbors[mvEntry.from][meta.counts[mvEntry.from]++] = mvEntry.to;
      meta.numSimpleMoves++;
    } else {
      const i8 slotIdx = meta.captureMoveCountByNode[mvEntry.from]++;
      meta.captureMoveIdxByNode[mvEntry.from][slotIdx] = static_cast<i16>(idx);
      meta.numCaptureMoves++;
    }
  }

  for (int i = 0; i < NUM_NODES; ++i) {
    std::sort(meta.neighbors[i].begin(), meta.neighbors[i].begin() + meta.counts[i]);
  }

  // Calculate total potential moves originating from each individual node.
  std::array<int, NUM_NODES> perNode{};
  for (int i = 0; i < NUM_NODES; ++i) {
    perNode[i] = meta.counts[i] + static_cast<int>(meta.captureMoveCountByNode[i]);
  }

  // Use the transparent greater<> functor to avoid explicit type repetition
  std::ranges::sort(perNode, std::greater<>{});

  // Sum the top 16 nodes using a view slice, starting from 1 for END_CHAIN_MOVE
  auto top16 = perNode | std::views::take(16);
  meta.maxMovesPerState = std::accumulate(top16.begin(), top16.end(), 1);

  return meta;
}

}  // namespace detail

/**
 * @brief Metadata representation of the board computed at compile time.
 */
constexpr detail::boardMetadata BOARD_METADATA = detail::build_metadata();

/**
 * @brief Compile-time derived number of simple moves.
 */
constexpr int NUM_SIMPLE_MOVES = BOARD_METADATA.numSimpleMoves;

/**
 * @brief Compile-time derived number of capture moves.
 */
constexpr int NUM_CAPTURE_MOVES = BOARD_METADATA.numCaptureMoves;

/**
 * @brief Compile-time derived total move count (including END_CHAIN_MOVE).
 */
constexpr int TOTAL_MOVE_COUNT = 1 + NUM_SIMPLE_MOVES + NUM_CAPTURE_MOVES;

/**
 * @brief Exact upper bound on valid moves in any single game state.
 * @details Sum of the 16 largest per-node total-move counts + 1 (END_CHAIN_MOVE),
 *          computed at compile time from the board topology.
 */
constexpr int MAX_MOVES_PER_STATE = BOARD_METADATA.maxMovesPerState;

}  // namespace kribu::board
