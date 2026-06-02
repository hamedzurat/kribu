/**
 * @file rules.hpp
 * @brief Rules verification, move validation, and game loop conditions for Sholo Guti.
 * @details This file implements the validation of sliding and jump moves, piece counting,
 *          multi-capture chain traversal, and game over detection.
 */

#pragma once

#include <array>
#include <bit>
#include <eve/eve.hpp>
#include <eve/module/core.hpp>
#include <stdexcept>

#include "board.hpp"
#include "types.hpp"
#include "zobrist.hpp"

namespace kribu {
// Thread-local configurations and game history moved or removed to avoid process-wide globals.
}  // namespace kribu

/**
 * @namespace kribu::sholoGuti
 * @brief Main namespace for the Sholo Guti engine rules and status calculations.
 */
namespace kribu::sholoGuti {

using namespace kribu::board;

/**
 * @enum GameStatus
 * @brief Represents the three possible final outcomes or the current state of a Sholo Guti game.
 */
enum class GameStatus : i8 {
  /**
   * @brief Opponent has won because the active player has no moves left (stalemate).
   */
  OPP_WINS_STALEMATE = -2,

  /**
   * @brief Opponent has won because the active player has lost all pieces.
   */
  OPP_WINS_ELIMINATION = -1,

  /**
   * @brief Game is still active and ongoing.
   */
  ONGOING = 0,

  /**
   * @brief Active player (me) has won because the opponent has lost all pieces.
   */
  ME_WINS_ELIMINATION = 1,

  /**
   * @brief Active player (me) has won because the opponent has no moves left (stalemate).
   */
  ME_WINS_STALEMATE = 2,

  /**
   * @brief Game is drawn due to progress limit (no capture for MAX_HISTORY_LIMIT moves).
   */
  DRAW_PROGRESS_RULE = 3,
};

/**
 * @struct MoveList
 * @brief A stack-allocated list of move IDs with a fixed capacity.
 * @details Avoids heap allocation to ensure high performance inside critical code paths.
 */
struct MoveList {
  /**
   * @brief Array buffer storing the move IDs.
   */
  std::array<i16, MAX_MOVES_PER_STATE> moves{};

  /**
   * @brief The number of elements currently stored in the list.
   */
  int count = 0;

  /**
   * @brief Appends a move ID to the list.
   * @param moveId The ID of the move to append.
   */
  constexpr void push(i16 moveId) noexcept { moves[count++] = moveId; }

  /**
   * @brief Returns a pointer to the beginning of the list.
   * @return Const iterator pointer to the first element.
   */
  [[nodiscard]] constexpr const i16* begin() const noexcept { return moves.data(); }

  /**
   * @brief Returns a pointer to the end of the list.
   * @return Const iterator pointer to one-past-the-last element.
   */
  [[nodiscard]] constexpr const i16* end() const noexcept { return moves.data() + count; }

  /**
   * @brief Checks if the list is empty.
   * @return True if the list has zero elements, false otherwise.
   */
  [[nodiscard]] bool empty() const noexcept { return count == 0; }

  /**
   * @brief Returns the number of move IDs in the list.
   * @return The size of the list.
   */
  [[nodiscard]] int size() const noexcept { return count; }
};

/**
 * @brief Counts the number of active pieces represented in a bitboard bitmask.
 * @param mask Bitmask representing player pieces.
 * @return Count of set bits (pieces).
 */
[[nodiscard]] inline i32 piece_count(u64 mask) noexcept {
  return static_cast<i32>(eve::popcount(mask));
}

/**
 * @brief Decodes a move ID to inspect its details (from, to, captured).
 * @param moveId The ID of the move to decode.
 * @return Decoded Move struct.
 * @throws std::out_of_range If moveId is invalid.
 */
[[nodiscard]] constexpr move decode_move(int moveId) {
  if (moveId < 0 || moveId >= TOTAL_MOVE_COUNT) {
    throw std::out_of_range("Invalid move ID");
  }
  return MOVE_TABLE[moveId];
}

/**
 * @brief Checks whether a move ID represents a simple (non-capturing) slide move.
 * @param moveId Move ID to check.
 * @return True if it is a simple move, false otherwise.
 */
[[nodiscard]] constexpr bool is_simple_move(int moveId) noexcept {
  return moveId > END_CHAIN_MOVE && moveId <= NUM_SIMPLE_MOVES;
}

/**
 * @brief Checks whether a move ID represents a jump-capturing move.
 * @param moveId Move ID to check.
 * @return True if it is a capture move, false otherwise.
 */
[[nodiscard]] constexpr bool is_capture_move(int moveId) noexcept {
  return moveId > NUM_SIMPLE_MOVES && moveId < TOTAL_MOVE_COUNT;
}

/**
 * @brief Finds the move ID corresponding to a given (from, destination) node pair.
 * @param from Source node index.
 * @param dst  Destination node index.
 * @return The move ID, or -1 if no such move exists in the static move table.
 */
[[nodiscard]] constexpr int find_move(i8 from, i8 dst) noexcept {
  if (from < 0 || from >= NUM_NODES || dst < 0 || dst >= NUM_NODES) {
    return -1;
  }
  return BOARD_METADATA.moveIdMap[from][dst];
}

/**
 * @brief Flips the board 180 degrees, swapping the active player and the opponent.
 * @details Used to evaluate the board from the opponent's perspective.
 * @param state Current BoardState.
 * @return Mirrored BoardState.
 */
[[nodiscard]] constexpr boardState flip_board(const boardState& state) noexcept {
  constexpr std::array<i8, NUM_NODES> FLIP_MAP = {36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24,
                                                  23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
                                                  10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0};
  boardState flipped;
  flipped.activeCaptureIdx = state.activeCaptureIdx != -1 ? FLIP_MAP[state.activeCaptureIdx] : static_cast<i8>(-1);

  // Loop through each set bit (piece) in state.me.
  // - bits &= bits - 1 is Brian Kernighan's algorithm, which clears the lowest set bit.
  //   Example: If bits = 0b10100 (indices 2 and 4 are set):
  //     bits - 1        = 0b10011
  //     bits & (bits-1) = 0b10000 (cleared lowest set bit at index 2)
  for (u64 bits = state.me; bits != 0U; bits &= bits - 1) {
    // - std::countr_zero(bits) finds trailing zeroes, which is the index of the lowest set bit.
    //   Example: std::countr_zero(0b10100) -> 2.
    // - FLIP_MAP[2] -> 34 (symmetrical 180-degree flipped position).
    // - 1ULL << 34 creates a mask with bit 34 set.
    // - |= sets this flipped bit in the opponent's mask (swapping players on flip).
    flipped.opp |= (1ULL << FLIP_MAP[std::countr_zero(bits)]);
  }

  // Symmetrically map state.opp set bits to flipped.me.
  for (u64 bits = state.opp; bits != 0U; bits &= bits - 1) {
    flipped.me |= (1ULL << FLIP_MAP[std::countr_zero(bits)]);
  }

  flipped.history = state.history;
  flipped.historyCount = state.historyCount;

  flipped.hash = kribu::zobrist::compute_hash(
      {.activePlayer = flipped.me, .opponentPlayer = flipped.opp, .activeCaptureIdx = flipped.activeCaptureIdx});
  return flipped;
}

/**
 * @brief Applies a move to the board state and returns the resulting board state.
 * @note  No validity checks are performed. The caller must verify legitimacy beforehand.
 * @param state  Current BoardState.
 * @param moveId A valid move ID.
 * @return Resulting BoardState.
 */
[[nodiscard]] constexpr boardState apply_move(const boardState& state, int moveId) noexcept {
  boardState next = state;

  if (moveId == END_CHAIN_MOVE) {
    if (state.activeCaptureIdx != -1) {
      next.hash ^= kribu::zobrist::KEYS.activeCapture[state.activeCaptureIdx];
      next.hash ^= kribu::zobrist::KEYS.activeCapture[37];
    }
    next.activeCaptureIdx = -1;
    next.historyCount = 0;
    return next;
  }

  const move& mov = MOVE_TABLE[moveId];
  next.me &= ~(1ULL << mov.from);
  next.me |= (1ULL << mov.to);

  // Update hash for moving active piece
  next.hash ^= kribu::zobrist::KEYS.me[mov.from];
  next.hash ^= kribu::zobrist::KEYS.me[mov.to];

  if (is_capture_move(moveId)) {
    next.opp &= ~(1ULL << mov.captured);
    next.hash ^= kribu::zobrist::KEYS.opp[mov.captured];

    i8 nextCaptureIdx = mov.to;

    int oldCap = (state.activeCaptureIdx == -1) ? 37 : state.activeCaptureIdx;
    int newCap = (nextCaptureIdx == -1) ? 37 : nextCaptureIdx;
    next.hash ^= kribu::zobrist::KEYS.activeCapture[oldCap];
    next.hash ^= kribu::zobrist::KEYS.activeCapture[newCap];

    next.activeCaptureIdx = nextCaptureIdx;
    next.historyCount = 0;
  } else {
    int oldCap = (state.activeCaptureIdx == -1) ? 37 : state.activeCaptureIdx;
    next.hash ^= kribu::zobrist::KEYS.activeCapture[oldCap];
    next.hash ^= kribu::zobrist::KEYS.activeCapture[37];

    next.activeCaptureIdx = -1;

    if (next.historyCount < MAX_HISTORY_LIMIT) {
      next.history[next.historyCount] = state.hash;
      next.historyCount++;
    } else {
      for (int i = 0; i < MAX_HISTORY_LIMIT - 1; ++i) {
        next.history[i] = next.history[i + 1];
      }
      next.history[MAX_HISTORY_LIMIT - 1] = state.hash;
    }
  }

  return next;
}

/**
 * @brief Computes the Zobrist hash of a flipped board state directly without full board construction.
 * @param state Current BoardState.
 * @return Flipped board state hash.
 */
[[nodiscard]] constexpr u64 compute_flipped_hash(const boardState& state) noexcept {
  u64 hash = 0;
  for (u64 bits = state.me; bits != 0U; bits &= bits - 1) {
    hash ^= kribu::zobrist::KEYS.opp[36 - std::countr_zero(bits)];
  }
  for (u64 bits = state.opp; bits != 0U; bits &= bits - 1) {
    hash ^= kribu::zobrist::KEYS.me[36 - std::countr_zero(bits)];
  }
  int capIdx = (state.activeCaptureIdx == -1) ? 37 : (36 - state.activeCaptureIdx);
  hash ^= kribu::zobrist::KEYS.activeCapture[capIdx];
  return hash;
}

/**
 * @brief Checks if a move results in a board state that violates threefold repetition rules.
 * @param state  Current BoardState.
 * @param moveId Move ID to check.
 * @return True if the move is repetition-legal, false otherwise.
 */
[[nodiscard]] constexpr bool is_repetition_legal(const boardState& state, int moveId) noexcept {
  boardState next = apply_move(state, moveId);
  u64 nextHash = 0;
  if (next.activeCaptureIdx == -1) {
    nextHash = compute_flipped_hash(next);
  } else {
    nextHash = next.hash;
  }
  int repetitions = 0;
  for (u32 i = 0; i < state.historyCount; ++i) {
    if (state.history[i] == nextHash) {
      repetitions++;
    }
  }
  return repetitions < 2;
}

/**
 * @brief Validates if a specific move ID is legal in the given board state.
 * @details Checks starting-piece ownership, target occupancy, jump-capture rules, and repetition rules.
 *          If a capture chain is active, it enforces that only the capturing piece can move.
 * @param state  Current BoardState.
 * @param moveId Move ID to validate.
 * @return True if the move is legal, false otherwise.
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] constexpr bool is_valid(const boardState& state, int moveId) noexcept {
  if (moveId < 0 || moveId >= TOTAL_MOVE_COUNT) {
    return false;
  }

  bool basicValid = false;
  if (state.activeCaptureIdx != -1) {
    if (moveId == END_CHAIN_MOVE) {
      basicValid = true;
    } else if (!is_capture_move(moveId)) {
      basicValid = false;
    } else {
      const move& mov = MOVE_TABLE[moveId];
      if (mov.from != state.activeCaptureIdx || (((state.me | state.opp) >> mov.to) & 1U) != 0U) {
        basicValid = false;
      } else {
        basicValid = ((state.opp >> mov.captured) & 1U) != 0U;
      }
    }
  } else {
    if (moveId == END_CHAIN_MOVE) {
      basicValid = false;
    } else {
      const move& mov = MOVE_TABLE[moveId];
      if (((state.me >> mov.from) & 1U) == 0U || (((state.me | state.opp) >> mov.to) & 1U) != 0U) {
        basicValid = false;
      } else if (is_simple_move(moveId)) {
        basicValid = true;
      } else {
        basicValid = ((state.opp >> mov.captured) & 1U) != 0U;
      }
    }
  }

  if (!basicValid) {
    return false;
  }

  return is_repetition_legal(state, moveId);
}

/**
 * @brief Checks if a further capture is possible from the given node index.
 * @details Examines all static capture moves starting from `fromNode` and checks if the destination is empty
 *          and the jumped-over node contains an opponent piece.
 * @param next     Board state after a capture.
 * @param fromNode Node where the capturing piece now resides.
 * @return True if at least one further capture is legal, false otherwise.
 */
[[nodiscard]] constexpr bool can_continue_capturing(const boardState& next, i8 fromNode) noexcept {
  // Bitmask of all occupied nodes on the board.
  const u64 occupied = next.me | next.opp;
  // To prevent the bugprone-signed-char-misuse lint error when converting signed i8 to int,
  // we cast it to unsigned u8 first, and then to int.
  const int captureMoveCount = static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[fromNode]));
  for (int k = 0; k < captureMoveCount; ++k) {
    const move& mov = MOVE_TABLE[BOARD_METADATA.captureMoveIdxByNode[fromNode][k]];
    // bitwise check: ((occupied >> mov.to) & 1U) == 0U verifies the destination node is empty.
    // bitwise check: ((next.opp >> mov.captured) & 1U) != 0U verifies the opponent piece is at the captured node.
    if (((occupied >> mov.to) & 1U) == 0U && ((next.opp >> mov.captured) & 1U) != 0U) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Generates all valid move IDs for the active player.
 * @details Evaluates all potential moves and returns a list of legal move IDs.
 *          If a capture chain is active, it only evaluates captures originating from the locked piece.
 * @param state Current BoardState.
 * @return Stack-allocated MoveList of valid move IDs.
 */
[[nodiscard]] constexpr MoveList all_possible_moves(const boardState& state) noexcept {
  MoveList list;

  if (state.activeCaptureIdx != -1) {
    list.push(static_cast<i16>(END_CHAIN_MOVE));
    const int captureMoveCount =
        static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[state.activeCaptureIdx]));
    for (int k = 0; k < captureMoveCount; ++k) {
      const i16 moveId = BOARD_METADATA.captureMoveIdxByNode[state.activeCaptureIdx][k];
      if (is_valid(state, moveId)) {
        list.push(moveId);
      }
    }
    return list;
  }

  for (int i = 1; i < TOTAL_MOVE_COUNT; ++i) {
    if (is_valid(state, i)) {
      list.push(static_cast<i16>(i));
    }
  }
  return list;
}

/**
 * @brief Evaluates and returns the current game status.
 * @details Detects win/loss based on piece counts or stalemate (no moves available).
 * @param state Current BoardState.
 * @return GameStatus::ME_WINS, GameStatus::OPP_WINS, or GameStatus::ONGOING.
 */
[[nodiscard]] constexpr GameStatus get_game_status(const boardState& state) noexcept {
  if (state.historyCount >= MAX_HISTORY_LIMIT) {
    return GameStatus::DRAW_PROGRESS_RULE;
  }

  if (piece_count(state.opp) == 0) {
    return GameStatus::ME_WINS_ELIMINATION;
  }
  if (piece_count(state.me) == 0) {
    return GameStatus::OPP_WINS_ELIMINATION;
  }

  if (state.activeCaptureIdx != -1) {
    return GameStatus::ONGOING;
  }

  if (all_possible_moves(state).empty()) {
    return GameStatus::OPP_WINS_STALEMATE;
  }
  if (all_possible_moves(flip_board(state)).empty()) {
    return GameStatus::ME_WINS_STALEMATE;
  }
  return GameStatus::ONGOING;
}

}  // namespace kribu::sholoGuti
