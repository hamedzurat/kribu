/**
 * @file zobrist.hpp
 * @brief Compile-time Zobrist hashing keys and utility functions.
 */

#pragma once

#include <array>
#include <bit>

#include "kribu/types.hpp"

namespace kribu::zobrist {

/**
 * @brief Default seed for Zobrist random key generation.
 */
constexpr u64 ZOBRIST_SEED = 0x9e3779b97f4a7c15ULL;

/**
 * @brief Simple constexpr pseudo-random number generator (LCG-like / Xorshift).
 */
constexpr u64 next_random(u64& state) noexcept {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 0x2545F4914F6CDD1DULL;
}

/**
 * @struct ZobristKeys
 * @brief Group of random keys for each element of the board state.
 */
struct ZobristKeys {
  std::array<u64, 37> me{};
  std::array<u64, 37> opp{};
  std::array<u64, 38> activeCapture{};  // 0-36, and 37 for -1
};

/**
 * @brief Generates unique Zobrist keys at compile time.
 */
constexpr ZobristKeys generate_keys() noexcept {
  ZobristKeys keys{};
  u64 state = ZOBRIST_SEED;
  for (int i = 0; i < 37; ++i) {
    keys.me[i] = next_random(state);
    keys.opp[i] = next_random(state);
    keys.activeCapture[i] = next_random(state);
  }
  keys.activeCapture[37] = next_random(state);  // -1 sentinel
  return keys;
}

/**
 * @brief Static global Zobrist keys.
 */
constexpr ZobristKeys KEYS = generate_keys();

/**
 * @struct HashInput
 * @brief Input parameters for computing board Zobrist hash.
 */
struct HashInput {
  u64 activePlayer;
  u64 opponentPlayer;
  i8 activeCaptureIdx;
};

/**
 * @brief Computes the Zobrist hash from scratch for a given state.
 */
[[nodiscard]] constexpr u64 compute_hash(HashInput input) noexcept {
  u64 hash = 0;
  for (u64 bits = input.activePlayer; bits != 0U; bits &= bits - 1) {
    hash ^= KEYS.me[std::countr_zero(bits)];
  }
  for (u64 bits = input.opponentPlayer; bits != 0U; bits &= bits - 1) {
    hash ^= KEYS.opp[std::countr_zero(bits)];
  }
  int capIdx = (input.activeCaptureIdx == -1) ? 37 : input.activeCaptureIdx;
  hash ^= KEYS.activeCapture[capIdx];
  return hash;
}

}  // namespace kribu::zobrist
