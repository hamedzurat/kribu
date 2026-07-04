/**
 * @file fast_rng.hpp
 * @brief Extremely fast pseudo-random number generator using Xorshift64.
 * @details This file defines FastRng, a 64-bit pseudo-random number generator
 *          and a global thread-local instance of it.
 */

#pragma once

#include <random>

#include "types.hpp"

/**
 * @class FastRng
 * @brief Extremely fast 64-bit pseudo-random number generator using Xorshift64.
 * @details This generator has a period of 2^64 - 1, a tiny 8-byte state, and meets
 *          the UniformRandomBitGenerator requirements for C++ distributions.
 */
class FastRng {
 public:
  /**
   * @brief The type of the generated unsigned integers.
   */
  using result_type = u64;

  /**
   * @brief Returns the minimum possible value generated.
   * @return 0.
   */
  static constexpr u64 min() noexcept { return 0; }

  /**
   * @brief Returns the maximum possible value generated.
   * @return UINT64_MAX.
   */
  static constexpr u64 max() noexcept { return ~static_cast<u64>(0); }

  /**
   * @brief Default constructor initializing the generator with a default seed.
   */
  constexpr FastRng() noexcept : state_(88172645463325252ULL) {}

  /**
   * @brief Constructor initializing the generator with a custom seed.
   * @param seed The initialization seed (if 0, a default seed is used).
   */
  explicit constexpr FastRng(u64 seed) noexcept : state_(seed == 0 ? 88172645463325252ULL : seed) {}

  /**
   * @brief Generates the next pseudo-random number.
   * @return The generated 64-bit unsigned integer.
   */
  u64 operator()() noexcept {
    u64 nextState = state_;
    nextState ^= nextState << 13;
    nextState ^= nextState >> 7;
    nextState ^= nextState << 17;
    state_ = nextState;
    return nextState;
  }

 private:
  /**
   * @brief The internal 64-bit state of the generator.
   */
  u64 state_;
};

/**
 * @brief Global thread-local random number generator seeded once per thread.
 */
inline thread_local FastRng rng{std::random_device{}()};

/**
 * @brief Returns a random index in the range [0, upperBound).
 *
 * @param upperBound Exclusive upper bound.
 * @return Random index, or 0 if upperBound <= 1.
 *
 * @details
 * Uses fast multiply-high reduction on compilers that support unsigned __int128.
 * This is faster than std::uniform_int_distribution and good enough for rollout
 * move selection.
 */
[[nodiscard]] inline int random_index(int upperBound) noexcept {
  if (upperBound <= 1) {
    return 0;
  }

  const u64 boundValue = static_cast<u64>(upperBound);

#ifdef __SIZEOF_INT128__
  const auto randomProduct = static_cast<unsigned __int128>(rng()) * static_cast<unsigned __int128>(boundValue);
  return static_cast<int>(randomProduct >> 64);
#else
  return static_cast<int>(rng() % boundValue);
#endif
}

/**
 * @brief Returns a random float in the range [0.0, 1.0).
 *
 * @return Random floating-point value.
 *
 * @details
 * Uses the top 24 bits of the random value, matching the precision of a float
 * mantissa. This avoids std::uniform_real_distribution in hot rollout paths.
 */
[[nodiscard]] inline f32 random_unit_float() noexcept {
  constexpr f32 inverseScale = 1.0F / 16777216.0F;
  return static_cast<f32>(rng() >> 40) * inverseScale;
}

/**
 * @brief Returns true with the given probability.
 *
 * @param probability Probability in the range [0.0, 1.0].
 * @return True with approximately the requested probability.
 */
[[nodiscard]] inline bool random_chance(f32 probability) noexcept {
  if (probability <= 0.0F) {
    return false;
  }

  if (probability >= 1.0F) {
    return true;
  }

  return random_unit_float() < probability;
}