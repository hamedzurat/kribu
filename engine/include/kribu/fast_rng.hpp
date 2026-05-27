/**
 * @file fast_rng.hpp
 * @brief Extremely fast pseudo-random number generator using Xorshift64.
 * @details This file defines FastRng, a 64-bit pseudo-random number generator
 *          and a global thread-local instance of it.
 */

#pragma once

#include <cstdint>
#include <random>

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
  using result_type = std::uint64_t;

  /**
   * @brief Returns the minimum possible value generated.
   * @return 0.
   */
  static constexpr result_type min() noexcept { return 0; }

  /**
   * @brief Returns the maximum possible value generated.
   * @return UINT64_MAX.
   */
  static constexpr result_type max() noexcept { return ~static_cast<result_type>(0); }

  /**
   * @brief Default constructor initializing the generator with a default seed.
   */
  constexpr FastRng() noexcept : state_(88172645463325252ULL) {}

  /**
   * @brief Constructor initializing the generator with a custom seed.
   * @param seed The initialization seed (if 0, a default seed is used).
   */
  explicit constexpr FastRng(std::uint64_t seed) noexcept : state_(seed == 0 ? 88172645463325252ULL : seed) {}

  /**
   * @brief Generates the next pseudo-random number.
   * @return The generated 64-bit unsigned integer.
   */
  result_type operator()() noexcept {
    std::uint64_t nextState = state_;
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
  std::uint64_t state_;
};

/**
 * @brief Global thread-local random number generator seeded once per thread.
 */
inline thread_local FastRng rng{std::random_device{}()};
