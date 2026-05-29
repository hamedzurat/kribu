/**
 * @file config.hpp
 * @brief Header for compile-time benchmark tournament configs.
 */

#pragma once

#include <array>
#include <string_view>

namespace kribu::benchmark {

/**
 * @brief Configurable limit for threefold (or N-fold) repetition detection.
 * @details Set to 0 to disable repetition detection.
 */
inline constexpr int REPETITION_LIMIT = 4;

/**
 * @brief If true, repetitions are resolved by playing a random valid move instead of drawing.
 * If false, repetitions immediately trigger a game DRAW.
 */
inline constexpr bool ALLOW_REPETITION = true;

struct MatchConfig {
  std::string_view player1Name;
  std::string_view player2Name;
  int games;
  int maxTurns;
};

}  // namespace kribu::benchmark

// Architecture:                x86_64
//   CPU op-mode(s):            32-bit, 64-bit
//   Address sizes:             48 bits physical, 48 bits virtual
//   Byte Order:                Little Endian
// CPU(s):                      32
//   On-line CPU(s) list:       0-31
// Vendor ID:                   AuthenticAMD
//   Model name:                AMD Ryzen 9 9950X3D 16-Core Processor
//     Thread(s) per core:      2
//     Core(s) per socket:      16
//     CPU max MHz:             5752.0000
// Caches (sum of all):
//   L1d:                       768 KiB
//   L1i:                       512 KiB
//   L2:                        16 MiB
//   L3:                        128 MiB

namespace kribu::benchmark {

/**
 * @brief Returns the matchup pairs (by name) to simulate.
 */
inline constexpr int THREAD_COUNT = 32;

/**
 * @brief Compile-time defined array of benchmark matchups.
 */
inline constexpr std::array BENCHMARK_MATCHUPS = {
    // clang-format off
    MatchConfig{.player1Name = "RandomPlayer", .player2Name = "RandomPlayer", .games = 32, .maxTurns = 2048},
    // clang-format on
};
}  // namespace kribu::benchmark
