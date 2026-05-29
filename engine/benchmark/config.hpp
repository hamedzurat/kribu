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
 * @brief Number of worker threads for parallel game execution.
 */
inline constexpr int THREAD_COUNT = 32;

/**
 * @brief Compile-time defined array of benchmark matchups.
 *
 * @details Two categories:
 *
 *  POLICY matchups (1000 games, maxTurns=1024)
 *    Strong-player moves are the training signal for the policy network.
 *    Filter by playerPlayed (exclude "ForcedRandom" / "MadPlayer") in post-processing.
 *
 *  VALUE matchups (200 games, maxTurns=512)
 *    Diverse skill gaps and outcome distributions for value-network training.
 *    All turns from all games are used; label is derived from the game outcome.
 */
inline constexpr std::array BENCHMARK_MATCHUPS = {
    // clang-format off

    // ── POLICY matchups (1000 games × maxTurns=1024) ─────────────────────────
    //    Only strong player moves are kept for policy training.
    MatchConfig{.player1Name = "MinimaxD8",         .player2Name = "MinimaxD8",           .games = 2, .maxTurns = 1024},
    MatchConfig{.player1Name = "MinimaxD8",         .player2Name = "MCTS_EpsGreedy1000",  .games = 2, .maxTurns = 1024},
    MatchConfig{.player1Name = "MinimaxD8",         .player2Name = "MCTS_EpsGreedy800",   .games = 2, .maxTurns = 1024},
    MatchConfig{.player1Name = "MinimaxD8",         .player2Name = "MCTS_Heuristic800",   .games = 2, .maxTurns = 1024},
    MatchConfig{.player1Name = "MinimaxD8",         .player2Name = "MinimaxD8_Mad2",      .games = 2, .maxTurns = 1024},

    // ── VALUE matchups (200 games × maxTurns=512) ────────────────────────────
    //    Diverse skill gaps produce diverse outcome labels for value training.
    MatchConfig{.player1Name = "MinimaxD8",         .player2Name = "RandomPlayer",        .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "MinimaxD8",         .player2Name = "GreedyPlayer",        .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "MinimaxD8",         .player2Name = "MinimaxD8_Mad5",      .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "MinimaxD8",         .player2Name = "MinimaxD8_Mad10",     .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "MCTS_EpsGreedy800", .player2Name = "RandomPlayer",        .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "MCTS_EpsGreedy800", .player2Name = "GreedyPlayer",        .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "MCTS_Random800",    .player2Name = "RandomPlayer",        .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "GreedyPlayer",      .player2Name = "RandomPlayer",        .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "GreedyPlayer_Mad30",.player2Name = "RandomPlayer",        .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "MinimaxD8_Mad2",    .player2Name = "MinimaxD8_Mad5",      .games = 2,  .maxTurns = 512},
    MatchConfig{.player1Name = "MinimaxD8_Mad5",    .player2Name = "GreedyPlayer",        .games = 2,  .maxTurns = 512},

    // clang-format on
};

}  // namespace kribu::benchmark
