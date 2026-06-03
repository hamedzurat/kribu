/**
 * @file config.hpp
 * @brief Header for compile-time benchmark tournament configs.
 */

#pragma once

#include <array>
#include <string_view>

namespace kribu::benchmark {

struct MatchConfig {
  /**
   * @brief Name of Player 1.
   */
  std::string_view player1Name;

  /**
   * @brief Name of Player 2.
   */
  std::string_view player2Name;

  /**
   * @brief Total number of games to play in this matchup.
   */
  int games;

  /**
   * @brief Maximum turns (plies) allowed per game.
   */
  int maxTurns;
};

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

/**
 * @brief Number of worker threads for parallel game execution.
 */
inline constexpr int THREAD_COUNT = 32;

// clang-format off
/**
 * @brief Compile-time defined array of benchmark matchups.
 *
 * ### Matchup Contributions
 *
 * - **POLICY Matchups** (Minimax/MCTS vs Minimax/MCTS):
 *   - **policy_data**: Yes (All non-exploratory turns)
 *   - **value_data**: Yes (All non-exploratory turns)
 *
 * - **VALUE Matchups** (Minimax/MCTS vs Random/Greedy):
 *   - **policy_data**: No (Filtered out because one of the player types is random/greedy)
 *   - **value_data**: Yes, but only on the expert's turn (Preserves turns where the Minimax/MCTS player was the active player; discards turns where the Random/Greedy player was active)
 */
inline constexpr std::array BENCHMARK_MATCHUPS = {
    // ── POLICY matchups ──────────────────────────────────────────────────────
    // Deterministic search-vs-search pairings are coverage seeds, not bulk generators.
    // Keep one explicit game per starting-side configuration and spend the bulk of the
    // benchmark budget on stochastic or weaker-opponent matchups that broaden coverage.
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "Minimax8",       .games = 1,     .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "Minimax4",       .games = 1,     .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax4",       .player2Name = "Minimax8",       .games = 1,     .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "Minimax8_Mad1",  .games = 4096,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad1",  .player2Name = "Minimax8",       .games = 4096,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "Minimax8_Mad2",  .games = 4096,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad2",  .player2Name = "Minimax8",       .games = 4096,  .maxTurns = 1024},

    // ── VALUE matchups ───────────────────────────────────────────────────────
    // Only keep matchups that can label Minimax8 positions directly.
    MatchConfig{.player1Name = "Minimax8",      .player2Name = "RandomPlayer",  .games = 2048,  .maxTurns = 1024},
    MatchConfig{.player1Name = "RandomPlayer",  .player2Name = "Minimax8",      .games = 2048,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8",      .player2Name = "GreedyPlayer",  .games = 2048,  .maxTurns = 1024},
    MatchConfig{.player1Name = "GreedyPlayer",  .player2Name = "Minimax8",      .games = 2048,  .maxTurns = 1024},
    // clang-format on
};

}  // namespace kribu::benchmark
