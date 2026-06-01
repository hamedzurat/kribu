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

/**
 * @brief Compile-time defined array of benchmark matchups.
 *
 * @details Two data-generation categories:
 *
 *  POLICY matchups (~12,800 games)
 *    Strong-player moves are the training signal for the policy network.
 *
 *    Structure:
 *      - Deterministic normal-vs-normal grid is included once per ordered pair
 *        because these games are reproducible and do not need repeated sampling.
 *      - Most policy diversity comes from Mad variants, where injected randomness
 *        produces different trajectories from otherwise strong players.
 *      - Full Mad-vs-Mad coverage is intentionally avoided because it is too
 *        expensive; only a small low-cost subset is included for extra stochastic
 *        cross-play diversity.
 *
 *    Post-processing:
 *      - Filter by playerPlayed.
 *      - Exclude moves made by "MadPlayer".
 *      - Keep only moves made by the underlying strong policy.
 *
 *  VALUE matchups (~5,120 games)
 *    Diverse skill gaps and outcome distributions for value-network training.
 *
 *    Structure:
 *      - Random/Greedy self-play provides weak and noisy baseline outcomes.
 *      - Random/Greedy vs strong players provides clear win/loss labels.
 *      - Both player orders are included to reduce first-player/side bias.
 *
 *    Post-processing:
 *      - All turns from all games are usable.
 *      - Label is derived from the final game outcome.
 *
 *  Target: ~17,920 games total.
 *    Expected turn count: ~1.8M to 3.6M turns total (assuming games generally run 100-200 turns).
 *    Note: maxTurns is set to 1024 to properly detect and resolve draws/repetition.
 *    Minimax8 is oversampled relative to MCTS800 (4:1 game ratio) because MCTS800 is much slower.
 */
inline constexpr std::array BENCHMARK_MATCHUPS = {
    // clang-format off
    // ── POLICY matchups ──────────────────────────────────────────────────────
    // zero randomness
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "Minimax8",           .games = 1,     .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "MCTS800",            .games = 1,     .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800",        .player2Name = "Minimax8",           .games = 1,     .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800",        .player2Name = "MCTS800",            .games = 1,     .maxTurns = 1024},

    // Same Mad vs Mad
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "Minimax8_Mad2",    .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad1",    .player2Name = "Minimax8_Mad1",    .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "MCTS800_Mad2",     .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800_Mad1",     .player2Name = "MCTS800_Mad1",     .games = 256,   .maxTurns = 1024},

    // Normal vs same Mad
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "Minimax8_Mad2",    .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "Minimax8",         .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "Minimax8_Mad1",    .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad1",    .player2Name = "Minimax8",         .games = 1024,  .maxTurns = 1024},

    MatchConfig{.player1Name = "MCTS800",          .player2Name = "MCTS800_Mad2",     .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "MCTS800",          .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800",          .player2Name = "MCTS800_Mad1",     .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800_Mad1",     .player2Name = "MCTS800",          .games = 256,   .maxTurns = 1024},

    // Normal vs other Mad
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "MCTS800_Mad2",     .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "Minimax8",         .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "MCTS800_Mad1",     .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800_Mad1",     .player2Name = "Minimax8",         .games = 256,   .maxTurns = 1024},
  
    MatchConfig{.player1Name = "MCTS800",          .player2Name = "Minimax8_Mad2",    .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "MCTS800",          .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800",          .player2Name = "Minimax8_Mad1",    .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad1",    .player2Name = "MCTS800",          .games = 256,   .maxTurns = 1024},

    // Low-cost sampled Mad vs Mad
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "Minimax8_Mad1",    .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad1",    .player2Name = "Minimax8_Mad2",    .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "MCTS800_Mad2",     .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "Minimax8_Mad2",    .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800_Mad1",     .player2Name = "Minimax8_Mad2",    .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8_Mad1",    .player2Name = "MCTS800_Mad2",     .games = 256,   .maxTurns = 1024},
 
    // ── VALUE matchups ───────────────────────────────────────────────────────
    //    Diverse skill gaps produce diverse outcome labels for value training.
    MatchConfig{.player1Name = "Minimax8",      .player2Name = "RandomPlayer",  .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "RandomPlayer",  .player2Name = "Minimax8",      .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "Minimax8",      .player2Name = "GreedyPlayer",  .games = 1024,  .maxTurns = 1024},
    MatchConfig{.player1Name = "GreedyPlayer",  .player2Name = "Minimax8",      .games = 1024,  .maxTurns = 1024},

    MatchConfig{.player1Name = "MCTS800",       .player2Name = "RandomPlayer",  .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "RandomPlayer",  .player2Name = "MCTS800",       .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "MCTS800",       .player2Name = "GreedyPlayer",  .games = 256,   .maxTurns = 1024},
    MatchConfig{.player1Name = "GreedyPlayer",  .player2Name = "MCTS800",       .games = 256,   .maxTurns = 1024},
    // clang-format on
};

}  // namespace kribu::benchmark
