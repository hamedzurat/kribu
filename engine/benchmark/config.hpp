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
 *  POLICY matchups (~45,600 games)
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
 *      - Exclude moves made by "ForcedRandom" / "MadPlayer".
 *      - Keep only moves made by the underlying strong policy.
 *
 *  VALUE matchups (~35,800 games)
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
 *  Target: ~81,400 games total.
 *    Runtime and generated turn count depend on average game length.
 *    Minimax8 and MCTS800 are intentionally oversampled because they are faster.
 */
inline constexpr std::array BENCHMARK_MATCHUPS = {
    // clang-format off
    // ── POLICY matchups ──────────────────────────────────────────────────────
    // zero randomness
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "Minimax8",       .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "Minimax12",      .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "MCTS800",        .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8",       .player2Name = "MCTS1200",       .games = 1,    .maxTurns = 2048},
    
    MatchConfig{.player1Name = "Minimax12",      .player2Name = "Minimax8",       .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12",      .player2Name = "Minimax12",      .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12",      .player2Name = "MCTS800",        .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12",      .player2Name = "MCTS1200",       .games = 1,    .maxTurns = 2048},

    MatchConfig{.player1Name = "MCTS800",        .player2Name = "Minimax8",       .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800",        .player2Name = "Minimax12",      .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800",        .player2Name = "MCTS800",        .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800",        .player2Name = "MCTS1200",       .games = 1,    .maxTurns = 2048},

    MatchConfig{.player1Name = "MCTS1200",       .player2Name = "Minimax8",       .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS1200",       .player2Name = "Minimax12",      .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS1200",       .player2Name = "MCTS800",        .games = 1,    .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS1200",       .player2Name = "MCTS1200",       .games = 1,    .maxTurns = 2048},

    // Madness variants

    // Same Mad vs Mad
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "Minimax8_Mad2",    .games = 8192,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad10",   .player2Name = "Minimax8_Mad10",   .games = 8192,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12_Mad2",   .player2Name = "Minimax12_Mad2",   .games = 1024,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12_Mad10",  .player2Name = "Minimax12_Mad10",  .games = 1024,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "MCTS800_Mad2",     .games =  512,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800_Mad10",    .player2Name = "MCTS800_Mad10",    .games =  256,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS1200_Mad2",    .player2Name = "MCTS1200_Mad2",    .games =  256,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS1200_Mad10",   .player2Name = "MCTS1200_Mad10",   .games =  128,  .maxTurns = 2048},

    // Normal vs same Mad
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "Minimax8_Mad2",    .games = 8192,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "Minimax8",         .games = 8192,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "Minimax8_Mad10",   .games = 8192,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad10",   .player2Name = "Minimax8",         .games = 8192,  .maxTurns = 2048},

    MatchConfig{.player1Name = "Minimax12",        .player2Name = "Minimax12_Mad2",   .games = 2048,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12_Mad2",   .player2Name = "Minimax12",        .games = 2048,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12",        .player2Name = "Minimax12_Mad10",  .games =  256,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12_Mad10",  .player2Name = "Minimax12",        .games =  256,  .maxTurns = 2048},

    MatchConfig{.player1Name = "MCTS800",          .player2Name = "MCTS800_Mad2",     .games =  512/8,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "MCTS800",          .games =  512/8,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800",          .player2Name = "MCTS800_Mad10",    .games =  128/8,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800_Mad10",    .player2Name = "MCTS800",          .games =  128/8,  .maxTurns = 2048},

    // MatchConfig{.player1Name = "MCTS1200",         .player2Name = "MCTS1200_Mad2",    .games =  128/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200_Mad2",    .player2Name = "MCTS1200",         .games =  128/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200",         .player2Name = "MCTS1200_Mad10",   .games =   64/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200_Mad10",   .player2Name = "MCTS1200",         .games =   64/8,  .maxTurns = 2048},

    // Normal vs other Mad
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "Minimax12_Mad2",   .games = 1024,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12_Mad2",   .player2Name = "Minimax8",         .games = 1024,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "Minimax12_Mad10",  .games =  512,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12_Mad10",  .player2Name = "Minimax8",         .games =  512,  .maxTurns = 2048},

    MatchConfig{.player1Name = "Minimax8",         .player2Name = "MCTS800_Mad2",     .games =  512,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "Minimax8",         .games =  512,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "MCTS1200_Mad2",    .games =  512,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS1200_Mad2",    .player2Name = "Minimax8",         .games =  512,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8",         .player2Name = "MCTS800_Mad10",    .games =  128,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800_Mad10",    .player2Name = "Minimax8",         .games =  128,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "Minimax8",         .player2Name = "MCTS1200_Mad10",   .games =  128,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200_Mad10",   .player2Name = "Minimax8",         .games =  128,  .maxTurns = 2048},

    MatchConfig{.player1Name = "Minimax12",        .player2Name = "Minimax8_Mad2",    .games = 1024,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "Minimax12",        .games = 1024,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12",        .player2Name = "Minimax8_Mad10",   .games = 1024,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad10",   .player2Name = "Minimax12",        .games = 1024,  .maxTurns = 2048},

    MatchConfig{.player1Name = "Minimax12",        .player2Name = "MCTS800_Mad2",     .games =  512/8,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "Minimax12",        .games =  512/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "Minimax12",        .player2Name = "MCTS1200_Mad2",    .games =  512/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200_Mad2",    .player2Name = "Minimax12",        .games =  512/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "Minimax12",        .player2Name = "MCTS800_Mad10",    .games =  256/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS800_Mad10",    .player2Name = "Minimax12",        .games =  256/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "Minimax12",        .player2Name = "MCTS1200_Mad10",   .games =  256/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200_Mad10",   .player2Name = "Minimax12",        .games =  256/8,  .maxTurns = 2048},

    MatchConfig{.player1Name = "MCTS800",          .player2Name = "MCTS1200_Mad2",    .games =  128/8,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS1200_Mad2",    .player2Name = "MCTS800",          .games =  128/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS800",          .player2Name = "MCTS1200_Mad10",   .games =   64/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200_Mad10",   .player2Name = "MCTS800",          .games =   64/8,  .maxTurns = 2048},
    
    MatchConfig{.player1Name = "MCTS800",          .player2Name = "Minimax8_Mad2",    .games =  512,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "MCTS800",          .games =  512,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800",          .player2Name = "Minimax12_Mad2",   .games =  512/8,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12_Mad2",   .player2Name = "MCTS800",          .games =  512/8,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800",          .player2Name = "Minimax8_Mad10",   .games =  256,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad10",   .player2Name = "MCTS800",          .games =  256,  .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800",          .player2Name = "Minimax12_Mad10",  .games =  256/8,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12_Mad10",  .player2Name = "MCTS800",          .games =  256/8,  .maxTurns = 2048},

    // MatchConfig{.player1Name = "MCTS1200",         .player2Name = "MCTS800_Mad2",     .games =   64/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "MCTS1200",         .games =   64/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200",         .player2Name = "MCTS800_Mad10",    .games =   32/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS800_Mad10",    .player2Name = "MCTS1200",         .games =   32/8,  .maxTurns = 2048},

    MatchConfig{.player1Name = "MCTS1200",         .player2Name = "Minimax8_Mad2",    .games =  512/8,  .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "MCTS1200",         .games =  512/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200",         .player2Name = "Minimax12_Mad2",   .games =  512/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "Minimax12_Mad2",   .player2Name = "MCTS1200",         .games =  512/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200",         .player2Name = "Minimax8_Mad10",   .games =  256,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "Minimax8_Mad10",   .player2Name = "MCTS1200",         .games =  256,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "MCTS1200",         .player2Name = "Minimax12_Mad10",  .games =  256/8,  .maxTurns = 2048},
    // MatchConfig{.player1Name = "Minimax12_Mad10",  .player2Name = "MCTS1200",         .games =  256/8,  .maxTurns = 2048},

    // Low-cost sampled Mad vs Mad
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "MCTS800_Mad2",     .games =  1024/4, .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800_Mad2",     .player2Name = "Minimax8_Mad2",    .games =  1024/4, .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad2",    .player2Name = "Minimax8_Mad10",   .games =  1024, .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad10",   .player2Name = "Minimax8_Mad2",    .games =  1024, .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800_Mad10",    .player2Name = "Minimax8_Mad2",    .games =  1024/4, .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8_Mad10",   .player2Name = "MCTS800_Mad2",     .games =  1024/4, .maxTurns = 2048},
 
    // ── VALUE matchups ───────────────────────────────────────────────────────
    //    Diverse skill gaps produce diverse outcome labels for value training.
    MatchConfig{.player1Name = "RandomPlayer",  .player2Name = "RandomPlayer",  .games =  512, .maxTurns = 4096},
    MatchConfig{.player1Name = "GreedyPlayer",  .player2Name = "GreedyPlayer",  .games =  512, .maxTurns = 4096},

    MatchConfig{.player1Name = "GreedyPlayer",  .player2Name = "RandomPlayer",  .games = 1024, .maxTurns = 2048},
    MatchConfig{.player1Name = "RandomPlayer",  .player2Name = "GreedyPlayer",  .games = 1024, .maxTurns = 2048},

    MatchConfig{.player1Name = "Minimax8",      .player2Name = "RandomPlayer",  .games = 4096, .maxTurns = 2048},
    MatchConfig{.player1Name = "RandomPlayer",  .player2Name = "Minimax8",      .games = 4096, .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax8",      .player2Name = "GreedyPlayer",  .games = 4096, .maxTurns = 2048},
    MatchConfig{.player1Name = "GreedyPlayer",  .player2Name = "Minimax8",      .games = 4096, .maxTurns = 2048},

    MatchConfig{.player1Name = "Minimax12",     .player2Name = "RandomPlayer",  .games = 1024, .maxTurns = 2048},
    MatchConfig{.player1Name = "RandomPlayer",  .player2Name = "Minimax12",     .games = 1024, .maxTurns = 2048},
    MatchConfig{.player1Name = "Minimax12",     .player2Name = "GreedyPlayer",  .games = 1024, .maxTurns = 2048},
    MatchConfig{.player1Name = "GreedyPlayer",  .player2Name = "Minimax12",     .games = 1024, .maxTurns = 2048},

    MatchConfig{.player1Name = "MCTS800",       .player2Name = "RandomPlayer",  .games =  512, .maxTurns = 2048},
    MatchConfig{.player1Name = "RandomPlayer",  .player2Name = "MCTS800",       .games =  512, .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS800",       .player2Name = "GreedyPlayer",  .games =  512, .maxTurns = 2048},
    MatchConfig{.player1Name = "GreedyPlayer",  .player2Name = "MCTS800",       .games =  512, .maxTurns = 2048},

    MatchConfig{.player1Name = "MCTS1200",      .player2Name = "RandomPlayer",  .games =   64, .maxTurns = 2048},
    MatchConfig{.player1Name = "RandomPlayer",  .player2Name = "MCTS1200",      .games =   64, .maxTurns = 2048},
    MatchConfig{.player1Name = "MCTS1200",      .player2Name = "GreedyPlayer",  .games =   64, .maxTurns = 2048},
    MatchConfig{.player1Name = "GreedyPlayer",  .player2Name = "MCTS1200",      .games =   64, .maxTurns = 2048},

    // clang-format on
};

}  // namespace kribu::benchmark
