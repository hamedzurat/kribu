/**
 * @file config.hpp
 * @brief Header for compile-time benchmark tournament configs.
 */

#pragma once

#include <array>

#include "kribu/benchmark.hpp"

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

/**
 * @brief Returns the matchup pairs (by name) to simulate.
 */
inline constexpr int THREAD_COUNT = 32;

/**
 * @brief Compile-time defined array of benchmark matchups.
 */
inline constexpr std::array BENCHMARK_MATCHUPS = {
    // clang-format off
    MatchConfig{.player1Name = "RandomPlayer", .player2Name = "RandomPlayer", .games = 20, .maxTurns = 1000},
    MatchConfig{.player1Name = "RandomPlayer", .player2Name = "GreedyPlayer", .games = 20, .maxTurns = 1000},
    MatchConfig{.player1Name = "GreedyPlayer", .player2Name = "RandomPlayer", .games = 20, .maxTurns = 1000},
    MatchConfig{.player1Name = "GreedyPlayer", .player2Name = "GreedyPlayer", .games = 20000, .maxTurns = 10000},
    // MatchConfig{.player1Name = "MinimaxPieceCountDepth4", .player2Name = "RandomPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxPieceCountDepth4", .player2Name = "GreedyPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxPositionDepth4", .player2Name = "RandomPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxPositionDepth4", .player2Name = "GreedyPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxMixedDepth4", .player2Name = "RandomPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxMixedDepth4", .player2Name = "GreedyPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxMobilityDepth4", .player2Name = "RandomPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxMobilityDepth4", .player2Name = "GreedyPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MCTSPlayerRandomIter1000", .player2Name = "RandomPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MCTSPlayerRandomIter1000", .player2Name = "GreedyPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MCTSPlayerHeuristicIter1000", .player2Name = "RandomPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MCTSPlayerHeuristicIter1000", .player2Name = "GreedyPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxPositionDepth4Mad20", .player2Name = "RandomPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxPositionDepth4Mad20", .player2Name = "GreedyPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxPositionDepth4Mad50", .player2Name = "RandomPlayer", .games = 2, .maxTurns = 1000},
    // MatchConfig{.player1Name = "MinimaxPositionDepth4Mad50", .player2Name = "GreedyPlayer", .games = 2, .maxTurns = 1000},

    // clang-format on
};
}  // namespace kribu::benchmark
