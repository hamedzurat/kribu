/**
 * @file config.cpp
 * @brief Implementations of compile-time configurations for Sholo Guti benchmarking.
 */

#include "config.hpp"

#include <array>
#include <span>

#include "kribu/benchmark.hpp"
#include "kribu/player/player.hpp"

namespace kribu::benchmark {

namespace {

/**
 * @brief Compile-time defined array of benchmark players.
 */
constexpr std::array<Player, 4> BENCHMARK_PLAYERS = {
    Player{.name = "RandomPlayer", .select = kribu::player::select_random_player},
    Player{.name = "GreedyPlayer", .select = kribu::player::select_greedy},
    Player{.name = "MinimaxPieceCount_Depth4", .select = kribu::player::select_minimax_piece_d4},
    Player{.name = "MinimaxPosition_Depth4", .select = kribu::player::select_minimax_position_d4},
};

/**
 * @brief Compile-time defined array of benchmark matchups.
 */
constexpr std::array<MatchConfig, 6> BENCHMARK_MATCHUPS = {
    MatchConfig{.player1Name = "RandomPlayer", .player2Name = "GreedyPlayer", .games = 2, .maxTurns = 5000},
    MatchConfig{.player1Name = "RandomPlayer", .player2Name = "MinimaxPieceCount_Depth4", .games = 2, .maxTurns = 5000},
    MatchConfig{.player1Name = "RandomPlayer", .player2Name = "MinimaxPosition_Depth4", .games = 2, .maxTurns = 5000},
    MatchConfig{.player1Name = "GreedyPlayer", .player2Name = "MinimaxPieceCount_Depth4", .games = 2, .maxTurns = 5000},
    MatchConfig{.player1Name = "GreedyPlayer", .player2Name = "MinimaxPosition_Depth4", .games = 2, .maxTurns = 5000},
    MatchConfig{.player1Name = "MinimaxPieceCount_Depth4",
                .player2Name = "MinimaxPosition_Depth4",
                .games = 2,
                .maxTurns = 5000},
};

}  // namespace

std::span<const Player> get_benchmark_players() {
  return BENCHMARK_PLAYERS;
}

std::span<const MatchConfig> get_benchmark_matchups() {
  return BENCHMARK_MATCHUPS;
}

}  // namespace kribu::benchmark
