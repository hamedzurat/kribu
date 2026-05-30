/**
 * @file players.hpp
 * @brief Centralized player configurations and factory instantiations for Sholo Guti benchmark.
 */

#pragma once

#include "benchmark.hpp"
#include "kribu/player/greedy.hpp"
#include "kribu/player/mcts.hpp"
#include "kribu/player/minimax.hpp"
#include "kribu/player/random.hpp"

namespace kribu::player {

using namespace kribu::benchmark;

/**
 * @brief Compile-time defined array of benchmark players.
 * @details Players are defined function pointers.
 */
inline constexpr std::array BENCHMARK_PLAYERS = {
    // clang-format off
    // ── Weak baselines ────────────────────────────────────────────────────────────────
    Player{.type = "random", .name = "RandomPlayer",       .select = select_random},
    Player{.type = "greedy", .name = "GreedyPlayer",       .select = greedy_player_maker},
    Player{.type = "greedy", .name = "GreedyPlayer_Mad10", .select = greedy_player_maker, .madness = 10},

    // ── Minimax ───────────────────────────────────────────────────────────────────────
    Player{.type = "minimax", .name = "Minimax8",       .select = minimax_player_maker<8>, .depth = 8},
    Player{.type = "minimax", .name = "Minimax8_Mad2",  .select = minimax_player_maker<8>, .depth = 8, .madness = 2},
    Player{.type = "minimax", .name = "Minimax8_Mad10", .select = minimax_player_maker<8>, .depth = 8, .madness = 10},
    
    Player{.type = "minimax", .name = "Minimax12",       .select = minimax_player_maker<12>, .depth = 12},
    Player{.type = "minimax", .name = "Minimax12_Mad2",  .select = minimax_player_maker<12>, .depth = 12, .madness = 2},
    Player{.type = "minimax", .name = "Minimax12_Mad10", .select = minimax_player_maker<12>, .depth = 12, .madness = 10},
    
    // Player{.type = "minimax", .name = "Minimax16",       .select = minimax_player_maker<16>, .depth = 16},
    // Player{.type = "minimax", .name = "Minimax16_Mad2",  .select = minimax_player_maker<16>, .depth = 16, .madness = 2},
    // Player{.type = "minimax", .name = "Minimax16_Mad10", .select = minimax_player_maker<16>, .depth = 16, .madness = 10},

    // ── MCTS ──────────────────────────────────────────────────────────────────────────
    Player{.type = "mcts", .name = "MCTS800",       .select = mcts_player_maker<800>, .depth = 800},
    Player{.type = "mcts", .name = "MCTS800_Mad2",  .select = mcts_player_maker<800>, .depth = 800, .madness = 2},
    Player{.type = "mcts", .name = "MCTS800_Mad10", .select = mcts_player_maker<800>, .depth = 800, .madness = 10},

    Player{.type = "mcts", .name = "MCTS1200",       .select = mcts_player_maker<1200>, .depth = 1200},
    Player{.type = "mcts", .name = "MCTS1200_Mad2",  .select = mcts_player_maker<1200>, .depth = 1200, .madness = 2},
    Player{.type = "mcts", .name = "MCTS1200_Mad10", .select = mcts_player_maker<1200>, .depth = 1200, .madness = 10},
    // clang-format on
};

}  // namespace kribu::player
