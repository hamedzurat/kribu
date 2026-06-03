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
    Player{.type = "greedy", .name = "GreedyPlayer_Mad2",  .select = greedy_player_maker, .madness = 2},

    // ── Minimax ───────────────────────────────────────────────────────────────────────
    Player{.type = "minimax", .name = "Minimax4",       .select = minimax_player_maker<4>, .depth = 4},
    Player{.type = "minimax", .name = "Minimax8",       .select = minimax_player_maker<8>, .depth = 8},
    Player{.type = "minimax", .name = "Minimax8_Mad1",  .select = minimax_player_maker<8>, .depth = 8, .madness = 1},
    Player{.type = "minimax", .name = "Minimax8_Mad2",  .select = minimax_player_maker<8>, .depth = 8, .madness = 2},

    // ── MCTS ──────────────────────────────────────────────────────────────────────────
    Player{.type = "mcts", .name = "MCTS800",       .select = mcts_player_maker<800>, .depth = 800},
    Player{.type = "mcts", .name = "MCTS800_Mad1",  .select = mcts_player_maker<800>, .depth = 800, .madness = 1},
    Player{.type = "mcts", .name = "MCTS800_Mad2",  .select = mcts_player_maker<800>, .depth = 800, .madness = 2},
    // clang-format on
};

}  // namespace kribu::player
