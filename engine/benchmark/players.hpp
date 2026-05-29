/**
 * @file players.hpp
 * @brief Centralized player configurations and factory instantiations for Sholo Guti benchmark.
 */

#pragma once

#include "benchmark.hpp"
#include "kribu/board.hpp"
#include "kribu/player/greedy.hpp"
#include "kribu/player/mcts.hpp"
#include "kribu/player/minimax.hpp"
#include "kribu/player/random.hpp"

namespace kribu::player {

using namespace kribu::benchmark;

/**
 * @brief Compile-time defined array of benchmark players.
 * @details Players are defined and named in place using stateless C++ lambdas, which decay to function pointers.
 *
 * Skill spectrum (weakest → strongest):
 *   RandomPlayer < GreedyPlayer < MadPlayers < MCTS variants < MinimaxD8
 *
 * mad% variants of MinimaxD8 inject random moves at the given probability, producing
 * a spectrum of near-perfect to imperfect play for value-network diversity.
 */
inline constexpr std::array BENCHMARK_PLAYERS = {
    // clang-format off

    // ── Weak baselines ──────────────────────────────────────────────────────
    Player{
        .name   = "RandomPlayer",
        .select = [](const boardState& state) { return select_random(state); },
    },
    Player{
        .name   = "GreedyPlayer",
        .select = [](const boardState& state) { return greedy_player_maker(state); },
    },

    // ── Strong player: pure Minimax Depth-8 ─────────────────────────────────
    Player{
        .name   = "MinimaxD8",
        .select = [](const boardState& state) { return minimax_player_maker<8>(state); },
    },

    // ── Minimax D8 with madness injections (for value-network diversity) ─────
    //    madness = % chance of replacing the selected move with a random one.
    Player{
        .name    = "MinimaxD8_Mad2",
        .select  = [](const boardState& state) { return minimax_player_maker<8>(state); },
        .madness = 2,
    },
    Player{
        .name    = "MinimaxD8_Mad5",
        .select  = [](const boardState& state) { return minimax_player_maker<8>(state); },
        .madness = 5,
    },
    Player{
        .name    = "MinimaxD8_Mad10",
        .select  = [](const boardState& state) { return minimax_player_maker<8>(state); },
        .madness = 10,
    },

    // ── MCTS variants @ 800 iterations ──────────────────────────────────────
    Player{
        .name   = "MCTS_Random800",
        .select = [](const boardState& state) { return mcts_player_maker<RandomRollout, 800>(state); },
    },
    Player{
        .name   = "MCTS_Heuristic800",
        .select = [](const boardState& state) { return mcts_player_maker<HeuristicRollout, 800>(state); },
    },
    Player{
        .name   = "MCTS_EpsGreedy800",
        .select = [](const boardState& state) { return mcts_player_maker<EpsilonGreedyRollout, 800>(state); },
    },

    // ── MCTS best variant @ 1000 iterations (better teacher signal) ──────────
    Player{
        .name   = "MCTS_EpsGreedy1000",
        .select = [](const boardState& state) { return mcts_player_maker<EpsilonGreedyRollout, 1000>(state); },
    },

    // ── Greedy with madness (extra weak/noisy games for value training) ───────
    Player{
        .name    = "GreedyPlayer_Mad30",
        .select  = [](const boardState& state) { return greedy_player_maker(state); },
        .madness = 30,
    },

    // clang-format on
};

}  // namespace kribu::player
