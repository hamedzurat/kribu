/**
 * @file players.hpp
 * @brief Centralized player configurations and factory instantiations for Sholo Guti benchmark.
 */

#pragma once

// #include "config.hpp"
#include "benchmark.hpp"
#include "kribu/board.hpp"
// #include "kribu/heuristic.hpp"
// #include "kribu/player/greedy.hpp"
// #include "kribu/player/mcts.hpp"
// #include "kribu/player/minimax.hpp"
#include "kribu/player/random.hpp"

namespace kribu::player {

// using namespace kribu::heuristics;
using namespace kribu::benchmark;

/**
 * @brief Compile-time defined array of benchmark players.
 * @details Players are defined and named in place using stateless C++ lambdas, which decay to function pointers.
 */
inline constexpr std::array BENCHMARK_PLAYERS = {
    // clang-format off
    Player{
        .name = "RandomPlayer",
        .select = [](const boardState& state) { return select_random(state); }
    },
    // clang-format on
};

}  // namespace kribu::player
