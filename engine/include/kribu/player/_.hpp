/**
 * @file player.hpp
 * @brief Centralized player configurations and factory instantiations for Sholo Guti.
 */

#pragma once

#include "kribu/benchmark.hpp"
#include "kribu/board.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/player/greedy.hpp"
#include "kribu/player/mcts.hpp"
#include "kribu/player/minimax.hpp"
#include "kribu/player/random.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

using namespace kribu::heuristics;
using namespace kribu::benchmark;

/**
 * @brief Compile-time defined array of benchmark players.
 * @details Players are defined and named in place using stateless C++ lambdas, which decay to function pointers.
 */
inline constexpr std::array BENCHMARK_PLAYERS = {
    // clang-format off
    Player{
        .name = "RandomPlayer",
        .select = [](const boardState& state, u64& nodes) { 
            return select_random(state, nodes); 
        }
    },
    Player{
        .name = "GreedyPlayer",
        .select = [](const boardState& state, u64& nodes) { 
            return greedy_player_maker<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>>(state, nodes);
        }
    },
    Player{
        .name = "MinimaxPieceCountDepth4",
        .select = [](const boardState& state, u64& nodes) { 
            return minimax_player_maker<evaluate_piece_count, 4>(state, nodes); 
        }
    },
    Player{
        .name = "MinimaxPositionDepth4",
        .select = [](const boardState& state, u64& nodes) { 
            return minimax_player_maker<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>, 4>(state, nodes);
        }
    },
    Player{
        .name = "MinimaxMixedDepth4",
        .select = [](const boardState& state, u64& nodes) { 
            return minimax_player_maker<evaluate_mixed, 4>(state, nodes);
        }
    },
    Player{
        .name = "MinimaxMobilityDepth4",
        .select = [](const boardState& state, u64& nodes) { 
            return minimax_player_maker<evaluate_mobility, 4>(state, nodes);
        }
    },
    Player{
        .name = "MCTSPlayerRandomIter1000",
        .select = [](const boardState& state, u64& nodes) { 
            return mcts_player_maker<RandomRollout, 1000>(state, nodes); 
        }
    },
    Player{
        .name = "MCTSPlayerHeuristicIter1000",
        .select = [](const boardState& state, u64& nodes) {
            return mcts_player_maker<HeuristicRollout<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>>, 1000>(state, nodes);
        }
    },
    Player{
        .name = "MinimaxPositionDepth4Mad20",
        .select = [](const boardState& state, u64& nodes) {
            return minimax_player_maker<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>, 4>(state, nodes, 20);
        }
    },
    Player{
        .name = "MinimaxPositionDepth4Mad50",
        .select = [](const boardState& state, u64& nodes) {
            return minimax_player_maker<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>, 4>(state, nodes, 50);
        }
    }
    // clang-format on
};

}  // namespace kribu::player
