/**
 * @file players.hpp
 * @brief Centralized player configurations and factory instantiations for Sholo Guti benchmark.
 */

#pragma once

#include "config.hpp"
#include "kribu/benchmark.hpp"
#include "kribu/board.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/player/greedy.hpp"
#include "kribu/player/mcts.hpp"
#include "kribu/player/minimax.hpp"
#include "kribu/player/random.hpp"

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
        .select = [](const boardState& state) { return select_random(state); }
    },
    // Greedy
    Player{
        .name = "GreedyPieceCount",
        .select = [](const boardState& state) { return greedy_player_maker<evaluate_piece_count>(state); }
    },
    Player{
        .name = "GreedyPosition",
        .select = [](const boardState& state) { return greedy_player_maker<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>>(state); }
    },
    Player{
        .name = "GreedyMobility",
        .select = [](const boardState& state) { return greedy_player_maker<evaluate_mobility>(state); }
    },
    Player{
        .name = "GreedyMixed",
        .select = [](const boardState& state) { return greedy_player_maker<evaluate_mixed>(state); }
    },
    // Minimax PieceCount
    Player{
        .name = "MinimaxPieceCountD4",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_piece_count, 4, MINIMAX_SEARCH_THREADS>(state); }
    },
    Player{
        .name = "MinimaxPieceCountD4M20",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_piece_count, 4, MINIMAX_SEARCH_THREADS>(state); },
        .madness = 20
    },
    Player{
        .name = "MinimaxPieceCountD8",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_piece_count, 8, MINIMAX_SEARCH_THREADS>(state); }
    },
    Player{
        .name = "MinimaxPieceCountD8M20",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_piece_count, 8, MINIMAX_SEARCH_THREADS>(state); },
        .madness = 20
    },
    // Minimax Position
    Player{
        .name = "MinimaxPositionD4",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>, 4, MINIMAX_SEARCH_THREADS>(state); }
    },
    Player{
        .name = "MinimaxPositionD4M20",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>, 4, MINIMAX_SEARCH_THREADS>(state); },
        .madness = 20
    },
    Player{
        .name = "MinimaxPositionD8",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>, 8, MINIMAX_SEARCH_THREADS>(state); }
    },
    Player{
        .name = "MinimaxPositionD8M20",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>, 8, MINIMAX_SEARCH_THREADS>(state); },
        .madness = 20
    },
    // Minimax Mixed
    Player{
        .name = "MinimaxMixedD4",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_mixed, 4, MINIMAX_SEARCH_THREADS>(state); }
    },
    Player{
        .name = "MinimaxMixedD4M20",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_mixed, 4, MINIMAX_SEARCH_THREADS>(state); },
        .madness = 20
    },
    Player{
        .name = "MinimaxMixedD8",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_mixed, 8, MINIMAX_SEARCH_THREADS>(state); }
    },
    Player{
        .name = "MinimaxMixedD8M20",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_mixed, 8, MINIMAX_SEARCH_THREADS>(state); },
        .madness = 20
    },
    // Minimax Mobility
    Player{
        .name = "MinimaxMobilityD4",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_mobility, 4, MINIMAX_SEARCH_THREADS>(state); }
    },
    Player{
        .name = "MinimaxMobilityD4M20",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_mobility, 4, MINIMAX_SEARCH_THREADS>(state); },
        .madness = 20
    },
    Player{
        .name = "MinimaxMobilityD8",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_mobility, 8, MINIMAX_SEARCH_THREADS>(state); }
    },
    Player{
        .name = "MinimaxMobilityD8M20",
        .select = [](const boardState& state) { return minimax_player_maker<evaluate_mobility, 8, MINIMAX_SEARCH_THREADS>(state); },
        .madness = 20
    },
    // MCTS Random
    Player{
        .name = "MctsRandom500",
        .select = [](const boardState& state) { return mcts_player_maker<RandomRollout, 500, 8>(state); }
    },
    Player{
        .name = "MctsRandom1000",
        .select = [](const boardState& state) { return mcts_player_maker<RandomRollout, 1000, 8>(state); }
    },
    // MCTS Heuristic
    Player{
        .name = "MctsHeuristic500",
        .select = [](const boardState& state) { return mcts_player_maker<HeuristicRollout<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>>, 500, 8>(state); }
    },
    Player{
        .name = "MctsHeuristic1000",
        .select = [](const boardState& state) { return mcts_player_maker<HeuristicRollout<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>>, 1000, 8>(state); }
    },
    // MCTS EpsilonGreedy
    Player{
        .name = "MctsEpsilonGreedy500",
        .select = [](const boardState& state) { return mcts_player_maker<EpsilonGreedyRollout<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>>, 500, 8>(state); }
    },
    Player{
        .name = "MctsEpsilonGreedy1000",
        .select = [](const boardState& state) { return mcts_player_maker<EpsilonGreedyRollout<evaluate_by_node_values<HEURISTIC_NODE_WEIGHTS>>, 1000, 8>(state); }
    }
    // clang-format on
};

}  // namespace kribu::player
