/**
 * @file mcts.hpp
 * @brief Monte Carlo Tree Search (MCTS) implementation with heuristic evaluation.
 */

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include "kribu/board.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/player/random.hpp"
#include "kribu/rules.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

using namespace kribu::board;
using namespace kribu::sholoGuti;

/**
 * @struct RandomRollout
 * @brief Playout policy that selects moves purely at random.
 */
struct RandomRollout {
  /**
   * @brief Selects a move randomly.
   */
  static int select_move(const boardState& state) {
    u64 dummy = 0;
    return select_random(state, dummy);
  }

  /**
   * @brief Simple static evaluation at the end of the playout using piece count.
   */
  static double evaluate(const boardState& state) noexcept {
    return static_cast<double>(heuristics::evaluate_piece_count(state));
  }
};

/**
 * @struct HeuristicRollout
 * @brief Playout policy that prioritizes captures (to keep rollouts realistic), otherwise random, and uses a custom
 * evaluation.
 */
template <auto EvalFunc>
struct HeuristicRollout {
  /**
   * @brief Selects a capture move if available, otherwise random.
   */
  static int select_move(const boardState& state) {
    const MoveList moves = all_possible_moves(state);
    if (moves.empty()) {
      return -1;
    }
    MoveList captures;
    for (int i = 0; i < moves.count; ++i) {
      if (is_capture_move(moves.moves[i])) {
        captures.push(moves.moves[i]);
      }
    }
    if (!captures.empty()) {
      std::uniform_int_distribution<int> dist(0, captures.size() - 1);
      return captures.moves[dist(rng)];
    }
    std::uniform_int_distribution<int> dist(0, moves.size() - 1);
    return moves.moves[dist(rng)];
  }

  /**
   * @brief Evaluates the board state using the provided EvalFunc heuristic function.
   */
  static double evaluate(const boardState& state) noexcept { return static_cast<double>(EvalFunc(state)); }
};

/**
 * @struct MCTSNode
 * @brief Represents a single state node in the MCTS tree.
 */
struct MCTSNode {
  boardState state;
  int moveId = -1;
  int parentIdx = -1;
  std::array<int, MAX_MOVES_PER_STATE> children{};
  int numChildren = 0;
  double visits = 0.0;
  double valueSum = 0.0;
  bool isFullyExpanded = false;
  bool isTerminal = false;
};

/**
 * @class MCTS
 * @brief Monte Carlo Tree Search player solver utilizing rollout policies.
 */
template <typename RolloutPolicy>
class MCTS {
 public:
  /**
   * @brief Constructs the MCTS solver.
   * @param iterations Fixed number of selection/simulation loops to run.
   */
  explicit MCTS(int iterations = 800) : maxIterations(iterations) {}

  /**
   * @brief Simulates and selects the best move for a given board state.
   */
  int select_move(const boardState& rootState, u64& nodesEvaluated) {
    nodesEvaluated = 0;

    // Clear and reserve pool to avoid dynamic heap allocation during iteration
    pool.clear();
    pool.reserve(static_cast<std::size_t>(maxIterations) + 100);

    // Initialize root node
    MCTSNode root;
    root.state = rootState;
    root.isTerminal = (get_game_status(rootState) != GameStatus::ONGOING);
    pool.push_back(root);

    for (int iter = 0; iter < maxIterations; ++iter) {
      int nodeIdx = select_leaf(0);
      double val = 0.0;
      if (!pool[static_cast<std::size_t>(nodeIdx)].isTerminal) {
        int expandedIdx = expand_node(nodeIdx);
        if (expandedIdx != -1) {
          nodeIdx = expandedIdx;
        }
        val = simulate(pool[static_cast<std::size_t>(nodeIdx)].state);
        nodesEvaluated++;
      } else {
        GameStatus status = get_game_status(pool[static_cast<std::size_t>(nodeIdx)].state);
        if (status == GameStatus::ME_WINS_ELIMINATION || status == GameStatus::ME_WINS_STALEMATE) {
          val = 1.0;
        } else if (status == GameStatus::OPP_WINS_ELIMINATION || status == GameStatus::OPP_WINS_STALEMATE) {
          val = 0.0;
        } else {
          val = 0.5;
        }
      }
      backpropagate(nodeIdx, val);
    }

    // Return the move with the highest visit count from the root node
    int bestMove = -1;
    double maxVisits = -1.0;
    for (int i = 0; i < pool[0].numChildren; ++i) {
      int childIdx = pool[0].children[static_cast<std::size_t>(i)];
      if (pool[static_cast<std::size_t>(childIdx)].visits > maxVisits) {
        maxVisits = pool[static_cast<std::size_t>(childIdx)].visits;
        bestMove = pool[static_cast<std::size_t>(childIdx)].moveId;
      }
    }

    if (bestMove == -1 && pool[0].numChildren > 0) {
      bestMove = pool[static_cast<std::size_t>(pool[0].children[0])].moveId;
    }

    return bestMove;
  }

 private:
  int select_leaf(int nodeIdx) {
    while (pool[static_cast<std::size_t>(nodeIdx)].isFullyExpanded
           && !pool[static_cast<std::size_t>(nodeIdx)].isTerminal
           && pool[static_cast<std::size_t>(nodeIdx)].numChildren > 0) {
      nodeIdx = best_child_uct(nodeIdx);
    }
    return nodeIdx;
  }

  int expand_node(int nodeIdx) {
    auto moves = all_possible_moves(pool[static_cast<std::size_t>(nodeIdx)].state);
    if (moves.empty()) {
      pool[static_cast<std::size_t>(nodeIdx)].isTerminal = true;
      return -1;
    }

    // Find first move not yet present in children list
    int chosenMoveId = -1;
    for (int i = 0; i < moves.count; ++i) {
      int moveId = moves.moves[i];
      bool found = false;
      for (int childIdxIter = 0; childIdxIter < pool[static_cast<std::size_t>(nodeIdx)].numChildren; ++childIdxIter) {
        if (pool[static_cast<std::size_t>(
                     pool[static_cast<std::size_t>(nodeIdx)].children[static_cast<std::size_t>(childIdxIter)])]
                .moveId
            == moveId) {
          found = true;
          break;
        }
      }
      if (!found) {
        chosenMoveId = moveId;
        break;
      }
    }

    if (chosenMoveId == -1) {
      pool[static_cast<std::size_t>(nodeIdx)].isFullyExpanded = true;
      return -1;
    }

    MCTSNode child;
    child.state = apply_move(pool[static_cast<std::size_t>(nodeIdx)].state, chosenMoveId);
    if (child.state.activeCaptureIdx == -1) {
      child.state = flip_board(child.state);
    }
    child.moveId = chosenMoveId;
    child.parentIdx = nodeIdx;
    child.isTerminal = (get_game_status(child.state) != GameStatus::ONGOING);

    int childIdx = static_cast<int>(pool.size());
    pool.push_back(child);

    pool[static_cast<std::size_t>(nodeIdx)]
        .children[static_cast<std::size_t>(pool[static_cast<std::size_t>(nodeIdx)].numChildren++)] = childIdx;

    if (pool[static_cast<std::size_t>(nodeIdx)].numChildren == moves.count) {
      pool[static_cast<std::size_t>(nodeIdx)].isFullyExpanded = true;
    }

    return childIdx;
  }

  double simulate(boardState state) {
    int steps = 0;
    constexpr int MAX_ROLLOUT_STEPS = 60;
    bool isP1Turn = true;

    while (steps < MAX_ROLLOUT_STEPS) {
      GameStatus status = get_game_status(state);
      if (status != GameStatus::ONGOING) {
        if (status == GameStatus::ME_WINS_ELIMINATION || status == GameStatus::ME_WINS_STALEMATE) {
          return isP1Turn ? 1.0 : 0.0;
        }
        if (status == GameStatus::OPP_WINS_ELIMINATION || status == GameStatus::OPP_WINS_STALEMATE) {
          return isP1Turn ? 0.0 : 1.0;
        }
        return 0.5;
      }

      int moveId = RolloutPolicy::select_move(state);
      if (moveId == -1) {
        return isP1Turn ? 0.0 : 1.0;
      }

      boardState nextState = apply_move(state, moveId);
      if (nextState.activeCaptureIdx == -1) {
        state = flip_board(nextState);
        isP1Turn = !isP1Turn;
      } else {
        state = nextState;
      }
      steps++;
    }

    double score = RolloutPolicy::evaluate(state);
    double val = 0.5 + (0.5 * std::tanh(score / 50.0));
    return isP1Turn ? val : (1.0 - val);
  }

  void backpropagate(int nodeIdx, double val) {
    while (nodeIdx != -1) {
      pool[static_cast<std::size_t>(nodeIdx)].visits += 1.0;
      pool[static_cast<std::size_t>(nodeIdx)].valueSum += val;

      int parentIdx = pool[static_cast<std::size_t>(nodeIdx)].parentIdx;
      if (parentIdx != -1) {
        bool turnFlipped = (pool[static_cast<std::size_t>(nodeIdx)].moveId == END_CHAIN_MOVE
                            || !is_capture_move(pool[static_cast<std::size_t>(nodeIdx)].moveId));
        if (turnFlipped) {
          val = 1.0 - val;
        }
      }
      nodeIdx = parentIdx;
    }
  }

  int best_child_uct(int nodeIdx) {
    double parentVisits = pool[static_cast<std::size_t>(nodeIdx)].visits;
    int bestIdx = -1;
    double bestUCT = -1e9;

    for (int i = 0; i < pool[static_cast<std::size_t>(nodeIdx)].numChildren; ++i) {
      int childIdx = pool[static_cast<std::size_t>(nodeIdx)].children[static_cast<std::size_t>(i)];
      double childVisits = pool[static_cast<std::size_t>(childIdx)].visits;
      double winRate = pool[static_cast<std::size_t>(childIdx)].valueSum / (childVisits + 1e-6);
      double uct = winRate + (std::numbers::sqrt2 * std::sqrt(std::log(parentVisits + 1.0) / (childVisits + 1e-6)));
      if (uct > bestUCT) {
        bestUCT = uct;
        bestIdx = childIdx;
      }
    }
    return bestIdx;
  }

  int maxIterations;
  std::vector<MCTSNode> pool;
};

/**
 * @brief Player maker function template for MCTS players.
 */
template <typename RolloutPolicy, int Iterations = 800>
[[nodiscard]] inline int mcts_player_maker(const boardState& state, u64& nodes) {
  thread_local MCTS<RolloutPolicy> solver(Iterations);
  return solver.select_move(state, nodes);
}

}  // namespace kribu::player
