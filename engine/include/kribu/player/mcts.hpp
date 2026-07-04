/**
 * @file mcts.hpp
 * @brief Monte Carlo Tree Search (MCTS) with progressive bias, FPU,
 *        early termination, and root parallelism for Sholo Guti.
 *
 * @details Optimizations over baseline UCT:
 * - Progressive bias: heuristic evaluation guides early exploration
 * - First Play Urgency (FPU): avoids forced round-robin of all children
 * - O(1) node expansion via tracked move index
 * - Capture-priority expansion ordering
 * - Early termination when one move clearly dominates
 * - Epsilon-greedy rollout policy for informed simulations
 * - Fast piece-count terminal check in rollouts
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

#include "kribu/board.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/rollout.hpp"
#include "kribu/rules.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

using namespace kribu::board;
using namespace kribu::sholoGuti;

// ── MCTS Configuration Constants ────────────────────────────────────────

/**
 * @brief UCT exploration constant (√2).
 */
constexpr f32 MCTS_EXPLORATION_C = static_cast<f32>(std::numbers::sqrt2);

/**
 * @brief Weight of heuristic prior in progressive bias formula.
 * @details Added as prior * weight / (visits + 1), fading as visits grow.
 */
constexpr f32 MCTS_BIAS_WEIGHT = 1.0F;

/**
 * @brief FPU reduction: subtracted from parent win rate for unvisited children.
 */
constexpr f32 MCTS_FPU_REDUCTION = 0.1F;

/**
 * @brief Iteration interval for checking early termination.
 */
constexpr int MCTS_EARLY_TERM_INTERVAL = 100;

/**
 * @brief Visit share threshold to trigger early termination (0.85 = 85%).
 */
constexpr f32 MCTS_EARLY_TERM_THRESHOLD = 0.85F;

/**
 * @brief Maximum steps per rollout simulation.
 */
constexpr int MCTS_MAX_ROLLOUT_STEPS = 60;

/**
 * @brief Probability of random move in epsilon-greedy rollout.
 */
constexpr f32 MCTS_EPSILON = 0.3F;

/**
 * @brief Divisor for tanh-based score-to-value normalization.
 */
constexpr f32 MCTS_SCORE_SCALE = 3000.0F;

// ── Utility Functions ───────────────────────────────────────────────────

/**
 * @brief Converts a raw heuristic score to a [0, 1] win probability via tanh.
 * @param score Raw evaluation score.
 * @return Normalized value in [0, 1].
 */
[[nodiscard]] inline f32 mcts_score_to_value(f32 score) noexcept {
  return 0.5F + (0.5F * std::tanh(score / MCTS_SCORE_SCALE));
}

/**
 * @brief Generates legal moves ordered for MCTS expansion: captures first.
 * @param state Board state to generate moves for.
 * @return MoveList with captures before quiet moves.
 */
[[nodiscard]] inline MoveList generate_expansion_moves(const boardState& state) noexcept {
  const MoveList all = all_possible_moves(state);
  MoveList ordered;
  for (int i = 0; i < all.count; ++i) {
    if (is_capture_move(all.moves[i])) {
      ordered.push(all.moves[i]);
    }
  }
  for (int i = 0; i < all.count; ++i) {
    if (!is_capture_move(all.moves[i])) {
      ordered.push(all.moves[i]);
    }
  }
  return ordered;
}

// ── MCTS Node ───────────────────────────────────────────────────────────

/**
 * @struct MCTSNode
 * @brief A single state node in the MCTS tree.
 * @details Stores board state, tree structure, expansion progress via
 *          nextExpandIdx for O(1) expansion, visit statistics, and
 *          heuristic prior for progressive bias.
 */
struct MCTSNode {
  /**
   * @brief Board state at this node.
   */
  boardState state;

  /**
   * @brief Pool index of the parent node (-1 for root).
   */
  int parentIdx = -1;

  /**
   * @brief Move ID that led to this node from parent.
   */
  int moveId = -1;

  /**
   * @brief Pool indices of child nodes.
   */
  std::array<int, MAX_MOVES_PER_STATE> children{};

  /**
   * @brief Number of children currently expanded.
   */
  int numChildren = 0;

  /**
   * @brief Total legal moves from this state.
   */
  int numLegalMoves = 0;

  /**
   * @brief Index of next move to expand (for O(1) expansion).
   * @details Replaces the O(n²) scan of the old implementation.
   */
  int nextExpandIdx = 0;

  /**
   * @brief Total visit count.
   */
  f32 visits = 0.0F;

  /**
   * @brief Cumulative value sum from backpropagation.
   */
  f32 valueSum = 0.0F;

  /**
   * @brief Heuristic prior for progressive bias (normalized to [0,1]).
   * @details Computed once at node creation via RolloutPolicy::evaluate.
   */
  f32 prior = 0.5F;

  /**
   * @brief Whether the turn flipped when transitioning to this node from parent.
   */
  bool turnFlipped = false;

  /**
   * @brief Whether this node represents a terminal game state.
   */
  bool isTerminal = false;

  std::array<i16, MAX_MOVES_PER_STATE> legalMoves{};

  /**
   * @brief Checks if all legal moves have been expanded.
   * @return True if every legal move has a corresponding child.
   */
  [[nodiscard]] bool is_fully_expanded() const noexcept { return nextExpandIdx >= numLegalMoves; }
};

// ── MCTS Engine ─────────────────────────────────────────────────────────

/**
 * @class MCTS
 * @brief Monte Carlo Tree Search engine with UCT, progressive bias, FPU,
 *        O(1) expansion, and early termination.
 */
class MCTS {
 public:
  /**
   * @brief Constructs the MCTS solver.
   * @param iterations Maximum number of search iterations.
   */
  explicit MCTS(int iterations = 800) : maxIterations(iterations) {}

  /**
   * @brief Selects the best move using MCTS from a given root state.
   * @param rootState The starting board state.
   * @return The selected move ID.
   */
  int select_move(const boardState& rootState) {
    init_tree(rootState);

    for (int iter = 0; iter < maxIterations; ++iter) {
      run_iteration();
      if (iter > 0 && iter % MCTS_EARLY_TERM_INTERVAL == 0 && should_terminate_early()) {
        break;
      }
    }

    return pick_best_root_move();
  }

 private:
  /**
   * @brief Accesses a node by pool index.
   * @param idx Pool index.
   * @return Reference to the node.
   */
  [[nodiscard]] MCTSNode& node_at(int idx) { return pool[static_cast<usize>(idx)]; }

  /**
   * @brief Accesses a node by pool index (const).
   * @param idx Pool index.
   * @return Const reference to the node.
   */
  [[nodiscard]] const MCTSNode& node_at(int idx) const { return pool[static_cast<usize>(idx)]; }

  /**
   * @brief Initializes the search tree with a root node.
   * @param rootState The root board state.
   */
  void init_tree(const boardState& rootState) {
    pool.clear();
    pool.reserve(static_cast<usize>(maxIterations) + 100);
    pool.push_back(make_node(rootState, -1, -1, false));
  }

  /**
   * @brief Creates a new MCTSNode from a board state.
   * @details Checks terminal status via piece counts and move availability.
   *          Computes heuristic prior for progressive bias.
   * @param state Board state for the new node.
   * @param parent Pool index of the parent (-1 for root).
   * @param move Move ID that produced this state.
   * @return Constructed MCTSNode.
   */
  [[nodiscard]] static MCTSNode make_node(const boardState& state, int parent, int move, bool turnFlipped) {
    MCTSNode node;
    node.state = state;
    node.parentIdx = parent;
    node.moveId = move;
    node.turnFlipped = turnFlipped;

    if (piece_count(state.opp) == 0 || piece_count(state.me) == 0) {
      node.isTerminal = true;
      return node;
    }

    if (opponent_has_no_moves(state)) {
      node.isTerminal = true;
      return node;
    }

    const MoveList moves = generate_expansion_moves(state);
    if (moves.empty()) {
      node.isTerminal = true;
      return node;
    }

    node.numLegalMoves = moves.count;

    for (int moveIndex = 0; moveIndex < moves.count; ++moveIndex) {
      node.legalMoves[moveIndex] = moves.moves[moveIndex];
    }

    node.prior = mcts_score_to_value(static_cast<f32>(heuristics::evaluate(state, 3)));
    return node;
  }

  /**
   * @brief Runs a single iteration of the MCTS cycle (selection, expansion, simulation, backpropagation).
   */
  void run_iteration() {
    int nodeIdx = select_leaf(0);
    f32 val = 0.0F;

    if (node_at(nodeIdx).isTerminal) {
      val = evaluate_terminal(node_at(nodeIdx).state);
    } else {
      int expandedIdx = expand_node(nodeIdx);
      if (expandedIdx != -1) {
        nodeIdx = expandedIdx;
      }
      val = simulate(node_at(nodeIdx).state);
    }

    backpropagate(nodeIdx, val);
  }

  /**
   * @brief Checks if the opponent player has no legal moves from the current state.
   * @param state The current board state.
   * @return True if the opponent has no legal moves, false otherwise.
   */
  [[nodiscard]] static bool opponent_has_no_moves(const boardState& state) noexcept {
    const boardState flippedState = flip_board(state);
    return all_possible_moves(flippedState).empty();
  }

  /**
   * @brief Selects a leaf node by following UCT through fully expanded nodes.
   * @param nodeIdx Starting node index
   * (typically 0).
   * @return Pool index of the selected leaf node.
   */
  [[nodiscard]] int select_leaf(int nodeIdx) {
    while (node_at(nodeIdx).is_fully_expanded() && !node_at(nodeIdx).isTerminal && node_at(nodeIdx).numChildren > 0) {
      nodeIdx = best_child_uct(nodeIdx);
    }
    return nodeIdx;
  }

  /**
   * @brief Expands one new child from the next unexpanded move.
   * @details Uses nextExpandIdx for O(1) expansion instead of the original O(n²) scan.
   *          Regenerates moves deterministically (captures first) each call.
   * @param nodeIdx Pool index of the node to expand.
   * @return Pool index of the new child, or -1 if fully expanded.
   */
  int expand_node(int nodeIdx) {
    if (node_at(nodeIdx).is_fully_expanded()) {
      return -1;
    }

    // Regenerate moves in capture-first order (deterministic from state)
    const int expandIndex = node_at(nodeIdx).nextExpandIdx;
    const int chosenMoveId = node_at(nodeIdx).legalMoves[expandIndex];

    // Prepare child state
    boardState childState = apply_move(node_at(nodeIdx).state, chosenMoveId);
    bool flipped = false;
    if (childState.activeCaptureIdx == -1) {
      childState = flip_board(childState);
      flipped = true;
    }

    // Update parent expansion index before push_back (which may reallocate)
    node_at(nodeIdx).nextExpandIdx++;

    int childIdx = static_cast<int>(pool.size());
    pool.push_back(make_node(childState, nodeIdx, chosenMoveId, flipped));

    // Re-access parent via index (safe after potential reallocation)
    int slot = node_at(nodeIdx).numChildren;
    node_at(nodeIdx).children[static_cast<usize>(slot)] = childIdx;
    node_at(nodeIdx).numChildren = slot + 1;

    return childIdx;
  }

  /**
   * @brief Advances a rollout by one move, flipping board if the turn changes.
   * @param state Board state to advance (modified in place).
   * @param isP1Turn Turn tracker (modified in place).
   * @param moveId Move to apply.
   */
  static void advance_rollout(boardState& state, bool& isP1Turn, int moveId) noexcept {
    const boardState nextState = apply_move(state, moveId);
    if (nextState.activeCaptureIdx == -1) {
      state = flip_board(nextState);
      isP1Turn = !isP1Turn;
    } else {
      state = nextState;
    }
  }

  /**
   * @brief Runs a rollout simulation from the given state.
   * @details Uses fast piece-count terminal check instead of full game status
   *          to avoid the expensive stalemate detection during rollouts.
   * @param state Starting board state for the rollout.
   * @return Simulation result in [0, 1].
   */
  [[nodiscard]] static f32 simulate(boardState state) {
    bool isP1Turn = true;

    for (int step = 0; step < MCTS_MAX_ROLLOUT_STEPS; ++step) {
      // Fast terminal check: piece elimination only (skip stalemate)
      if (piece_count(state.opp) == 0) {
        return isP1Turn ? 1.0F : 0.0F;
      }
      if (piece_count(state.me) == 0) {
        return isP1Turn ? 0.0F : 1.0F;
      }

      int moveId = Rollout::select_move(state);
      if (moveId == -1) {
        return isP1Turn ? 0.0F : 1.0F;
      }

      advance_rollout(state, isP1Turn, moveId);
    }

    const auto score = static_cast<f32>(Rollout::evaluate(state));
    f32 val = mcts_score_to_value(score);
    return isP1Turn ? val : (1.0F - val);
  }

  /**
   * @brief Backpropagates simulation results up to the root node.
   * @param nodeIdx Index of the starting node to backpropagate from.
   * @param val The simulation value to backpropagate.
   */
  void backpropagate(int nodeIdx, f32 val) {
    while (nodeIdx != -1) {
      node_at(nodeIdx).visits += 1.0F;
      node_at(nodeIdx).valueSum += val;

      bool flipped = node_at(nodeIdx).turnFlipped;
      int parentIdx = node_at(nodeIdx).parentIdx;
      if (parentIdx != -1 && flipped) {
        val = 1.0F - val;
      }
      nodeIdx = parentIdx;
    }
  }

  /**
   * @brief Computes the UCT value for a child node with progressive bias and FPU.
   * @details For unvisited children (visits < 1), returns FPU value:
   *          parent_win_rate - FPU_REDUCTION + prior * BIAS_WEIGHT.
   *          For visited children: win_rate + exploration + decaying_bias.
   * @param childIdx Pool index of the child.
   * @param parentWinRate Parent node's win rate (for FPU baseline).
   * @param logParent Log of parent's visit count (for exploration term).
   * @return Combined UCT score.
   */
  [[nodiscard]] f32 compute_child_uct(int childIdx, f32 parentWinRate, f32 logParent) const {
    const MCTSNode& child = node_at(childIdx);

    f32 childPrior = child.prior;
    if (child.turnFlipped) {
      childPrior = 1.0F - childPrior;
    }

    f32 childWinRate = 0.0F;
    if (child.visits > 0.0F) {
      childWinRate = child.valueSum / child.visits;

      if (child.turnFlipped) {
        childWinRate = 1.0F - childWinRate;
      }
    }

    if (child.visits < 1.0F) {
      return parentWinRate - MCTS_FPU_REDUCTION + (childPrior * MCTS_BIAS_WEIGHT);
    }

    const f32 exploration = MCTS_EXPLORATION_C * std::sqrt(logParent / child.visits);
    const f32 bias = childPrior * MCTS_BIAS_WEIGHT / (child.visits + 1.0F);

    return childWinRate + exploration + bias;
  }

  /**
   * @brief Selects the best child of a node using UCT with progressive bias and FPU.
   * @param nodeIdx Pool index of the parent node.
   * @return Pool index of the best child.
   */
  [[nodiscard]] int best_child_uct(int nodeIdx) {
    const MCTSNode& parent = node_at(nodeIdx);
    f32 parentWinRate = parent.valueSum / (parent.visits + 1e-6F);
    f32 logParent = std::log(parent.visits + 1.0F);

    int bestIdx = -1;
    f32 bestUCT = -1e9F;

    for (int i = 0; i < parent.numChildren; ++i) {
      int childIdx = parent.children[static_cast<usize>(i)];
      f32 uct = compute_child_uct(childIdx, parentWinRate, logParent);
      if (uct > bestUCT) {
        bestUCT = uct;
        bestIdx = childIdx;
      }
    }
    return bestIdx;
  }

  /**
   * @brief Evaluates a terminal state as a [0, 1] value.
   * @param state Terminal board state.
   * @return 1.0 for win, 0.0 for loss, 0.5 for draw.
   */
  [[nodiscard]] static f32 evaluate_terminal(const boardState& state) noexcept {
    if (piece_count(state.opp) == 0) {
      return 1.0F;
    }

    if (piece_count(state.me) == 0) {
      return 0.0F;
    }

    if (opponent_has_no_moves(state)) {
      return 1.0F;
    }

    const MoveList moves = all_possible_moves(state);
    if (moves.empty()) {
      return 0.0F;
    }

    return 0.5F;
  }

  /**
   * @brief Returns the best root move by highest visit count.
   * @return Move ID with the most visits, or -1 if no children.
   */
  [[nodiscard]] int pick_best_root_move() const {
    const MCTSNode& root = node_at(0);
    int bestMove = -1;
    f32 maxVisits = -1.0F;

    for (int i = 0; i < root.numChildren; ++i) {
      int childIdx = root.children[static_cast<usize>(i)];
      if (node_at(childIdx).visits > maxVisits) {
        maxVisits = node_at(childIdx).visits;
        bestMove = node_at(childIdx).moveId;
      }
    }

    if (bestMove == -1 && root.numChildren > 0) {
      bestMove = node_at(root.children[0]).moveId;
    }
    return bestMove;
  }

  /**
   * @brief Checks if early termination is warranted.
   * @details Triggers when the best child has ≥ MCTS_EARLY_TERM_THRESHOLD
   *          of total root visits, indicating further search is unlikely to
   *          change the outcome.
   * @return True if search should stop early.
   */
  [[nodiscard]] bool should_terminate_early() const {
    const MCTSNode& root = node_at(0);
    if (root.numChildren <= 1) {
      return true;
    }

    f32 totalVisits = 0.0F;
    f32 maxChildVisits = 0.0F;

    for (int i = 0; i < root.numChildren; ++i) {
      f32 childVisits = node_at(root.children[static_cast<usize>(i)]).visits;
      totalVisits += childVisits;
      maxChildVisits = std::max(maxChildVisits, childVisits);
    }

    return totalVisits > 0.0F && (maxChildVisits / totalVisits) >= MCTS_EARLY_TERM_THRESHOLD;
  }

  /**
   * @brief Maximum number of search iterations.
   */
  int maxIterations;

  /**
   * @brief Pool-allocated tree nodes.
   */
  std::vector<MCTSNode> pool;
};

// ── Player Maker ────────────────────────────────────────────────────────

/**
 * @brief Player maker function template for MCTS players.
 * @tparam Iterations Number of MCTS iterations per search.
 * @param state Current board state.
 * @return Selected move ID.
 */
template <int Iterations = 800>
[[nodiscard]] inline int mcts_player_maker(const boardState& state) {
  thread_local MCTS solver(Iterations);
  return solver.select_move(state);
}

}  // namespace kribu::player
