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
 * - Root parallelism via independent trees with majority voting
 * - Epsilon-greedy rollout policy for informed simulations
 * - Fast piece-count terminal check in rollouts
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <thread>
#include <vector>

#include "kribu/board.hpp"
#include "kribu/fast_rng.hpp"
#include "kribu/heuristic.hpp"
#include "kribu/player/random.hpp"
#include "kribu/rules.hpp"
#include "kribu/types.hpp"

namespace kribu::player {

using namespace kribu::board;
using namespace kribu::sholoGuti;

// ── MCTS Configuration Constants ────────────────────────────────────────

/**
 * @brief UCT exploration constant (√2).
 */
constexpr float MCTS_EXPLORATION_C = static_cast<float>(std::numbers::sqrt2);

/**
 * @brief Weight of heuristic prior in progressive bias formula.
 * @details Added as prior * weight / (visits + 1), fading as visits grow.
 */
constexpr float MCTS_BIAS_WEIGHT = 1.0F;

/**
 * @brief FPU reduction: subtracted from parent win rate for unvisited children.
 */
constexpr float MCTS_FPU_REDUCTION = 0.1F;

/**
 * @brief Iteration interval for checking early termination.
 */
constexpr int MCTS_EARLY_TERM_INTERVAL = 50;

/**
 * @brief Visit share threshold to trigger early termination (0.85 = 85%).
 */
constexpr float MCTS_EARLY_TERM_THRESHOLD = 0.85F;

/**
 * @brief Maximum steps per rollout simulation.
 */
constexpr int MCTS_MAX_ROLLOUT_STEPS = 60;

/**
 * @brief Probability of random move in epsilon-greedy rollout.
 */
constexpr float MCTS_EPSILON = 0.3F;

/**
 * @brief Divisor for tanh-based score-to-value normalization.
 */
constexpr float MCTS_SCORE_SCALE = 50.0F;

// ── Utility Functions ───────────────────────────────────────────────────

/**
 * @brief Converts a raw heuristic score to a [0, 1] win probability via tanh.
 * @param score Raw evaluation score.
 * @return Normalized value in [0, 1].
 */
[[nodiscard]] inline float mcts_score_to_value(float score) noexcept {
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

// ── Rollout Policies ────────────────────────────────────────────────────

/**
 * @struct RandomRollout
 * @brief Playout policy that selects moves purely at random.
 */
struct RandomRollout {
  /**
   * @brief Selects a move randomly.
   * @param state Current board state.
   * @return Random legal move ID, or -1 if none.
   */
  static int select_move(const boardState& state) {
    u64 dummy = 0;
    return select_random(state, dummy);
  }

  /**
   * @brief Simple static evaluation at the end of the playout using piece count.
   * @param state Board state to evaluate.
   * @return Piece count evaluation score.
   */
  static double evaluate(const boardState& state) noexcept {
    return static_cast<double>(heuristics::evaluate_piece_count(state));
  }
};

/**
 * @struct HeuristicRollout
 * @brief Playout policy that prioritizes captures (to keep rollouts realistic), otherwise random,
 *        and uses a custom evaluation.
 * @tparam EvalFunc Custom evaluation function.
 */
template <auto EvalFunc>
struct HeuristicRollout {
  /**
   * @brief Selects a capture move if available, otherwise random.
   * @param state Current board state.
   * @return Selected move ID, or -1 if none.
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
   * @param state Board state to evaluate.
   * @return Evaluation score.
   */
  static double evaluate(const boardState& state) noexcept { return static_cast<double>(EvalFunc(state)); }
};

/**
 * @struct EpsilonGreedyRollout
 * @brief Playout policy that greedily picks the best-evaluated move with
 *        probability (1 - ε), or a random move with probability ε.
 * @details Produces more realistic rollouts than pure random by exploiting
 *        domain knowledge, while ε-randomness prevents deterministic loops.
 * @tparam EvalFunc Heuristic evaluation function.
 */
template <auto EvalFunc>
struct EpsilonGreedyRollout {
  /**
   * @brief Selects a move using epsilon-greedy strategy.
   * @param state Current board state.
   * @return Selected move ID, or -1 if none.
   */
  static int select_move(const boardState& state) {
    const MoveList moves = all_possible_moves(state);
    if (moves.empty()) {
      return -1;
    }
    std::uniform_real_distribution<float> prob(0.0F, 1.0F);
    if (prob(rng) < MCTS_EPSILON) {
      std::uniform_int_distribution<int> dist(0, moves.size() - 1);
      return moves.moves[dist(rng)];
    }
    return pick_best_move(state, moves);
  }

  /**
   * @brief Evaluates the board state using the provided EvalFunc.
   * @param state Board state to evaluate.
   * @return Evaluation score.
   */
  static double evaluate(const boardState& state) noexcept { return static_cast<double>(EvalFunc(state)); }

 private:
  /**
   * @brief Picks the move with the highest immediate heuristic value.
   * @param state Current board state.
   * @param moves Available legal moves.
   * @return Move ID with best evaluation.
   */
  static int pick_best_move(const boardState& state, const MoveList& moves) {
    int bestMove = moves.moves[0];
    i32 bestVal = -999999;
    for (int i = 0; i < moves.count; ++i) {
      const boardState next = apply_move(state, moves.moves[i]);
      i32 val = 0;
      if (next.activeCaptureIdx == -1) {
        val = -EvalFunc(flip_board(next));
      } else {
        val = EvalFunc(next);
      }
      if (val > bestVal) {
        bestVal = val;
        bestMove = moves.moves[i];
      }
    }
    return bestMove;
  }
};

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
  float visits = 0.0F;

  /**
   * @brief Cumulative value sum from backpropagation.
   */
  float valueSum = 0.0F;

  /**
   * @brief Heuristic prior for progressive bias (normalized to [0,1]).
   * @details Computed once at node creation via RolloutPolicy::evaluate.
   */
  float prior = 0.5F;

  /**
   * @brief Whether the turn flipped when transitioning to this node from parent.
   */
  bool turnFlipped = false;

  /**
   * @brief Whether this node represents a terminal game state.
   */
  bool isTerminal = false;

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
 * @tparam RolloutPolicy Policy defining move selection and evaluation during rollouts.
 */
template <typename RolloutPolicy>
class MCTS {
 public:
  /**
   * @brief Constructs the MCTS solver.
   * @param iterations Maximum number of search iterations.
   */
  explicit MCTS(int iterations = 800) : maxIterations(iterations) {}

  /**
   * @brief Runs MCTS search and returns the best move.
   * @param rootState The board state to search from.
   * @param nodesEvaluated Out-parameter for rollout count.
   * @return Best move ID selected by visit count.
   */
  int select_move(const boardState& rootState, u64& nodesEvaluated) {
    nodesEvaluated = 0;
    init_tree(rootState);

    for (int iter = 0; iter < maxIterations; ++iter) {
      run_iteration(nodesEvaluated);
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
  [[nodiscard]] MCTSNode& node_at(int idx) { return pool[static_cast<std::size_t>(idx)]; }

  /**
   * @brief Accesses a node by pool index (const).
   * @param idx Pool index.
   * @return Const reference to the node.
   */
  [[nodiscard]] const MCTSNode& node_at(int idx) const { return pool[static_cast<std::size_t>(idx)]; }

  /**
   * @brief Initializes the search tree with a root node.
   * @param rootState The root board state.
   */
  void init_tree(const boardState& rootState) {
    pool.clear();
    pool.reserve(static_cast<std::size_t>(maxIterations) + 100);
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
  [[nodiscard]] MCTSNode make_node(const boardState& state, int parent, int move, bool turnFlipped) {
    MCTSNode node;
    node.state = state;
    node.parentIdx = parent;
    node.moveId = move;
    node.turnFlipped = turnFlipped;

    // Fast terminal check by piece elimination
    if (piece_count(state.me) == 0 || piece_count(state.opp) == 0) {
      node.isTerminal = true;
      return node;
    }

    // Check stalemate (no legal moves)
    const MoveList moves = all_possible_moves(state);
    if (moves.empty()) {
      node.isTerminal = true;
      return node;
    }

    node.numLegalMoves = moves.count;
    node.prior = mcts_score_to_value(static_cast<float>(RolloutPolicy::evaluate(state)));
    return node;
  }

  /**
   * @brief Executes one MCTS iteration: select → expand → simulate → backpropagate.
   * @param nodesEvaluated Counter incremented per rollout.
   */
  void run_iteration(u64& nodesEvaluated) {
    int nodeIdx = select_leaf(0);
    float val = 0.0F;

    if (node_at(nodeIdx).isTerminal) {
      val = evaluate_terminal(node_at(nodeIdx).state);
    } else {
      int expandedIdx = expand_node(nodeIdx);
      if (expandedIdx != -1) {
        nodeIdx = expandedIdx;
      }
      val = simulate(node_at(nodeIdx).state);
      nodesEvaluated++;
    }

    backpropagate(nodeIdx, val);
  }

  /**
   * @brief Selects a leaf node by following UCT from the root.
   * @param nodeIdx Starting node index (typically 0).
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
    const MoveList moves = generate_expansion_moves(node_at(nodeIdx).state);
    int expandIdx = node_at(nodeIdx).nextExpandIdx;
    int chosenMoveId = moves.moves[expandIdx];

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
    node_at(nodeIdx).children[static_cast<std::size_t>(slot)] = childIdx;
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
  [[nodiscard]] float simulate(boardState state) {
    bool isP1Turn = true;

    for (int step = 0; step < MCTS_MAX_ROLLOUT_STEPS; ++step) {
      // Fast terminal check: piece elimination only (skip stalemate)
      if (piece_count(state.opp) == 0) {
        return isP1Turn ? 1.0F : 0.0F;
      }
      if (piece_count(state.me) == 0) {
        return isP1Turn ? 0.0F : 1.0F;
      }

      int moveId = RolloutPolicy::select_move(state);
      if (moveId == -1) {
        return isP1Turn ? 0.0F : 1.0F;
      }

      advance_rollout(state, isP1Turn, moveId);
    }

    const auto score = static_cast<float>(RolloutPolicy::evaluate(state));
    float val = mcts_score_to_value(score);
    return isP1Turn ? val : (1.0F - val);
  }

  void backpropagate(int nodeIdx, float val) {
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
  [[nodiscard]] float compute_child_uct(int childIdx, float parentWinRate, float logParent) const {
    const MCTSNode& child = node_at(childIdx);

    float childWinRate = 0.0F;
    if (child.visits > 0.0F) {
      childWinRate = child.valueSum / child.visits;
      // If the turn flipped between parent and child, the child's value is from the
      // opponent's perspective. The parent wants to minimize the opponent's win rate.
      if (child.turnFlipped) {
        childWinRate = 1.0F - childWinRate;
      }
    }

    // First Play Urgency: skip round-robin, use informed default
    if (child.visits < 1.0F) {
      return parentWinRate - MCTS_FPU_REDUCTION + (child.prior * MCTS_BIAS_WEIGHT);
    }

    float exploration = MCTS_EXPLORATION_C * std::sqrt(logParent / child.visits);
    float bias = child.prior * MCTS_BIAS_WEIGHT / (child.visits + 1.0F);

    return childWinRate + exploration + bias;
  }

  /**
   * @brief Selects the best child of a node using UCT with progressive bias and FPU.
   * @param nodeIdx Pool index of the parent node.
   * @return Pool index of the best child.
   */
  [[nodiscard]] int best_child_uct(int nodeIdx) {
    const MCTSNode& parent = node_at(nodeIdx);
    float parentWinRate = parent.valueSum / (parent.visits + 1e-6F);
    float logParent = std::log(parent.visits + 1.0F);

    int bestIdx = -1;
    float bestUCT = -1e9F;

    for (int i = 0; i < parent.numChildren; ++i) {
      int childIdx = parent.children[static_cast<std::size_t>(i)];
      float uct = compute_child_uct(childIdx, parentWinRate, logParent);
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
  [[nodiscard]] static float evaluate_terminal(const boardState& state) noexcept {
    if (piece_count(state.opp) == 0) {
      return 1.0F;
    }
    if (piece_count(state.me) == 0) {
      return 0.0F;
    }
    // Stalemate: active player has no moves → loss
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
    float maxVisits = -1.0F;

    for (int i = 0; i < root.numChildren; ++i) {
      int childIdx = root.children[static_cast<std::size_t>(i)];
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

    float totalVisits = 0.0F;
    float maxChildVisits = 0.0F;

    for (int i = 0; i < root.numChildren; ++i) {
      float childVisits = node_at(root.children[static_cast<std::size_t>(i)]).visits;
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

// ── Root Parallelism ────────────────────────────────────────────────────

/**
 * @brief Tallies majority votes from parallel MCTS results.
 * @tparam N Number of results.
 * @param results Array of move IDs from each thread.
 * @return Move ID with the most votes.
 */
template <std::size_t N>
[[nodiscard]] inline int tally_votes(const std::array<int, N>& results) {
  std::array<int, TOTAL_MOVE_COUNT> votes{};
  for (std::size_t voteIdx = 0; voteIdx < N; ++voteIdx) {
    int move = results[voteIdx];
    if (move >= 0 && move < TOTAL_MOVE_COUNT) {
      votes[static_cast<std::size_t>(move)]++;
    }
  }

  int bestMove = results[0];
  int bestVotes = 0;
  for (int i = 0; i < TOTAL_MOVE_COUNT; ++i) {
    if (votes[static_cast<std::size_t>(i)] > bestVotes) {
      bestVotes = votes[static_cast<std::size_t>(i)];
      bestMove = i;
    }
  }
  return bestMove;
}

/**
 * @brief Runs MCTS with root parallelism via majority voting.
 * @details Spawns NumThreads independent MCTS searches on the same root
 *          position. Each thread runs the full iteration count. The move
 *          receiving the most votes across threads is selected.
 * @tparam RolloutPolicy Rollout policy type.
 * @tparam Iterations Iterations per thread.
 * @tparam NumThreads Number of parallel search threads.
 * @param state Root board state.
 * @param nodes Out-parameter for total nodes evaluated.
 * @return Best move ID by majority vote.
 */
template <typename RolloutPolicy, int Iterations, int NumThreads>
[[nodiscard]] inline int mcts_root_parallel(const boardState& state, u64& nodes) {
  static_assert(NumThreads >= 2, "Use mcts_player_maker directly for single-threaded search.");

  std::array<int, NumThreads> results{};
  std::array<u64, NumThreads> nodeCounts{};
  std::array<std::thread, NumThreads> threads{};

  for (int threadIdx = 0; threadIdx < NumThreads; ++threadIdx) {
    threads[static_cast<std::size_t>(threadIdx)] = std::thread([&state, &results, &nodeCounts, threadIdx]() {
      MCTS<RolloutPolicy> solver(Iterations);
      u64 localNodes = 0;
      results[static_cast<std::size_t>(threadIdx)] = solver.select_move(state, localNodes);
      nodeCounts[static_cast<std::size_t>(threadIdx)] = localNodes;
    });
  }

  for (auto& thr : threads) {
    thr.join();
  }

  nodes = 0;
  for (int threadIdx = 0; threadIdx < NumThreads; ++threadIdx) {
    nodes += nodeCounts[static_cast<std::size_t>(threadIdx)];
  }

  return tally_votes(results);
}

// ── Player Maker ────────────────────────────────────────────────────────

/**
 * @brief Player maker function template for MCTS players.
 * @tparam RolloutPolicy Policy type for rollout simulation.
 * @tparam Iterations Number of MCTS iterations per search.
 * @tparam NumThreads Number of parallel threads (1 = sequential, default).
 * @param state Current board state.
 * @param nodes Out-parameter for total nodes evaluated.
 * @return Selected move ID.
 */
template <typename RolloutPolicy, int Iterations = 800, int NumThreads = 1>
[[nodiscard]] inline int mcts_player_maker(const boardState& state, u64& nodes) {
  if constexpr (NumThreads <= 1) {
    thread_local MCTS<RolloutPolicy> solver(Iterations);
    return solver.select_move(state, nodes);
  } else {
    return mcts_root_parallel<RolloutPolicy, Iterations, NumThreads>(state, nodes);
  }
}

}  // namespace kribu::player
