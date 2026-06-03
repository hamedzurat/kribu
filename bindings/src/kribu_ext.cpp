/**
 * @file kribu_ext.cpp
 * @brief Python bindings for the Kribu Sholo Guti engine.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <kribu/board.hpp>
#include <kribu/player/greedy.hpp>
#include <kribu/player/mcts.hpp>
#include <kribu/player/minimax.hpp>
#include <kribu/player/random.hpp>
#include <kribu/rules.hpp>
#include <kribu/types.hpp>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nb = nanobind;
using namespace kribu::board;
using namespace kribu::player;

/**
 * @brief Converts a MoveList to a std::vector<int> for Python interop.
 * @param state Current board state.
 * @return Vector of valid move IDs.
 */
static std::vector<int> all_possible_moves_py(const boardState& state) {
  MoveList moves = all_possible_moves(state);
  std::vector<int> result;
  result.reserve(static_cast<usize>(moves.size()));
  for (i16 mid : moves) {
    result.push_back(static_cast<int>(mid));
  }
  return result;
}

static MinimaxResult minimax_py(const boardState& state, int depth) {
  return minimax(state, depth, -INFINITY_VAL, INFINITY_VAL);
}

static int minimax_player_8_py(const boardState& state) {
  return minimax_player_maker<8>(state);
}

static int minimax_player_4_py(const boardState& state) {
  return minimax_player_maker<4>(state);
}

static int minimax_player_8_mad2_py(const boardState& state) {
  thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, 99);
  if (dist(rng) < 2) {
    return select_random(state);
  }
  return minimax_player_maker<8>(state);
}

static int mcts_player_800_py(const boardState& state) {
  return mcts_player_maker<800>(state);
}

static int greedy_player_py(const boardState& state) {
  return greedy_player_maker(state);
}

/**
 * @brief Count how many entries in the repetition history equal the target hash.
 * @param state Current board state with its rolling repetition history.
 * @param targetHash Zobrist hash to count inside the valid history window.
 * @return Number of occurrences of @p targetHash in `state.history[0:historyCount)`.
 */
static int count_history_matches(const boardState& state, u64 targetHash) {
  int matches = 0;
  for (u32 index = 0; index < state.historyCount; ++index) {
    if (state.history[index] == targetHash) {
      matches++;
    }
  }
  return matches;
}

/**
 * @brief Return compact repetition features for one board state.
 * @param state Current board state.
 * @return Tuple of `(history_count, current_repeat_count, current_flip_repeat_count)`.
 */
static std::tuple<int, int, int> repetition_features_py(const boardState& state) {
  return {
      static_cast<int>(state.historyCount),
      count_history_matches(state, state.hash),
      count_history_matches(state, compute_flipped_hash(state)),
  };
}

/**
 * @brief Reconstruct repetition features for the ordered turn sequence of one game.
 * @param meMasks Active-player bitboards for each turn.
 * @param oppMasks Opponent-player bitboards for each turn.
 * @param activeCaptureIndices Active capture indices for each turn.
 * @param chosenMoves Recorded chosen move IDs for each turn.
 * @return Tuple of per-turn vectors `(history_count, current_repeat_count, current_flip_repeat_count)`.
 */
static std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> annotate_repetition_features_py(
    const std::vector<u64>& meMasks,
    const std::vector<u64>& oppMasks,
    const std::vector<i8>& activeCaptureIndices,
    const std::vector<int>& chosenMoves) {
  const usize turnCount = meMasks.size();
  if (oppMasks.size() != turnCount || activeCaptureIndices.size() != turnCount || chosenMoves.size() != turnCount) {
    throw std::invalid_argument("annotate_repetition_features expects equally sized input vectors");
  }

  std::vector<int> historyCounts;
  std::vector<int> currentRepeatCounts;
  std::vector<int> currentFlipRepeatCounts;
  historyCounts.reserve(turnCount);
  currentRepeatCounts.reserve(turnCount);
  currentFlipRepeatCounts.reserve(turnCount);

  if (turnCount == 0) {
    return {historyCounts, currentRepeatCounts, currentFlipRepeatCounts};
  }

  boardState state;
  state.me = meMasks[0];
  state.opp = oppMasks[0];
  state.activeCaptureIdx = activeCaptureIndices[0];
  state.hash = kribu::zobrist::compute_hash(
      {.activePlayer = state.me, .opponentPlayer = state.opp, .activeCaptureIdx = state.activeCaptureIdx});

  for (usize turnIndex = 0; turnIndex < turnCount; ++turnIndex) {
    if (state.me != meMasks[turnIndex] || state.opp != oppMasks[turnIndex]
        || state.activeCaptureIdx != activeCaptureIndices[turnIndex]) {
      throw std::runtime_error("annotate_repetition_features encountered a turn sequence mismatch");
    }

    historyCounts.push_back(static_cast<int>(state.historyCount));
    currentRepeatCounts.push_back(count_history_matches(state, state.hash));
    currentFlipRepeatCounts.push_back(count_history_matches(state, compute_flipped_hash(state)));

    boardState nextState = apply_move(state, chosenMoves[turnIndex]);
    state = nextState.activeCaptureIdx == -1 ? flip_board(nextState) : nextState;
  }

  return {historyCounts, currentRepeatCounts, currentFlipRepeatCounts};
}

/**
 * @brief Binding module definition for kribu_ext.
 */
NB_MODULE(kribu_ext, module) {  // NOLINT(readability-identifier-length, modernize-avoid-c-arrays)
  module.doc() = "Kribu Sholo Guti AI extension module";

  module.attr("NUM_NODES") = NUM_NODES;
  module.attr("TOTAL_MOVE_COUNT") = TOTAL_MOVE_COUNT;
  module.attr("END_CHAIN_MOVE") = END_CHAIN_MOVE;
  module.attr("NUM_SIMPLE_MOVES") = NUM_SIMPLE_MOVES;
  module.attr("NUM_CAPTURE_MOVES") = NUM_CAPTURE_MOVES;
  module.attr("MAX_MOVES_PER_STATE") = MAX_MOVES_PER_STATE;

  nb::class_<boardState>(module, "boardState")
      .def(nb::init<>())
      .def_rw("me", &boardState::me)
      .def_rw("opp", &boardState::opp)
      .def_rw("activeCaptureIdx", &boardState::activeCaptureIdx)
      .def_rw("hash", &boardState::hash)
      .def_rw("history", &boardState::history)
      .def_rw("historyCount", &boardState::historyCount)
      .def("__eq__", &boardState::operator==);

  nb::class_<move>(module, "move")
      .def(nb::init<>())
      .def_rw("fromNode", &move::from)
      .def_rw("toNode", &move::to)
      .def_rw("captured", &move::captured);

  nb::enum_<GameStatus>(module, "GameStatus")
      .value("OPP_WINS_ELIMINATION", GameStatus::OPP_WINS_ELIMINATION)
      .value("OPP_WINS_STALEMATE", GameStatus::OPP_WINS_STALEMATE)
      .value("ONGOING", GameStatus::ONGOING)
      .value("ME_WINS_STALEMATE", GameStatus::ME_WINS_STALEMATE)
      .value("ME_WINS_ELIMINATION", GameStatus::ME_WINS_ELIMINATION)
      .value("DRAW_PROGRESS_RULE", GameStatus::DRAW_PROGRESS_RULE)
      .export_values();

  module.attr("INITIAL_STATE") = INITIAL_STATE;
  module.def("piece_count", &piece_count, "Count pieces in a bitmask");
  module.def("decode_move", &decode_move, "Decode a move ID into a Move object");
  module.def("find_move", &find_move, "Find move ID for a (from, dst) node pair, or -1");
  module.def("is_simple_move", &is_simple_move, "True if moveId is a simple (non-capture) move");
  module.def("is_capture_move", &is_capture_move, "True if moveId is a capture move");
  module.def("flip_board", &flip_board, "Flip the board 180 degrees and swap players");
  module.def("is_valid", &is_valid, "Check if a move is valid for the given state");
  module.def("all_possible_moves", &all_possible_moves_py, "Get all possible valid move IDs for the given state");
  module.def("apply_move", &apply_move, "Apply a move and return the new state (no validity check)");
  module.def("get_game_status", &get_game_status, "Get the full GameStatus enum value");
  module.def("repetition_features",
             &repetition_features_py,
             nb::arg("state"),
             "Return (history_count, current_repeat_count, current_flip_repeat_count) for one state");
  module.def("annotate_repetition_features",
             &annotate_repetition_features_py,
             nb::arg("me_masks"),
             nb::arg("opp_masks"),
             nb::arg("active_capture_indices"),
             nb::arg("chosen_moves"),
             "Reconstruct repetition features for the ordered turns of one recorded game");

  nb::class_<MinimaxResult>(module, "MinimaxResult")
      .def(nb::init<>())
      .def_rw("score", &MinimaxResult::score)
      .def_rw("moveId", &MinimaxResult::moveId);

  module.def("minimax", &minimax_py, nb::arg("state"), nb::arg("depth"), "Run minimax search and return MinimaxResult");
  module.def(
      "minimax_player_4", &minimax_player_4_py, nb::arg("state"), "Run minimax search with depth 4 and return move ID");
  module.def(
      "minimax_player_8", &minimax_player_8_py, nb::arg("state"), "Run minimax search with depth 8 and return move ID");
  module.def("minimax_player_8_mad2",
             &minimax_player_8_mad2_py,
             nb::arg("state"),
             "Run depth-8 minimax with benchmark-style 2% random move injection and return move ID");
  module.def("mcts_player_800",
             &mcts_player_800_py,
             nb::arg("state"),
             "Run MCTS search with 800 iterations and return move ID");
  module.def("greedy_player", &greedy_player_py, nb::arg("state"), "Run greedy player maker and return move ID");
}
