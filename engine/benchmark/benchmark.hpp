/**
 * @file benchmark.hpp
 * @brief Multithreaded benchmarking and matchmaking framework for Sholo Guti engine.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#include "config.hpp"
#include "kribu/board.hpp"
#include "kribu/player/random.hpp"
#include "kribu/rules.hpp"
#include "kribu/types.hpp"

namespace kribu::benchmark {

using namespace kribu::board;
using namespace kribu::sholoGuti;

/**
 * @struct TurnRecord
 * @brief Records the state and decision of a single turn/ply for Parquet serialization.
 */
struct TurnRecord {
  boardState state;
  bool isP1Turn = false;
  int chosenMove = -1;
  std::string_view playerPlayed;
};

/**
 * @enum GameResult
 * @brief Represents the outcome of a single match.
 */
enum class GameResult : u8 { P1_WINS, P2_WINS, DRAW };

/**
 * @enum WinReason
 * @brief Represents the reason why a player won or the game ended.
 */
enum class WinReason : u8 { ELIMINATION, STALEMATE, INVALID_MOVE, DRAW_MAX_TURNS, REPETITION };

/**
 * @struct GameOutcome
 * @brief Bundles game result and the reason.
 */
struct GameOutcome {
  GameResult result;
  WinReason reason;
  int winMargin = 0;
};

/**
 * @struct Player
 * @brief Abstract representation of an AI player in the tournament.
 */
struct Player {
  std::string_view type = "unknown";
  std::string_view name;
  int (*select)(const boardState&) = nullptr;
  int depth = 0;
  int madness = 0;
};

/**
 * @struct MatchStats
 * @brief Aggregate statistics from a matchup between two players.
 */
struct MatchStats {
  std::string_view player1Name;
  std::string_view player2Name;
  int p1Wins = 0;
  int p2Wins = 0;
  int draws = 0;
  int p1EliminationWins = 0;
  int p1StalemateWins = 0;
  int p1InvalidMoveWins = 0;
  int p2EliminationWins = 0;
  int p2StalemateWins = 0;
  int p2InvalidMoveWins = 0;
  f64 p1TotalCpuTimeSeconds = 0.0;
  f64 p2TotalCpuTimeSeconds = 0.0;
  u64 totalTurns = 0;
};

/**
 * @struct PlayerPerformance
 * @brief Performance telemetry for a player during a game.
 */
struct PlayerPerformance {
  f64 cpuTimeSeconds = 0.0;
};

/**
 * @struct GamePerf
 * @brief Performance metrics for both players in a single match.
 */
struct GamePerf {
  PlayerPerformance p1;
  PlayerPerformance p2;
};

/**
 * @struct TournamentConfig
 * @brief Parameters configuring a matchup tournament.
 */
struct TournamentConfig {
  int totalGames = 0;
  int maxTurns = 1000;
  int threadCount = 0;
  std::atomic<int>* completedGames = nullptr;
  std::atomic<int>* globalGameId = nullptr;
  std::atomic<int>* p1Wins = nullptr;
  std::atomic<int>* p2Wins = nullptr;
  std::atomic<int>* draws = nullptr;
  std::atomic<bool>* abortRequested = nullptr;
};

/**
 * @brief Saves a completed game dataset to DuckDB.
 * @details Implemented in benchmark_main.cpp.
 */
void save_game(std::string_view player1Name,
               std::string_view player2Name,
               int gameId,
               const GameOutcome& outcome,
               const std::vector<TurnRecord>& history);

/**
 * @brief Thread-local accumulator for win statistics.
 */
struct LocalWins {
  int wins = 0;
  int elimination = 0;
  int stalemate = 0;
  int invalidMove = 0;
};

/**
 * @brief Helper to record outcomes of a game.
 */
constexpr void record_outcome(const GameOutcome& outcome, LocalWins& localWins) noexcept {
  localWins.wins++;
  if (outcome.reason == WinReason::ELIMINATION) {
    localWins.elimination++;
  } else if (outcome.reason == WinReason::STALEMATE) {
    localWins.stalemate++;
  } else if (outcome.reason == WinReason::INVALID_MOVE) {
    localWins.invalidMove++;
  }
}

/**
 * @brief Helper to query the player move choice.
 */
inline int get_player_move(
    const Player& player1, const Player& player2, bool isP1Turn, const boardState& state, bool& isMadMove) {
  const Player& player = isP1Turn ? player1 : player2;
  if (player.madness > 0) {
    std::uniform_int_distribution<int> dist(0, 99);
    if (dist(rng) < player.madness) {
      isMadMove = true;
      return kribu::player::select_random(state);
    }
  }
  return player.select(state);
}

/**
 * @brief Helper to update telemetry performance statistics (visited nodes and CPU time).
 */
constexpr void update_perf_stats(GamePerf& perf, bool isP1Turn, f64 elapsed) noexcept {
  if (isP1Turn) {
    perf.p1.cpuTimeSeconds += elapsed;
  } else {
    perf.p2.cpuTimeSeconds += elapsed;
  }
}

/**
 * @brief Helper to advance turn state after a move.
 */
constexpr void advance_turn_state(boardState& state, bool& isP1Turn, const boardState& nextState) noexcept {
  if (nextState.activeCaptureIdx == -1) {
    state = flip_board(nextState);
    isP1Turn = !isP1Turn;
  } else {
    state = nextState;
  }
}

/**
 * @brief Helper function to execute a single move and record metrics.
 * @return The move ID played, or -1 if invalid or error.
 */
inline int execute_move(boardState& state,
                        bool& isP1Turn,
                        const Player& player1,
                        const Player& player2,
                        GamePerf& perf,
                        bool forceRandom,
                        bool& isMadMove) {
  int moveId = -1;

  auto startTime = std::chrono::high_resolution_clock::now();
  if (forceRandom) {
    moveId = kribu::player::select_random(state);
  } else {
    moveId = get_player_move(player1, player2, isP1Turn, state, isMadMove);
  }
  auto endTime = std::chrono::high_resolution_clock::now();
  f64 elapsed{0.0};
  elapsed = std::chrono::duration<f64>(endTime - startTime).count();

  update_perf_stats(perf, isP1Turn, elapsed);

  if (moveId == -1 || !is_valid(state, moveId)) {
    return -1;
  }

  advance_turn_state(state, isP1Turn, apply_move(state, moveId));
  return moveId;
}

/**
 * @brief Helper to count the occurrences of a board state hash in game history.
 */
inline int count_repetitions(u64 hash, const std::vector<u64>& gameHistoryHashes) noexcept {
  int repetitions = 0;
  for (u64 prevHash : gameHistoryHashes) {
    if (prevHash == hash) {
      repetitions++;
    }
  }
  return repetitions;
}

/**
 * @brief Helper to record history hash.
 */
inline void record_history_hash(u64 hash, std::vector<u64>& gameHistoryHashes) noexcept {
  gameHistoryHashes.push_back(hash);
}

/**
 * @brief Helper to handle game over scenario, determining the winner and win reason.
 */
constexpr GameOutcome handle_game_over(GameStatus status, bool isP1Turn) noexcept {
  bool meWins = (status == GameStatus::ME_WINS_ELIMINATION || status == GameStatus::ME_WINS_STALEMATE);
  GameResult result = (meWins == isP1Turn) ? GameResult::P1_WINS : GameResult::P2_WINS;

  bool elimination = (status == GameStatus::ME_WINS_ELIMINATION || status == GameStatus::OPP_WINS_ELIMINATION);
  WinReason reason = elimination ? WinReason::ELIMINATION : WinReason::STALEMATE;

  return GameOutcome{.result = result, .reason = reason};
}

/**
 * @brief Helper to handle invalid move scenario.
 */
constexpr GameOutcome handle_invalid_move(bool isP1Turn) noexcept {
  GameResult res = isP1Turn ? GameResult::P2_WINS : GameResult::P1_WINS;
  return GameOutcome{.result = res, .reason = WinReason::INVALID_MOVE};
}

/**
 * @brief Helper to compute and set victory margin (difference in piece count).
 */
inline void set_win_margin(GameOutcome& outcome, const boardState& state, bool isP1Turn) noexcept {
  if (outcome.result == GameResult::DRAW) {
    outcome.winMargin = 0;
    return;
  }
  int p1Pieces = isP1Turn ? piece_count(state.me) : piece_count(state.opp);
  int p2Pieces = isP1Turn ? piece_count(state.opp) : piece_count(state.me);

  if (outcome.result == GameResult::P1_WINS) {
    outcome.winMargin = p1Pieces - p2Pieces;
  } else {
    outcome.winMargin = p2Pieces - p1Pieces;
  }
}

/**
 * @brief Computes the number of consecutive forced random turns to play to escape a repetition loop.
 */
constexpr int calculate_forced_random_turns(int repetitions) noexcept {
  // Option 1: Linear scaling
  // return (repetitions - 1) * 2;

  // Option 2: Quadratic scaling
  return (repetitions - 1) * (repetitions - 1) * 2;

  // Option 3: Exponential scaling (power of 2)
  // return (1 << (repetitions - 1));

  // Option 4: Constant scaling
  return 4;
}

/**
 * @brief Checks if the number of repetitions has reached the limit.
 * @param repetitions The current repetition count.
 * @return True if the limit is reached, false otherwise.
 */
inline bool is_repetition_limit_reached(int repetitions) noexcept {
  return repetitions >= REPETITION_LIMIT - 1;
}

/**
 * @brief Handles the repetition limits and sets consecutive random turns if repetition occurs.
 * @param state The current board state.
 * @param gameHistoryHashes Record of Zobrist hashes in the current game.
 * @param forcedRandomTurnsLeft Reference to the counter of forced random turns.
 * @param outcome Reference to the GameOutcome to set in case of repetition limit.
 * @return True if repetition limit was reached and the game should end, false otherwise.
 */
inline bool handle_repetition(const boardState& state,
                              std::vector<u64>& gameHistoryHashes,
                              i32& forcedRandomTurnsLeft,
                              GameOutcome& outcome) noexcept {
  i32 repetitions{0};
  repetitions = count_repetitions(state.hash, gameHistoryHashes);
  if (is_repetition_limit_reached(repetitions)) {
    if (!ALLOW_REPETITION) {
      outcome = GameOutcome{.result = GameResult::DRAW, .reason = WinReason::REPETITION};
      return true;
    }
  }

  if (repetitions >= 2 && ALLOW_REPETITION) {
    forcedRandomTurnsLeft = std::max(forcedRandomTurnsLeft, calculate_forced_random_turns(repetitions));
  }
  record_history_hash(state.hash, gameHistoryHashes);
  return false;
}

/**
 * @brief Executes a single turn of the game, including player move choice,
 * validation, and state progression.
 * @param state The current board state.
 * @param isP1Turn Reference to flag tracking if it is Player 1's turn.
 * @param player1 The first player.
 * @param player2 The second player.
 * @param perf Reference to performance accumulator.
 * @param forcedRandomTurnsLeft Reference to remaining consecutive random turns.
 * @param history Vector tracking the game turn history.
 * @param outcome Reference to the output game outcome if the game finishes this turn.
 * @return True if the game is completed on this turn, false if it continues.
 */
inline bool play_single_turn(boardState& state,
                             bool& isP1Turn,
                             const Player& player1,
                             const Player& player2,
                             GamePerf& perf,
                             i32& forcedRandomTurnsLeft,
                             std::vector<TurnRecord>& history,
                             GameOutcome& outcome) {
  const GameStatus status = get_game_status(state);
  if (status != GameStatus::ONGOING) {
    outcome = handle_game_over(status, isP1Turn);
    set_win_margin(outcome, state, isP1Turn);
    return true;
  }

  TurnRecord record{.state = state, .isP1Turn = isP1Turn, .chosenMove = -1, .playerPlayed = ""};

  const bool forceRandom = (forcedRandomTurnsLeft > 0);
  if (forcedRandomTurnsLeft > 0) {
    forcedRandomTurnsLeft--;
  }

  const bool playerWasP1 = isP1Turn;
  const std::string_view playerNameBeforeMove = playerWasP1 ? player1.name : player2.name;

  bool isMadMove = false;
  const i32 moveId = execute_move(state, isP1Turn, player1, player2, perf, forceRandom, isMadMove);
  if (moveId == -1) {
    outcome = handle_invalid_move(isP1Turn);
    set_win_margin(outcome, state, isP1Turn);
    return true;
  }

  record.chosenMove = moveId;
  if (forceRandom) {
    record.playerPlayed = "ForcedRandom";
  } else if (isMadMove) {
    record.playerPlayed = "MadPlayer";
  } else {
    record.playerPlayed = playerNameBeforeMove;
  }
  history.push_back(record);
  return false;
}

/**
 * @brief Plays a single game between two players, saving turn history.
 * @param player1 The first player.
 * @param player2 The second player.
 * @param perf Accumulator for timing and node counts for both players.
 * @param maxTurns Maximum number of turns (plies) allowed before a draw is declared.
 * @param history Output vector where each turn's state/decision is recorded.
 * @return The outcome of the game (winner/draw and margin).
 */
inline GameOutcome play_single_game(
    const Player& player1, const Player& player2, GamePerf& perf, i32 maxTurns, std::vector<TurnRecord>& history) {
  boardState state = INITIAL_STATE;
  i32 turnCount = 0;
  bool isP1Turn = true;
  i32 forcedRandomTurnsLeft = 0;

  std::vector<u64> gameHistoryHashes;
  gameHistoryHashes.reserve(static_cast<usize>(maxTurns));

  history.clear();
  history.reserve(static_cast<usize>(maxTurns));

  while (turnCount < maxTurns) {
    GameOutcome outcome;
    if (handle_repetition(state, gameHistoryHashes, forcedRandomTurnsLeft, outcome)) {
      return outcome;
    }

    if (play_single_turn(state, isP1Turn, player1, player2, perf, forcedRandomTurnsLeft, history, outcome)) {
      return outcome;
    }
    turnCount++;
  }

  return GameOutcome{.result = GameResult::DRAW, .reason = WinReason::DRAW_MAX_TURNS};
}

/**
 * @brief Runs a tournament matchup between two players in a multithreaded fashion.
 */
inline MatchStats run_matchup_multithreaded(const Player& player1,  // NOLINT(readability-function-cognitive-complexity)
                                            const Player& player2,
                                            const TournamentConfig& config) {
  MatchStats stats{.player1Name = player1.name, .player2Name = player2.name};
  std::atomic<int> nextGameIdx{0};
  std::mutex statsMutex;

  auto worker = [&]() {  // NOLINT(readability-function-cognitive-complexity)
    LocalWins localP1;
    LocalWins localP2;
    int localDraws = 0;
    f64 localP1Time = 0.0;
    f64 localP2Time = 0.0;
    u64 localTotalTurns = 0;

    while (true) {
      if (config.abortRequested && config.abortRequested->load(std::memory_order_relaxed)) {
        break;
      }
      int gameIdx = nextGameIdx.fetch_add(1, std::memory_order_relaxed);
      if (gameIdx >= config.totalGames) {
        break;
      }

      int globalId = config.globalGameId ? config.globalGameId->fetch_add(1, std::memory_order_relaxed) : gameIdx;
      GamePerf perf;
      std::vector<TurnRecord> gameHistory;

      GameOutcome outcome = play_single_game(player1, player2, perf, config.maxTurns, gameHistory);

      save_game(player1.name, player2.name, globalId, outcome, gameHistory);

      if (outcome.result == GameResult::P1_WINS) {
        record_outcome(outcome, localP1);
        if (config.p1Wins) {
          config.p1Wins->fetch_add(1, std::memory_order_relaxed);
        }
      } else if (outcome.result == GameResult::P2_WINS) {
        record_outcome(outcome, localP2);
        if (config.p2Wins) {
          config.p2Wins->fetch_add(1, std::memory_order_relaxed);
        }
      } else {
        localDraws++;
        if (config.draws) {
          config.draws->fetch_add(1, std::memory_order_relaxed);
        }
      }

      localP1Time += perf.p1.cpuTimeSeconds;
      localP2Time += perf.p2.cpuTimeSeconds;
      localTotalTurns += gameHistory.size();

      if (config.completedGames) {
        config.completedGames->fetch_add(1, std::memory_order_relaxed);
      }
    }

    std::scoped_lock lock(statsMutex);
    stats.p1Wins += localP1.wins;
    stats.p2Wins += localP2.wins;
    stats.draws += localDraws;
    stats.p1EliminationWins += localP1.elimination;
    stats.p1StalemateWins += localP1.stalemate;
    stats.p1InvalidMoveWins += localP1.invalidMove;
    stats.p2EliminationWins += localP2.elimination;
    stats.p2StalemateWins += localP2.stalemate;
    stats.p2InvalidMoveWins += localP2.invalidMove;
    stats.p1TotalCpuTimeSeconds += localP1Time;
    stats.p2TotalCpuTimeSeconds += localP2Time;
    stats.totalTurns += localTotalTurns;
  };

  std::vector<std::jthread> threads;
  threads.reserve(static_cast<usize>(config.threadCount));
  for (int i = 0; i < config.threadCount; ++i) {
    threads.emplace_back(worker);
  }

  // std::jthread automatically joins on destruction when leaving scope.
  // Explicitly clear vector to ensure all threads finish before return.
  threads.clear();

  return stats;
}

}  // namespace kribu::benchmark
