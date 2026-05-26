/**
 * @file benchmark.hpp
 * @brief Multithreaded benchmarking and matchmaking framework for Sholo Guti engine.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include "board.hpp"
#include "rules.hpp"
#include "types.hpp"

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
  MoveList possibleMoves;
};

struct MatchConfig {
  std::string_view player1Name;
  std::string_view player2Name;
  int games = 10;
  int maxTurns = 5000;
};

/**
 * @enum GameResult
 * @brief Represents the outcome of a single match.
 */
enum class GameResult : std::uint8_t { P1_WINS, P2_WINS, DRAW };

/**
 * @enum WinReason
 * @brief Represents the reason why a player won or the game ended.
 */
enum class WinReason : std::uint8_t { ELIMINATION, STALEMATE, INVALID_MOVE, DRAW_MAX_TURNS };

/**
 * @struct GameOutcome
 * @brief Bundles game result and the reason.
 */
struct GameOutcome {
  GameResult result;
  WinReason reason;
};

/**
 * @struct Player
 * @brief Abstract representation of an AI player in the tournament.
 */
struct Player {
  std::string_view name;
  int (*select)(const boardState&, u64&) = nullptr;
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
  u64 p1TotalNodes = 0;
  u64 p2TotalNodes = 0;
  double p1TotalCpuTimeSeconds = 0.0;
  double p2TotalCpuTimeSeconds = 0.0;
  u64 totalTurns = 0;
};

/**
 * @struct PlayerPerformance
 * @brief Performance telemetry for a player during a game.
 */
struct PlayerPerformance {
  u64 nodes = 0;
  double cpuTimeSeconds = 0.0;
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
  int maxTurns = 5000;
  int threadCount = 0;
  std::atomic<int>* completedGames = nullptr;
};

/**
 * @brief Helper function to execute a single move and record metrics.
 * @return The move ID played, or -1 if invalid or error.
 */
constexpr int execute_move(
    boardState& state, bool& isP1Turn, const Player& player1, const Player& player2, GamePerf& perf) {
  int moveId = -1;
  u64 moveNodes = 0;
  double elapsed = 0.0;

  if (std::is_constant_evaluated()) {
    if (isP1Turn) {
      moveId = player1.select(state, moveNodes);
      perf.p1.nodes += moveNodes;
    } else {
      moveId = player2.select(state, moveNodes);
      perf.p2.nodes += moveNodes;
    }
  } else {
    auto startTime = std::chrono::high_resolution_clock::now();
    if (isP1Turn) {
      moveId = player1.select(state, moveNodes);
      perf.p1.nodes += moveNodes;
    } else {
      moveId = player2.select(state, moveNodes);
      perf.p2.nodes += moveNodes;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double>(endTime - startTime).count();
  }

  if (isP1Turn) {
    perf.p1.cpuTimeSeconds += elapsed;
  } else {
    perf.p2.cpuTimeSeconds += elapsed;
  }

  if (moveId == -1 || !is_valid(state, moveId)) {
    return -1;
  }

  boardState nextState = apply_move(state, moveId);
  if (nextState.activeCaptureIdx == -1) {
    state = flip_board(nextState);
    isP1Turn = !isP1Turn;
  } else {
    state = nextState;
  }
  return moveId;
}

/**
 * @brief Helper to determine the game winner based on the active player and game status.
 */
[[nodiscard]] constexpr GameResult determine_winner(GameStatus status, bool isP1Turn) noexcept {
  if (status == GameStatus::ME_WINS_ELIMINATION || status == GameStatus::ME_WINS_STALEMATE) {
    return isP1Turn ? GameResult::P1_WINS : GameResult::P2_WINS;
  }
  return isP1Turn ? GameResult::P2_WINS : GameResult::P1_WINS;
}

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
 * @brief Plays a single game between two players, saving turn history.
 */
constexpr GameOutcome play_single_game(const Player& player1,
                                       const Player& player2,
                                       bool p1StartsFirst,
                                       GamePerf& perf,
                                       int maxTurns,
                                       std::vector<TurnRecord>& history) {
  boardState state = INITIAL_STATE;
  int turnCount = 0;
  bool isP1Turn = p1StartsFirst;

  history.clear();
  history.reserve(static_cast<std::size_t>(maxTurns));

  while (turnCount < maxTurns) {
    GameStatus status = get_game_status(state);
    if (status != GameStatus::ONGOING) {
      GameResult res = determine_winner(status, isP1Turn);
      WinReason reason = (status == GameStatus::ME_WINS_ELIMINATION || status == GameStatus::OPP_WINS_ELIMINATION)
                             ? WinReason::ELIMINATION
                             : WinReason::STALEMATE;
      return GameOutcome{.result = res, .reason = reason};
    }

    TurnRecord record;
    record.state = state;
    record.isP1Turn = isP1Turn;
    record.possibleMoves = all_possible_moves(state);

    int moveId = execute_move(state, isP1Turn, player1, player2, perf);
    if (moveId == -1) {
      GameResult res = isP1Turn ? GameResult::P2_WINS : GameResult::P1_WINS;
      return GameOutcome{.result = res, .reason = WinReason::INVALID_MOVE};
    }

    record.chosenMove = moveId;
    history.push_back(record);
    turnCount++;
  }

  return GameOutcome{.result = GameResult::DRAW, .reason = WinReason::DRAW_MAX_TURNS};
}

/**
 * @brief Plays a single game between two players (without history tracking).
 */
constexpr GameOutcome play_single_game(
    const Player& player1, const Player& player2, bool p1StartsFirst, GamePerf& perf, int maxTurns) {
  std::vector<TurnRecord> dummy;
  return play_single_game(player1, player2, p1StartsFirst, perf, maxTurns, dummy);
}

/**
 * @brief Saves a completed game dataset directly to a Parquet file.
 * @details Implemented in benchmark_main.cpp using Apache Arrow.
 */
void save_game_parquet(std::string_view player1Name,
                       std::string_view player2Name,
                       int gameIdx,
                       const GameOutcome& outcome,
                       const std::vector<TurnRecord>& history);

/**
 * @brief Runs a tournament matchup between two players in a multithreaded fashion.
 */
// TODO: make sure to add the last wining board state
inline MatchStats run_matchup_multithreaded(const Player& player1,
                                            const Player& player2,
                                            const TournamentConfig& config) {
  MatchStats stats{.player1Name = player1.name, .player2Name = player2.name};
  std::atomic<int> nextGameIdx{0};
  std::mutex statsMutex;

  auto worker = [&]() {
    LocalWins localP1;
    LocalWins localP2;
    int localDraws = 0;
    u64 localP1Nodes = 0;
    u64 localP2Nodes = 0;
    double localP1Time = 0.0;
    double localP2Time = 0.0;
    u64 localTotalTurns = 0;

    while (true) {
      int gameIdx = nextGameIdx.fetch_add(1, std::memory_order_relaxed);
      if (gameIdx >= config.totalGames) {
        break;
      }

      bool p1Starts = (gameIdx % 2 == 0);
      GamePerf perf;
      std::vector<TurnRecord> gameHistory;

      GameOutcome outcome = play_single_game(player1, player2, p1Starts, perf, config.maxTurns, gameHistory);

      save_game_parquet(player1.name, player2.name, gameIdx, outcome, gameHistory);

      if (outcome.result == GameResult::P1_WINS) {
        record_outcome(outcome, localP1);
      } else if (outcome.result == GameResult::P2_WINS) {
        record_outcome(outcome, localP2);
      } else {
        localDraws++;
      }

      localP1Nodes += perf.p1.nodes;
      localP2Nodes += perf.p2.nodes;
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
    stats.p1TotalNodes += localP1Nodes;
    stats.p2TotalNodes += localP2Nodes;
    stats.p1TotalCpuTimeSeconds += localP1Time;
    stats.p2TotalCpuTimeSeconds += localP2Time;
    stats.totalTurns += localTotalTurns;
  };

  std::vector<std::jthread> threads;
  threads.reserve(static_cast<std::size_t>(config.threadCount));
  for (int i = 0; i < config.threadCount; ++i) {
    threads.emplace_back(worker);
  }

  // std::jthread automatically joins on destruction when leaving scope.
  // Explicitly clear vector to ensure all threads finish before return.
  threads.clear();

  return stats;
}

}  // namespace kribu::benchmark
