/**
 * @file benchmark_main.cpp
 * @brief Tournament runner for Sholo Guti engine benchmarking with duckdb.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <duckdb.hpp>  // NOLINT(misc-include-cleaner)
#include <exception>
#include <filesystem>
#include <fstream>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <tabulate/color.hpp>
#include <tabulate/font_align.hpp>
#include <tabulate/font_style.hpp>
#include <tabulate/table.hpp>
#include <thread>
#include <utility>
#include <vector>

#include "benchmark.hpp"
#include "config.hpp"
#include "kribu/types.hpp"
#include "players.hpp"  // NOLINT(misc-include-cleaner)

using namespace kribu::board;
using namespace kribu::sholoGuti;
using namespace kribu::benchmark;

namespace kribu::benchmark {

namespace {

std::atomic<int> globalGameId{1};

/**
 * @struct CpuTimes
 * @brief Holds CPU idle and total times.
 */
struct CpuTimes {
  unsigned long long idle = 0;   ///< Idle time ticks.
  unsigned long long total = 0;  ///< Total time ticks.
};

/**
 * @brief Reads the current CPU times from /proc/stat.
 * @return CpuTimes struct containing current cpu tick counts.
 */
CpuTimes get_cpu_times() {
  std::ifstream file("/proc/stat");
  std::string line;
  if (std::getline(file, line)) {
    std::istringstream ssInput(line);
    std::string cpu;
    ssInput >> cpu;
    if (cpu == "cpu") {
      unsigned long long cpuUser{0};
      unsigned long long cpuNice{0};
      unsigned long long cpuSystem{0};
      unsigned long long cpuIdle{0};
      unsigned long long cpuIowait{0};
      unsigned long long cpuIrq{0};
      unsigned long long cpuSoftirq{0};
      unsigned long long cpuSteal{0};
      if (ssInput >> cpuUser >> cpuNice >> cpuSystem >> cpuIdle >> cpuIowait >> cpuIrq >> cpuSoftirq >> cpuSteal) {
        CpuTimes times;
        times.idle = cpuIdle + cpuIowait;
        times.total = cpuUser + cpuNice + cpuSystem + cpuIdle + cpuIowait + cpuIrq + cpuSoftirq + cpuSteal;
        return times;
      }
    }
  }
  return {};
}

/**
 * @brief Computes CPU usage fraction between two measurements.
 * @param prev The previous measurement.
 * @param curr The current measurement.
 * @return CPU usage fraction (0.0 to 1.0).
 */
double calculate_cpu_usage(const CpuTimes& prev, const CpuTimes& curr) {
  unsigned long long totalDiff = curr.total - prev.total;
  unsigned long long idleDiff = curr.idle - prev.idle;
  if (totalDiff == 0) {
    return 0.0;
  }
  return static_cast<double>(totalDiff - idleDiff) / static_cast<double>(totalDiff);
}

/**
 * @brief Computes the system RAM usage fraction using /proc/meminfo.
 * @return RAM usage fraction (0.0 to 1.0).
 */
double get_ram_usage_fraction() {
  std::ifstream file("/proc/meminfo");
  std::string line;
  double memTotal = 0.0;
  double memAvailable = 0.0;
  while (std::getline(file, line)) {
    if (line.starts_with("MemTotal:")) {
      std::istringstream ssMemTotal(line.substr(9));
      ssMemTotal >> memTotal;
    } else if (line.starts_with("MemAvailable:")) {
      std::istringstream ssMemAvailable(line.substr(13));
      ssMemAvailable >> memAvailable;
    }
  }
  if (memTotal > 0.0) {
    return (memTotal - memAvailable) / memTotal;
  }
  return 0.0;
}

/**
 * @brief Converts a WinReason to its string representation.
 * @param reason The win reason enum value.
 * @return String representation ("ELIMINATION", "STALEMATE", "INVALID_MOVE", or "DRAW_MAX_TURNS").
 */
std::string reason_to_string(WinReason reason) {
  if (reason == WinReason::ELIMINATION) {
    return "ELIMINATION";
  }
  if (reason == WinReason::STALEMATE) {
    return "STALEMATE";
  }
  if (reason == WinReason::INVALID_MOVE) {
    return "INVALID_MOVE";
  }
  if (reason == WinReason::REPETITION) {
    return "REPETITION";
  }
  return "DRAW_MAX_TURNS";
}

struct CompletedGame {
  std::string p1Name;
  std::string p2Name;
  int gameId;
  GameOutcome outcome;
  std::vector<TurnRecord> history;
};

/**
 * @class GameWriterQueue
 * @brief Thread-safe blocking queue for completed games waiting to be written to DuckDB.
 */
class GameWriterQueue {
 private:
  /**
   * @brief Internal queue storing completed games.
   */
  std::queue<CompletedGame> queue_;

  /**
   * @brief Mutex protecting queue access.
   */
  std::mutex mutex_;

  /**
   * @brief Condition variable for consumer blocking.
   */
  std::condition_variable cv_;

  /**
   * @brief Flag indicating the tournament is completed and no more games will be pushed.
   */
  bool done_ = false;

 public:
  /**
   * @brief Pushes a completed game into the queue and notifies the consumer.
   * @param game Rvalue reference of the completed game.
   */
  void push(CompletedGame&& game) {
    {
      std::scoped_lock lock(mutex_);
      queue_.push(std::move(game));
    }
    cv_.notify_one();
  }

  /**
   * @brief Pops a completed game from the queue, blocking if empty until notified or done.
   * @param game Output reference where the popped game will be stored.
   * @return True if a game was popped successfully, false if the queue is empty and done.
   */
  bool pop(CompletedGame& game) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() || done_; });
    if (queue_.empty()) {
      return false;
    }
    game = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  /**
   * @brief Signals that no more games will be pushed, waking up any blocked pop callers.
   */
  void set_done() {
    {
      std::scoped_lock lock(mutex_);
      done_ = true;
    }
    cv_.notify_all();
  }
};

namespace {
/**
 * @brief Global queue instance for DuckDB writing.
 */
GameWriterQueue gameQueue;
}  // namespace

/**
 * @brief Initializes the DuckDB database schema and creates games, players, and turns tables.
 * @param con Reference to the DuckDB connection.
 */
void initialize_duckdb(duckdb::Connection& con) {  // NOLINT(misc-include-cleaner)
  con.Query(
      "CREATE TABLE IF NOT EXISTS players ("
      "name        VARCHAR PRIMARY KEY,"
      "player_type VARCHAR,"
      "depth       INTEGER,"
      "madness     INTEGER"
      ")");

  con.Query(
      "CREATE TABLE IF NOT EXISTS games ("
      "game_id             INTEGER PRIMARY KEY,"
      "p1_name             VARCHAR,"
      "p2_name             VARCHAR,"
      "outcome             TINYINT,"
      "reason              VARCHAR,"
      "total_turns         INTEGER,"
      "win_margin          INTEGER,"
      "forced_random_turns INTEGER,"
      "mad_turns           INTEGER"
      ")");

  con.Query(
      "CREATE TABLE IF NOT EXISTS turns ("
      "game_id             INTEGER,"
      "turn_idx            INTEGER,"
      "me                  BIGINT,"
      "opp                 BIGINT,"
      "active_capture_idx  TINYINT,"
      "is_p1_turn          BOOLEAN,"
      "chosen_move         SMALLINT,"
      "player_played       VARCHAR,"
      "PRIMARY KEY (game_id, turn_idx),"
      "FOREIGN KEY (player_played) REFERENCES players(name)"
      ")");
}

/**
 * @brief Checks if a game record with a given game ID already exists in the database.
 * @param con Reference to the DuckDB connection.
 * @param gameId The game ID to check.
 * @return True if the game exists, false otherwise.
 */
bool game_exists(duckdb::Connection& con, int gameId) {
  auto res = con.Query("SELECT 1 FROM games WHERE game_id = " + std::to_string(gameId));
  return res->RowCount() > 0;
}

/**
 * @brief Registers players in the database players table.
 * @param con Reference to the DuckDB connection.
 * @param players Map of players to insert.
 */
void insert_players(duckdb::Connection& con, const std::map<std::string, Player>& players) {
  con.Query(
      "INSERT INTO players (name, player_type, depth, madness) VALUES ('ForcedRandom', 'forced_random', 0, 0) ON "
      "CONFLICT (name) DO NOTHING");
  con.Query(
      "INSERT INTO players (name, player_type, depth, madness) VALUES ('MadPlayer', 'mad_player', 0, 0) ON CONFLICT "
      "(name) DO NOTHING");

  for (const auto& [name, player] : players) {
    auto prep = con.Prepare(
        "INSERT INTO players (name, player_type, depth, madness) VALUES (?, ?, ?, ?) ON CONFLICT (name) DO NOTHING");
    prep->Execute(name, std::string(player.type), player.depth, player.madness);
  }
}

/**
 * @brief The database writer thread worker function.
 * @details Dequeues completed games from the GameWriterQueue and writes them to the DuckDB database.
 */
void db_writer_thread_func() {
  duckdb::DuckDB duckDb("benchmark/dataset.duckdb");  // NOLINT(misc-include-cleaner)
  duckdb::Connection con(duckDb);                     // NOLINT(misc-include-cleaner)
  initialize_duckdb(con);

  CompletedGame game;
  while (gameQueue.pop(game)) {
    if (game_exists(con, game.gameId)) {
      continue;
    }

    int outcomeInt = 2;
    if (game.outcome.result == GameResult::P1_WINS) {
      outcomeInt = 0;
    } else if (game.outcome.result == GameResult::P2_WINS) {
      outcomeInt = 1;
    }

    std::string reasonStr = reason_to_string(game.outcome.reason);

    int madTurnsCount = 0;
    int forcedRandomTurnsCount = 0;
    for (const auto& rec : game.history) {
      if (rec.playerPlayed == "MadPlayer") {
        madTurnsCount++;
      } else if (rec.playerPlayed == "ForcedRandom") {
        forcedRandomTurnsCount++;
      }
    }

    {
      auto prep = con.Prepare(
          "INSERT INTO games (game_id, p1_name, p2_name, outcome, reason, total_turns, win_margin, "
          "forced_random_turns, mad_turns) "
          "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
          "ON CONFLICT (game_id) DO NOTHING");
      prep->Execute(game.gameId,
                    game.p1Name,
                    game.p2Name,
                    outcomeInt,
                    reasonStr,
                    static_cast<int>(game.history.size()),
                    game.outcome.winMargin,
                    forcedRandomTurnsCount,
                    madTurnsCount);
    }

    try {
      duckdb::Appender appender(con, "turns");  // NOLINT(misc-include-cleaner)
      for (size_t i = 0; i < game.history.size(); ++i) {
        const auto& turn = game.history[i];
        appender.BeginRow();
        appender.Append<int32_t>(game.gameId);
        appender.Append<int32_t>(static_cast<int32_t>(i));
        appender.Append<int64_t>(static_cast<int64_t>(turn.state.me));
        appender.Append<int64_t>(static_cast<int64_t>(turn.state.opp));
        appender.Append<int8_t>(static_cast<int8_t>(turn.state.activeCaptureIdx));
        appender.Append<bool>(turn.isP1Turn);
        appender.Append<int16_t>(static_cast<int16_t>(turn.chosenMove));
        appender.Append(std::string(turn.playerPlayed).c_str());
        appender.EndRow();
      }
      appender.Close();
    } catch (const std::exception& e) {
      std::cerr << "Error writing turns for game " << game.gameId << ": " << e.what() << "\n";
    }
  }
}

/**
 * @brief Retrieves the maximum game ID currently stored in the database.
 * @return The maximum game ID, or 0 if empty or database is uninitialized.
 */
int get_max_game_id() {
  try {
    duckdb::DuckDB duckDb("benchmark/dataset.duckdb");  // NOLINT(misc-include-cleaner)
    duckdb::Connection con(duckDb);                     // NOLINT(misc-include-cleaner)
    initialize_duckdb(con);
    auto res = con.Query("SELECT COALESCE(MAX(game_id), 0) FROM games");
    if (res->RowCount() > 0) {
      return res->GetValue(0, 0).GetValue<int32_t>();
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch)
  }
  return 0;
}

/**
 * @struct MatchupState
 * @brief Thread-safe progress and telemetry statistics for a single tournament matchup.
 */
struct MatchupState {
  std::string p1Name;                  ///< Name of Player 1.
  std::string p2Name;                  ///< Name of Player 2.
  int games = 0;                       ///< Total games to play.
  int maxTurns = 0;                    ///< Maximum turns per game.
  int initialCompletedGames = 0;       ///< Number of games completed before current run.
  std::atomic<int> completedGames{0};  ///< Atomic count of completed games.
  std::atomic<int> p1Wins{0};          ///< Atomic count of Player 1 wins.
  std::atomic<int> p2Wins{0};          ///< Atomic count of Player 2 wins.
  std::atomic<int> draws{0};           ///< Atomic count of draws.
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;  ///< Matchup start time.
  std::chrono::time_point<std::chrono::high_resolution_clock> endTime;    ///< Matchup end time.
  bool active = false;                                                    ///< True if matchup is currently running.
  bool completed = false;                                                 ///< True if matchup is fully completed.
  MatchStats stats;                                                       ///< Final matchup statistics.
  bool registered = true;  ///< True if both players are registered in players map.
};

/**
 * @brief Renders a single row in the matchup list panel.
 * @param match The matchup state.
 * @param index The matchup index.
 * @param isActive True if this matchup is currently active.
 * @return FTXUI Element representing the matchup row.
 */
ftxui::Element render_matchup_row(const MatchupState& match, int index, bool isActive) {
  using namespace ftxui;
  std::string statusStr = "[PENDING] ";
  Color statusColor = Color::GrayDark;

  if (match.completed) {
    if (!match.registered) {
      statusStr = "[SKIPPED] ";
      statusColor = Color::Yellow;
    } else {
      statusStr = "[DONE]    ";
      statusColor = Color::Green;
    }
  } else if (match.active) {
    statusStr = "[ACTIVE]  ";
    statusColor = Color::Cyan;
  }

  auto statusEl = bold(text(statusStr)) | color(statusColor);
  auto playersEl =
      hbox({text(match.p1Name) | color(Color::Cyan), text(" vs "), text(match.p2Name) | color(Color::Magenta)});

  std::string statsStr;
  Color statsColor = Color::GrayLight;
  if (match.completed && match.registered) {
    statsStr = "  (" + std::to_string(match.stats.p1Wins) + "-" + std::to_string(match.stats.p2Wins) + "-"
               + std::to_string(match.stats.draws) + " / " + std::to_string(match.games) + ")";
  } else if (match.active) {
    statsStr = "  (" + std::to_string(match.p1Wins.load()) + "-" + std::to_string(match.p2Wins.load()) + "-"
               + std::to_string(match.draws.load()) + " / " + std::to_string(match.games) + ")";
    statsColor = Color::Yellow;
  } else {
    statsStr = "  (0 / " + std::to_string(match.games) + ")";
  }
  Element statsEl = text(statsStr) | color(statsColor);

  Element rowEl =
      hbox({text(std::to_string(index + 1) + ". ") | color(Color::GrayLight), statusEl, playersEl, statsEl});

  if (isActive) {
    rowEl = bold(rowEl) | bgcolor(Color::Blue);
  }
  return rowEl;
}

/**
 * @brief Renders the active matchup panel with live telemetry and progress gauges.
 * @param match The active matchup state.
 * @return FTXUI Element representing the active panel.
 */
ftxui::Element render_active_panel(const MatchupState& match) {
  using namespace ftxui;
  int completed = match.completedGames.load();
  float progress = (match.games > 0) ? static_cast<float>(completed) / static_cast<float>(match.games) : 0.0F;

  auto now = std::chrono::high_resolution_clock::now();
  double elapsedSecs{0.0};
  elapsedSecs = std::chrono::duration<double>(now - match.startTime).count();
  double etaSecs = 0.0;
  int runCompleted = completed - match.initialCompletedGames;
  if (runCompleted > 0 && completed < match.games) {
    etaSecs = (elapsedSecs / runCompleted) * (match.games - completed);
  }

  auto progressGauge = gauge(progress) | color(Color::Cyan);
  std::string progressText = std::to_string(static_cast<int>(progress * 100.0F)) + "% (" + std::to_string(completed)
                             + "/" + std::to_string(match.games) + ")";

  auto timeInfo = hbox({text("Elapsed: ") | color(Color::GrayLight),
                        bold(text(std::to_string(static_cast<int>(elapsedSecs)) + "s")) | color(Color::White),
                        text(" | ETA: ") | color(Color::GrayLight),
                        bold(text(completed == match.games ? "0s" : std::to_string(static_cast<int>(etaSecs)) + "s"))
                            | color(Color::Yellow)});

  auto statsInfo =
      vbox({hbox({text("P1 Wins: ") | color(Color::Cyan), bold(text(std::to_string(match.p1Wins.load())))}),
            hbox({text("P2 Wins: ") | color(Color::Magenta), bold(text(std::to_string(match.p2Wins.load())))}),
            hbox({text("Draws:   ") | color(Color::Yellow), bold(text(std::to_string(match.draws.load())))})});

  return border(vbox(
             {center(bold(text("ACTIVE MATCHUP DETAILS"))) | color(Color::Yellow),
              separator(),
              hbox({text("Player 1 (P1): ") | color(Color::GrayLight), bold(text(match.p1Name)) | color(Color::Cyan)}),
              hbox({text("Player 2 (P2): ") | color(Color::GrayLight),
                    bold(text(match.p2Name)) | color(Color::Magenta)}),
              separator(),
              hbox({text("Progress: "), progressGauge, text(" " + progressText)}),
              separator(),
              timeInfo,
              separator(),
              statsInfo,
              filler()}))
         | color(Color::Blue) | flex;
}

}  // namespace

void save_game(std::string_view player1Name,
               std::string_view player2Name,
               int gameId,
               const GameOutcome& outcome,
               const std::vector<TurnRecord>& history) {
  CompletedGame game{.p1Name = std::string(player1Name),
                     .p2Name = std::string(player2Name),
                     .gameId = gameId,
                     .outcome = outcome,
                     .history = history};
  gameQueue.push(std::move(game));
}

}  // namespace kribu::benchmark

/**
 * @brief Main entrypoint running the compile-time matchup tournament.
 * @return 0 on success, non-zero on error.
 */
int main() {  // NOLINT(readability-function-cognitive-complexity)
  using namespace ftxui;

  try {
    std::filesystem::create_directories("benchmark");

    int lastId = kribu::benchmark::get_max_game_id();
    globalGameId.store(lastId + 1, std::memory_order_relaxed);

    const auto playerList = kribu::player::BENCHMARK_PLAYERS;
    std::map<std::string, Player> players;
    for (const auto& playerEntry : playerList) {
      players[std::string(playerEntry.name)] = playerEntry;
    }

    struct DbStats {
      int count = 0;
      int p1Wins = 0;
      int p2Wins = 0;
      int draws = 0;
      int p1Elim = 0;
      int p1Stalemate = 0;
      int p1Invalid = 0;
      int p2Elim = 0;
      int p2Stalemate = 0;
      int p2Invalid = 0;
      u64 totalTurns = 0;
    };
    std::map<std::pair<std::string, std::string>, DbStats> existingStats;

    // Initialize DuckDB schema and insert players first
    {
      duckdb::DuckDB duckDb("benchmark/dataset.duckdb");  // NOLINT(misc-include-cleaner)
      duckdb::Connection con(duckDb);                     // NOLINT(misc-include-cleaner)
      kribu::benchmark::initialize_duckdb(con);
      kribu::benchmark::insert_players(con, players);

      try {
        auto res = con.Query(
            "SELECT p1_name, p2_name, CAST(COUNT(*) AS INTEGER), "
            "CAST(SUM(CASE WHEN outcome = 0 THEN 1 ELSE 0 END) AS INTEGER), "
            "CAST(SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) AS INTEGER), "
            "CAST(SUM(CASE WHEN outcome = 2 THEN 1 ELSE 0 END) AS INTEGER), "
            "CAST(SUM(CASE WHEN outcome = 0 AND reason = 'ELIMINATION' THEN 1 ELSE 0 END) AS INTEGER), "
            "CAST(SUM(CASE WHEN outcome = 0 AND reason = 'STALEMATE' THEN 1 ELSE 0 END) AS INTEGER), "
            "CAST(SUM(CASE WHEN outcome = 0 AND reason = 'INVALID_MOVE' THEN 1 ELSE 0 END) AS INTEGER), "
            "CAST(SUM(CASE WHEN outcome = 1 AND reason = 'ELIMINATION' THEN 1 ELSE 0 END) AS INTEGER), "
            "CAST(SUM(CASE WHEN outcome = 1 AND reason = 'STALEMATE' THEN 1 ELSE 0 END) AS INTEGER), "
            "CAST(SUM(CASE WHEN outcome = 1 AND reason = 'INVALID_MOVE' THEN 1 ELSE 0 END) AS INTEGER), "
            "CAST(SUM(total_turns) AS BIGINT) "
            "FROM games GROUP BY p1_name, p2_name");

        for (size_t i = 0; i < res->RowCount(); ++i) {
          std::string player1Name = res->GetValue(0, i).GetValue<std::string>();
          std::string player2Name = res->GetValue(1, i).GetValue<std::string>();
          DbStats dbStats;
          dbStats.count = res->GetValue(2, i).GetValue<int32_t>();
          dbStats.p1Wins = res->GetValue(3, i).GetValue<int32_t>();
          dbStats.p2Wins = res->GetValue(4, i).GetValue<int32_t>();
          dbStats.draws = res->GetValue(5, i).GetValue<int32_t>();
          dbStats.p1Elim = res->GetValue(6, i).GetValue<int32_t>();
          dbStats.p1Stalemate = res->GetValue(7, i).GetValue<int32_t>();
          dbStats.p1Invalid = res->GetValue(8, i).GetValue<int32_t>();
          dbStats.p2Elim = res->GetValue(9, i).GetValue<int32_t>();
          dbStats.p2Stalemate = res->GetValue(10, i).GetValue<int32_t>();
          dbStats.p2Invalid = res->GetValue(11, i).GetValue<int32_t>();
          dbStats.totalTurns = res->GetValue(12, i).GetValue<int64_t>();
          existingStats[{player1Name, player2Name}] = dbStats;
        }
      } catch (...) {  // NOLINT(bugprone-empty-catch)
      }
    }

    // Start background DB writer thread
    std::thread dbWriterThread(kribu::benchmark::db_writer_thread_func);

    const auto matchups = BENCHMARK_MATCHUPS;

    // Convert matchups to states for FTXUI tracking
    std::vector<std::unique_ptr<MatchupState>> matchupsList;
    matchupsList.reserve(matchups.size());
    for (const auto& match : matchups) {
      auto state = std::make_unique<MatchupState>();
      state->p1Name = std::string(match.player1Name);
      state->p2Name = std::string(match.player2Name);
      state->games = match.games;
      state->maxTurns = match.maxTurns;

      if (!players.contains(state->p1Name) || !players.contains(state->p2Name)) {
        state->registered = false;
        state->completed = true;
      } else {
        auto existing = existingStats[{state->p1Name, state->p2Name}];
        state->completedGames.store(existing.count);
        state->initialCompletedGames = existing.count;
        state->p1Wins.store(existing.p1Wins);
        state->p2Wins.store(existing.p2Wins);
        state->draws.store(existing.draws);

        state->stats.p1Wins = existing.p1Wins;
        state->stats.p2Wins = existing.p2Wins;
        state->stats.draws = existing.draws;
        state->stats.p1EliminationWins = existing.p1Elim;
        state->stats.p1StalemateWins = existing.p1Stalemate;
        state->stats.p1InvalidMoveWins = existing.p1Invalid;
        state->stats.p2EliminationWins = existing.p2Elim;
        state->stats.p2StalemateWins = existing.p2Stalemate;
        state->stats.p2InvalidMoveWins = existing.p2Invalid;
        state->stats.totalTurns = existing.totalTurns;

        if (existing.count >= state->games) {
          state->completed = true;
        }
      }
      matchupsList.push_back(std::move(state));
    }

    tabulate::Table summaryTable;
    summaryTable.add_row({"Player 1",
                          "Player 2",
                          "P1 Wins (Elim/Stale/Inv)",
                          "P2 Wins (Elim/Stale/Inv)",
                          "Draws",
                          "Avg Turns",
                          "P1 Avg CPU",
                          "P2 Avg CPU"});

    auto screen = ScreenInteractive::Fullscreen();
    std::atomic<bool> tournamentCompleted{false};
    std::atomic<bool> abortRequested{false};

    // System Telemetry History
    std::vector<double> cpuHistory(128, 0.0);
    std::vector<double> ramHistory(128, 0.0);
    CpuTimes prevCpu = get_cpu_times();

    // background runner thread
    std::thread tournamentThread([&]() {
      for (const auto& matchPtr : matchupsList) {
        if (abortRequested.load(std::memory_order_relaxed)) {
          break;
        }
        auto& match = *matchPtr;
        if (!match.registered) {
          continue;
        }

        int existing = match.completedGames.load();
        int gamesToPlay = match.games > existing ? match.games - existing : 0;

        if (gamesToPlay > 0) {
          match.active = true;
          match.initialCompletedGames = match.completedGames.load();
          match.startTime = std::chrono::high_resolution_clock::now();

          const auto& playerFirst = players.at(match.p1Name);
          const auto& playerSecond = players.at(match.p2Name);

          auto runStats = run_matchup_multithreaded(playerFirst,
                                                    playerSecond,
                                                    TournamentConfig{.totalGames = gamesToPlay,
                                                                     .maxTurns = match.maxTurns,
                                                                     .threadCount = THREAD_COUNT,
                                                                     .completedGames = &match.completedGames,
                                                                     .globalGameId = &globalGameId,
                                                                     .p1Wins = &match.p1Wins,
                                                                     .p2Wins = &match.p2Wins,
                                                                     .draws = &match.draws,
                                                                     .abortRequested = &abortRequested});

          match.endTime = std::chrono::high_resolution_clock::now();
          match.active = false;
          match.completed = true;

          match.stats.p1Wins += runStats.p1Wins;
          match.stats.p2Wins += runStats.p2Wins;
          match.stats.draws += runStats.draws;
          match.stats.p1EliminationWins += runStats.p1EliminationWins;
          match.stats.p1StalemateWins += runStats.p1StalemateWins;
          match.stats.p1InvalidMoveWins += runStats.p1InvalidMoveWins;
          match.stats.p2EliminationWins += runStats.p2EliminationWins;
          match.stats.p2StalemateWins += runStats.p2StalemateWins;
          match.stats.p2InvalidMoveWins += runStats.p2InvalidMoveWins;
          match.stats.totalTurns += runStats.totalTurns;
          match.stats.p1TotalCpuTimeSeconds += runStats.p1TotalCpuTimeSeconds;
          match.stats.p2TotalCpuTimeSeconds += runStats.p2TotalCpuTimeSeconds;
        }

        int totalPlayed = std::max(1, match.completedGames.load());
        f64 p1AvgCpuMs = (match.stats.p1TotalCpuTimeSeconds * 1000.0) / totalPlayed;
        f64 p2AvgCpuMs = (match.stats.p2TotalCpuTimeSeconds * 1000.0) / totalPlayed;
        f64 avgTurns = static_cast<f64>(match.stats.totalTurns) / totalPlayed;

        summaryTable.add_row({match.p1Name,
                              match.p2Name,
                              std::to_string(match.stats.p1Wins) + " (" + std::to_string(match.stats.p1EliminationWins)
                                  + "/" + std::to_string(match.stats.p1StalemateWins) + "/"
                                  + std::to_string(match.stats.p1InvalidMoveWins) + ")",
                              std::to_string(match.stats.p2Wins) + " (" + std::to_string(match.stats.p2EliminationWins)
                                  + "/" + std::to_string(match.stats.p2StalemateWins) + "/"
                                  + std::to_string(match.stats.p2InvalidMoveWins) + ")",
                              std::to_string(match.stats.draws),
                              std::to_string(static_cast<int>(avgTurns)),
                              std::to_string(static_cast<int>(p1AvgCpuMs)) + " ms",
                              std::to_string(static_cast<int>(p2AvgCpuMs)) + " ms"});

        screen.PostEvent(Event::Custom);
      }
      tournamentCompleted.store(true);
      screen.PostEvent(Event::Custom);
    });

    // ui component definition
    auto renderer = Renderer([&]() {
      Elements listElements;
      int activeIndex = -1;
      int completedCount = 0;

      for (size_t i = 0; i < matchupsList.size(); ++i) {
        if (matchupsList[i]->active) {
          activeIndex = static_cast<int>(i);
        }
        if (matchupsList[i]->completed) {
          completedCount++;
        }
      }

      int termWidth = screen.dimx() <= 0 ? 80 : screen.dimx();
      int termHeight = screen.dimy() <= 0 ? 24 : screen.dimy();

      for (size_t i = 0; i < matchupsList.size(); ++i) {
        listElements.push_back(
            render_matchup_row(*matchupsList[i], static_cast<int>(i), std::cmp_equal(i, activeIndex)));
      }

      Element activePanel = border(center(text("No active matchup"))) | flex;
      if (activeIndex != -1) {
        activePanel = render_active_panel(*matchupsList[activeIndex]);
      } else if (std::cmp_equal(completedCount, matchupsList.size())) {
        activePanel = border(vbox({center(bold(text("TOURNAMENT COMPLETED!"))) | color(Color::Green),
                                   center(text("Finalizing summary table...")) | color(Color::GrayLight),
                                   filler()}))
                      | color(Color::Green) | flex;
      }

      float overallProgress = static_cast<float>(completedCount) / static_cast<float>(matchupsList.size());
      auto overallGauge = gauge(overallProgress) | color(Color::Green);
      std::string overallText =
          std::to_string(completedCount) + "/" + std::to_string(matchupsList.size()) + " matchups done";

      auto header = border(
          vbox({center(bold(text(" KRIBU ENGINE BENCHMARK TOURNAMENT "))) | color(Color::Yellow) | bgcolor(Color::Blue),
                center(hbox({text(" Threads: ") | color(Color::GrayLight),
                             bold(text(std::to_string(THREAD_COUNT))) | color(Color::White),
                             text(" | Global ID start: ") | color(Color::GrayLight),
                             bold(text(std::to_string(lastId + 1))) | color(Color::White)}))}));

      auto footer = border(vbox({hbox({text("Overall Tournament Progress: "), overallGauge, text(" " + overallText)}),
                                 center(text("Press 'q' twice to stop")) | color(Color::GrayDark)}))
                    | color(Color::Green);

      auto sysPanel = border(vbox({center(bold(text("SYSTEM TELEMETRY"))) | color(Color::Yellow),
                                   separator(),
                                   hbox({text("CPU Usage: ") | color(Color::GrayLight),
                                         gauge(static_cast<float>(cpuHistory.back())) | color(Color::Green) | flex,
                                         text(" " + std::to_string(static_cast<int>(cpuHistory.back() * 100.0)) + "%")
                                             | color(Color::Green)}),
                                   hbox({text("RAM Usage: ") | color(Color::GrayLight),
                                         gauge(static_cast<float>(ramHistory.back())) | color(Color::Yellow) | flex,
                                         text(" " + std::to_string(static_cast<int>(ramHistory.back() * 100.0)) + "%")
                                             | color(Color::Yellow)})}))
                      | color(Color::GrayDark);

      auto leftColumn = vbox({header, activePanel, sysPanel, footer}) | flex;

      size_t halfIndex = (listElements.size() + 1) / 2;
      Elements leftColElements;
      Elements rightColElements;
      for (size_t i = 0; i < listElements.size(); ++i) {
        if (i < halfIndex) {
          leftColElements.push_back(std::move(listElements[i]));
        } else {
          rightColElements.push_back(std::move(listElements[i]));
        }
      }

      auto rightColumn =
          border(vbox(
              {center(bold(text("Matchup Queue Window"))) | color(Color::Cyan),
               separator(),
               hbox({vbox(std::move(leftColElements)) | flex, separator(), vbox(std::move(rightColElements)) | flex})
                   | flex,
               filler()}))
          | color(Color::Cyan) | flex;

      return hbox({leftColumn, rightColumn}) | size(WIDTH, EQUAL, termWidth) | size(HEIGHT, EQUAL, termHeight);
    });

    int qPressCount = 0;
    auto component = renderer | CatchEvent([&](const Event& event) {
                       if (event.is_character() && event.character() == "q") {
                         qPressCount++;
                         if (qPressCount >= 2) {
                           abortRequested.store(true);
                           tournamentCompleted.store(true);
                           screen.ExitLoopClosure()();
                           return true;
                         }
                       } else {
                         qPressCount = 0;
                       }
                       return false;
                     });

    std::thread refreshThread([&]() {
      while (!tournamentCompleted.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        CpuTimes currCpu = get_cpu_times();
        double cpuUsage = calculate_cpu_usage(prevCpu, currCpu);
        prevCpu = currCpu;

        double ramUsage = get_ram_usage_fraction();

        std::ranges::rotate(cpuHistory, cpuHistory.begin() + 1);
        cpuHistory.back() = cpuUsage;

        std::ranges::rotate(ramHistory, ramHistory.begin() + 1);
        ramHistory.back() = ramUsage;

        screen.PostEvent(Event::Custom);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      screen.ExitLoopClosure()();
    });

    screen.Loop(component);

    if (tournamentThread.joinable()) {
      tournamentThread.join();
    }
    if (refreshThread.joinable()) {
      refreshThread.join();
    }

    // Stop and join the DuckDB writer thread
    kribu::benchmark::gameQueue.set_done();
    if (dbWriterThread.joinable()) {
      dbWriterThread.join();
    }

    // Format summaryTable for premium terminal DX
    summaryTable[0]
        .format()
        .font_style({tabulate::FontStyle::bold})
        .font_align(tabulate::FontAlign::center)
        .font_color(tabulate::Color::yellow);

    for (usize i = 1; i < summaryTable.size(); ++i) {
      summaryTable[i][0].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::cyan);
      summaryTable[i][1].format().font_style({tabulate::FontStyle::bold}).font_color(tabulate::Color::magenta);
    }

    std::cout << "\n=== Tournament Summary ===\n";
    std::cout << summaryTable << "\n";
  } catch (const std::exception& e) {
    std::cerr << "Exception occurred in main: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception occurred in main\n";
    return 1;
  }
  return 0;
}
