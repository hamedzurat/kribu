/**
 * @file benchmark_main.cpp
 * @brief Tournament runner for Sholo Guti engine benchmarking with Apache Arrow/Parquet.
 */

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>
#include <parquet/properties.h>
#include <sys/ioctl.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/canvas.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/terminal.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <tabulate/color.hpp>
#include <tabulate/font_align.hpp>
#include <tabulate/font_style.hpp>
#include <tabulate/table.hpp>
#include <thread>
#include <vector>

#include "config.hpp"
#include "kribu/benchmark.hpp"
#include "kribu/rules.hpp"
#include "players.hpp"

using namespace kribu::board;
using namespace kribu::sholoGuti;
using namespace kribu::benchmark;

namespace kribu::benchmark {

namespace {

std::mutex metadataMutex;
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
    std::istringstream ss(line);
    std::string cpu;
    ss >> cpu;
    if (cpu == "cpu") {
      unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
      if (ss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal) {
        CpuTimes times;
        times.idle = idle + iowait;
        times.total = user + nice + system + idle + iowait + irq + softirq + steal;
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
  return static_cast<double>(totalDiff - idleDiff) / totalDiff;
}

/**
 * @brief Computes the system RAM usage fraction.
 * @return RAM usage fraction (0.0 to 1.0).
 */
double get_ram_usage_fraction() {
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    double total = static_cast<double>(info.totalram) * info.mem_unit;
    double freeMem = static_cast<double>(info.freeram) * info.mem_unit;
    double buffered = static_cast<double>(info.bufferram) * info.mem_unit;
    double used = total - freeMem - buffered;
    return used / total;
  }
  return 0.0;
}

/**
 * @brief Renders a high-resolution line graph of telemetry history using ftxui::Canvas.
 * @param history Usage history data (fractions 0.0 to 1.0).
 * @param width Canvas width in characters.
 * @param height Canvas height in characters.
 * @param color Color of the graph line.
 * @return FTXUI Element containing the graph.
 */
ftxui::Element draw_graph(const std::vector<double>& history, int width, int height, ftxui::Color color) {
  using namespace ftxui;
  Canvas c(width, height);

  int ptWidth = width * 2;
  int ptHeight = height * 4;

  // Draw background grid lines (subtle dashed horizontal lines)
  for (int y = ptHeight / 4; y < ptHeight; y += ptHeight / 4) {
    for (int x = 0; x < ptWidth; x += 4) {
      c.DrawPoint(x, y, true, Color::GrayDark);
    }
  }

  // Draw the history plot line
  for (size_t i = 1; i < history.size(); ++i) {
    int x1 = static_cast<int>((i - 1) * static_cast<double>(ptWidth) / history.size());
    int y1 = ptHeight - 1 - static_cast<int>(history[i - 1] * (ptHeight - 1));
    int x2 = static_cast<int>(i * static_cast<double>(ptWidth) / history.size());
    int y2 = ptHeight - 1 - static_cast<int>(history[i] * (ptHeight - 1));

    x1 = std::max(0, std::min(x1, ptWidth - 1));
    y1 = std::max(0, std::min(y1, ptHeight - 1));
    x2 = std::max(0, std::min(x2, ptWidth - 1));
    y2 = std::max(0, std::min(y2, ptHeight - 1));

    c.DrawPointLine(x1, y1, x2, y2, color);
  }

  return canvas(std::move(c));
}

/**
 * @brief Reads the highest game ID from the CSV manifest, if it exists.
 */
int find_max_id_from_csv(const std::string& filepath) {
  std::ifstream infile(filepath);
  if (!infile.is_open()) {
    return 0;
  }
  std::string line;
  int maxId = 0;
  // skip header
  std::getline(infile, line);
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }
    auto pos = line.find(',');
    if (pos != std::string::npos) {
      try {
        int parsedId = std::stoi(line.substr(0, pos));
        maxId = std::max(parsedId, maxId);
      } catch (const std::exception& e) {
        std::cerr << "Warning: failed to parse manifest CSV id: " << e.what() << "\n";
      }
    }
  }
  return maxId;
}

/**
 * @brief Holds fully-finished Arrow arrays for a single game.
 */
struct GameArrays {
  std::shared_ptr<arrow::Array> me;
  std::shared_ptr<arrow::Array> opp;
  std::shared_ptr<arrow::Array> activeCaptureIdx;
  std::shared_ptr<arrow::Array> isP1Turn;
  std::shared_ptr<arrow::Array> chosenMove;
  std::shared_ptr<arrow::Array> possibleMoves;
  std::shared_ptr<arrow::Array> playerPlayed;
};

/**
 * @brief Appends all turn records into Arrow builders and finishes them.
 * @param history The game turn history.
 * @return Finished Arrow arrays for each column.
 */
GameArrays build_turn_arrays(  // NOLINT(readability-function-cognitive-complexity)
    const std::vector<TurnRecord>& history) {
  arrow::Int64Builder meBuilder;
  arrow::Int64Builder oppBuilder;
  arrow::Int8Builder activeCaptureIdxBuilder;
  arrow::BooleanBuilder isP1TurnBuilder;
  arrow::Int16Builder chosenMoveBuilder;
  auto valueBuilder = std::make_shared<arrow::Int16Builder>();
  arrow::ListBuilder possibleMovesBuilder(arrow::default_memory_pool(), valueBuilder);
  arrow::StringBuilder playerPlayedBuilder;

  for (const auto& turn : history) {
    PARQUET_THROW_NOT_OK(meBuilder.Append(static_cast<i64>(turn.state.me)));
    PARQUET_THROW_NOT_OK(oppBuilder.Append(static_cast<i64>(turn.state.opp)));
    PARQUET_THROW_NOT_OK(activeCaptureIdxBuilder.Append(static_cast<i8>(turn.state.activeCaptureIdx)));
    PARQUET_THROW_NOT_OK(isP1TurnBuilder.Append(turn.isP1Turn));
    PARQUET_THROW_NOT_OK(chosenMoveBuilder.Append(static_cast<i16>(turn.chosenMove)));
    PARQUET_THROW_NOT_OK(possibleMovesBuilder.Append());
    for (int i = 0; i < turn.possibleMoves.size(); ++i) {
      PARQUET_THROW_NOT_OK(valueBuilder->Append(static_cast<i16>(turn.possibleMoves.moves[i])));
    }
    PARQUET_THROW_NOT_OK(playerPlayedBuilder.Append(std::string(turn.playerPlayed)));
  }

  GameArrays arrays;
  PARQUET_THROW_NOT_OK(meBuilder.Finish(&arrays.me));
  PARQUET_THROW_NOT_OK(oppBuilder.Finish(&arrays.opp));
  PARQUET_THROW_NOT_OK(activeCaptureIdxBuilder.Finish(&arrays.activeCaptureIdx));
  PARQUET_THROW_NOT_OK(isP1TurnBuilder.Finish(&arrays.isP1Turn));
  PARQUET_THROW_NOT_OK(chosenMoveBuilder.Finish(&arrays.chosenMove));
  PARQUET_THROW_NOT_OK(possibleMovesBuilder.Finish(&arrays.possibleMoves));
  PARQUET_THROW_NOT_OK(playerPlayedBuilder.Finish(&arrays.playerPlayed));
  return arrays;
}

/**
 * @brief Bundles game-level metadata to avoid adjacent swappable parameters.
 */
struct GameTableParams {
  std::string_view player1Name;  ///< Name of Player 1.
  std::string_view player2Name;  ///< Name of Player 2.
  int gameId = 0;                ///< Global game ID.
  std::string winner;            ///< Name of the winner.
  std::string loser;             ///< Name of the loser.
  std::string outcomeStr;        ///< Outcome string (P1_WINS / P2_WINS / DRAW).
  std::string reasonStr;         ///< Reason string (ELIMINATION / STALEMATE / etc.).
  int totalTurns = 0;            ///< Total number of turns in the game.
  int winMargin = 0;             ///< Victory margin (pieces remaining difference).
  int forcedRandomTurns = 0;     ///< Number of forced random turns.
};

/**
 * @brief Assembles an Arrow Table from finished arrays and game-level schema metadata.
 * @param arrays Finished column arrays.
 * @param params Game-level metadata for schema annotation.
 * @return Constructed Arrow Table with schema metadata.
 */
std::shared_ptr<arrow::Table> build_game_table(const GameArrays& arrays, const GameTableParams& params) {
  auto metadata = std::make_shared<arrow::KeyValueMetadata>();
  metadata->Append("id", std::to_string(params.gameId));
  metadata->Append("p1Name", std::string(params.player1Name));
  metadata->Append("p2Name", std::string(params.player2Name));
  metadata->Append("winner", params.winner);
  metadata->Append("loser", params.loser);
  metadata->Append("outcome", params.outcomeStr);
  metadata->Append("reason", params.reasonStr);
  metadata->Append("totalTurns", std::to_string(params.totalTurns));
  metadata->Append("winMargin", std::to_string(params.winMargin));
  metadata->Append("forcedRandomTurns", std::to_string(params.forcedRandomTurns));

  auto schema = arrow::schema({arrow::field("me", arrow::int64()),
                               arrow::field("opp", arrow::int64()),
                               arrow::field("activeCaptureIdx", arrow::int8()),
                               arrow::field("isP1Turn", arrow::boolean()),
                               arrow::field("chosenMove", arrow::int16()),
                               arrow::field("possibleMoves", arrow::list(arrow::int16())),
                               arrow::field("playerPlayed", arrow::utf8())},
                              metadata);

  return arrow::Table::Make(schema,
                            {arrays.me,
                             arrays.opp,
                             arrays.activeCaptureIdx,
                             arrays.isP1Turn,
                             arrays.chosenMove,
                             arrays.possibleMoves,
                             arrays.playerPlayed});
}

/**
 * @brief Writes an Arrow Table to a Parquet file, preserving Arrow schema metadata in the footer.
 * @param table The Arrow Table to write.
 * @param filename The output file path.
 */
void write_parquet_file(const arrow::Table& table, const std::string& filename) {
  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(filename));
  auto arrowProps = parquet::ArrowWriterProperties::Builder().store_schema()->build();
  PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
      table, arrow::default_memory_pool(), outfile, 1024LL * 1024LL, parquet::default_writer_properties(), arrowProps));
}

/**
 * @brief Converts a GameResult to its string representation.
 * @param result The game result enum value.
 * @return String representation ("P1_WINS", "P2_WINS", or "DRAW").
 */
std::string outcome_to_string(GameResult result) {
  if (result == GameResult::P1_WINS) {
    return "P1_WINS";
  }
  if (result == GameResult::P2_WINS) {
    return "P2_WINS";
  }
  return "DRAW";
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

/**
 * @brief Prints the tournament matchup progress bar.
 * @param completed Number of completed games.
 * @param total Total number of games in the matchup.
 * @param p1Name Name of Player 1.
 * @param p2Name Name of Player 2.
 */
/**
 * @struct MatchupState
 * @brief Thread-safe progress and telemetry statistics for a single tournament matchup.
 */
struct MatchupState {
  std::string p1Name;                                                     ///< Name of Player 1.
  std::string p2Name;                                                     ///< Name of Player 2.
  int games = 0;                                                          ///< Total games to play.
  int maxTurns = 0;                                                       ///< Maximum turns per game.
  std::atomic<int> completedGames{0};                                     ///< Atomic count of completed games.
  std::atomic<int> p1Wins{0};                                             ///< Atomic count of Player 1 wins.
  std::atomic<int> p2Wins{0};                                             ///< Atomic count of Player 2 wins.
  std::atomic<int> draws{0};                                              ///< Atomic count of draws.
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

  Element statsEl = text("");
  if (match.completed && match.registered) {
    statsEl = text("  (" + std::to_string(match.stats.p1Wins) + "-" + std::to_string(match.stats.p2Wins) + "-"
                   + std::to_string(match.stats.draws) + ")")
              | color(Color::GrayLight);
  } else if (match.active) {
    statsEl = text("  (" + std::to_string(match.p1Wins.load()) + "-" + std::to_string(match.p2Wins.load()) + "-"
                   + std::to_string(match.draws.load()) + ")")
              | color(Color::Yellow);
  }

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
  float progress = (match.games > 0) ? static_cast<float>(completed) / match.games : 0.0F;

  auto now = std::chrono::high_resolution_clock::now();
  double elapsedSecs = std::chrono::duration<double>(now - match.startTime).count();
  double etaSecs = 0.0;
  if (completed > 0 && completed < match.games) {
    etaSecs = (elapsedSecs / completed) * (match.games - completed);
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

/**
 * @brief Saves a completed game dataset directly to a Parquet file, storing metadata in the schema footer.
 * @param player1Name Name of Player 1.
 * @param player2Name Name of Player 2.
 * @param gameId The unique game index in the tournament matchup.
 * @param outcome The result and win reason of the game.
 * @param history The sequence of moves played during the game.
 */
void save_game_parquet(std::string_view player1Name,
                       std::string_view player2Name,
                       int gameId,
                       const GameOutcome& outcome,
                       const std::vector<TurnRecord>& history) {
  std::string winner = "DRAW";
  std::string loser = "DRAW";
  std::string outcomeStr = outcome_to_string(outcome.result);

  if (outcomeStr == "P1_WINS") {
    winner = player1Name;
    loser = player2Name;
  } else if (outcomeStr == "P2_WINS") {
    winner = player2Name;
    loser = player1Name;
  }

  GameTableParams params{.player1Name = player1Name,
                         .player2Name = player2Name,
                         .gameId = gameId,
                         .winner = winner,
                         .loser = loser,
                         .outcomeStr = outcomeStr,
                         .reasonStr = reason_to_string(outcome.reason),
                         .totalTurns = static_cast<int>(history.size()),
                         .winMargin = outcome.winMargin,
                         .forcedRandomTurns = outcome.forcedRandomTurns};

  const GameArrays arrays = build_turn_arrays(history);
  const auto table = build_game_table(arrays, params);

  const std::string filename = "benchmark/" + std::to_string(gameId) + ".parquet";
  write_parquet_file(*table, filename);

  {
    std::scoped_lock lock(metadataMutex);
    std::string csvPath = "benchmark/manifest.csv";
    bool fileExists = std::filesystem::exists(csvPath);
    std::ofstream outfile(csvPath, std::ios::app);
    if (outfile.is_open()) {
      if (!fileExists) {
        outfile << "id,winner,loser,outcome,reason,totalTurns,winMargin,forcedRandomTurns\n";
      }
      outfile << gameId << "," << winner << "," << loser << "," << params.outcomeStr << "," << params.reasonStr << ","
              << params.totalTurns << "," << params.winMargin << "," << params.forcedRandomTurns << "\n";
    }
  }
}

}  // namespace kribu::benchmark

/**
 * @brief Main entrypoint running the compile-time matchup tournament.
 * @return 0 on success, non-zero on error.
 */
int main() {
  using namespace ftxui;

  kribu::maxRepetitions = kribu::benchmark::REPETITION_LIMIT;
  kribu::allowRepetition = kribu::benchmark::ALLOW_REPETITION;
  try {
    std::filesystem::create_directories("benchmark");

    int lastId = kribu::benchmark::find_max_id_from_csv("benchmark/manifest.csv");
    globalGameId.store(lastId + 1, std::memory_order_relaxed);

    const auto playerList = kribu::player::BENCHMARK_PLAYERS;
    std::map<std::string, Player> players;
    for (const auto& playerEntry : playerList) {
      players[std::string(playerEntry.name)] = playerEntry;
    }

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
      for (size_t i = 0; i < matchupsList.size(); ++i) {
        if (abortRequested.load(std::memory_order_relaxed)) {
          break;
        }
        auto& match = *matchupsList[i];
        if (!match.registered) {
          continue;
        }

        match.active = true;
        match.startTime = std::chrono::high_resolution_clock::now();

        const auto& playerFirst = players.at(match.p1Name);
        const auto& playerSecond = players.at(match.p2Name);

        match.stats = run_matchup_multithreaded(playerFirst,
                                                playerSecond,
                                                TournamentConfig{.totalGames = match.games,
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

        f64 p1AvgCpuMs = (match.stats.p1TotalCpuTimeSeconds * 1000.0) / match.games;
        f64 p2AvgCpuMs = (match.stats.p2TotalCpuTimeSeconds * 1000.0) / match.games;
        f64 avgTurns = static_cast<f64>(match.stats.totalTurns) / match.games;

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

      int graphHeight = std::max(6, std::min(12, termHeight / 4));

      for (size_t i = 0; i < matchupsList.size(); ++i) {
        listElements.push_back(
            render_matchup_row(*matchupsList[i], static_cast<int>(i), static_cast<int>(i) == activeIndex));
      }

      Element activePanel = border(center(text("No active matchup"))) | flex;
      if (activeIndex != -1) {
        activePanel = render_active_panel(*matchupsList[activeIndex]);
      } else if (completedCount == static_cast<int>(matchupsList.size())) {
        activePanel = border(vbox({center(bold(text("TOURNAMENT COMPLETED!"))) | color(Color::Green),
                                   center(text("Finalizing summary table...")) | color(Color::GrayLight),
                                   filler()}))
                      | color(Color::Green) | flex;
      }

      float overallProgress = static_cast<float>(completedCount) / matchupsList.size();
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

      auto body = hbox({border(vbox({center(bold(text("Matchup Queue Window"))) | color(Color::Cyan),
                                     separator(),
                                     vbox(std::move(listElements)),
                                     filler()}))
                            | color(Color::Cyan) | flex,
                        activePanel});

      int graphWidth = std::max(20, (termWidth - 8) / 2);
      auto cpuGraph = draw_graph(cpuHistory, graphWidth, graphHeight, Color::Green);
      auto ramGraph = draw_graph(ramHistory, graphWidth, graphHeight, Color::Yellow);

      auto sysPanel =
          border(hbox({flex(vbox({hbox({bold(text("CPU Usage: ")),
                                        bold(text(std::to_string(static_cast<int>(cpuHistory.back() * 100.0)) + "%"))
                                            | color(Color::Green)}),
                                  cpuGraph})),
                       separator(),
                       flex(vbox({hbox({bold(text("RAM Usage: ")),
                                        bold(text(std::to_string(static_cast<int>(ramHistory.back() * 100.0)) + "%"))
                                            | color(Color::Yellow)}),
                                  ramGraph}))}))
          | color(Color::GrayDark);

      return vbox({header, flex(body), sysPanel, footer}) | size(WIDTH, EQUAL, termWidth)
             | size(HEIGHT, EQUAL, termHeight);
    });

    int qPressCount = 0;
    auto component = renderer | CatchEvent([&](Event event) {
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

        std::rotate(cpuHistory.begin(), cpuHistory.begin() + 1, cpuHistory.end());
        cpuHistory.back() = cpuUsage;

        std::rotate(ramHistory.begin(), ramHistory.begin() + 1, ramHistory.end());
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
