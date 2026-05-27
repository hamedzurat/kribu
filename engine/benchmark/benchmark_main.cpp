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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
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
#include "kribu/player/_.hpp"
#include "kribu/rules.hpp"

using namespace kribu::board;
using namespace kribu::sholoGuti;
using namespace kribu::benchmark;

namespace kribu::benchmark {

namespace {

std::mutex metadataMutex;
std::atomic<int> globalGameId{1};

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
    PARQUET_THROW_NOT_OK(meBuilder.Append(static_cast<int64_t>(turn.state.me)));
    PARQUET_THROW_NOT_OK(oppBuilder.Append(static_cast<int64_t>(turn.state.opp)));
    PARQUET_THROW_NOT_OK(activeCaptureIdxBuilder.Append(static_cast<int8_t>(turn.state.activeCaptureIdx)));
    PARQUET_THROW_NOT_OK(isP1TurnBuilder.Append(turn.isP1Turn));
    PARQUET_THROW_NOT_OK(chosenMoveBuilder.Append(static_cast<int16_t>(turn.chosenMove)));
    PARQUET_THROW_NOT_OK(possibleMovesBuilder.Append());
    for (int i = 0; i < turn.possibleMoves.size(); ++i) {
      PARQUET_THROW_NOT_OK(valueBuilder->Append(static_cast<int16_t>(turn.possibleMoves.moves[i])));
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
void print_progress(int completed,
                    int total,
                    std::string_view p1Name,
                    std::string_view p2Name,
                    std::chrono::time_point<std::chrono::high_resolution_clock> start) noexcept {
  constexpr int BAR_WIDTH = 20;
  float progress = static_cast<float>(completed) / static_cast<float>(total);
  int pos = static_cast<int>(BAR_WIDTH * progress);

  auto now = std::chrono::high_resolution_clock::now();
  auto elapsedSecs = std::chrono::duration<double>(now - start).count();
  double etaSecs = 0.0;
  if (completed > 0 && completed < total) {
    etaSecs = (elapsedSecs / completed) * (total - completed);
  }

  std::cout << "\rMatchup: " << p1Name << " vs " << p2Name << " | [";
  for (int i = 0; i < BAR_WIDTH; ++i) {
    if (i < pos) {
      std::cout << "=";
    } else if (i == pos) {
      std::cout << ">";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "] " << static_cast<int>(progress * 100.0) << "% (" << completed << "/" << total << ")";

  if (completed > 0 && completed < total) {
    std::cout << " ETA: " << std::fixed << std::setprecision(1) << etaSecs << "s";
  } else if (completed == total) {
    std::cout << " Time: " << std::fixed << std::setprecision(1) << elapsedSecs << "s";
  }

  std::cout << "       " << std::flush;
}

/**
 * @brief Runs a tournament matchup between two players in a multithreaded fashion.
 * @param match Configuration for the matchup.
 * @param players Map of registered players.
 * @param summaryTable Table to write results summary row to.
 */
void run_matchup(const MatchConfig& match,
                 const std::map<std::string, Player>& players,
                 tabulate::Table& summaryTable) {
  if (!players.contains(std::string(match.player1Name)) || !players.contains(std::string(match.player2Name))) {
    std::cerr << "Skipping matchup " << match.player1Name << " vs " << match.player2Name
              << ": one or both players not registered.\n";
    return;
  }

  const auto& playerFirst = players.at(std::string(match.player1Name));
  const auto& playerSecond = players.at(std::string(match.player2Name));

  std::atomic<int> completedGames{0};
  auto startTime = std::chrono::high_resolution_clock::now();

  std::thread progressThread([&completedGames, &match, &playerFirst, &playerSecond, startTime]() {
    while (true) {
      int completed = completedGames.load(std::memory_order_relaxed);
      print_progress(completed, match.games, playerFirst.name, playerSecond.name, startTime);
      if (completed >= match.games) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });

  MatchStats stats = run_matchup_multithreaded(playerFirst,
                                               playerSecond,
                                               TournamentConfig{.totalGames = match.games,
                                                                .maxTurns = match.maxTurns,
                                                                .threadCount = THREAD_COUNT,
                                                                .completedGames = &completedGames,
                                                                .globalGameId = &globalGameId});

  if (progressThread.joinable()) {
    progressThread.join();
  }
  std::cout << "\rMatchup: " << playerFirst.name << " vs " << playerSecond.name
            << " | Done!                                    \n";

  double p1AvgNodes = static_cast<double>(stats.p1TotalNodes) / match.games;
  double p2AvgNodes = static_cast<double>(stats.p2TotalNodes) / match.games;
  double p1AvgCpuMs = (stats.p1TotalCpuTimeSeconds * 1000.0) / match.games;
  double p2AvgCpuMs = (stats.p2TotalCpuTimeSeconds * 1000.0) / match.games;
  double avgTurns = static_cast<double>(stats.totalTurns) / match.games;

  summaryTable.add_row(
      {std::string(playerFirst.name),
       std::string(playerSecond.name),
       std::to_string(stats.p1Wins) + " (" + std::to_string(stats.p1EliminationWins) + "/"
           + std::to_string(stats.p1StalemateWins) + "/" + std::to_string(stats.p1InvalidMoveWins) + ")",
       std::to_string(stats.p2Wins) + " (" + std::to_string(stats.p2EliminationWins) + "/"
           + std::to_string(stats.p2StalemateWins) + "/" + std::to_string(stats.p2InvalidMoveWins) + ")",
       std::to_string(stats.draws),
       std::to_string(static_cast<int>(avgTurns)),
       std::to_string(static_cast<int>(p1AvgNodes)),
       std::to_string(static_cast<int>(p2AvgNodes)),
       std::to_string(static_cast<int>(p1AvgCpuMs)) + " ms",
       std::to_string(static_cast<int>(p2AvgCpuMs)) + " ms"});
}

}  // namespace

/**
 * @brief Saves a completed game dataset directly to a Parquet file, storing metadata in the schema footer.
 * @param player1Name Name of Player 1.
 * @param player2Name Name of Player 2.
 * @param gameIdx The unique game index in the tournament matchup.
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
  kribu::maxRepetitions = kribu::benchmark::REPETITION_LIMIT;
  kribu::allowRepetition = kribu::benchmark::ALLOW_REPETITION;
  try {
    std::filesystem::create_directories("benchmark");

    int lastId = find_max_id_from_csv("benchmark/manifest.csv");
    globalGameId.store(lastId + 1, std::memory_order_relaxed);

    const auto playerList = kribu::player::BENCHMARK_PLAYERS;
    std::map<std::string, Player> players;
    for (const auto& playerEntry : playerList) {
      players[std::string(playerEntry.name)] = playerEntry;
      std::cout << "Registered player: " << playerEntry.name << "\n";
    }

    const auto matchups = BENCHMARK_MATCHUPS;

    std::cout << "\nStarting " << matchups.size() << " matchups with " << THREAD_COUNT << " threads (starting from ID "
              << (lastId + 1) << ")...\n\n";

    tabulate::Table summaryTable;
    summaryTable.add_row({"Player 1",
                          "Player 2",
                          "P1 Wins (Elim/Stale/Inv)",
                          "P2 Wins (Elim/Stale/Inv)",
                          "Draws",
                          "Avg Turns",
                          "P1 Avg Nodes",
                          "P2 Avg Nodes",
                          "P1 Avg CPU",
                          "P2 Avg CPU"});

    for (const auto& match : matchups) {
      kribu::benchmark::run_matchup(match, players, summaryTable);
    }

    // Format summaryTable for premium terminal DX
    summaryTable[0]
        .format()
        .font_style({tabulate::FontStyle::bold})
        .font_align(tabulate::FontAlign::center)
        .font_color(tabulate::Color::yellow);

    for (size_t i = 1; i < summaryTable.size(); ++i) {
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
