/**
 * @file transposition_table.hpp
 * @brief Thread-local/instance-level transposition table for minimax search.
 */

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "kribu/types.hpp"

namespace kribu {

/**
 * @enum TTFlag
 * @brief Represents the bounding type of the stored score.
 */
enum class TTFlag : u8 {
  EXACT,  ///< Exact score evaluation.
  ALPHA,  ///< Score is an upper bound.
  BETA    ///< Score is a lower bound.
};

/**
 * @struct TTEntry
 * @brief Transposition Table cache entry.
 */
struct TTEntry {
  u64 hash = 0;
  i32 score = 0;
  i16 moveId = -1;
  u8 depth = 0;
  TTFlag flag = TTFlag::EXACT;
};

/**
 * @class TranspositionTable
 * @brief Lock-free, pre-allocated transposition table optimized for parallel searches.
 */
class TranspositionTable {
 public:
  /**
   * @brief Constructs the transposition table with a fixed capacity.
   * @param size Size of the table (number of elements).
   */
  explicit TranspositionTable(usize size = 1048576) : table(size) {}

  /**
   * @brief Clears the transposition table.
   */
  void clear() noexcept { std::ranges::fill(table, TTEntry{}); }

  /**
   * @brief Probes the table for a cached match.
   * @return True if a valid cutoff or exact score is found, false otherwise.
   */
  [[nodiscard]] bool probe(u64 hash, int depth, i32 alpha, i32 beta, i32& score, int& moveId) const noexcept {
    usize idx = hash % table.size();
    const auto& entry = table[idx];
    if (entry.hash == hash && std::cmp_greater_equal(entry.depth, depth)) {
      if (entry.flag == TTFlag::EXACT) {
        score = entry.score;
        moveId = entry.moveId;
        return true;
      }
      if (entry.flag == TTFlag::ALPHA && entry.score <= alpha) {
        score = alpha;
        moveId = entry.moveId;
        return true;
      }
      if (entry.flag == TTFlag::BETA && entry.score >= beta) {
        score = beta;
        moveId = entry.moveId;
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Stores a search result in the transposition table.
   */
  void store(u64 hash, int depth, i32 score, int moveId, TTFlag flag) noexcept {
    usize idx = hash % table.size();
    auto& entry = table[idx];
    // Depth-preferred replacement scheme
    if (entry.hash != hash || std::cmp_greater_equal(depth, entry.depth)) {
      entry.hash = hash;
      entry.score = score;
      entry.moveId = static_cast<i16>(moveId);
      entry.depth = static_cast<u8>(depth);
      entry.flag = flag;
    }
  }

 private:
  std::vector<TTEntry> table;
};

}  // namespace kribu
