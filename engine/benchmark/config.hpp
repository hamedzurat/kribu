/**
 * @file config.hpp
 * @brief Header for compile-time benchmark tournament configs.
 */

#pragma once

#include <span>

#include "kribu/benchmark.hpp"

namespace kribu::benchmark {

/**
 * @brief Returns the list of benchmark player definitions.
 */
std::span<const Player> get_benchmark_players();

/**
 * @brief Returns the matchup pairs (by name) to simulate.
 */
constexpr int THREAD_COUNT = 6;

/**
 * @brief Returns the matchup definitions defined by hand.
 */
std::span<const MatchConfig> get_benchmark_matchups();

}  // namespace kribu::benchmark
