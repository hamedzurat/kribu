/**
 * @file test_fast_rng.cpp
 * @brief Unit tests for the FastRng pseudo-random number generator.
 */

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <kribu/fast_rng.hpp>

TEST_CASE("FastRng Basic Properties", "[fast_rng]") {
  FastRng myRng(42);

  REQUIRE(myRng.min() == 0);
  REQUIRE(myRng.max() == ~static_cast<std::uint64_t>(0));

  std::uint64_t val1 = myRng();
  std::uint64_t val2 = myRng();
  std::uint64_t val3 = myRng();

  // Values should not be identical (unlikely with period of 2^64-1)
  REQUIRE(val1 != val2);
  REQUIRE(val2 != val3);
  REQUIRE(val1 != val3);
}

TEST_CASE("FastRng Thread Local rng", "[fast_rng]") {
  // Test that the global thread-local rng operates and produces valid values
  std::uint64_t val1 = rng();
  std::uint64_t val2 = rng();
  REQUIRE(val1 != val2);
}
