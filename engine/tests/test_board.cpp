#include <array>
#include <catch2/catch_test_macros.hpp>
#include <kribu/board.hpp>
#include <kribu/rules.hpp>
#include <kribu/types.hpp>

using namespace kribu::board;
using namespace kribu::sholoGuti;

namespace {
constexpr std::array<i8, NUM_NODES> FLIP_MAP = {36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24,
                                                23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
                                                10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0};
}  // namespace

// ---------------------------------------------------------------------------
// Board topology
// ---------------------------------------------------------------------------

TEST_CASE("Board Topology Validation", "[board]") {  // NOLINT(readability-function-cognitive-complexity)
  SECTION("Degree Sums") {
    int degreeSum = 0;
    for (int i = 0; i < NUM_NODES; ++i) {
      degreeSum += BOARD_METADATA.counts[i];
    }
    REQUIRE(degreeSum == 152);  // 76 edges * 2
  }

  SECTION("Edge Deduplication / Merging") {
    int count1In2 = 0;
    for (int i = 0; i < BOARD_METADATA.counts[2]; ++i) {
      if (BOARD_METADATA.neighbors[2][i] == 1) {
        count1In2++;
      }
    }
    REQUIRE(count1In2 == 1);

    int count0In3 = 0;
    for (int i = 0; i < BOARD_METADATA.counts[3]; ++i) {
      if (BOARD_METADATA.neighbors[3][i] == 0) {
        count0In3++;
      }
    }
    REQUIRE(count0In3 == 1);
  }

  SECTION("Flip Symmetry - Adjacency") {
    for (int i = 0; i < NUM_NODES; ++i) {
      int flipId = static_cast<u8>(FLIP_MAP[i]);
      REQUIRE(BOARD_METADATA.counts[i] == BOARD_METADATA.counts[flipId]);

      // Check that edges mirror
      for (int j = 0; j < BOARD_METADATA.counts[i]; ++j) {
        int neighbor = static_cast<u8>(BOARD_METADATA.neighbors[i][j]);
        int flipNeighbor = static_cast<u8>(FLIP_MAP[neighbor]);

        bool found = false;
        for (int k = 0; k < BOARD_METADATA.counts[flipId]; ++k) {
          if (BOARD_METADATA.neighbors[flipId][k] == flipNeighbor) {
            found = true;
            break;
          }
        }
        REQUIRE(found);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Move generation counts
// ---------------------------------------------------------------------------

TEST_CASE("Move Generation Counts", "[board]") {  // NOLINT(readability-function-cognitive-complexity)
  SECTION("Move Table Counts") {
    int simpleMoves = 0;
    int captureMoves = 0;
    int endChain = 0;

    for (int i = 0; i < TOTAL_MOVE_COUNT; ++i) {
      move mvEntry = MOVE_TABLE[i];
      if (mvEntry.captured == -1 && mvEntry.from != -1) {
        simpleMoves++;
      } else if (mvEntry.captured != -1) {
        captureMoves++;
      } else if (i == END_CHAIN_MOVE) {
        endChain++;
      }
    }

    REQUIRE(simpleMoves == NUM_SIMPLE_MOVES);
    REQUIRE(captureMoves == NUM_CAPTURE_MOVES);
    REQUIRE(endChain == 1);
  }
}

// ---------------------------------------------------------------------------
// Precomputed metadata: capture index and moveIdMap
// ---------------------------------------------------------------------------

TEST_CASE("Board Metadata - Capture Index", "[board]") {  // NOLINT(readability-function-cognitive-complexity)
  SECTION("Capture move count matches MOVE_TABLE") {
    // Independently count captures per node from MOVE_TABLE.
    std::array<int, NUM_NODES> expected{};
    for (int i = 0; i < TOTAL_MOVE_COUNT; ++i) {
      const move& mvEntry = MOVE_TABLE[i];
      if (mvEntry.from >= 0 && mvEntry.captured != -1) {
        expected[mvEntry.from]++;
      }
    }
    for (int node = 0; node < NUM_NODES; ++node) {
      const int computed = static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[node]));
      REQUIRE(computed == expected[node]);
    }
  }

  SECTION("captureMoveIdxByNode entries point to correct MOVE_TABLE rows") {
    for (int node = 0; node < NUM_NODES; ++node) {
      const int cnt = static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[node]));
      for (int k = 0; k < cnt; ++k) {
        const i16 idx = BOARD_METADATA.captureMoveIdxByNode[node][k];
        const move& mvEntry = MOVE_TABLE[idx];
        REQUIRE(mvEntry.from == static_cast<i8>(node));
        REQUIRE(mvEntry.captured != -1);
      }
    }
  }

  SECTION("Node 8 has exactly 8 captures (hub symmetry)") {
    REQUIRE(static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[8])) == 8);
    REQUIRE(static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[18])) == 8);
    REQUIRE(static_cast<int>(static_cast<u8>(BOARD_METADATA.captureMoveCountByNode[28])) == 8);
  }
}

TEST_CASE("Board Metadata - Move ID Map", "[board]") {  // NOLINT(readability-function-cognitive-complexity)
  SECTION("moveIdMap is -1 for non-existent moves") {
    REQUIRE(BOARD_METADATA.moveIdMap[0][5] == -1);
    REQUIRE(BOARD_METADATA.moveIdMap[1][3] == -1);
  }

  SECTION("moveIdMap returns correct index for known moves") {
    // Node 0 -> 1 is a simple move
    const i16 idx01 = BOARD_METADATA.moveIdMap[0][1];
    REQUIRE(idx01 > 0);
    const move& mv01 = MOVE_TABLE[idx01];
    REQUIRE(mv01.from == 0);
    REQUIRE(mv01.to == 1);
    REQUIRE(mv01.captured == -1);

    // Node 0 -> 2 is a capture over 1
    const i16 idx02 = BOARD_METADATA.moveIdMap[0][2];
    REQUIRE(idx02 > 0);
    const move& mv02 = MOVE_TABLE[idx02];
    REQUIRE(mv02.from == 0);
    REQUIRE(mv02.to == 2);
    REQUIRE(mv02.captured == 1);
  }

  SECTION("moveIdMap is consistent with find_move") {
    REQUIRE(find_move(0, 1) == BOARD_METADATA.moveIdMap[0][1]);
    REQUIRE(find_move(0, 2) == BOARD_METADATA.moveIdMap[0][2]);
    REQUIRE(find_move(8, 0) == BOARD_METADATA.moveIdMap[8][0]);
    REQUIRE(find_move(-1, 0) == -1);
    REQUIRE(find_move(0, static_cast<i8>(NUM_NODES)) == -1);
  }
}

// ---------------------------------------------------------------------------
// Move classifiers
// ---------------------------------------------------------------------------

TEST_CASE("Move Classifiers", "[board]") {  // NOLINT(readability-function-cognitive-complexity)
  SECTION("END_CHAIN_MOVE is neither simple nor capture") {
    REQUIRE_FALSE(is_simple_move(END_CHAIN_MOVE));
    REQUIRE_FALSE(is_capture_move(END_CHAIN_MOVE));
  }

  SECTION("Simple moves are in range [1, NUM_SIMPLE_MOVES]") {
    for (int i = 1; i <= NUM_SIMPLE_MOVES; ++i) {
      REQUIRE(is_simple_move(i));
      REQUIRE_FALSE(is_capture_move(i));
    }
  }

  SECTION("Capture moves are in range [NUM_SIMPLE_MOVES+1, TOTAL_MOVE_COUNT)") {
    for (int i = NUM_SIMPLE_MOVES + 1; i < TOTAL_MOVE_COUNT; ++i) {
      REQUIRE(is_capture_move(i));
      REQUIRE_FALSE(is_simple_move(i));
    }
  }

  SECTION("Out-of-range IDs are neither") {
    REQUIRE_FALSE(is_simple_move(-1));
    REQUIRE_FALSE(is_capture_move(-1));
    REQUIRE_FALSE(is_simple_move(TOTAL_MOVE_COUNT));
    REQUIRE_FALSE(is_capture_move(TOTAL_MOVE_COUNT));
  }
}

// ---------------------------------------------------------------------------
// MoveList
// ---------------------------------------------------------------------------

TEST_CASE("MoveList", "[board]") {  // NOLINT(readability-function-cognitive-complexity)
  SECTION("Starts empty") {
    MoveList list;
    REQUIRE(list.empty());
  }

  SECTION("Push and iterate") {
    MoveList list;
    list.push(1);
    list.push(5);
    list.push(100);
    REQUIRE_FALSE(list.empty());
    REQUIRE(list.size() == 3);
    REQUIRE(list.begin()[0] == 1);
    REQUIRE(list.begin()[1] == 5);
    REQUIRE(list.begin()[2] == 100);
  }

  SECTION("Range-for works") {
    MoveList list;
    list.push(10);
    list.push(20);
    int sum = 0;
    for (i16 moveId : list) {
      sum += moveId;
    }
    REQUIRE(sum == 30);
  }
}

// ---------------------------------------------------------------------------
// Rule engine — initial state
// ---------------------------------------------------------------------------

TEST_CASE("Rule Engine - Initial State", "[rules]") {  // NOLINT(readability-function-cognitive-complexity)
  boardState state = INITIAL_STATE;

  REQUIRE(piece_count(state.me) == 16);
  REQUIRE(piece_count(state.opp) == 16);
  REQUIRE(state.activeCaptureIdx == -1);

  // Check initial positions
  for (int i = 0; i <= 15; ++i) {
    REQUIRE(((state.opp >> i) & 1) == 1);
  }
  for (int i = 21; i <= 36; ++i) {
    REQUIRE(((state.me >> i) & 1) == 1);
  }

  // Empty center
  for (int i = 16; i <= 20; ++i) {
    REQUIRE(((state.me >> i) & 1) == 0);
    REQUIRE(((state.opp >> i) & 1) == 0);
  }

  REQUIRE(get_game_status(state) == GameStatus::ONGOING);
  REQUIRE(static_cast<int>(get_game_status(state)) == 0);
}

// ---------------------------------------------------------------------------
// Rule engine — basic moves
// ---------------------------------------------------------------------------

TEST_CASE("Rule Engine - Basic Moves", "[rules]") {  // NOLINT(readability-function-cognitive-complexity)
  boardState state = INITIAL_STATE;

  SECTION("Valid simple move into empty center") {
    // me pieces are 21-36. 21 connects to 16.
    int moveId = -1;
    for (int i = 1; i <= NUM_SIMPLE_MOVES; ++i) {
      move mvEntry = MOVE_TABLE[i];
      if (mvEntry.from == 21 && mvEntry.to == 16) {
        moveId = i;
        break;
      }
    }
    REQUIRE(moveId != -1);
    REQUIRE(is_valid(state, moveId));

    boardState next = apply_move(state, moveId);
    REQUIRE(piece_count(next.me) == 16);
    REQUIRE(((next.me >> 21) & 1) == 0);
    REQUIRE(((next.me >> 16) & 1) == 1);
    REQUIRE(next.activeCaptureIdx == -1);
  }

  SECTION("Invalid move - blocked") {
    // me piece at 36 connects to 33 and 35. both are occupied by 'me'
    int moveId = -1;
    for (int i = 1; i <= NUM_SIMPLE_MOVES; ++i) {
      move mvEntry = MOVE_TABLE[i];
      if (mvEntry.from == 36 && mvEntry.to == 33) {
        moveId = i;
        break;
      }
    }
    REQUIRE(moveId != -1);
    REQUIRE(!is_valid(state, moveId));
  }

  SECTION("find_move returns same ID as table scan") {
    int moveId = find_move(21, 16);
    REQUIRE(moveId != -1);
    REQUIRE(is_valid(state, moveId));
    const move& mvEntry = MOVE_TABLE[moveId];
    REQUIRE(mvEntry.from == 21);
    REQUIRE(mvEntry.to == 16);
  }
}

// ---------------------------------------------------------------------------
// Rule engine — flip board
// ---------------------------------------------------------------------------

TEST_CASE("Rule Engine - Flip Board", "[rules]") {  // NOLINT(readability-function-cognitive-complexity)
  boardState state = INITIAL_STATE;

  int moveId = find_move(21, 16);
  REQUIRE(moveId != -1);
  boardState next = apply_move(state, moveId);

  boardState flipped = flip_board(next);
  REQUIRE(piece_count(flipped.opp) == 16);
  REQUIRE(((flipped.opp >> FLIP_MAP[16]) & 1) == 1);
  REQUIRE(((flipped.opp >> FLIP_MAP[21]) & 1) == 0);
  REQUIRE(piece_count(flipped.me) == 16);

  SECTION("Double flip returns to original") {
    boardState doubleFlipped = flip_board(flipped);
    REQUIRE(doubleFlipped == next);
  }
}

// ---------------------------------------------------------------------------
// Rule engine — game status
// ---------------------------------------------------------------------------

TEST_CASE("Rule Engine - Game Status", "[rules]") {  // NOLINT(readability-function-cognitive-complexity)
  SECTION("Win by capturing all opponent pieces") {
    boardState state;
    state.me = 1U << 0;
    state.opp = 0;
    REQUIRE(get_game_status(state) == GameStatus::ME_WINS_ELIMINATION);
    REQUIRE(get_game_status(state) != GameStatus::ONGOING);
    REQUIRE(static_cast<int>(get_game_status(state)) == 1);
  }

  SECTION("Loss when own pieces are all gone") {
    boardState state;
    state.me = 0;
    state.opp = 1U << 0;
    REQUIRE(get_game_status(state) == GameStatus::OPP_WINS_ELIMINATION);
    REQUIRE(static_cast<int>(get_game_status(state)) == -1);
  }

  SECTION("Ongoing in initial state") {
    REQUIRE(get_game_status(INITIAL_STATE) == GameStatus::ONGOING);
  }
}
