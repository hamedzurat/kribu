import importlib.util
from pathlib import Path

import duckdb


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "merge_benchmark_duckdb.py"
SPEC = importlib.util.spec_from_file_location("merge_benchmark_duckdb", SCRIPT_PATH)
merge_benchmark_duckdb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge_benchmark_duckdb)


def create_raw_benchmark_db(
    db_path: Path,
    *,
    players: list[tuple[str, str, int, int]],
    games: list[tuple[int, str, str, int, str, int, int, int]],
    turns: list[tuple[int, int, int, int, int, bool, int, str]],
) -> None:
    """Create a minimal raw benchmark DuckDB for merge-script tests."""
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE players (
                name VARCHAR PRIMARY KEY,
                player_type VARCHAR,
                depth INTEGER,
                madness INTEGER
            )
            """
        )
        con.execute(
            """
            CREATE TABLE games (
                game_id INTEGER PRIMARY KEY,
                p1_name VARCHAR,
                p2_name VARCHAR,
                outcome TINYINT,
                reason VARCHAR,
                total_turns INTEGER,
                win_margin INTEGER,
                mad_turns INTEGER
            )
            """
        )
        con.execute(
            """
            CREATE TABLE turns (
                game_id INTEGER,
                turn_idx INTEGER,
                me BIGINT,
                opp BIGINT,
                active_capture_idx TINYINT,
                is_p1_turn BOOLEAN,
                chosen_move SMALLINT,
                player_played VARCHAR
            )
            """
        )
        con.executemany("INSERT INTO players VALUES (?, ?, ?, ?)", players)
        con.executemany("INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)", games)
        con.executemany("INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?)", turns)
    finally:
        con.close()


def test_build_merged_benchmark_duckdb_deduplicates_exact_games(tmp_path):
    source_a = tmp_path / "source_a.duckdb"
    source_b = tmp_path / "source_b.duckdb"
    output_path = tmp_path / "merged.duckdb"

    create_raw_benchmark_db(
        source_a,
        players=[
            ("MinimaxA", "minimax", 8, 0),
            ("MinimaxB", "minimax", 8, 0),
        ],
        games=[
            (1, "MinimaxA", "MinimaxB", 0, "ELIMINATION", 2, 3, 0),
            (2, "MinimaxB", "MinimaxA", 1, "ELIMINATION", 1, 1, 0),
        ],
        turns=[
            (1, 0, 1, 2, -1, True, 10, "MinimaxA"),
            (1, 1, 2, 1, -1, False, 11, "MinimaxB"),
            (2, 0, 3, 4, -1, True, 12, "MinimaxB"),
        ],
    )
    create_raw_benchmark_db(
        source_b,
        players=[
            ("MinimaxA", "minimax", 8, 0),
            ("MinimaxB", "minimax", 8, 0),
            ("MCTS800", "mcts", 800, 0),
        ],
        games=[
            (9, "MinimaxA", "MinimaxB", 0, "ELIMINATION", 2, 3, 0),
            (10, "MCTS800", "MinimaxA", 0, "ELIMINATION", 1, 2, 0),
        ],
        turns=[
            (9, 0, 1, 2, -1, True, 10, "MinimaxA"),
            (9, 1, 2, 1, -1, False, 11, "MinimaxB"),
            (10, 0, 5, 6, -1, True, 13, "MCTS800"),
        ],
    )

    duplicate_games, duplicate_turns = merge_benchmark_duckdb.build_merged_benchmark_duckdb(
        [source_a, source_b],
        output_path,
        deduplicate=True,
    )

    assert duplicate_games == 1
    assert duplicate_turns == 2

    con = duckdb.connect(str(output_path), read_only=True)
    try:
        players = con.execute("SELECT name FROM players ORDER BY name").fetchall()
        games = con.execute("SELECT p1_name, p2_name, total_turns FROM games ORDER BY game_id").fetchall()
        turns = con.execute("SELECT game_id, turn_idx, chosen_move FROM turns ORDER BY game_id, turn_idx").fetchall()
        source_games = con.execute(
            "SELECT source_index, source_path, original_game_id FROM source_games ORDER BY merged_game_id"
        ).fetchall()
    finally:
        con.close()

    assert players == [("MCTS800",), ("MinimaxA",), ("MinimaxB",)]
    assert games == [
        ("MinimaxA", "MinimaxB", 2),
        ("MinimaxB", "MinimaxA", 1),
        ("MCTS800", "MinimaxA", 1),
    ]
    assert turns == [
        (1, 0, 10),
        (1, 1, 11),
        (2, 0, 12),
        (4, 0, 13),
    ]
    assert source_games == [
        (0, str(source_a), 1),
        (0, str(source_a), 2),
        (1, str(source_b), 10),
    ]


def test_build_merged_benchmark_duckdb_can_keep_duplicate_games(tmp_path):
    source_a = tmp_path / "source_a.duckdb"
    source_b = tmp_path / "source_b.duckdb"
    output_path = tmp_path / "merged.duckdb"

    create_raw_benchmark_db(
        source_a,
        players=[("MinimaxA", "minimax", 8, 0)],
        games=[(1, "MinimaxA", "MinimaxA", 0, "ELIMINATION", 1, 1, 0)],
        turns=[(1, 0, 1, 1, -1, True, 10, "MinimaxA")],
    )
    create_raw_benchmark_db(
        source_b,
        players=[("MinimaxA", "minimax", 8, 0)],
        games=[(2, "MinimaxA", "MinimaxA", 0, "ELIMINATION", 1, 1, 0)],
        turns=[(2, 0, 1, 1, -1, True, 10, "MinimaxA")],
    )

    duplicate_games, duplicate_turns = merge_benchmark_duckdb.build_merged_benchmark_duckdb(
        [source_a, source_b],
        output_path,
        deduplicate=False,
    )

    assert duplicate_games == 0
    assert duplicate_turns == 0

    con = duckdb.connect(str(output_path), read_only=True)
    try:
        game_count = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        turn_count = con.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    finally:
        con.close()

    assert game_count == 2
    assert turn_count == 2
