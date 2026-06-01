import importlib.util
from pathlib import Path

import duckdb


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "create_training_duckdb.py"
SPEC = importlib.util.spec_from_file_location("create_training_duckdb", SCRIPT_PATH)
create_training_duckdb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(create_training_duckdb)


def create_source_duckdb(dbPath: Path) -> None:
    """Create a small raw benchmark DuckDB for training-table tests."""
    con = duckdb.connect(str(dbPath))
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
                forced_random_turns INTEGER,
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
        con.execute(
            """
            INSERT INTO players VALUES
                ('MinimaxA', 'minimax', 2, 0),
                ('MctsA', 'mcts', 2, 0),
                ('GreedyA', 'greedy', 0, 0),
                ('ForcedRandom', 'forced_random', 0, 0),
                ('MadPlayer', 'mad_player', 0, 0)
            """
        )
        con.execute(
            """
            INSERT INTO games VALUES
                (1, 'MinimaxA', 'MctsA', 0, 'WIN', 3, 4, 1, 0),
                (2, 'MinimaxA', 'GreedyA', 0, 'WIN', 1, 4, 0, 0),
                (3, 'MinimaxA', 'MctsA', 1, 'INVALID_MOVE', 1, 4, 0, 0),
                (4, 'MinimaxA', 'MctsA', 0, 'WIN', 1, 4, 0, 0)
            """
        )
        con.execute(
            """
            INSERT INTO turns VALUES
                (1, 0, 10, 20, -1, true, 11, 'MinimaxA'),
                (1, 1, 99, 88, -1, false, 77, 'ForcedRandom'),
                (1, 2, 30, 40, 0, true, 12, 'MctsA'),
                (2, 0, 50, 60, 1, true, 13, 'MinimaxA'),
                (3, 0, 70, 80, 2, true, 14, 'MinimaxA'),
                (4, 0, 30, 40, 0, true, 12, 'MinimaxA')
            """
        )
    finally:
        con.close()


def test_build_training_duckdb_materializes_only_randomized_training_tables(tmp_path):
    sourcePath = tmp_path / "source.duckdb"
    outputPath = tmp_path / "training.duckdb"
    create_source_duckdb(sourcePath)

    policyCount, valueCount = create_training_duckdb.build_training_duckdb(sourcePath, outputPath)

    assert policyCount == 1
    assert valueCount == 1

    con = duckdb.connect(str(outputPath), read_only=True)
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        policyRows = con.execute("SELECT * FROM policy_data").fetchall()
        valueRows = con.execute("SELECT * FROM value_data").fetchall()
    finally:
        con.close()

    assert sorted(tableName for (tableName,) in tables) == ["policy_data", "value_data"]
    assert policyRows == [(30, 40, 0, 12)]
    assert valueRows == [(30, 40, 0, 1.0)]


def test_build_training_duckdb_with_filter_no_forced_random(tmp_path):
    sourcePath = tmp_path / "source.duckdb"
    outputPath = tmp_path / "training.duckdb"
    create_source_duckdb(sourcePath)

    con = duckdb.connect(str(sourcePath))
    try:
        con.execute(
            """
            INSERT INTO games VALUES
                (5, 'MinimaxA', 'MctsA', 0, 'WIN', 1, 4, 0, 0)
            """
        )
        con.execute(
            """
            INSERT INTO turns VALUES
                (5, 0, 500, 600, 1, true, 99, 'MinimaxA')
            """
        )
    finally:
        con.close()

    policyCount, valueCount = create_training_duckdb.build_training_duckdb(
        sourcePath, outputPath, filterNoForcedRandom=True
    )

    # Game 5 has 0 forced_random_turns, so its turns should be excluded.
    # Only Game 1's turn (which has 1 forced_random_turn) should remain.
    assert policyCount == 1
    assert valueCount == 1

    con = duckdb.connect(str(outputPath), read_only=True)
    try:
        policyRows = con.execute("SELECT * FROM policy_data").fetchall()
        valueRows = con.execute("SELECT * FROM value_data").fetchall()
    finally:
        con.close()

    assert policyRows == [(30, 40, 0, 12)]
    assert valueRows == [(30, 40, 0, 1.0)]
