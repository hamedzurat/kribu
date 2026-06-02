import importlib.util
from pathlib import Path

import duckdb
import kribu


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "create_training_duckdb.py"
SPEC = importlib.util.spec_from_file_location("create_training_duckdb", SCRIPT_PATH)
create_training_duckdb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(create_training_duckdb)


def create_source_duckdb(dbPath: Path) -> None:
    """Create a small raw benchmark DuckDB for training-table tests."""
    initialState = kribu.INITIAL_STATE
    initialMove = kribu.all_possible_moves(initialState)[0]
    initialNextState = kribu.flip_board(kribu.apply_move(initialState, initialMove))
    replyMove = kribu.all_possible_moves(initialNextState)[0]

    greedyState = kribu.boardState()
    greedyState.me = 1 << 16
    greedyState.opp = 1 << 0
    greedyMove = kribu.all_possible_moves(greedyState)[0]

    drawState = kribu.boardState()
    drawState.me = 1 << 18
    drawState.opp = 1 << 5
    drawMove = kribu.all_possible_moves(drawState)[0]

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
                ('MadPlayer', 'mad_player', 0, 0)
            """
        )
        con.execute(
            """
            INSERT INTO games VALUES
                (1, 'MinimaxA', 'MctsA', 0, 'WIN', 2, 4, 0),
                (2, 'MinimaxA', 'GreedyA', 0, 'WIN', 1, 4, 0),
                (3, 'MinimaxA', 'MctsA', 1, 'INVALID_MOVE', 1, 4, 0),
                (4, 'MinimaxA', 'MctsA', 0, 'WIN', 2, 4, 0),
                (5, 'MinimaxA', 'MctsA', 2, 'DRAW_MAX_TURNS', 1, 0, 0)
            """
        )
        turnRows = [
            (
                1,
                0,
                int(initialState.me),
                int(initialState.opp),
                int(initialState.activeCaptureIdx),
                True,
                int(initialMove),
                "MinimaxA",
            ),
            (
                1,
                1,
                int(initialNextState.me),
                int(initialNextState.opp),
                int(initialNextState.activeCaptureIdx),
                True,
                int(replyMove),
                "MctsA",
            ),
            (
                2,
                0,
                int(greedyState.me),
                int(greedyState.opp),
                int(greedyState.activeCaptureIdx),
                True,
                int(greedyMove),
                "MinimaxA",
            ),
            (
                3,
                0,
                int(initialState.me),
                int(initialState.opp),
                int(initialState.activeCaptureIdx),
                True,
                int(initialMove),
                "MinimaxA",
            ),
            (
                4,
                0,
                int(initialState.me),
                int(initialState.opp),
                int(initialState.activeCaptureIdx),
                True,
                int(initialMove),
                "MinimaxA",
            ),
            (
                4,
                1,
                int(initialNextState.me),
                int(initialNextState.opp),
                int(initialNextState.activeCaptureIdx),
                True,
                int(replyMove),
                "MinimaxA",
            ),
            (
                5,
                0,
                int(drawState.me),
                int(drawState.opp),
                int(drawState.activeCaptureIdx),
                True,
                int(drawMove),
                "MinimaxA",
            ),
        ]
        con.executemany("INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?)", turnRows)
    finally:
        con.close()


def test_build_training_duckdb_materializes_only_randomized_training_tables(tmp_path):
    sourcePath = tmp_path / "source.duckdb"
    outputPath = tmp_path / "training.duckdb"
    create_source_duckdb(sourcePath)
    initialState = kribu.INITIAL_STATE
    initialMove = kribu.all_possible_moves(initialState)[0]
    initialNextState = kribu.flip_board(kribu.apply_move(initialState, initialMove))
    replyMove = kribu.all_possible_moves(initialNextState)[0]
    greedyState = kribu.boardState()
    greedyState.me = 1 << 16
    greedyState.opp = 1 << 0
    drawState = kribu.boardState()
    drawState.me = 1 << 18
    drawState.opp = 1 << 5

    policyCount, valueCount = create_training_duckdb.build_training_duckdb(sourcePath, outputPath)

    assert policyCount == 2
    assert valueCount == 4

    con = duckdb.connect(str(outputPath), read_only=True)
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        policyRows = con.execute("SELECT * FROM policy_data").fetchall()
        valueRows = con.execute("SELECT * FROM value_data").fetchall()
    finally:
        con.close()

    assert sorted(tableName for (tableName,) in tables) == ["policy_data", "value_data"]
    assert sorted(policyRows) == sorted(
        [
            (int(initialState.me), int(initialState.opp), -1, 0, 0, 0, int(initialMove), 2, 1, 2, 1.0),
            (int(initialNextState.me), int(initialNextState.opp), -1, 1, 0, 0, int(replyMove), 2, 1, 2, 1.0),
        ]
    )
    assert sorted(valueRows) == sorted(
        [
            (int(initialState.me), int(initialState.opp), -1, 0, 0, 0, 1.0, 2, 1.0, 1.0, False, True),
            (int(initialNextState.me), int(initialNextState.opp), -1, 1, 0, 0, 1.0, 2, 1.0, 1.0, False, True),
            (int(greedyState.me), int(greedyState.opp), -1, 0, 0, 0, 1.0, 1, 1.0, 1.0, False, True),
            (int(drawState.me), int(drawState.opp), -1, 0, 0, 0, 0.5, 1, 0.5, 0.5, True, False),
        ]
    )


def test_build_training_duckdb_can_filter_low_support_policy_and_loop_heavy_draw_states(tmp_path):
    sourcePath = tmp_path / "source.duckdb"
    outputPath = tmp_path / "training.duckdb"
    create_source_duckdb(sourcePath)

    con = duckdb.connect(str(sourcePath))
    try:
        con.execute(
            """
            INSERT INTO games VALUES
                (6, 'MinimaxA', 'MctsA', 0, 'WIN', 1, 4, 0),
                (7, 'MinimaxA', 'MctsA', 0, 'WIN', 1, 4, 0),
                (8, 'MinimaxA', 'MctsA', 2, 'DRAW_MAX_TURNS', 1, 0, 0),
                (9, 'MinimaxA', 'MctsA', 2, 'DRAW_MAX_TURNS', 1, 0, 0),
                (10, 'MinimaxA', 'MctsA', 2, 'DRAW_MAX_TURNS', 1, 0, 0)
            """
        )
        alternateMove = kribu.all_possible_moves(kribu.INITIAL_STATE)[1]
        drawState = kribu.boardState()
        drawState.me = 1 << 18
        drawState.opp = 1 << 5
        drawMove = kribu.all_possible_moves(drawState)[0]
        extraTurns = [
            (6, 0, int(kribu.INITIAL_STATE.me), int(kribu.INITIAL_STATE.opp), -1, True, int(alternateMove), "MinimaxA"),
            (7, 0, int(kribu.INITIAL_STATE.me), int(kribu.INITIAL_STATE.opp), -1, True, int(alternateMove), "MinimaxA"),
            (8, 0, int(drawState.me), int(drawState.opp), -1, True, int(drawMove), "MinimaxA"),
            (9, 0, int(drawState.me), int(drawState.opp), -1, True, int(drawMove), "MinimaxA"),
            (10, 0, int(drawState.me), int(drawState.opp), -1, True, int(drawMove), "MinimaxA"),
        ]
        con.executemany("INSERT INTO turns VALUES (?, ?, ?, ?, ?, ?, ?, ?)", extraTurns)
    finally:
        con.close()

    policyCount, valueCount = create_training_duckdb.build_training_duckdb(
        sourcePath,
        outputPath,
        minPolicyVisits=2,
        minPolicySupport=0.75,
        maxDrawOnlyVisits=2,
    )

    assert policyCount == 1
    assert valueCount == 3

    con = duckdb.connect(str(outputPath), read_only=True)
    try:
        policyRows = con.execute(
            """
            SELECT
                me,
                opp,
                active_capture_idx,
                history_count,
                current_repeat_count,
                current_flip_repeat_count,
                chosen_move,
                visit_count,
                move_support
            FROM policy_data
            """
        ).fetchall()
        valueRows = con.execute(
            """
            SELECT
                me,
                opp,
                active_capture_idx,
                history_count,
                current_repeat_count,
                current_flip_repeat_count,
                value_label,
                visit_count
            FROM value_data
            """
        ).fetchall()
    finally:
        con.close()

    initialState = kribu.INITIAL_STATE
    initialMove = kribu.all_possible_moves(initialState)[0]
    initialNextState = kribu.flip_board(kribu.apply_move(initialState, initialMove))
    replyMove = kribu.all_possible_moves(initialNextState)[0]
    assert policyRows == [(int(initialNextState.me), int(initialNextState.opp), -1, 1, 0, 0, int(replyMove), 2, 1.0)]
    assert sorted(valueRows) == sorted(
        [
            (int(initialNextState.me), int(initialNextState.opp), -1, 1, 0, 0, 1.0, 2),
            (1 << 16, 1 << 0, -1, 0, 0, 0, 1.0, 1),
            (int(initialState.me), int(initialState.opp), -1, 0, 0, 0, 1.0, 4),
        ]
    )


def test_build_training_duckdb_can_filter_to_named_teacher(tmp_path):
    sourcePath = tmp_path / "source.duckdb"
    outputPath = tmp_path / "training.duckdb"
    create_source_duckdb(sourcePath)

    policyCount, valueCount = create_training_duckdb.build_training_duckdb(
        sourcePath,
        outputPath,
        teacherNames=["MinimaxA"],
    )

    assert policyCount == 2
    assert valueCount == 4

    con = duckdb.connect(str(outputPath), read_only=True)
    try:
        policyRows = con.execute(
            """
            SELECT me, opp, active_capture_idx, history_count, chosen_move, visit_count
            FROM policy_data
            ORDER BY me, opp, history_count
            """
        ).fetchall()
        valueRows = con.execute(
            """
            SELECT me, opp, active_capture_idx, history_count, value_label, visit_count
            FROM value_data
            ORDER BY me, opp, history_count
            """
        ).fetchall()
    finally:
        con.close()

    initialState = kribu.INITIAL_STATE
    initialMove = kribu.all_possible_moves(initialState)[0]
    initialNextState = kribu.flip_board(kribu.apply_move(initialState, initialMove))
    replyMove = kribu.all_possible_moves(initialNextState)[0]
    greedyState = kribu.boardState()
    greedyState.me = 1 << 16
    greedyState.opp = 1 << 0
    drawState = kribu.boardState()
    drawState.me = 1 << 18
    drawState.opp = 1 << 5

    assert policyRows == [
        (int(initialState.me), int(initialState.opp), -1, 0, int(initialMove), 2),
        (int(initialNextState.me), int(initialNextState.opp), -1, 1, int(replyMove), 1),
    ]
    assert valueRows == [
        (int(greedyState.me), int(greedyState.opp), -1, 0, 1.0, 1),
        (int(drawState.me), int(drawState.opp), -1, 0, 0.5, 1),
        (int(initialState.me), int(initialState.opp), -1, 0, 1.0, 2),
        (int(initialNextState.me), int(initialNextState.opp), -1, 1, 1.0, 1),
    ]
