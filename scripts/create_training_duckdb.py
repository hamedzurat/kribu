#!/usr/bin/env python3
"""Create randomized supervised training tables from the raw benchmark DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import kribu
import pandas as pd


DEFAULT_SOURCE_PATH = Path("benchmark/dataset.duckdb")
DEFAULT_OUTPUT_PATH = Path("benchmark/training_dataset.duckdb")
DEFAULT_MIN_POLICY_VISITS = 1
DEFAULT_MIN_POLICY_SUPPORT = 0.0
DEFAULT_MAX_DRAW_ONLY_VISITS = 0
DEFAULT_HISTORY_BATCH_GAMES = 256


def quote_identifier(identifier: str) -> str:
    """Return a DuckDB-safe quoted identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    """Return a DuckDB-safe quoted string literal."""
    return "'" + value.replace("'", "''") + "'"


def drop_existing_objects(con: duckdb.DuckDBPyConnection) -> None:
    """Remove existing tables and views from the output database."""
    objects = con.execute(
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """
    ).fetchall()

    for tableName, tableType in objects:
        objectType = "VIEW" if tableType == "VIEW" else "TABLE"
        con.execute(f"DROP {objectType} IF EXISTS {quote_identifier(tableName)}")


def create_turn_history_features(
    con: duckdb.DuckDBPyConnection,
    *,
    batchGames: int = DEFAULT_HISTORY_BATCH_GAMES,
) -> None:
    """Reconstruct compact repetition features for each recorded turn.

    @param con Reference to the DuckDB connection with the source DB attached as `source`.
    @param batchGames Number of games to annotate per SQL fetch batch.
    """
    if batchGames < 1:
        raise ValueError("batchGames must be at least 1")

    con.execute(
        """
        CREATE TABLE turn_history_features (
            game_id INTEGER,
            turn_idx INTEGER,
            history_count INTEGER,
            current_repeat_count INTEGER,
            current_flip_repeat_count INTEGER,
            PRIMARY KEY (game_id, turn_idx)
        )
        """
    )

    gameIds = [row[0] for row in con.execute("SELECT game_id FROM source.games ORDER BY game_id").fetchall()]

    for startIndex in range(0, len(gameIds), batchGames):
        batchIds = gameIds[startIndex : startIndex + batchGames]
        placeholders = ", ".join("?" for _ in batchIds)
        turnRows = con.execute(
            f"""
            SELECT game_id, turn_idx, me, opp, active_capture_idx, chosen_move
            FROM source.turns
            WHERE game_id IN ({placeholders})
            ORDER BY game_id, turn_idx
            """,
            batchIds,
        ).fetchall()

        featureRows: list[tuple[int, int, int, int, int]] = []
        currentGameId = None
        meMasks: list[int] = []
        oppMasks: list[int] = []
        activeCaptureIndices: list[int] = []
        chosenMoves: list[int] = []
        turnIndices: list[int] = []

        def flush_game() -> None:
            if currentGameId is None:
                return

            historyCounts, currentRepeats, currentFlipRepeats = kribu.annotate_repetition_features(
                meMasks,
                oppMasks,
                activeCaptureIndices,
                chosenMoves,
            )
            for rowIndex, turnIndex in enumerate(turnIndices):
                featureRows.append(
                    (
                        int(currentGameId),
                        int(turnIndex),
                        int(historyCounts[rowIndex]),
                        int(currentRepeats[rowIndex]),
                        int(currentFlipRepeats[rowIndex]),
                    )
                )

        for gameId, turnIdx, me, opp, activeCaptureIdx, chosenMove in turnRows:
            if currentGameId is not None and gameId != currentGameId:
                flush_game()
                meMasks = []
                oppMasks = []
                activeCaptureIndices = []
                chosenMoves = []
                turnIndices = []

            currentGameId = gameId
            meMasks.append(int(me))
            oppMasks.append(int(opp))
            activeCaptureIndices.append(int(activeCaptureIdx))
            chosenMoves.append(int(chosenMove))
            turnIndices.append(int(turnIdx))

        flush_game()

        if featureRows:
            featureFrame = pd.DataFrame(
                featureRows,
                columns=[
                    "game_id",
                    "turn_idx",
                    "history_count",
                    "current_repeat_count",
                    "current_flip_repeat_count",
                ],
            )
            con.register("turn_history_feature_batch", featureFrame)
            try:
                con.execute("INSERT INTO turn_history_features SELECT * FROM turn_history_feature_batch")
            finally:
                con.unregister("turn_history_feature_batch")


def create_policy_data(
    con: duckdb.DuckDBPyConnection,
    *,
    keepDuplicates: bool,
    minPolicyVisits: int,
    minPolicySupport: float,
) -> None:
    """Create the randomized policy_data table.

    @param con Reference to the DuckDB connection.
    @param keepDuplicates Whether to keep duplicate rows (has no effect if filtering/averaging).
    @param minPolicyVisits Minimum number of raw visits required to keep a policy state.
    @param minPolicySupport Minimum majority support required to keep a policy state.
    """
    if minPolicyVisits < 1:
        raise ValueError("minPolicyVisits must be at least 1")
    if minPolicySupport < 0.0 or minPolicySupport > 1.0:
        raise ValueError("minPolicySupport must be within [0.0, 1.0]")

    con.execute(
        f"""
        CREATE TABLE policy_data AS
        WITH usable_turns AS (
            SELECT
                t.me,
                t.opp,
                t.active_capture_idx,
                hf.history_count,
                hf.current_repeat_count,
                hf.current_flip_repeat_count,
                t.chosen_move
            FROM source.turns AS t
            JOIN turn_history_features AS hf
                ON t.game_id = hf.game_id
                AND t.turn_idx = hf.turn_idx
            JOIN source.players AS actor ON t.player_played = actor.name
            JOIN source.games AS g ON t.game_id = g.game_id
            JOIN source.players AS p1 ON g.p1_name = p1.name
            JOIN source.players AS p2 ON g.p2_name = p2.name
            WHERE
                g.reason != 'INVALID_MOVE'
                AND g.reason != 'DRAW_MAX_TURNS'
                AND t.player_played != 'MadPlayer'
                AND actor.player_type IN ('minimax', 'mcts')
                AND p1.player_type IN ('minimax', 'mcts')
                AND p2.player_type IN ('minimax', 'mcts')
        ),
        move_counts AS (
            SELECT
                me,
                opp,
                active_capture_idx,
                history_count,
                current_repeat_count,
                current_flip_repeat_count,
                chosen_move,
                COUNT(*) AS chosen_move_count,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        me,
                        opp,
                        active_capture_idx,
                        history_count,
                        current_repeat_count,
                        current_flip_repeat_count
                    ORDER BY COUNT(*) DESC, random()
                ) AS rn
            FROM usable_turns
            GROUP BY
                me,
                opp,
                active_capture_idx,
                history_count,
                current_repeat_count,
                current_flip_repeat_count,
                chosen_move
        ),
        state_stats AS (
            SELECT
                me,
                opp,
                active_capture_idx,
                history_count,
                current_repeat_count,
                current_flip_repeat_count,
                SUM(chosen_move_count) AS visit_count,
                COUNT(*) AS distinct_move_count,
                MAX(chosen_move_count) AS majority_move_count
            FROM move_counts
            GROUP BY
                me,
                opp,
                active_capture_idx,
                history_count,
                current_repeat_count,
                current_flip_repeat_count
        )
        SELECT
            mc.me,
            mc.opp,
            mc.active_capture_idx,
            mc.history_count,
            mc.current_repeat_count,
            mc.current_flip_repeat_count,
            mc.chosen_move,
            ss.visit_count,
            ss.distinct_move_count,
            mc.chosen_move_count,
            mc.chosen_move_count * 1.0 / ss.visit_count AS move_support
        FROM move_counts AS mc
        JOIN state_stats AS ss
            ON mc.me = ss.me
            AND mc.opp = ss.opp
            AND mc.active_capture_idx = ss.active_capture_idx
            AND mc.history_count = ss.history_count
            AND mc.current_repeat_count = ss.current_repeat_count
            AND mc.current_flip_repeat_count = ss.current_flip_repeat_count
        WHERE
            mc.rn = 1
            AND ss.visit_count >= {minPolicyVisits}
            AND mc.chosen_move_count * 1.0 / ss.visit_count >= {minPolicySupport}
        ORDER BY random()
        """
    )


def create_value_data(
    con: duckdb.DuckDBPyConnection,
    *,
    keepDuplicates: bool,
    maxDrawOnlyVisits: int,
) -> None:
    """Create the randomized value_data table.

    @param con Reference to the DuckDB connection.
    @param keepDuplicates Whether to keep duplicate rows (has no effect if filtering/averaging).
    @param maxDrawOnlyVisits Maximum raw visits allowed for draw-only 0.5 states; 0 disables this filter.
    """
    if maxDrawOnlyVisits < 0:
        raise ValueError("maxDrawOnlyVisits must be non-negative")

    drawVisitFilter = ""
    if maxDrawOnlyVisits > 0:
        drawVisitFilter = f"""
            AND NOT (
                seen_draw_game
                AND NOT seen_non_draw_game
                AND value_label = 0.5
                AND visit_count > {maxDrawOnlyVisits}
            )
        """

    con.execute(
        f"""
        CREATE TABLE value_data AS
        WITH usable_turns AS (
            SELECT
                t.me,
                t.opp,
                t.active_capture_idx,
                hf.history_count,
                hf.current_repeat_count,
                hf.current_flip_repeat_count,
                t.is_p1_turn,
                g.outcome,
                g.reason
            FROM source.turns AS t
            JOIN turn_history_features AS hf
                ON t.game_id = hf.game_id
                AND t.turn_idx = hf.turn_idx
            JOIN source.players AS actor ON t.player_played = actor.name
            JOIN source.games AS g ON t.game_id = g.game_id
            WHERE
                g.reason != 'INVALID_MOVE'
                AND t.player_played != 'MadPlayer'
                AND actor.player_type IN ('minimax', 'mcts')
        ),
        turn_values AS (
            SELECT
                me,
                opp,
                active_capture_idx,
                history_count,
                current_repeat_count,
                current_flip_repeat_count,
                reason,
                CASE
                    WHEN outcome = 2 THEN 0.5
                    WHEN is_p1_turn AND outcome = 0 THEN 1.0
                    WHEN NOT is_p1_turn AND outcome = 1 THEN 1.0
                    ELSE 0.0
                END AS value_label
            FROM usable_turns
        )
        SELECT
            me,
            opp,
            active_capture_idx,
            history_count,
            current_repeat_count,
            current_flip_repeat_count,
            AVG(value_label) AS value_label,
            COUNT(*) AS visit_count,
            MIN(value_label) AS min_value_label,
            MAX(value_label) AS max_value_label,
            BOOL_OR(reason = 'DRAW_MAX_TURNS') AS seen_draw_game,
            BOOL_OR(reason != 'DRAW_MAX_TURNS') AS seen_non_draw_game
        FROM turn_values
        GROUP BY
            me,
            opp,
            active_capture_idx,
            history_count,
            current_repeat_count,
            current_flip_repeat_count
        HAVING TRUE
            {drawVisitFilter}
        ORDER BY random()
        """
    )


def build_training_duckdb(
    sourcePath: Path,
    outputPath: Path,
    *,
    keepDuplicates: bool = False,
    minPolicyVisits: int = DEFAULT_MIN_POLICY_VISITS,
    minPolicySupport: float = DEFAULT_MIN_POLICY_SUPPORT,
    maxDrawOnlyVisits: int = DEFAULT_MAX_DRAW_ONLY_VISITS,
) -> tuple[int, int]:
    """Build randomized policy_data and value_data tables in a DuckDB file.

    @param sourcePath Path to the raw source DuckDB.
    @param outputPath Path to the destination training DuckDB.
    @param keepDuplicates Whether to keep exact duplicate rows.
    @param minPolicyVisits Minimum number of raw visits required to keep a policy state.
    @param minPolicySupport Minimum majority support required to keep a policy state.
    @param maxDrawOnlyVisits Maximum raw visits allowed for draw-only 0.5 states; 0 disables this filter.
    """
    if not sourcePath.exists():
        raise FileNotFoundError(f"source DuckDB not found: {sourcePath}")
    if sourcePath.resolve() == outputPath.resolve():
        raise ValueError("source and output DuckDB paths must be different")

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(outputPath))
    try:
        con.execute(f"ATTACH {quote_literal(str(sourcePath))} AS source (READ_ONLY)")
        drop_existing_objects(con)
        create_turn_history_features(con)
        create_policy_data(
            con,
            keepDuplicates=keepDuplicates,
            minPolicyVisits=minPolicyVisits,
            minPolicySupport=minPolicySupport,
        )
        create_value_data(
            con,
            keepDuplicates=keepDuplicates,
            maxDrawOnlyVisits=maxDrawOnlyVisits,
        )
        con.execute("DROP TABLE turn_history_features")

        policyCount = con.execute("SELECT count(*) FROM policy_data").fetchone()[0]
        valueCount = con.execute("SELECT count(*) FROM value_data").fetchone()[0]
        con.execute("DETACH source")
    finally:
        con.close()

    return policyCount, valueCount


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH, help="Raw benchmark DuckDB path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Training DuckDB path to write.")
    parser.add_argument("--keep-duplicates", action="store_true", help="Keep exact duplicate supervised rows.")
    parser.add_argument(
        "--min-policy-visits",
        type=int,
        default=DEFAULT_MIN_POLICY_VISITS,
        help="Drop policy states seen fewer than this many times.",
    )
    parser.add_argument(
        "--min-policy-support",
        type=float,
        default=DEFAULT_MIN_POLICY_SUPPORT,
        help="Drop policy states whose majority move support is below this fraction.",
    )
    parser.add_argument(
        "--max-draw-only-visits",
        type=int,
        default=DEFAULT_MAX_DRAW_ONLY_VISITS,
        help="Drop draw-only value states with more than this many visits; 0 disables the filter.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training DuckDB build."""
    args = parse_args()
    policyCount, valueCount = build_training_duckdb(
        args.source,
        args.output,
        keepDuplicates=args.keep_duplicates,
        minPolicyVisits=args.min_policy_visits,
        minPolicySupport=args.min_policy_support,
        maxDrawOnlyVisits=args.max_draw_only_visits,
    )
    print(f"wrote {args.output}")
    print(f"policy_data rows: {policyCount}")
    print(f"value_data rows: {valueCount}")


if __name__ == "__main__":
    main()
