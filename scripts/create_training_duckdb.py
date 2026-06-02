#!/usr/bin/env python3
"""Create randomized supervised training tables from the raw benchmark DuckDB."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_SOURCE_PATH = Path("benchmark/dataset.duckdb")
DEFAULT_OUTPUT_PATH = Path("benchmark/training_dataset.duckdb")


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


def create_policy_data(con: duckdb.DuckDBPyConnection, *, keepDuplicates: bool) -> None:
    """Create the randomized policy_data table.

    @param con Reference to the DuckDB connection.
    @param keepDuplicates Whether to keep duplicate rows (has no effect if filtering/averaging).
    """
    con.execute(
        """
        CREATE TABLE policy_data AS
        WITH usable_turns AS (
            SELECT
                t.me,
                t.opp,
                t.active_capture_idx,
                t.chosen_move
            FROM source.turns AS t
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
                chosen_move,
                COUNT(*) as cnt,
                ROW_NUMBER() OVER (
                    PARTITION BY me, opp, active_capture_idx 
                    ORDER BY COUNT(*) DESC, random()
                ) as rn
            FROM usable_turns
            GROUP BY me, opp, active_capture_idx, chosen_move
        )
        SELECT
            me,
            opp,
            active_capture_idx,
            chosen_move
        FROM move_counts
        WHERE rn = 1
        ORDER BY random()
        """
    )


def create_value_data(con: duckdb.DuckDBPyConnection, *, keepDuplicates: bool) -> None:
    """Create the randomized value_data table.

    @param con Reference to the DuckDB connection.
    @param keepDuplicates Whether to keep duplicate rows (has no effect if filtering/averaging).
    """
    con.execute(
        """
        CREATE TABLE value_data AS
        WITH usable_turns AS (
            SELECT
                t.me,
                t.opp,
                t.active_capture_idx,
                t.is_p1_turn,
                g.outcome
            FROM source.turns AS t
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
            AVG(value_label) AS value_label
        FROM turn_values
        GROUP BY me, opp, active_capture_idx
        ORDER BY random()
        """
    )


def build_training_duckdb(
    sourcePath: Path,
    outputPath: Path,
    *,
    keepDuplicates: bool = False,
) -> tuple[int, int]:
    """Build randomized policy_data and value_data tables in a DuckDB file.

    @param sourcePath Path to the raw source DuckDB.
    @param outputPath Path to the destination training DuckDB.
    @param keepDuplicates Whether to keep exact duplicate rows.
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
        create_policy_data(con, keepDuplicates=keepDuplicates)
        create_value_data(con, keepDuplicates=keepDuplicates)

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
    return parser.parse_args()


def main() -> None:
    """Run the training DuckDB build."""
    args = parse_args()
    policyCount, valueCount = build_training_duckdb(
        args.source,
        args.output,
        keepDuplicates=args.keep_duplicates,
    )
    print(f"wrote {args.output}")
    print(f"policy_data rows: {policyCount}")
    print(f"value_data rows: {valueCount}")


if __name__ == "__main__":
    main()
