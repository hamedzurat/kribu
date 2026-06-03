#!/usr/bin/env python3
"""Merge one or more raw benchmark DuckDB files into a single deduplicated database."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


DEFAULT_OUTPUT_PATH = Path("benchmark/dataset_merged.duckdb")


def quote_identifier(identifier: str) -> str:
    """Return a DuckDB-safe quoted identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    """Return a DuckDB-safe quoted string literal."""
    return "'" + value.replace("'", "''") + "'"


def table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    """Return ordered column names for a DuckDB table."""
    return [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]


def drop_existing_objects(con: duckdb.DuckDBPyConnection) -> None:
    """Remove any existing tables/views from the output database."""
    objects = con.execute(
        """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """
    ).fetchall()

    for table_name, table_type in objects:
        object_type = "VIEW" if table_type == "VIEW" else "TABLE"
        con.execute(f"DROP {object_type} IF EXISTS {quote_identifier(table_name)}")


def create_output_schema(con: duckdb.DuckDBPyConnection, source_alias: str) -> None:
    """Create empty output tables using the first source schema."""
    con.execute(f"CREATE TABLE players AS SELECT * FROM {source_alias}.players LIMIT 0")
    con.execute(f"CREATE TABLE games AS SELECT * FROM {source_alias}.games LIMIT 0")
    con.execute(f"CREATE TABLE turns AS SELECT * FROM {source_alias}.turns LIMIT 0")
    if "forced_random_turns" not in table_columns(con, "games"):
        con.execute("ALTER TABLE games ADD COLUMN forced_random_turns INTEGER")
    con.execute(
        """
        CREATE TABLE source_games (
            source_index INTEGER,
            source_path VARCHAR,
            original_game_id INTEGER,
            merged_game_id INTEGER
        )
        """
    )


def merge_source(con: duckdb.DuckDBPyConnection, source_alias: str, source_path: Path, source_index: int) -> None:
    """Append one source database into the merged output, remapping game ids."""
    current_max_game_id = int(con.execute("SELECT COALESCE(MAX(game_id), 0) FROM games").fetchone()[0])
    target_game_columns = table_columns(con, "games")
    source_game_columns = set(table_columns(con, f"{source_alias}.games"))

    staged_game_columns = []
    for column_name in target_game_columns:
        if column_name == "game_id":
            continue
        if column_name in source_game_columns:
            expression = quote_identifier(column_name)
        elif column_name == "forced_random_turns":
            expression = "0"
        else:
            expression = "NULL"
        staged_game_columns.append(f"{expression} AS {quote_identifier(column_name)}")

    con.execute("DROP TABLE IF EXISTS staged_games")
    con.execute(
        f"""
        CREATE TEMP TABLE staged_games AS
        SELECT
            {current_max_game_id} + ROW_NUMBER() OVER (ORDER BY game_id) AS merged_game_id,
            game_id AS original_game_id,
            {", ".join(staged_game_columns)}
        FROM {source_alias}.games
        """
    )

    con.execute(
        f"""
        INSERT INTO players
        SELECT src.*
        FROM {source_alias}.players AS src
        WHERE NOT EXISTS (
            SELECT 1
            FROM players AS existing
            WHERE existing.name = src.name
        )
        """
    )

    con.execute(
        f"""
        INSERT INTO games ({", ".join(quote_identifier(column_name) for column_name in target_game_columns)})
        SELECT
            merged_game_id AS game_id,
            {", ".join(quote_identifier(column_name) for column_name in target_game_columns if column_name != 'game_id')}
        FROM staged_games
        """
    )

    con.execute(
        f"""
        INSERT INTO turns
        SELECT
            sg.merged_game_id AS game_id,
            t.turn_idx,
            t.me,
            t.opp,
            t.active_capture_idx,
            t.is_p1_turn,
            t.chosen_move,
            t.player_played
        FROM {source_alias}.turns AS t
        JOIN staged_games AS sg
            ON t.game_id = sg.original_game_id
        """
    )

    con.execute(
        """
        INSERT INTO source_games
        SELECT ?, ?, original_game_id, merged_game_id
        FROM staged_games
        """,
        [source_index, str(source_path),],
    )

    con.execute("DROP TABLE staged_games")


def deduplicate_exact_games(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """Drop duplicate full-game transcripts while keeping the earliest merged copy."""
    con.execute("DROP TABLE IF EXISTS duplicate_game_ids")
    con.execute(
        """
        CREATE TEMP TABLE duplicate_game_ids AS
        WITH transcript_hashes AS (
            SELECT
                g.game_id,
                g.p1_name,
                g.p2_name,
                g.outcome,
                g.reason,
                g.total_turns,
                g.win_margin,
                g.forced_random_turns,
                g.mad_turns,
                md5(COALESCE(string_agg(CAST(t.chosen_move AS VARCHAR), ',' ORDER BY t.turn_idx), '')) AS transcript_hash
            FROM games AS g
            LEFT JOIN turns AS t
                ON g.game_id = t.game_id
            GROUP BY
                g.game_id,
                g.p1_name,
                g.p2_name,
                g.outcome,
                g.reason,
                g.total_turns,
                g.win_margin,
                g.forced_random_turns,
                g.mad_turns
        ),
        ranked AS (
            SELECT
                game_id,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        p1_name,
                        p2_name,
                        outcome,
                        reason,
                        total_turns,
                        win_margin,
                        forced_random_turns,
                        mad_turns,
                        transcript_hash
                    ORDER BY game_id
                ) AS rn
            FROM transcript_hashes
        )
        SELECT game_id
        FROM ranked
        WHERE rn > 1
        """
    )

    duplicate_game_count = int(con.execute("SELECT COUNT(*) FROM duplicate_game_ids").fetchone()[0])
    duplicate_turn_count = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM turns
            WHERE game_id IN (SELECT game_id FROM duplicate_game_ids)
            """
        ).fetchone()[0]
    )

    con.execute("DELETE FROM turns WHERE game_id IN (SELECT game_id FROM duplicate_game_ids)")
    con.execute("DELETE FROM source_games WHERE merged_game_id IN (SELECT game_id FROM duplicate_game_ids)")
    con.execute("DELETE FROM games WHERE game_id IN (SELECT game_id FROM duplicate_game_ids)")
    con.execute("DROP TABLE duplicate_game_ids")

    return duplicate_game_count, duplicate_turn_count


def build_merged_benchmark_duckdb(
    source_paths: list[Path],
    output_path: Path,
    *,
    deduplicate: bool = True,
) -> tuple[int, int]:
    """Build one merged raw benchmark database from multiple source databases."""
    if not source_paths:
        raise ValueError("At least one source DuckDB is required")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(output_path))
    try:
        drop_existing_objects(con)
        for source_index, source_path in enumerate(source_paths):
            source_alias = f"src_{source_index}"
            con.execute(
                f"ATTACH {quote_literal(str(source_path))} AS {quote_identifier(source_alias)} (READ_ONLY)"
            )
            try:
                if source_index == 0:
                    create_output_schema(con, source_alias)
                merge_source(con, source_alias, source_path, source_index)
            finally:
                con.execute(f"DETACH {quote_identifier(source_alias)}")

        duplicate_games = 0
        duplicate_turns = 0
        if deduplicate:
            duplicate_games, duplicate_turns = deduplicate_exact_games(con)

        con.execute("CHECKPOINT")
        return duplicate_games, duplicate_turns
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Raw benchmark DuckDB to merge. Repeat for multiple files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Merged DuckDB path to write.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep exact duplicate game transcripts instead of deduplicating them.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    duplicate_games, duplicate_turns = build_merged_benchmark_duckdb(
        args.source,
        args.output,
        deduplicate=not args.keep_duplicates,
    )

    print(f"wrote {args.output}")
    print(f"deduplicated games: {duplicate_games}")
    print(f"deduplicated turns: {duplicate_turns}")


if __name__ == "__main__":
    main()
