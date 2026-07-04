"""Unit tests for the fightclub tournament runner."""

import os
import csv
from fightclub.__main__ import calculate_elo, calculate_time_stats, play_game_worker, write_summary_file


def test_calculate_elo_basic(tmp_path):
    csv_path = os.path.join(tmp_path, "test_elo.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_id",
                "player1",
                "player2",
                "winner",
                "win_reason",
                "turns",
                "p1_time",
                "p2_time",
                "p1_moves",
                "p2_moves",
            ]
        )
        # Player 1 wins
        writer.writerow([1, "A", "B", "player1", "elimination", 10, 1.0, 1.0, 5, 5])
        # Player 2 wins
        writer.writerow([2, "A", "B", "player2", "elimination", 10, 1.0, 1.0, 5, 5])
        # Draw
        writer.writerow([3, "A", "B", "draw", "max_turns", 100, 1.0, 1.0, 50, 50])

    ratings = calculate_elo(csv_path, ["A", "B"])
    # After one win each and a draw, ratings should be close to 1500
    assert ratings["A"] > 0
    assert ratings["B"] > 0
    assert abs(ratings["A"] - 1500) < 50
    assert abs(ratings["B"] - 1500) < 50


def test_calculate_time_stats(tmp_path):
    csv_path = os.path.join(tmp_path, "test_time.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_id",
                "player1",
                "player2",
                "winner",
                "win_reason",
                "turns",
                "p1_time",
                "p2_time",
                "p1_moves",
                "p2_moves",
            ]
        )
        writer.writerow([1, "A", "B", "player1", "elimination", 10, 2.5, 1.5, 5, 3])

    avg_times = calculate_time_stats(csv_path, ["A", "B"])
    # A took 2.5s for 5 moves -> 0.5s/move
    # B took 1.5s for 3 moves -> 0.5s/move
    assert avg_times["A"] == 0.5
    assert avg_times["B"] == 0.5


def test_play_game_basic():
    # Verify that a game between random and greedy players can run
    idx, p1, p2, winner, reason, turns, p1_time, p2_time, p1_moves, p2_moves = play_game_worker(
        (1, "random", "greedy", None, None, 5)
    )
    assert winner in ("player1", "player2", "draw")
    assert reason in ("elimination", "stalemate", "progress_rule", "max_turns", "unknown")
    assert turns > 0
    assert p1_time >= 0
    assert p2_time >= 0
    assert p1_moves >= 0
    assert p2_moves >= 0


def test_calculate_elo_custom_scores(tmp_path):
    import fightclub.config as config

    csv_path = os.path.join(tmp_path, "test_custom_elo.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_id",
                "player1",
                "player2",
                "winner",
                "win_reason",
                "turns",
                "p1_time",
                "p2_time",
                "p1_moves",
                "p2_moves",
            ]
        )
        # Custom draw score for max_turns
        writer.writerow([1, "A", "B", "draw", "max_turns", 100, 1.0, 1.0, 50, 50])

    original_scores = config.SCORES.copy()
    try:
        # Give a penalty to draws
        config.SCORES["draw_max_turns"] = (0.2, 0.2)
        ratings = calculate_elo(csv_path, ["A", "B"])

        # Draw with 0.2 score for each instead of 0.5 should result in lower final ratings than 1500
        assert ratings["A"] < 1500
        assert ratings["B"] < 1500
    finally:
        config.SCORES = original_scores


def test_write_summary_file(tmp_path):
    ratings = {"A": 1600.0, "B": 1450.0}
    avgTimes = {"A": 0.123, "B": 0.456}
    savePath = os.path.join(tmp_path, "summary.md")
    write_summary_file(ratings, avgTimes, savePath)

    assert os.path.exists(savePath)
    with open(savePath, "r", encoding="utf-8") as f:
        summaryContent = f.read()

    assert "# Fight Club Tournament Summary" in summaryContent
    assert "A" in summaryContent
    assert "B" in summaryContent
    assert "1600.0" in summaryContent
    assert "0.123" in summaryContent
