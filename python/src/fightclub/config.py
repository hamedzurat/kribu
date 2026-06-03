import os

# File path to save/resume tournament matches
CSV_PATH = "fightclub_results.csv"

# File path to save the generated Elo vs time graph
PLOT_PATH = "elo_time_tradeoff.png"

# File path to save the tournament summary (for AI readability)
SUMMARY_PATH = "fightclub_summary.md"

# Directory containing the .pt model files
MODELS_DIR = "models"

# Maximum turns per game.
# Lower than benchmark mode so human-facing comparison runs finish faster and less often devolve into long slogs.
MAX_TURNS = 512

# Total games to play per pair.
# This should be an even number so that starting color is exactly split 50/50.
# Keep this modest so Fight Club stays responsive as the model roster grows.
GAMES_PER_PAIR = 20

# Leave one core free so the machine remains usable while tournaments are running.
NUM_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))

# Elo update scores for different ending states.
# Format: (winner_score, loser_score) or (draw_score_p1, draw_score_p2)
SCORES = {
    "win_elimination": (1.0, 0.0),
    "win_stalemate": (1.0, 0.0),
    "draw_progress_rule": (0.5, 0.5),
    "draw_max_turns": (0.5, 0.5),
    "draw_unknown": (0.5, 0.5),
}
