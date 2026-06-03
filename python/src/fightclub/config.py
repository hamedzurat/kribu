import os

# File path to save/resume tournament matches
CSV_PATH = "fightclub_results.csv"

# File path to save the generated Elo vs time graph
PLOT_PATH = "elo_time_tradeoff.png"

# Directory containing the .pt model files
MODELS_DIR = "models"

# Maximum turns per game
MAX_TURNS = 1024

# Total games to play per pair.
# This should be an even number so that starting color is exactly split 50/50.
GAMES_PER_PAIR = 100

NUM_WORKERS = os.cpu_count() or 4

# Elo update scores for different ending states.
# Format: (winner_score, loser_score) or (draw_score_p1, draw_score_p2)
SCORES = {
    "win_elimination": (1.0, 0.0),
    "win_stalemate": (1.0, 0.0),
    "draw_progress_rule": (0.5, 0.5),
    "draw_max_turns": (0.5, 0.5),
    "draw_unknown": (0.5, 0.5),
}
