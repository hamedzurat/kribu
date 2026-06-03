"""Configuration settings for the Fight Club tournament."""

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
GAMES_PER_PAIR = 2

# Number of parallel worker processes for running games
import os

NUM_WORKERS = os.cpu_count() or 4
