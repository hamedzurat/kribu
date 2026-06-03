"""!
@file __main__.py
@brief Multiprocessed tournament runner for Fight Club.
@details Executes matchups in parallel using ProcessPoolExecutor to leverage multiple CPU cores,
         maintaining CSV persistence, Elo ratings, and generating the performance plot.
"""

import csv
import glob
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn

# Add python/src to sys.path if not present
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kribu
import fightclub.config as config
from arena.player import NeuralPlayer
from fightclub.worker import play_game_worker


## @brief Computes ELO ratings for all players based on the match history in the CSV file.
#  @param csv_path Path to the CSV match history.
#  @param player_names List of all player names to track.
#  @param initial_elo The starting ELO score (default 1500).
#  @param k_factor The update K-factor (default 32).
#  @return Dictionary mapping player names to their computed ELO rating.
def calculate_elo(
    csv_path: str,
    player_names: list[str],
    initial_elo: float = 1500.0,
    k_factor: float = 32.0,
) -> dict[str, float]:
    ratings = {name: initial_elo for name in player_names}
    if not os.path.exists(csv_path):
        return ratings

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p1 = row["player1"]
            p2 = row["player2"]
            winner = row["winner"]

            if p1 not in ratings:
                ratings[p1] = initial_elo
            if p2 not in ratings:
                ratings[p2] = initial_elo

            r1 = ratings[p1]
            r2 = ratings[p2]

            e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
            e2 = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / 400.0))

            if winner == "player1":
                s1, s2 = 1.0, 0.0
            elif winner == "player2":
                s1, s2 = 0.0, 1.0
            else:
                s1, s2 = 0.5, 0.5

            ratings[p1] = r1 + k_factor * (s1 - e1)
            ratings[p2] = r2 + k_factor * (s2 - e2)

    return ratings


## @brief Computes decision speed statistics for all players.
#  @param csv_path Path to the CSV match history.
#  @param player_names List of all player names.
#  @return Dictionary mapping player names to their average decision time per move (seconds).
def calculate_time_stats(csv_path: str, player_names: list[str]) -> dict[str, float]:
    total_time = {name: 0.0 for name in player_names}
    total_moves = {name: 0 for name in player_names}

    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p1 = row["player1"]
                p2 = row["player2"]

                if p1 in total_time:
                    total_time[p1] += float(row.get("p1_time", 0.0))
                    total_moves[p1] += int(row.get("p1_moves", 0))
                if p2 in total_time:
                    total_time[p2] += float(row.get("p2_time", 0.0))
                    total_moves[p2] += int(row.get("p2_moves", 0))

    avg_times = {}
    for name in player_names:
        if total_moves[name] > 0:
            avg_times[name] = total_time[name] / total_moves[name]
        else:
            avg_times[name] = 0.0
    return avg_times


## @brief Generates an ELO vs. Decision Time scatter plot and saves it as an image.
#  @param ratings Dictionary mapping player names to ELO ratings.
#  @param avg_times Dictionary mapping player names to average time per move.
#  @param save_path Path to save the generated image.
def plot_elo_time(ratings: dict[str, float], avg_times: dict[str, float], save_path: str) -> None:
    # Use a clean, modern style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

    # Convert to arrays
    names = list(ratings.keys())
    x = [avg_times.get(name, 0.0) for name in names]
    y = [ratings[name] for name in names]

    # Handle zero times to avoid log scale errors
    min_nonzero = min([val for val in x if val > 0] or [1e-6])
    x_adjusted = [val if val > 0 else min_nonzero * 0.1 for val in x]

    # Color map
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(names)))

    scatter = ax.scatter(x_adjusted, y, c=colors, s=200, alpha=0.85, edgecolors="black", linewidths=1.2)

    # Annotate player names
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (x_adjusted[i], y[i]),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="grey", lw=0.5),
        )

    ax.set_title("Fight Club Performance Trade-off: Elo vs. Decision Time", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Elo Rating", fontsize=12, fontweight="bold")

    # Set x-scale to log if there is a massive range
    if max(x_adjusted) / min(x_adjusted) > 10.0:
        ax.set_xscale("log")
        ax.set_xlabel("Average Time per Move (seconds, Log Scale)", fontsize=12, fontweight="bold")
    else:
        ax.set_xlabel("Average Time per Move (seconds)", fontsize=12, fontweight="bold")

    # Layout adjustment and saving
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


## @brief Main entry point to register players, run matches, update ratings, and plot.
def main() -> None:
    console = Console()
    console.print("[bold magenta]Starting FIGHT CLUB Engine (Parallel Multiprocessed)...[/bold magenta]")

    # Create directories if they don't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    # 1. Register players
    player_names = [
        "random",
        "greedy",
        "mcts_200",
        "mcts_400",
        "mcts_600",
        "mcts_800",
        "minimax_2",
        "minimax_4",
        "minimax_8",
        "minimax_2_mad2",
        "minimax_4_mad2",
        "minimax_8_mad2",
    ]

    # Model file tracking
    pt_files = glob.glob(os.path.join(config.MODELS_DIR, "*.pt"))
    model_paths = {}
    for pt_path in pt_files:
        name = os.path.splitext(os.path.basename(pt_path))[0]
        model_paths[name] = pt_path
        player_names.append(name)
        console.print(f"[green]Registered model player: {name} ({pt_path})[/green]")

    player_names = sorted(list(set(player_names)))
    console.print(f"Registered [cyan]{len(player_names)}[/cyan] players: {', '.join(player_names)}\n")

    # 2. Build full matchup schedule
    # Each pair plays GAMES_PER_PAIR times (50/50 starting split)
    full_schedule = []
    for i in range(len(player_names)):
        for j in range(i + 1, len(player_names)):
            p1, p2 = player_names[i], player_names[j]
            half_games = config.GAMES_PER_PAIR // 2
            # p1 starts, p2 second
            for _ in range(half_games):
                full_schedule.append((p1, p2))
            # p2 starts, p1 second
            for _ in range(half_games):
                full_schedule.append((p2, p1))

    # Shuffle to distribute matches evenly
    random.seed(42)
    random.shuffle(full_schedule)

    # 3. Read existing progress from CSV to resume
    completed_counts = {(p1, p2): 0 for p1 in player_names for p2 in player_names}
    if os.path.exists(config.CSV_PATH):
        with open(config.CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p1 = row.get("player1")
                p2 = row.get("player2")
                if (p1, p2) in completed_counts:
                    completed_counts[(p1, p2)] += 1

    # Filter schedule
    remaining_schedule = []
    current_completed_counts = completed_counts.copy()
    for p1, p2 in full_schedule:
        if current_completed_counts[(p1, p2)] > 0:
            current_completed_counts[(p1, p2)] -= 1
        else:
            remaining_schedule.append((p1, p2))

    total_games = len(full_schedule)
    played_games = total_games - len(remaining_schedule)

    console.print(f"Total planned games: [yellow]{total_games}[/yellow]")
    console.print(f"Already completed games: [green]{played_games}[/green]")
    console.print(f"Remaining games to play: [cyan]{len(remaining_schedule)}[/cyan]\n")

    # Create CSV file with header if not present
    if not os.path.exists(config.CSV_PATH):
        with open(config.CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
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

    # 4. Run tournament match-ups in parallel processes
    try:
        elo_ratings = calculate_elo(config.CSV_PATH, player_names)
        avg_times = calculate_time_stats(config.CSV_PATH, player_names)
        plot_elo_time(elo_ratings, avg_times, config.PLOT_PATH)

        if not remaining_schedule:
            console.print("[bold green]All tournament games have already been completed![/bold green]")
            console.print(f"Elo vs time plot updated at: [bold yellow]{config.PLOT_PATH}[/bold yellow]")
            return

        with Live(
            Panel("Initializing parallel match execution...", border_style="cyan"),
            console=console,
            refresh_per_second=4,
        ) as live:
            progress_bar = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("Played: {task.completed}/{task.total}"),
            )
            task_id = progress_bar.add_task("Matches Progress", completed=played_games, total=total_games)

            # Start process pool executor
            with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
                # Submit all remaining games
                futures = {}
                for idx, (p1, p2) in enumerate(remaining_schedule, start=played_games + 1):
                    task_args = (idx, p1, p2, model_paths.get(p1), model_paths.get(p2), config.MAX_TURNS)
                    fut = executor.submit(play_game_worker, task_args)
                    futures[fut] = (p1, p2)

                completed_games_count = played_games

                # As each game finishes
                for fut in as_completed(futures):
                    p1, p2 = futures[fut]
                    try:
                        res_idx, res_p1, res_p2, winner_id, reason, turns, p1_time, p2_time, p1_moves, p2_moves = (
                            fut.result()
                        )

                        # Append result to CSV
                        with open(config.CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [
                                    res_idx,
                                    res_p1,
                                    res_p2,
                                    winner_id,
                                    reason,
                                    turns,
                                    p1_time,
                                    p2_time,
                                    p1_moves,
                                    p2_moves,
                                ]
                            )

                        # Recalculate Elo and speeds
                        elo_ratings = calculate_elo(config.CSV_PATH, player_names)
                        avg_times = calculate_time_stats(config.CSV_PATH, player_names)

                        # Periodically update plot
                        completed_games_count += 1
                        if completed_games_count % 10 == 0 or completed_games_count == total_games:
                            plot_elo_time(elo_ratings, avg_times, config.PLOT_PATH)

                        # Update progress bar
                        progress_bar.update(task_id, completed=completed_games_count)

                        # Rebuild live standings table
                        dashboard_table = Table(title="Live ELO Standings", expand=True)
                        dashboard_table.add_column("Rank", justify="center", style="cyan")
                        dashboard_table.add_column("Player", style="bold white")
                        dashboard_table.add_column("Elo Rating", justify="right", style="green")
                        dashboard_table.add_column("Avg Time/Move (s)", justify="right", style="yellow")

                        # Sort by Elo
                        sorted_players = sorted(player_names, key=lambda name: elo_ratings[name], reverse=True)
                        for rank, name in enumerate(sorted_players, 1):
                            dashboard_table.add_row(
                                str(rank),
                                name,
                                f"{elo_ratings[name]:.1f}",
                                f"{avg_times.get(name, 0.0):.5f}",
                            )

                        layout_grid = Table.grid(padding=(1, 0))
                        layout_grid.add_row(Panel(progress_bar, title="Tournament Progress", border_style="cyan"))
                        layout_grid.add_row(Panel(dashboard_table, title="Current Standings", border_style="magenta"))

                        winner_name = (
                            res_p1 if winner_id == "player1" else (res_p2 if winner_id == "player2" else "draw")
                        )
                        layout_grid.add_row(
                            f"Last finished: [bold cyan]{res_p1}[/bold cyan] vs [bold yellow]{res_p2}[/bold yellow] -> Winner: [bold green]{winner_name}[/bold green] ({reason})"
                        )

                        live.update(Panel(layout_grid, title="[bold yellow]FIGHT CLUB LIVE DASHBOARD[/bold yellow]"))
                    except Exception as exc:
                        console.print(f"[red]Match {p1} vs {p2} generated an exception: {exc}[/red]")

        # Final plot update
        plot_elo_time(elo_ratings, avg_times, config.PLOT_PATH)
        console.print("\n[bold green]Tournament completed successfully![/bold green]")
        console.print(f"Results saved to [bold yellow]{config.CSV_PATH}[/bold yellow].")
        console.print(f"Performance chart saved to [bold yellow]{config.PLOT_PATH}[/bold yellow].")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Tournament interrupted by user (Ctrl+C). Standings saved.[/bold yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
