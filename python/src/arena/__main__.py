"""Arena evaluator to test neural network models against search or baseline players."""

import argparse
import csv
import os
import random
import sys
import time
import shutil

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn

# Add python/src to sys.path if not present
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kribu
from arena.player import NeuralPlayer


def count_bits(mask: int) -> int:
    """Counts set bits (pieces) in a uint64 mask."""
    return bin(mask).count("1")


def get_winner_and_reason(status, is_model_turn):
    """
    Determines the winner relative to the model (model vs opponent vs draw)
    and the reason for the outcome.
    """
    if status == kribu.GameStatus.ONGOING:
        return None, None

    if status in (kribu.GameStatus.ME_WINS_ELIMINATION, kribu.GameStatus.ME_WINS_STALEMATE):
        winner = "model" if is_model_turn else "opponent"
        reason = "elimination" if status == kribu.GameStatus.ME_WINS_ELIMINATION else "stalemate"
        return winner, reason
    elif status in (kribu.GameStatus.OPP_WINS_ELIMINATION, kribu.GameStatus.OPP_WINS_STALEMATE):
        winner = "opponent" if is_model_turn else "model"
        reason = "elimination" if status == kribu.GameStatus.OPP_WINS_ELIMINATION else "stalemate"
        return winner, reason
    elif status == kribu.GameStatus.DRAW_PROGRESS_RULE:
        return "draw", "progress_rule"

    return "draw", "unknown"


def make_dashboard(games_played, args, model_wins, opp_wins, draws, current_game_idx, current_game_info, history_log):
    """Creates a beautiful Layout dashboard of the arena status."""
    layout = Layout()
    layout.split_column(Layout(name="header", size=4), Layout(name="main", ratio=1), Layout(name="footer", size=3))

    # Count draw reasons dynamically from history_log for detailed top bar summary
    draw_max_turns = sum(1 for gh in history_log if gh["winner"] == "draw" and gh["reason"] == "max_turns")
    draw_progress = sum(1 for gh in history_log if gh["winner"] == "draw" and gh["reason"] == "progress_rule")

    draw_details = []
    if draw_max_turns > 0:
        draw_details.append(f"max turns: {draw_max_turns}")
    if draw_progress > 0:
        draw_details.append(f"progress rule: {draw_progress}")

    draw_details_str = f" ({', '.join(draw_details)})" if draw_details else ""

    # Header Panel
    win_rate = (model_wins / games_played * 100) if games_played > 0 else 0.0
    header_text = (
        f"[bold yellow]SHOLO GUTI ARENA[/bold yellow] | Model: [green]{args.model_path}[/green] | Opponent: [red]{args.opponent.upper()}[/red] | Games: [cyan]{games_played}/{args.games}[/cyan] | Max Turns: [magenta]{args.max_turns}[/magenta]\n"
        f"Model Win Rate: [bold green]{win_rate:.1f}%[/bold green] | Record: [green]{model_wins}[/green]-[red]{opp_wins}[/red]-[white]{draws}{draw_details_str}[/white] | CSV: [white]{args.csv_path}[/white]"
    )
    header_panel = Panel(
        header_text,
        style="bold white",
        border_style="cyan",
    )
    layout["header"].update(header_panel)

    # Main area split into Left (stats table) and Right (current game detail)
    layout["main"].split_row(Layout(name="stats_panel", ratio=1), Layout(name="game_panel", ratio=1))

    # Stats Panel (Left)
    stats_table = Table(title="Completed Games History", expand=True)
    stats_table.add_column("Game ID", justify="center")
    stats_table.add_column("Winner", justify="center")
    stats_table.add_column("Reason", justify="center")
    stats_table.add_column("Turns", justify="right")
    stats_table.add_column("Win By", justify="center")
    stats_table.add_column("Time (M/O)", justify="center")

    # Determine dynamic history size based on terminal height
    terminal_height = shutil.get_terminal_size().lines
    history_size = max(1, terminal_height - 12)

    for idx, gh in enumerate(history_log[-history_size:]):  # Display dynamically sized history
        winner_style = "green" if gh["winner"] == "model" else ("red" if gh["winner"] == "opponent" else "white")
        game_id_str = f"{gh['game_id']}*" if gh["model_started"] == "Yes" else str(gh["game_id"])
        win_by_pieces = abs(gh.get("model_pieces", 0) - gh.get("opp_pieces", 0))
        win_by_str = f"{win_by_pieces} pcs" if gh["winner"] in ("model", "opponent") else "-"

        m_time = gh.get("model_time", 0.0)
        o_time = gh.get("opp_time", 0.0)
        ratio = (o_time / m_time) if m_time > 0 else 0.0
        speed_suffix = f" {ratio:.1f}x" if m_time > 0 and o_time > 0 else ""
        time_str = f"{m_time:.3f} / {o_time:.3f}{speed_suffix}"

        stats_table.add_row(
            game_id_str,
            f"[{winner_style}]{gh['winner'].upper()}[/{winner_style}]",
            gh["reason"],
            str(gh["turns"]),
            win_by_str,
            time_str,
        )
    layout["main"]["stats_panel"].update(Panel(stats_table, border_style="yellow"))

    # Current Game Panel (Right)
    game_details = Table.grid(padding=(0, 2))
    game_details.add_column(style="bold cyan")
    game_details.add_column()

    game_details.add_row("Active Game ID:", f"[bold white]{current_game_idx}[/bold white]")
    game_details.add_row(
        "Model Started:", f"[bold magenta]{current_game_info.get('model_started', 'N/A')}[/bold magenta]"
    )
    game_details.add_row("Current Turn:", f"[bold yellow]{current_game_info.get('turn_idx', 0)}[/bold yellow]")
    game_details.add_row("Model Pieces:", f"[green]{current_game_info.get('model_pieces', 16)}[/green]")
    game_details.add_row("Opponent Pieces:", f"[red]{current_game_info.get('opp_pieces', 16)}[/red]")
    game_details.add_row("Model Speed (avg):", f"[cyan]{current_game_info.get('model_speed', 'N/A')}[/cyan]")
    game_details.add_row("Last Move Played:", f"[bold white]{current_game_info.get('last_move', 'None')}[/bold white]")
    game_details.add_row(
        "Active Capture Node:", f"[bold yellow]{current_game_info.get('active_cap', 'None')}[/bold yellow]"
    )
    game_details.add_row("Move History:", f"{current_game_info.get('moves_history', '')}")

    layout["main"]["game_panel"].update(Panel(game_details, title="Current Game Live Stats", border_style="magenta"))

    # Footer Progress Panel
    progress_bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    _ = progress_bar.add_task("Match Progress", completed=games_played, total=args.games)
    layout["footer"].update(Panel(progress_bar, border_style="cyan"))

    return layout


def main():
    parser = argparse.ArgumentParser(description="Test Sholo Guti PyTorch model against search or baseline opponents.")
    parser.add_argument(
        "--model-path", "-m", type=str, default="checkpoints/best_model.pt", help="Path to .pt checkpoint file."
    )
    parser.add_argument(
        "--opponent",
        "-o",
        type=str,
        choices=["minimax", "minimax4", "minimax_mad", "random", "mcts", "greedy"],
        default="minimax",
        help="Opponent type: 'minimax', 'minimax4', 'minimax_mad', 'random', 'mcts', or 'greedy'.",
    )
    parser.add_argument("--games", "-n", type=int, default=128, help="Number of games to play.")
    parser.add_argument("--max-turns", "-t", type=int, default=1024, help="Max turns per game.")
    parser.add_argument(
        "--csv-path", "-c", type=str, default="arena_results.csv", help="Path to save results CSV file."
    )

    args = parser.parse_args()

    console = Console()

    console.print("[bold cyan]Initializing Arena...[/bold cyan]")
    console.print(f"Loading NeuralPlayer from: [bold yellow]{args.model_path}[/bold yellow]")
    try:
        player = NeuralPlayer(model_path=args.model_path)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        console.print("[bold yellow]Please train the model first or specify a valid checkpoint using -m.[/bold yellow]")
        sys.exit(1)

    console.print(f"Opponent: [bold yellow]{args.opponent.upper()}[/bold yellow]")
    console.print(f"Games to Play: [bold yellow]{args.games}[/bold yellow]")
    console.print(f"Max Turns per Game: [bold yellow]{args.max_turns}[/bold yellow]\n")

    history_log = []
    model_wins = 0
    opp_wins = 0
    draws = 0

    # Write CSV header if not exists or overwrite
    with open(args.csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_id",
                "model_started",
                "opponent_type",
                "winner",
                "win_reason",
                "turns_played",
                "model_pieces_left",
                "opponent_pieces_left",
            ]
        )

    current_game_info = {}

    try:
        with Live(
            Panel("Starting arena simulation...", border_style="cyan"), console=console, refresh_per_second=10
        ) as live:
            for game_idx in range(1, args.games + 1):
                # Alternating colors: model plays first in odd games, second in even games
                model_is_p1 = game_idx % 2 != 0
                model_started_str = "Yes" if model_is_p1 else "No"

                state = kribu.INITIAL_STATE
                is_p1_turn = True
                turn_idx = 0
                last_move_str = "None"

                # Track pieces
                model_pieces = 16
                opp_pieces = 16

                # Track time
                model_time = 0.0
                opp_time = 0.0

                # Track moves/decisions count
                model_moves = 0

                # Track moves played in the current game
                moves_played = []

                while True:
                    # Check status
                    status = kribu.get_game_status(state)
                    is_model_turn = is_p1_turn == model_is_p1

                    if status != kribu.GameStatus.ONGOING:
                        winner, reason = get_winner_and_reason(status, is_model_turn)
                        break

                    if turn_idx >= args.max_turns:
                        winner, reason = "draw", "max_turns"
                        break

                    # Count pieces from active perspective
                    p_me = count_bits(state.me)
                    p_opp = count_bits(state.opp)
                    if is_model_turn:
                        model_pieces = p_me
                        opp_pieces = p_opp
                    else:
                        model_pieces = p_opp
                        opp_pieces = p_me

                    # Update live display info
                    current_game_info = {
                        "model_started": model_started_str,
                        "turn_idx": turn_idx,
                        "model_pieces": model_pieces,
                        "opp_pieces": opp_pieces,
                        "last_move": last_move_str,
                        "active_cap": str(state.activeCaptureIdx) if state.activeCaptureIdx != -1 else "None",
                        "moves_history": " -> ".join(moves_played),
                        "model_speed": f"{(opp_time / model_time):.1f}x" if model_time > 0 else "0.0x",
                    }
                    live.update(
                        make_dashboard(
                            game_idx - 1,
                            args,
                            model_wins,
                            opp_wins,
                            draws,
                            game_idx,
                            current_game_info,
                            history_log,
                        )
                    )

                    # Retrieve legal moves
                    valid_moves = list(kribu.all_possible_moves(state))
                    if not valid_moves:
                        # Stalemate, handled by get_game_status next iter
                        turn_idx += 1
                        continue

                    # Player makes move selection
                    start_t = time.perf_counter()
                    if is_model_turn:
                        # Model chooses move using NN policy with policy masking
                        move_idx, _ = player.get_move(
                            me_mask=state.me,
                            opp_mask=state.opp,
                            active_capture_idx=state.activeCaptureIdx,
                            valid_moves=valid_moves,
                            state=state,
                        )
                        model_time += time.perf_counter() - start_t
                        model_moves += 1
                    else:
                        # Opponent selection
                        if args.opponent == "minimax":
                            move_idx = kribu.minimax_player_8(state)
                        elif args.opponent == "minimax4":
                            move_idx = kribu.minimax_player_4(state)
                        elif args.opponent == "minimax_mad":
                            move_idx = kribu.minimax_player_8_mad2(state)
                        elif args.opponent == "mcts":
                            move_idx = kribu.mcts_player_800(state)
                        elif args.opponent == "greedy":
                            move_idx = kribu.greedy_player(state)
                        else:
                            move_idx = random.choice(valid_moves)

                        if move_idx not in valid_moves:
                            move_idx = valid_moves[0]
                        opp_time += time.perf_counter() - start_t

                    # Format move description
                    if move_idx == kribu.END_CHAIN_MOVE:
                        last_move_str = "Pass/End Chain"
                        move_short_str = "Pass"
                    else:
                        m = kribu.decode_move(move_idx)
                        last_move_str = f"{m.fromNode} -> {m.toNode}"
                        if m.captured != -1:
                            last_move_str += f" (Capture {m.captured})"

                        player_color = "green" if is_model_turn else "red"
                        move_short_str = f"[{player_color}]{m.fromNode}->{m.toNode}[/{player_color}]"

                    moves_played.append(move_short_str)

                    # Apply move (this also automatically appends hash history inside apply_move C++)
                    next_state = kribu.apply_move(state, move_idx)

                    # Transition turn
                    if next_state.activeCaptureIdx == -1:
                        state = kribu.flip_board(next_state)
                        is_p1_turn = not is_p1_turn
                    else:
                        state = next_state

                    turn_idx += 1

                # Update stats
                if winner == "model":
                    model_wins += 1
                elif winner == "opponent":
                    opp_wins += 1
                else:
                    draws += 1

                # Add to history log
                game_result = {
                    "game_id": game_idx,
                    "model_started": "Yes" if model_is_p1 else "No",
                    "winner": winner,
                    "reason": reason,
                    "turns": turn_idx,
                    "model_pieces": model_pieces,
                    "opp_pieces": opp_pieces,
                    "model_time": model_time,
                    "opp_time": opp_time,
                    "model_moves": model_moves,
                }
                history_log.append(game_result)

                # Persist to CSV file
                with open(args.csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            game_idx,
                            "Yes" if model_is_p1 else "No",
                            args.opponent,
                            winner,
                            reason,
                            turn_idx,
                            model_pieces,
                            opp_pieces,
                        ]
                    )

                # Update live display after game completes
                live.update(
                    make_dashboard(
                        game_idx, args, model_wins, opp_wins, draws, game_idx, current_game_info, history_log
                    )
                )
                time.sleep(0.5)

        console.print("\n[bold green]Arena evaluation completed successfully![/bold green]")
        console.print(f"Results saved to [bold yellow]{args.csv_path}[/bold yellow].")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Evaluation interrupted by user (Ctrl+C). Exiting gracefully...[/bold yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
