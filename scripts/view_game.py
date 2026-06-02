#!/usr/bin/env python3
"""
Interactive Sholo Guti game viewer for Parquet history logs.
Reads footer metadata and step-by-step game logs to visualize game playbacks.
"""

import os
import sys
import tty
import termios
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns

# Add python/src to PYTHONPATH dynamically so we can import kribu
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python", "src"))
import kribu


def count_bits(mask: int) -> int:
    """Counts the number of active bits (pieces) in the bitmask."""
    return bin(mask).count("1")


def flip_bits(mask: int) -> int:
    """Flips a 37-node bitmask 180 degrees symmetrically."""
    flipped = 0
    for i in range(37):
        if (mask & (1 << i)) != 0:
            flipped |= 1 << (36 - i)
    return flipped


def get_key() -> str:
    """Reads a single keypress from standard input without blocking."""
    if not sys.stdin.isatty():
        ch = sys.stdin.read(1)
        if not ch:
            return "q"  # Quit if EOF reached
        return ch

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # Escape or arrow key sequence
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                if ch3 == "A":
                    return "up"
                elif ch3 == "B":
                    return "down"
                elif ch3 == "C":
                    return "right"
                elif ch3 == "D":
                    return "left"
            return "esc"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def format_move(move_id: int, is_p1_turn: bool) -> str:
    """Decodes and formats a move ID into physical board coordinates."""
    if move_id == kribu.END_CHAIN_MOVE:
        return "End capture chain / Pass turn"

    try:
        m = kribu.decode_move(move_id)
        if is_p1_turn:
            from_node = m.fromNode
            to_node = m.toNode
            captured = m.captured
        else:
            from_node = 36 - m.fromNode
            to_node = 36 - m.toNode
            captured = 36 - m.captured if m.captured != -1 else -1

        if captured != -1:
            return f"{from_node} ──(jump over {captured})──> {to_node} (Capture)"
        return f"{from_node} ──> {to_node}"
    except Exception:
        return f"Unknown Move (ID: {move_id})"


def render_board(green_mask: int, red_mask: int, active_capture_idx: int) -> str:
    """Renders the physical board state from P1's fixed perspective (Green at bottom, Red at top)."""

    def char(node_id):
        is_me = (green_mask & (1 << node_id)) != 0
        is_opp = (red_mask & (1 << node_id)) != 0
        if is_me:
            return "[bold green]M[/bold green]"
        if is_opp:
            return "[bold red]O[/bold red]"
        if active_capture_idx == node_id:
            return "[bold yellow]*[/bold yellow]"
        return "[grey50].[/grey50]"

    c = [char(i) for i in range(37)]
    return f"""
   {c[0]}───────────────{c[1]}───────────────{c[2]}
     ╲             │             ╱
       ╲           │           ╱
         ╲         │         ╱
           {c[3]}───────{c[4]}───────{c[5]}
             ╲      │      ╱
               ╲    │    ╱
                 ╲  │  ╱
    {c[6]}───────{c[7]}───────{c[8]}───────{c[9]}───────{c[10]}
    │ ╲     │     ╱ │ ╲     │     ╱ │
    │   ╲   │   ╱   │   ╲   │   ╱   │
    │     ╲ │ ╱     │     ╲ │ ╱     │
    {c[11]}───────{c[12]}───────{c[13]}───────{c[14]}───────{c[15]}
    │     ╱ │ ╲     │     ╱ │ ╲     │
    │   ╱   │   ╲   │   ╱   │   ╲   │
    │ ╱     │     ╲ │ ╱     │     ╲ │
    {c[16]}───────{c[17]}───────{c[18]}───────{c[19]}───────{c[20]}
    │ ╲     │     ╱ │ ╲     │     ╱ │
    │   ╲   │   ╱   │   ╲   │   ╱   │
    │     ╲ │ ╱     │     ╲ │ ╱     │
    {c[21]}───────{c[22]}───────{c[23]}───────{c[24]}───────{c[25]}
    │     ╱ │ ╲     │     ╱ │ ╲     │
    │   ╱   │   ╲   │   ╱   │   ╲   │
    │ ╱     │     ╲ │ ╱     │     ╲ │
    {c[26]}───────{c[27]}───────{c[28]}───────{c[29]}───────{c[30]}
                 ╱  │  ╲
               ╱    │    ╲
             ╱      │      ╲
            {c[31]}───────{c[32]}───────{c[33]}
         ╱          │         ╲
       ╱            │           ╲
     ╱              │             ╲
    {c[34]}───────────────{c[35]}───────────────{c[36]}
    """


def select_game_from_duckdb(console: Console) -> int:
    """Lists simulated games from DuckDB and prompts user to pick one."""
    db_path = "benchmark/dataset.duckdb"
    if not os.path.exists(db_path):
        console.print("[bold red]Error: No dataset.duckdb found.[/bold red]")
        sys.exit(1)

    import duckdb

    con = duckdb.connect(db_path)
    rows = con.execute(
        "SELECT game_id, p1_name, p2_name, outcome, reason, total_turns, win_margin FROM games"
    ).fetchall()

    if not rows:
        console.print("[bold yellow]Database is empty. No games simulated yet.[/bold yellow]")
        sys.exit(0)

    outcome_map = {0: "P1_WINS", 1: "P2_WINS", 2: "DRAW"}

    games = []
    for row in rows:
        games.append(
            {
                "id": str(row[0]),
                "p1Name": row[1],
                "p2Name": row[2],
                "outcome": outcome_map.get(row[3], "DRAW"),
                "reason": row[4],
                "totalTurns": str(row[5]),
                "winMargin": str(row[6]),
            }
        )

    # Sort games by Winner (outcome) primarily, then by ID
    games.sort(key=lambda g: (g["outcome"], int(g["id"])))

    # Split the games list to render two tables side-by-side
    num_cols = 3
    chunk_size = (len(games) + num_cols - 1) // num_cols
    tables = []

    reason_map = {
        "ELIMINATION": "ELIM",
        "STALEMATE": "STAL",
        "INVALID_MOVE": "INVL",
        "DRAW_MAX_TURNS": "DRAW",
        "DRAW_PROGRESS_RULE": "DRAW",
    }

    for col_idx in range(num_cols):
        start_idx = col_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(games))
        if start_idx >= len(games):
            break

        table = Table(title=f"Games {start_idx + 1} - {end_idx}", header_style="bold yellow", expand=True)
        table.add_column("No.", justify="right", style="cyan")
        table.add_column("Winner", style="green")
        table.add_column("Loser", style="red")
        table.add_column("Reason", style="blue")
        table.add_column("Turns", justify="right", style="white")
        table.add_column("Win", justify="right", style="yellow")

        for idx in range(start_idx, end_idx):
            g = games[idx]
            win_by = g.get("winMargin", "0") if g.get("outcome") != "DRAW" else "-"
            short_reason = reason_map.get(g["reason"], g["reason"])

            if g["outcome"] == "P1_WINS":
                winner = g["p1Name"]
                loser = g["p2Name"]
            else:
                winner = g["p2Name"]
                loser = g["p1Name"]

            table.add_row(
                g["id"],
                winner,
                loser,
                short_reason,
                g["totalTurns"],
                win_by,
            )
        tables.append(table)

    console.print(Columns(tables, equal=True, expand=True))

    valid_ids = {g["id"] for g in games}

    while True:
        choice = input("\nSelect a game ID to view or [q] to quit: ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        if choice in valid_ids:
            return int(choice)
        console.print(f"[red]Invalid selection. Game ID {choice} not found.[/red]")


def main():
    console = Console()

    # Resolve file path or game ID
    game_id = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if "/" in arg or "." in arg:
            import re

            m = re.search(r"\d+", os.path.basename(arg))
            if m:
                game_id = int(m.group(0))
        else:
            try:
                game_id = int(arg)
            except ValueError:
                pass

    if game_id is None:
        game_id = select_game_from_duckdb(console)

    db_path = "benchmark/dataset.duckdb"
    if not os.path.exists(db_path):
        console.print("[bold red]Error: No dataset.duckdb found.[/bold red]")
        sys.exit(1)

    import duckdb

    # Read from DuckDB
    con = duckdb.connect(db_path)

    # Query game metadata
    game_row = con.execute(
        "SELECT p1_name, p2_name, outcome, reason, total_turns, win_margin FROM games WHERE game_id = ?",
        [game_id],
    ).fetchone()

    if not game_row:
        console.print(f"[bold red]Error: Game ID {game_id} not found in database.[/bold red]")
        sys.exit(1)

    p1_name, p2_name, outcome_int, reason, total_turns, win_margin = game_row
    outcome_map = {0: "P1_WINS", 1: "P2_WINS", 2: "DRAW"}
    outcome = outcome_map.get(outcome_int, "DRAW")

    # Query turns data
    turns = con.execute(
        "SELECT me, opp, active_capture_idx, is_p1_turn, chosen_move, player_played FROM turns WHERE game_id = ? ORDER BY turn_idx",
        [game_id],
    ).fetchall()

    if not turns:
        console.print(f"[bold red]Error: No turns found for Game ID {game_id} in database.[/bold red]")
        sys.exit(1)

    # Read column vectors
    me_col = [row[0] for row in turns]
    opp_col = [row[1] for row in turns]
    active_cap_col = [row[2] for row in turns]
    is_p1_turn_col = [row[3] for row in turns]
    chosen_move_col = [row[4] for row in turns]
    player_played_col = [row[5] for row in turns]

    turn_idx = 0
    actual_length = len(me_col)

    while True:
        # Determine fixed physical board piece masks
        is_p1_turn = is_p1_turn_col[turn_idx]
        me_mask = me_col[turn_idx]
        opp_mask = opp_col[turn_idx]
        active_cap = active_cap_col[turn_idx]

        if is_p1_turn:
            green_mask = me_mask
            red_mask = opp_mask
            active_cap_phys = active_cap
        else:
            green_mask = flip_bits(opp_mask)
            red_mask = flip_bits(me_mask)
            active_cap_phys = 36 - active_cap if active_cap != -1 else -1

        # Clear terminal screen
        os.system("clear" if os.name == "posix" else "cls")

        # Header Info
        header_text = Text()
        header_text.append("SHOLO GUTI GAME PLAYBACK ANALYSIS\n", style="bold yellow")
        header_text.append("Matchup: ", style="bold")
        header_text.append(f"{p1_name} (P1/Green)", style="green")
        header_text.append(" vs ", style="white")
        header_text.append(f"{p2_name} (P2/Red)\n", style="red")
        header_text.append(f"Result: {outcome} ({reason})", style="bold magenta")
        if outcome != "DRAW":
            header_text.append(f" | Margin: Winner by {win_margin} pieces", style="bold yellow")

        console.print(Panel(header_text, border_style="cyan"))

        # Render board
        console.print(render_board(green_mask, red_mask, active_cap_phys))

        # Turn statistics
        active_player_name = p1_name if is_p1_turn else p2_name
        active_player_color = "green" if is_p1_turn else "red"

        info_table = Table.grid(padding=(0, 2))
        info_table.add_column(style="bold cyan")
        info_table.add_column()

        info_table.add_row("Turn/Ply Index:", f"{turn_idx + 1} / {actual_length}")
        info_table.add_row("Total Turns (Game):", f"[bold white]{total_turns}[/bold white]")

        info_table.add_row(
            "Active Turn:", f"[bold {active_player_color}]{active_player_name}[/bold {active_player_color}]"
        )
        info_table.add_row("Move Decided By:", f"[bold magenta]{player_played_col[turn_idx]}[/bold magenta]")
        info_table.add_row("Green Pieces Remaining (P1):", f"[green]{count_bits(green_mask)}[/green]")
        info_table.add_row("Red Pieces Remaining (P2):", f"[red]{count_bits(red_mask)}[/red]")

        chosen_m_id = chosen_move_col[turn_idx]
        info_table.add_row("Move Played:", f"[bold yellow]{format_move(chosen_m_id, is_p1_turn)}[/bold yellow]")

        console.print(Panel(info_table, title="Turn Information", border_style="yellow"))

        # Footer instructions
        console.print(
            "\n[bold reverse] [Left Arrow / a]: Prev Turn | [Right Arrow / d / Space / Enter]: Next Turn | [r]: Reset | [q / Esc]: Quit [/bold reverse]"
        )

        # Process input
        key = get_key()
        if key in ("right", "d", " ", "\r", "\n"):
            turn_idx = min(turn_idx + 1, actual_length - 1)
        elif key in ("left", "a"):
            turn_idx = max(turn_idx - 1, 0)
        elif key == "r":
            turn_idx = 0
        elif key in ("q", "esc"):
            break


if __name__ == "__main__":
    main()
