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
import pyarrow.parquet as pq

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


def select_game_from_manifest(console: Console) -> str:
    """Lists simulated games from manifest.json and prompts user to pick one."""
    manifest_path = "benchmark/manifest.json"
    if not os.path.exists(manifest_path):
        console.print("[bold red]Error: No manifest.json found, and no parquet file path was provided.[/bold red]")
        sys.exit(1)

    import json

    with open(manifest_path, "r") as f:
        games = json.load(f)

    if not games:
        console.print("[bold yellow]Manifest is empty. No games simulated yet.[/bold yellow]")
        sys.exit(0)

    table = Table(title="Simulated Benchmark Games", header_style="bold yellow")
    table.add_column("No.", justify="right", style="cyan")
    table.add_column("Player 1 (Green)", style="green")
    table.add_column("Player 2 (Red)", style="red")
    table.add_column("Outcome", style="magenta")
    table.add_column("Reason", style="blue")
    table.add_column("Turns", justify="right", style="white")

    for idx, g in enumerate(games):
        table.add_row(str(idx + 1), g["p1Name"], g["p2Name"], g["outcome"], g["reason"], str(g["totalTurns"]))

    console.print(table)

    while True:
        try:
            choice = input(f"\nSelect a game index to view (1-{len(games)}) or [q] to quit: ").strip()
            if choice.lower() == "q":
                sys.exit(0)
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(games):
                return games[choice_idx]["filePath"]
            console.print(f"[red]Invalid selection. Please enter 1 to {len(games)}.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")


def main():
    console = Console()

    # Resolve file path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = select_game_from_manifest(console)

    if not os.path.exists(file_path):
        console.print(f"[bold red]Error: File {file_path} not found.[/bold red]")
        sys.exit(1)

    # Read Parquet
    console.print(f"[cyan]Loading {file_path}...[/cyan]")
    table = pq.read_table(file_path)

    # Extract schema-level metadata
    metadata = table.schema.metadata
    meta = {}
    if metadata:
        meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in metadata.items()}

    p1_name = meta.get("p1Name", "Player 1")
    p2_name = meta.get("p2Name", "Player 2")
    outcome = meta.get("outcome", "N/A")
    reason = meta.get("reason", "N/A")
    total_turns = int(meta.get("totalTurns", 0))

    # Read column vectors
    me_col = table.column("me").to_pylist()
    opp_col = table.column("opp").to_pylist()
    active_cap_col = table.column("activeCaptureIdx").to_pylist()
    is_p1_turn_col = table.column("isP1Turn").to_pylist()
    chosen_move_col = table.column("chosenMove").to_pylist()
    possible_moves_col = table.column("possibleMoves").to_pylist()

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
        info_table.add_row("Green Pieces Remaining (P1):", f"[green]{count_bits(green_mask)}[/green]")
        info_table.add_row("Red Pieces Remaining (P2):", f"[red]{count_bits(red_mask)}[/red]")

        chosen_m_id = chosen_move_col[turn_idx]
        info_table.add_row("Move Played:", f"[bold yellow]{format_move(chosen_m_id, is_p1_turn)}[/bold yellow]")

        console.print(Panel(info_table, title="Turn Information", border_style="yellow"))

        # Display list of available choices
        moves_list = possible_moves_col[turn_idx]
        formatted_moves = [format_move(m_id, is_p1_turn) for m_id in moves_list]
        moves_str = ", ".join(formatted_moves)
        if len(moves_str) > 100:
            moves_str = moves_str[:100] + " ... (truncated)"

        console.print(f"[bold cyan]Available Move Options ({len(moves_list)}):[/bold cyan] {moves_str}")

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
