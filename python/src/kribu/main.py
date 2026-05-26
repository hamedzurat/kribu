"""
Simulation script demonstrating the Kribu Sholo Guti engine functionality.
"""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import kribu


def render_board(state: kribu.boardState) -> str:
    """
    Renders the boardState as a colorized ASCII board layout.
    """

    def char(nodeId):
        isMe = (state.me & (1 << nodeId)) != 0
        isOpp = (state.opp & (1 << nodeId)) != 0
        if isMe:
            return "[bold green]M[/bold green]"
        if isOpp:
            return "[bold red]O[/bold red]"
        if state.activeCaptureIdx == nodeId:
            return "[bold yellow]*[/bold yellow]"
        return "[grey50].[/grey50]"

    c = [char(i) for i in range(37)]

    # We lay out the nodes precisely using a formatted string template
    boardStr = f"""
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
    return boardStr


def render_reference_board() -> str:
    """
    Renders a reference board showing the node indices (0-36).
    """
    c = [f"{i:02d}" for i in range(37)]
    boardStr = f"""
   {c[0]}──────────────{c[1]}──────────────{c[2]}
     ╲              │             ╱
       ╲            │           ╱
         ╲          │         ╱
         {c[3]}────────{c[4]}────────{c[5]}
             ╲      │      ╱
               ╲    │    ╱
                 ╲  │  ╱
   {c[6]}──────{c[7]}──────{c[8]}──────{c[9]}──────{c[10]}
    │ ╲     │     ╱ │ ╲     │     ╱ │
    │   ╲   │   ╱   │   ╲   │   ╱   │
    │     ╲ │ ╱     │     ╲ │ ╱     │
   {c[11]}──────{c[12]}──────{c[13]}──────{c[14]}──────{c[15]}
    │     ╱ │ ╲     │     ╱ │ ╲     │
    │   ╱   │   ╲   │   ╱   │   ╲   │
    │ ╱     │     ╲ │ ╱     │     ╲ │
   {c[16]}──────{c[17]}──────{c[18]}──────{c[19]}──────{c[20]}
    │ ╲     │     ╱ │ ╲     │     ╱ │
    │   ╲   │   ╱   │   ╲   │   ╱   │
    │     ╲ │ ╱     │     ╲ │ ╱     │
   {c[21]}──────{c[22]}──────{c[23]}──────{c[24]}──────{c[25]}
    │     ╱ │ ╲     │     ╱ │ ╲     │
    │   ╱   │   ╲   │   ╱   │   ╲   │
    │ ╱     │     ╲ │ ╱     │     ╲ │
   {c[26]}──────{c[27]}──────{c[28]}──────{c[29]}──────{c[30]}
                 ╱  │  ╲
               ╱    │    ╲
             ╱      │      ╲
         {c[31]}────────{c[32]}────────{c[33]}
        ╱           │           ╲
      ╱             │             ╲
    ╱               │               ╲
 {c[34]}────────────────{c[35]}────────────────{c[36]}
    """
    return boardStr


## @brief Selects the best move for the AI using either minimax or the neural network.
#  @param state The current board state.
#  @param ai_type Type of AI ("minimax" or "nn").
#  @param search_depth Search depth for minimax.
#  @param nn_model The neural network model, or None if using minimax.
#  @return Chosen move ID.
def select_ai_move(state, ai_type: str, search_depth: int, nn_model) -> int:
    if ai_type == "minimax":
        res = kribu.minimax(state, search_depth)
        return res.moveId
    else:
        import torch
        from kribu.train import prepare_state_input

        inp = prepare_state_input(state)
        with torch.no_grad():
            policy_logits, _ = nn_model(inp)

        valid_moves = kribu.all_possible_moves(state)
        if not valid_moves:
            return -1

        # Mask out invalid moves
        mask = torch.full_like(policy_logits[0], float("-inf"))
        for m_id in valid_moves:
            mask[m_id] = 0.0

        masked_logits = policy_logits[0] + mask
        return int(torch.argmax(masked_logits).item())


def play_game(user_role: str, search_depth: int, ai_type: str):
    """
    Launches an interactive Sholo Guti game: User vs AI.
    """
    console = Console()

    nn_model = None
    if ai_type == "nn":
        import os
        import torch
        from kribu.train import SholoGutiNet

        if not os.path.exists("model.pt"):
            raise FileNotFoundError("model.pt not found. Please train the model first.")
        nn_model = SholoGutiNet()
        nn_model.load_state_dict(torch.load("model.pt", map_location="cpu"))
        nn_model.eval()

    state = kribu.INITIAL_STATE
    current_player = "A"

    console.print("\n")
    console.print(
        Panel(
            Text("GAME STARTED!", style="bold green", justify="center"),
            border_style="green",
        )
    )
    console.print("\n[bold yellow]Reference Board with Node Indices (0-36):[/bold yellow]")
    console.print(render_reference_board())

    while True:
        # Check game over conditions
        status = kribu.get_game_status(state)
        if status != kribu.GameStatus.ONGOING:
            console.print("\n" + "=" * 50)
            console.print(render_board(state))
            console.print("\n")
            if status in (kribu.GameStatus.ME_WINS_ELIMINATION, kribu.GameStatus.ME_WINS_STALEMATE):
                console.print(
                    Panel(
                        Text(
                            "GAME OVER: Player A (Green) Wins!",
                            style="bold green",
                            justify="center",
                        ),
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        Text(
                            "GAME OVER: Player B (Red) Wins!",
                            style="bold red",
                            justify="center",
                        ),
                        border_style="red",
                    )
                )
            break

        console.print("\n" + "=" * 60)
        user_suffix = " (You)" if current_player == user_role else " (AI)"
        player_color = "green" if current_player == "A" else "red"
        console.print(f"Current Turn: [bold {player_color}]Player {current_player}{user_suffix}[/bold {player_color}]")
        console.print(
            f"Remaining Pieces -> Player A (Green): [bold green]{kribu.piece_count(state.me)}[/bold green] | Player B (Red): [bold red]{kribu.piece_count(state.opp)}[/bold red]"
        )

        if state.activeCaptureIdx != -1:
            active_node = 36 - state.activeCaptureIdx if current_player == "B" else state.activeCaptureIdx
            console.print(f"[bold yellow]Capture Chain: Piece locked at node {active_node}[/bold yellow]")

        console.print(render_board(state))

        is_user_turn = current_player == user_role

        if is_user_turn:
            # User turn move generation
            if current_player == "A":
                moves = kribu.all_possible_moves(state)
            else:
                flipped = kribu.flip_board(state)
                moves = kribu.all_possible_moves(flipped)

            if not moves:
                console.print("[bold red]No legal moves available![/bold red]")
                break

            console.print("[bold yellow]Your Available Moves:[/bold yellow]")
            move_options = []
            for idx, mId in enumerate(moves):
                if mId == kribu.END_CHAIN_MOVE:
                    option_text = "End capture chain / Pass turn"
                else:
                    m = kribu.decode_move(mId)
                    from_node = 36 - m.fromNode if current_player == "B" else m.fromNode
                    to_node = 36 - m.toNode if current_player == "B" else m.toNode
                    captured_node = (36 - m.captured) if (current_player == "B" and m.captured != -1) else m.captured

                    if m.captured != -1:
                        option_text = f"{from_node} ──(jump over {captured_node})──> {to_node}"
                    else:
                        option_text = f"{from_node} ──> {to_node}"
                move_options.append((mId, option_text))
                console.print(f"  [bold cyan]{idx + 1}[/bold cyan]: {option_text}")

            while True:
                try:
                    choice_str = input(f"Choose a move (1-{len(moves)}): ").strip()
                    choice = int(choice_str) - 1
                    if 0 <= choice < len(moves):
                        chosen_mId, selected_text = move_options[choice]
                        console.print(f"You played: [bold green]{selected_text}[/bold green]")
                        break
                    else:
                        console.print(f"[red]Invalid index. Choose 1 to {len(moves)}.[/red]")
                except ValueError:
                    console.print("[red]Invalid input. Please enter a number.[/red]")

            # Apply user move
            if current_player == "A":
                state = kribu.apply_move(state, chosen_mId)
                if state.activeCaptureIdx == -1:
                    current_player = "B"
            else:
                flipped = kribu.flip_board(state)
                next_flipped = kribu.apply_move(flipped, chosen_mId)
                state = kribu.flip_board(next_flipped)
                if next_flipped.activeCaptureIdx == -1:
                    current_player = "A"

        else:
            # AI turn
            console.print("[bold yellow]AI is calculating best move...[/bold yellow]")
            if current_player == "A":
                chosen_mId = select_ai_move(state, ai_type, search_depth, nn_model)
                m = kribu.decode_move(chosen_mId)
                if chosen_mId == kribu.END_CHAIN_MOVE:
                    console.print("AI decided to: [bold red]End capture chain / Pass turn[/bold red]")
                else:
                    if m.captured != -1:
                        console.print(
                            f"AI played: [bold red]{m.fromNode} ──(jump over {m.captured})──> {m.toNode}[/bold red]"
                        )
                    else:
                        console.print(f"AI played: [bold red]{m.fromNode} ──> {m.toNode}[/bold red]")
                state = kribu.apply_move(state, chosen_mId)
                if state.activeCaptureIdx == -1:
                    current_player = "B"
            else:
                flipped = kribu.flip_board(state)
                chosen_mId = select_ai_move(flipped, ai_type, search_depth, nn_model)
                m = kribu.decode_move(chosen_mId)
                if chosen_mId == kribu.END_CHAIN_MOVE:
                    console.print("AI decided to: [bold red]End capture chain / Pass turn[/bold red]")
                else:
                    from_node = 36 - m.fromNode
                    to_node = 36 - m.toNode
                    captured_node = 36 - m.captured if m.captured != -1 else -1
                    if m.captured != -1:
                        console.print(
                            f"AI played: [bold red]{from_node} ──(jump over {captured_node})──> {to_node}[/bold red]"
                        )
                    else:
                        console.print(f"AI played: [bold red]{from_node} ──> {to_node}[/bold red]")
                next_flipped = kribu.apply_move(flipped, chosen_mId)
                state = kribu.flip_board(next_flipped)
                if next_flipped.activeCaptureIdx == -1:
                    current_player = "A"


def run_simulation():
    """
    Runs the original simulation script highlighting core engine capabilities.
    """
    console = Console()

    # 1. Structural/Compile-time tables metadata
    metaTable = Table(title="Board Structural Statistics", box=box.ROUNDED, border_style="blue")
    metaTable.add_column("Property", style="bold magenta")
    metaTable.add_column("Value", style="green")
    metaTable.add_row("Total Nodes", f"{kribu.NUM_NODES}")
    metaTable.add_row("Simple Moves Count", f"{kribu.NUM_SIMPLE_MOVES}")
    metaTable.add_row("Capture Moves Count", f"{kribu.NUM_CAPTURE_MOVES}")
    metaTable.add_row("Total Moves (Including End-Chain)", f"{kribu.TOTAL_MOVE_COUNT}")
    console.print(metaTable)
    console.print("\n")

    # 2. Show all decoded moves to illustrate the stability and layout
    console.print("[bold yellow]Decoded Move Table (All Simple & Capture Moves):[/bold yellow]")
    sampleTable = Table(box=box.SIMPLE, header_style="bold blue")
    sampleTable.add_column("Move ID", justify="right")
    sampleTable.add_column("Type")
    sampleTable.add_column("From Node")
    sampleTable.add_column("To Node")
    sampleTable.add_column("Captured Node")

    # All simple moves
    for i in range(1, kribu.NUM_SIMPLE_MOVES + 1):
        m = kribu.decode_move(i)
        sampleTable.add_row(f"{i}", "Simple", f"{m.fromNode}", f"{m.toNode}", "-")

    # All capture moves
    startCapture = kribu.NUM_SIMPLE_MOVES + 1
    for i in range(startCapture, kribu.TOTAL_MOVE_COUNT):
        m = kribu.decode_move(i)
        sampleTable.add_row(
            f"{i}",
            "Capture",
            f"{m.fromNode}",
            f"{m.toNode}",
            f"{m.captured}",
        )

    console.print(sampleTable)
    console.print("\n")

    # 3. Simulate Initial State
    console.print(
        Panel(
            Text("Simulating Code Run & Functions", style="bold green"),
            box=box.ROUNDED,
        )
    )

    state = kribu.INITIAL_STATE
    console.print("[bold cyan]1. Initial Board State (`INITIAL_STATE`):[/bold cyan]")
    console.print(
        f"Me Piece Count: [green]{kribu.piece_count(state.me)}[/green], Opponent Piece Count: [red]{kribu.piece_count(state.opp)}[/red]"
    )
    console.print(render_board(state))
    console.print("\n")

    # 4. Generate & Display possible moves from initial state
    moves = kribu.all_possible_moves(state)
    console.print(
        "[bold cyan]2. Generating all possible moves from Initial State (`all_possible_moves()`):[/bold cyan]"
    )
    console.print(f"Total valid moves available: [green]{len(moves)}[/green]")
    console.print(f"Sample available move IDs: {moves[:10]}...")
    console.print("\n")

    # 5. Apply a normal move (e.g. Me moves 21 to 16)
    moveId = -1
    for mId in moves:
        m = kribu.decode_move(mId)
        if m.fromNode == 21 and m.toNode == 16:
            moveId = mId
            break

    if moveId != -1:
        console.print(f"[bold cyan]3. Applying move 21 -> 16 (Move ID {moveId}) via `apply_move()`:[/bold cyan]")
        state = kribu.apply_move(state, moveId)
        console.print(
            f"Me Piece Count: [green]{kribu.piece_count(state.me)}[/green], Opponent Piece Count: [red]{kribu.piece_count(state.opp)}[/red]"
        )
        console.print(render_board(state))
        console.print("\n")

    # 6. Show board flipping (flip_board)
    console.print("[bold cyan]4. Rotating board 180 degrees (`flip_board()`):[/bold cyan]")
    console.print("This flips the board, mapping nodes symmetrically, and swaps active player (Me) and opponent (Opp).")
    flipped = kribu.flip_board(state)
    console.print("Flipped Board State (Me is now the former Opponent):")
    console.print(
        f"Me Piece Count: [green]{kribu.piece_count(flipped.me)}[/green], Opponent Piece Count: [red]{kribu.piece_count(flipped.opp)}[/red]"
    )
    console.print(render_board(flipped))
    console.print("\n")

    # 7. Setup and simulate a multi-capture chain!
    console.print(
        Panel(
            Text("Simulating Multi-Capture Chaining Scenario", style="bold green"),
            box=box.ROUNDED,
        )
    )

    chainState = kribu.boardState()
    chainState.me = 1 << 16
    chainState.opp = (1 << 17) | (1 << 19)
    chainState.activeCaptureIdx = -1

    console.print("[bold cyan]Custom Board Setup for Capture Chain:[/bold cyan]")
    console.print("Me piece at node 16. Opponent pieces at nodes 17 and 19.")
    console.print(render_board(chainState))
    console.print("\n")

    chainMoves = kribu.all_possible_moves(chainState)
    console.print(f"Available Move IDs from setup: {chainMoves}")
    for mId in chainMoves:
        m = kribu.decode_move(mId)
        console.print(f"  - Move ID {mId}: {m.fromNode} -> {m.toNode} (Capture {m.captured})")

    cap1Id = -1
    for mId in chainMoves:
        m = kribu.decode_move(mId)
        if m.fromNode == 16 and m.toNode == 18:
            cap1Id = mId
            break

    if cap1Id != -1:
        console.print(f"\n[bold cyan]Applying first capture (16 -> 18, Move ID {cap1Id}):[/bold cyan]")
        chainState = kribu.apply_move(chainState, cap1Id)
        console.print(
            f"Active Capture Index: [yellow]{chainState.activeCaptureIdx}[/yellow] "
            "(Piece is locked at node 18 for multi-capture)"
        )
        console.print(render_board(chainState))
        console.print("\n")

        nextMoves = kribu.all_possible_moves(chainState)
        console.print(f"Available Move IDs in chain: {nextMoves}")
        for mId in nextMoves:
            if mId == kribu.END_CHAIN_MOVE:
                console.print(f"  - Move ID {mId}: END_CHAIN_MOVE (Stop capturing and end turn)")
            else:
                m = kribu.decode_move(mId)
                console.print(f"  - Move ID {mId}: {m.fromNode} -> {m.toNode} (Capture {m.captured})")

        cap2Id = -1
        for mId in nextMoves:
            if mId != kribu.END_CHAIN_MOVE:
                m = kribu.decode_move(mId)
                if m.fromNode == 18 and m.toNode == 20:
                    cap2Id = mId
                    break

        if cap2Id != -1:
            console.print(f"\n[bold cyan]Applying second capture (18 -> 20, Move ID {cap2Id}):[/bold cyan]")
            chainState = kribu.apply_move(chainState, cap2Id)
            console.print(
                f"Active Capture Index: [yellow]{chainState.activeCaptureIdx}[/yellow] "
                "(-1 means chain has automatically ended as no further captures are possible)"
            )
            console.print(render_board(chainState))
            console.print("\n")

    # 8. Check Game Over / Winner
    console.print(
        Panel(
            Text("Game Over & Win Condition Verification", style="bold green"),
            box=box.ROUNDED,
        )
    )
    status = kribu.get_game_status(chainState)
    isGameOver = status != kribu.GameStatus.ONGOING
    console.print(f"Is game over on capture chain board? [green]{isGameOver}[/green]")
    winnerStr = (
        "Me"
        if status in (kribu.GameStatus.ME_WINS_ELIMINATION, kribu.GameStatus.ME_WINS_STALEMATE)
        else "Opponent"
        if status in (kribu.GameStatus.OPP_WINS_ELIMINATION, kribu.GameStatus.OPP_WINS_STALEMATE)
        else "No one (game not over)"
    )
    console.print(f"Winner: [bold green]{winnerStr}[/bold green]")
    console.print("\n")


def main():
    """
    Interactive main menu to start gameplay or run original engine simulations.
    """
    console = Console()

    # Title Banner
    console.print("\n")
    console.print(
        Panel(
            Text(
                "KRIBU - SHOLO GUTI ENGINE & PLAYGROUND",
                style="bold cyan",
                justify="center",
            ),
            subtitle="Coordinate-Free Declarative Compile-Time Rule Engine & Minimax AI",
            box=box.DOUBLE,
            border_style="cyan",
        )
    )
    console.print("\n")

    console.print("[bold yellow]Available Modes:[/bold yellow]")
    console.print("  [bold cyan]1[/bold cyan]: Play as Player A (Green - Starts bottom row, goes first)")
    console.print("  [bold cyan]2[/bold cyan]: Play as Player B (Red - Starts top row, goes second)")
    console.print("  [bold cyan]3[/bold cyan]: Run original engine simulation")
    console.print("\n")

    while True:
        mode = input("Choose mode (1-3): ").strip()
        if mode in ("1", "2", "3"):
            break
        console.print("[red]Invalid choice. Please select 1, 2, or 3.[/red]")

    if mode == "3":
        run_simulation()
    else:
        import os

        has_model = os.path.exists("model.pt")

        console.print("[bold yellow]Select AI Opponent Type:[/bold yellow]")
        console.print("  [bold cyan]1[/bold cyan]: Minimax (Alpha-Beta Search)")
        if has_model:
            console.print("  [bold cyan]2[/bold cyan]: Neural Network (Supervised Learning model.pt)")
        else:
            console.print(
                "  [bold cyan]2[/bold cyan]: [grey50]Neural Network (model.pt not found - run train.py first)[/bold cyan]"
            )

        while True:
            ai_choice = input(f"Choose AI type (1-{'2' if has_model else '1'}): ").strip()
            if ai_choice == "1":
                ai_type = "minimax"
                break
            elif ai_choice == "2" and has_model:
                ai_type = "nn"
                break
            console.print("[red]Invalid choice. Please select 1 or 2.[/red]")

        depth = 0
        if ai_type == "minimax":
            # Prompt for difficulty
            while True:
                try:
                    depth_str = input("Select AI search depth/difficulty (1-8) [default: 6]: ").strip()
                    if not depth_str:
                        depth = 6
                        break
                    depth = int(depth_str)
                    if 1 <= depth <= 8:
                        break
                    else:
                        console.print("[red]Please enter a depth between 1 and 8.[/red]")
                except ValueError:
                    console.print("[red]Invalid input. Please enter a valid number.[/red]")

        user_role = "A" if mode == "1" else "B"
        play_game(user_role, depth, ai_type)


if __name__ == "__main__":
    main()
