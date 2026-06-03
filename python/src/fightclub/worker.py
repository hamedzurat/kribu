"""!
@file worker.py
@brief Multiprocessing worker function for Fight Club.
@details Separated to resolve import errors when spawning child processes.
"""

import random
import time
import kribu
from arena.player import NeuralPlayer


## @brief Worker function to execute a single game in a separate process.
#  @param task_args Tuple of (idx, p1, p2, p1_model_path, p2_model_path, max_turns)
#  @return Tuple of (idx, p1, p2, winner, reason, turns, p1_time, p2_time, p1_moves, p2_moves)
def play_game_worker(task_args) -> tuple[int, str, str, str, str, int, float, float, int, int]:
    idx, p1, p2, p1_model_path, p2_model_path, max_turns = task_args

    # Limit PyTorch to 1 thread per worker process to avoid context switching
    # overhead/CPU thrashing on multi-core servers.
    if p1_model_path or p2_model_path:
        import torch

        torch.set_num_threads(1)

    # Register standard players
    builtins = {
        "random": lambda state: (
            random.choice(list(kribu.all_possible_moves(state))) if list(kribu.all_possible_moves(state)) else -1
        ),
        "greedy": kribu.greedy_player,
        "mcts_200": kribu.mcts_player_200,
        "mcts_400": kribu.mcts_player_400,
        "mcts_600": kribu.mcts_player_600,
        "mcts_800": kribu.mcts_player_800,
        "minimax_2": kribu.minimax_player_2,
        "minimax_3": kribu.minimax_player_3,
        "minimax_4": kribu.minimax_player_4,
        "minimax_6": kribu.minimax_player_6,
        "minimax_8": kribu.minimax_player_8,
        "minimax_2_mad2": kribu.minimax_player_2_mad2,
        "minimax_3_mad2": kribu.minimax_player_3_mad2,
        "minimax_4_mad2": kribu.minimax_player_4_mad2,
        "minimax_6_mad2": kribu.minimax_player_6_mad2,
        "minimax_8_mad2": kribu.minimax_player_8_mad2,
    }

    # Resolve player 1 function
    if p1 in builtins:
        p1_fn = builtins[p1]
    elif p1_model_path:
        np1 = NeuralPlayer(model_path=p1_model_path)

        def p1_fn(state):
            return np1.get_move(
                me_mask=state.me,
                opp_mask=state.opp,
                active_capture_idx=state.activeCaptureIdx,
                valid_moves=list(kribu.all_possible_moves(state)),
                state=state,
            )[0]
    else:
        raise ValueError(f"Unknown player: {p1}")

    # Resolve player 2 function
    if p2 in builtins:
        p2_fn = builtins[p2]
    elif p2_model_path:
        np2 = NeuralPlayer(model_path=p2_model_path)

        def p2_fn(state):
            return np2.get_move(
                me_mask=state.me,
                opp_mask=state.opp,
                active_capture_idx=state.activeCaptureIdx,
                valid_moves=list(kribu.all_possible_moves(state)),
                state=state,
            )[0]
    else:
        raise ValueError(f"Unknown player: {p2}")

    # Game loop
    state = kribu.boardState()
    state.me = kribu.INITIAL_STATE.me
    state.opp = kribu.INITIAL_STATE.opp
    state.activeCaptureIdx = kribu.INITIAL_STATE.activeCaptureIdx
    state.hash = kribu.INITIAL_STATE.hash
    state.historyCount = kribu.INITIAL_STATE.historyCount

    is_p1_turn = True
    turn_idx = 0

    p1_time = 0.0
    p2_time = 0.0
    p1_moves = 0
    p2_moves = 0

    while True:
        status = kribu.get_game_status(state)

        if status != kribu.GameStatus.ONGOING:
            if status in (kribu.GameStatus.ME_WINS_ELIMINATION, kribu.GameStatus.ME_WINS_STALEMATE):
                winner = "player1" if is_p1_turn else "player2"
                reason = "elimination" if status == kribu.GameStatus.ME_WINS_ELIMINATION else "stalemate"
            elif status in (kribu.GameStatus.OPP_WINS_ELIMINATION, kribu.GameStatus.OPP_WINS_STALEMATE):
                winner = "player2" if is_p1_turn else "player1"
                reason = "elimination" if status == kribu.GameStatus.OPP_WINS_ELIMINATION else "stalemate"
            elif status == kribu.GameStatus.DRAW_PROGRESS_RULE:
                winner = "draw"
                reason = "progress_rule"
            else:
                winner = "draw"
                reason = "unknown"
            break

        if turn_idx >= max_turns:
            winner = "draw"
            reason = "max_turns"
            break

        valid_moves = list(kribu.all_possible_moves(state))
        if not valid_moves:
            turn_idx += 1
            continue

        start_t = time.perf_counter()
        if is_p1_turn:
            move_idx = p1_fn(state)
            p1_time += time.perf_counter() - start_t
            p1_moves += 1
        else:
            move_idx = p2_fn(state)
            p2_time += time.perf_counter() - start_t
            p2_moves += 1

        if move_idx not in valid_moves:
            move_idx = valid_moves[0]

        next_state = kribu.apply_move(state, move_idx)

        if next_state.activeCaptureIdx == -1:
            state = kribu.flip_board(next_state)
            is_p1_turn = not is_p1_turn
        else:
            state = next_state

        turn_idx += 1

    return idx, p1, p2, winner, reason, turn_idx, p1_time, p2_time, p1_moves, p2_moves
