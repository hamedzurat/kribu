from kribu import (
    INITIAL_STATE,
    minimax_player_8,
    all_possible_moves,
)


def test_minimax_player_8():
    # Verify that minimax_player_8 is callable and returns a valid move index
    move_id = minimax_player_8(INITIAL_STATE)
    assert isinstance(move_id, int)

    valid_moves = all_possible_moves(INITIAL_STATE)
    assert move_id in valid_moves
