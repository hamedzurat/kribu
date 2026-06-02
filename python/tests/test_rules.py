from kribu import (
    boardState,
    INITIAL_STATE,
    is_valid,
    apply_move,
    flip_board,
    get_game_status,
    GameStatus,
    piece_count,
    decode_move,
    all_possible_moves,
    TOTAL_MOVE_COUNT,
    END_CHAIN_MOVE,
    NUM_SIMPLE_MOVES,
    NUM_CAPTURE_MOVES,
    repetition_features,
    annotate_repetition_features,
)


def is_game_over(state):
    return get_game_status(state) != GameStatus.ONGOING


def get_winner(state):
    return get_game_status(state).value


def test_initial_state():
    state = INITIAL_STATE
    assert piece_count(state.me) == 16
    assert piece_count(state.opp) == 16
    assert state.activeCaptureIdx == -1
    assert not is_game_over(state)
    assert get_winner(state) == 0


def test_flip_board():
    state = INITIAL_STATE
    flipped = flip_board(state)
    assert piece_count(flipped.opp) == 16
    assert piece_count(flipped.me) == 16
    assert flipped.activeCaptureIdx == -1


def test_repetition_features_start_empty():
    assert repetition_features(INITIAL_STATE) == (0, 0, 0)


def test_annotate_repetition_features_tracks_quiet_history():
    moveId = all_possible_moves(INITIAL_STATE)[0]
    nextState = flip_board(apply_move(INITIAL_STATE, moveId))
    features = annotate_repetition_features(
        [int(INITIAL_STATE.me), int(nextState.me)],
        [int(INITIAL_STATE.opp), int(nextState.opp)],
        [int(INITIAL_STATE.activeCaptureIdx), int(nextState.activeCaptureIdx)],
        [int(moveId), int(all_possible_moves(nextState)[0])],
    )

    assert features[0] == [0, 1]
    assert features[1] == [0, 0]
    assert features[2] == [0, 0]


def test_decode_move():
    # Test valid range
    assert TOTAL_MOVE_COUNT == 1 + NUM_SIMPLE_MOVES + NUM_CAPTURE_MOVES
    m = decode_move(1)
    assert m.fromNode != -1

    mEnd = decode_move(END_CHAIN_MOVE)
    assert mEnd.fromNode == -1
    assert mEnd.toNode == -1
    assert mEnd.captured == -1


def test_moves():
    state = INITIAL_STATE
    moves = all_possible_moves(state)
    assert len(moves) > 0
    assert END_CHAIN_MOVE not in moves

    # Check simple move validity
    moveId = moves[0]
    assert is_valid(state, moveId)
    m = decode_move(moveId)
    assert m.captured == -1  # initial moves are simple

    # apply move
    nextState = apply_move(state, moveId)
    assert piece_count(nextState.me) == 16
    assert nextState.activeCaptureIdx == -1
    assert nextState != state


def test_invalid_move():
    state = INITIAL_STATE
    assert not is_valid(state, END_CHAIN_MOVE)
    assert not is_valid(state, -1)
    assert not is_valid(state, TOTAL_MOVE_COUNT)

    # trying to move opponent piece
    # opp pieces are at top (0..15). Move from 10 to 15 (wait, 15 is occupied)
    # just find a move where from is opponent
    for i in range(1, NUM_SIMPLE_MOVES + 1):
        m = decode_move(i)
        if m.fromNode < 16:
            assert not is_valid(state, i)


def test_stalemate_condition():
    # If a player has no moves, they lose
    # Create a state where me has 1 piece blocked by 4 opp pieces
    # Center node 18 is connected to 12, 13, 14, 17, 19, 22, 23, 24
    state = boardState()
    state.me = 1 << 18
    # Surround 18 completely with opp pieces
    for i in [12, 13, 14, 17, 19, 22, 23, 24]:
        state.opp |= 1 << i

    # Are there any jump captures from 18?
    # lines from 18:
    # 16-17-18-19-20 (H). Jump over 17 to 16? 16 is empty. Wait!
    # Let's just fill the jump destinations too.
    # 16, 20
    # 6, 8, 10
    # 26, 28, 30
    state.opp |= (1 << 16) | (1 << 20) | (1 << 6) | (1 << 8) | (1 << 10) | (1 << 26) | (1 << 28) | (1 << 30)

    moves = all_possible_moves(state)
    assert len(moves) == 0
    assert is_game_over(state)
    assert get_winner(state) == -2  # I have no moves, I lose (stalemate)


def test_win_condition():
    # Opponent has 0 pieces
    state = boardState()
    state.me = 1 << 0
    state.opp = 0
    assert is_game_over(state)
    assert get_winner(state) == 1


def test_multi_capture():
    state = boardState()
    # set up a scenario where 'me' can capture twice
    # Line: 16-17-18-19-20
    # me at 16, opp at 17, empty at 18, opp at 19, empty at 20
    state.me = 1 << 16
    state.opp = (1 << 17) | (1 << 19)

    moves = all_possible_moves(state)

    # find the capture move from 16 to 18 over 17
    capture1 = -1
    for mId in moves:
        m = decode_move(mId)
        if m.fromNode == 16 and m.toNode == 18 and m.captured == 17:
            capture1 = mId

    assert capture1 != -1

    # Apply first capture
    nextState = apply_move(state, capture1)
    assert piece_count(nextState.opp) == 1
    assert nextState.activeCaptureIdx == 18  # chaining!

    # Check valid moves in next state
    moves2 = all_possible_moves(nextState)
    assert END_CHAIN_MOVE in moves2

    capture2 = -1
    for mId in moves2:
        if mId != END_CHAIN_MOVE:
            m = decode_move(mId)
            if m.fromNode == 18 and m.toNode == 20 and m.captured == 19:
                capture2 = mId

    assert capture2 != -1

    # Apply second capture
    finalState = apply_move(nextState, capture2)
    assert piece_count(finalState.opp) == 0
    # The chain remains active at the destination node until ended explicitly
    assert finalState.activeCaptureIdx == 20

    finalMoves = all_possible_moves(finalState)
    assert END_CHAIN_MOVE in finalMoves
    assert len(finalMoves) == 1  # Only END_CHAIN_MOVE is available

    endedState = apply_move(finalState, END_CHAIN_MOVE)
    assert endedState.activeCaptureIdx == -1


def test_auto_end_chain():
    state = boardState()
    # Only one capture available: 16 -> 18 over 17
    # 19 is empty, so no second capture
    state.me = 1 << 16
    state.opp = 1 << 17

    # Need another piece for me just so game doesn't end immediately? Actually game over logic triggers if opp=0
    # But apply_move doesn't care about game_over
    moves = all_possible_moves(state)

    capture1 = -1
    for mId in moves:
        m = decode_move(mId)
        if m.fromNode == 16 and m.toNode == 18 and m.captured == 17:
            capture1 = mId

    assert capture1 != -1

    # Apply capture
    nextState = apply_move(state, capture1)
    assert piece_count(nextState.opp) == 0
    # The chain remains active at the destination node until ended explicitly
    assert nextState.activeCaptureIdx == 18

    nextMoves = all_possible_moves(nextState)
    assert END_CHAIN_MOVE in nextMoves
    assert len(nextMoves) == 1  # Only END_CHAIN_MOVE is available

    endedState = apply_move(nextState, END_CHAIN_MOVE)
    assert endedState.activeCaptureIdx == -1


def test_edge_deduplication():
    # If edge deduplication/merging failed, NUM_SIMPLE_MOVES would exceed 160
    assert NUM_SIMPLE_MOVES == 152
