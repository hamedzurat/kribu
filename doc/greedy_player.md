# Greedy Player

The **Greedy Player** is a 1-step lookahead agent that evaluates all legal moves immediately available and chooses the one that results in the highest board evaluation.

## Overview

The greedy player is implemented in [greedy.hpp](../engine/include/kribu/player/greedy.hpp). It utilizes the standard evaluation heuristic `kribu::heuristics::evaluate` at a fixed depth of 3 to select moves.

______________________________________________________________________

## Algorithm Details

### `greedy_player_maker`

```cpp
int greedy_player_maker(const boardState& state)
```

The function executes the following process:

1. Generates all possible legal moves using `all_possible_moves(state)`. If no moves are available, it returns `-1`.
1. Initializes tracking variables:
   - `bestMove = -1`
   - `bestVal = -999999`
1. For each candidate move:
   - Applies the move to get `nextState`.
   - Determines the evaluation perspective:
     - **Turn Flipped**: If the move ends the current turn (`nextState.activeCaptureIdx == -1`), the board is flipped to the opponent's perspective. The evaluation is negated: `val = -heuristics::evaluate(flip_board(nextState), 3)`.
     - **Same Turn (Capture Chain)**: If the player can continue capturing (`nextState.activeCaptureIdx != -1`), the turn does not flip. The evaluation is kept positive: `val = heuristics::evaluate(nextState, 3)`.
   - Compares the evaluation:
     - If `val > bestVal`, updates `bestVal` and sets `bestMove` to the current move.
     - If `val == bestVal`, tie-breaking occurs: there is a 50% probability (`(rng() & 1) == 0`) that the current move replaces `bestMove`.
1. Returns the `bestMove`.

> [!TIP]
> **50/50 Tie-breaking** ensures that if multiple moves yield the exact same maximum evaluation, the agent selects one randomly, providing game variability without search overhead.

______________________________________________________________________

## Control Flow

Below is the control flow of the greedy player:

```mermaid
flowchart TD
    Start([Start: greedy_player_maker]) --> GenMoves[Get all legal moves via all_possible_moves]
    GenMoves --> CheckEmpty{Are moves empty?}
    CheckEmpty -- Yes --> ReturnNoMove[Return -1]
    CheckEmpty -- No --> InitVariables[Initialize bestMove=-1, bestVal=-999999]

    InitVariables --> LoopStart{For each move in moves}

    LoopStart -- No more moves --> ReturnBest[Return bestMove]
    LoopStart -- Yes --> Apply[Apply move to get nextState]
    Apply --> CheckChain{Is nextState.activeCaptureIdx == -1?}

    CheckChain -- "Yes (Turn Flipped)" --> EvalFlip["val = -EvalFunc(flip_board(nextState))"]
    CheckChain -- "No (Same Turn)" --> EvalNoFlip["val = EvalFunc(nextState)"]

    EvalFlip --> CompareVal
    EvalNoFlip --> CompareVal

    CompareVal{Is val > bestVal?}
    CompareVal -- "Yes (New Max)" --> UpdateMax[bestVal = val, bestMove = moveId]
    CompareVal -- No --> CompareEqual{Is val == bestVal?}

    CompareEqual -- "Yes (Tie)" --> TieBreak{Is rng & 1 == 0?}
    CompareEqual -- No --> LoopStart

    TieBreak -- Yes --> Replace[bestMove = moveId]
    TieBreak -- No --> LoopStart
    UpdateMax --> LoopStart
    Replace --> LoopStart

    style Start fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
    style CheckEmpty fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    style ReturnNoMove fill:#F44336,stroke:#D32F2F,stroke-width:2px,color:#fff
    style CheckChain fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    style CompareVal fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    style ReturnBest fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
```
