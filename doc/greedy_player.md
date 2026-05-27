# Greedy Player

The **Greedy Player** is a 1-step lookahead agent that evaluates all legal moves immediately available and chooses the one that results in the highest board evaluation.

## Overview

The greedy player is implemented in \[greedy.hpp\](file:///home/hz/file/git/kribu/engine/include/kribu/player/greedy.hpp). It is parameterized by a template evaluation function `EvalFunc`, allowing it to be instantiated with different evaluation metrics (e.g., piece count or node values).

## Algorithm Details

### `greedy_player_maker`

```cpp
template <auto EvalFunc>
int greedy_player_maker(const boardState& state, u64& nodes)
```

The function executes the following process:

1. Generates all possible legal moves using `all_possible_moves(state)`. If no moves are available, it throws a `std::runtime_error`.
1. Initializes tracking variables:
   - `bestMove = -1`
   - `bestVal = -999999`
   - `bestCount = 0` (used for tie-breaking)
1. For each candidate move:
   - Increments the `nodes` counter.
   - Applies the move to get `nextState`.
   - Determines the evaluation perspective:
     - **Turn Flipped**: If the move ends the current turn (`nextState.activeCaptureIdx == -1`), the board is flipped to the opponent's perspective. The evaluation is negated: `val = -EvalFunc(flip_board(nextState))`.
     - **Same Turn (Capture Chain)**: If the player can continue capturing (`nextState.activeCaptureIdx != -1`), the turn does not flip. The evaluation is kept positive: `val = EvalFunc(nextState)`.
   - Compares the evaluation:
     - If `val > bestVal`, updates `bestVal`, sets `bestMove` to the current move, and resets `bestCount = 1`.
     - If `val == bestVal`, increments `bestCount` and performs reservoir sampling: replaces `bestMove` with the current move with a probability of `1 / bestCount` (`(rng() % bestCount) == 0`).
1. Returns the `bestMove`.

> [!TIP]
> **Reservoir Sampling** ensures that if multiple moves yield the exact same maximum evaluation, one is selected uniformly at random without requiring a secondary array to store the tied moves.

______________________________________________________________________

## Control Flow

Below is the control flow of the greedy player:

```mermaid
flowchart TD
    Start([Start: greedy_player_maker]) --> GenMoves[Get all legal moves via all_possible_moves]
    GenMoves --> CheckEmpty{Are moves empty?}
    CheckEmpty -- Yes --> ThrowError[Throw std::runtime_error]
    CheckEmpty -- No --> InitVariables[Initialize bestMove=-1, bestVal=-999999, bestCount=0]

    InitVariables --> LoopStart{For each move in moves}

    LoopStart -- No more moves --> ReturnBest[Return bestMove]
    LoopStart -- Yes --> IncNodes[Increment nodes]
    IncNodes --> Apply[Apply move to get nextState]
    Apply --> CheckChain{Is nextState.activeCaptureIdx == -1?}

    CheckChain -- "Yes (Turn Flipped)" --> EvalFlip["val = -EvalFunc(flip_board(nextState))"]
    CheckChain -- "No (Same Turn)" --> EvalNoFlip["val = EvalFunc(nextState)"]

    EvalFlip --> CompareVal
    EvalNoFlip --> CompareVal

    CompareVal{Is val > bestVal?}
    CompareVal -- "Yes (New Max)" --> UpdateMax[bestVal = val, bestMove = moveId, bestCount = 1]
    CompareVal -- No --> CompareEqual{Is val == bestVal?}

    CompareEqual -- "Yes (Tie)" --> Reservoir[Increment bestCount<br/>Replace bestMove with probability 1/bestCount]
    CompareEqual -- No --> LoopStart
    UpdateMax --> LoopStart
    Reservoir --> LoopStart

    style Start fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
    style CheckEmpty fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    style ThrowError fill:#F44336,stroke:#D32F2F,stroke-width:2px,color:#fff
    style CheckChain fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    style CompareVal fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    style ReturnBest fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
```
