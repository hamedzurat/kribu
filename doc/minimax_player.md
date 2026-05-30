# Minimax Player

The **Minimax Player** is a highly optimized search-based agent that uses the minimax algorithm with alpha-beta pruning. It is enhanced with modern game-tree search techniques to search deeper and more efficiently.

## Overview

The minimax player is implemented in [minimax.hpp](../engine/include/kribu/player/minimax.hpp). It supports templates for custom evaluation functions and is configured with optimizations that match modern chess/checkers engines.

______________________________________________________________________

## Core Optimizations

The minimax engine implements the following search enhancements:

### 1. Thread-Local Search Context

The minimax player executes single-threaded searches using a thread-local `TranspositionTable` and `SearchContext`.

- Spawning threads is avoided at the player level to eliminate synchronization overhead.
- Instead, the matchmaking tournament runs multiple games in parallel across separate threads, each using their own thread-local Transposition Table.
- This ensures maximum CPU utilization across multiple processor cores without complex concurrency locks inside the player logic.

### 2. Iterative Deepening & Aspiration Windows

Instead of searching directly to the target depth, the engine searches incrementally from depth $1, 2, \\dots, D$.

- **Transposition Table Warmup**: Shallower searches populate the TT, ensuring that deeper searches start with highly accurate move ordering.
- **Aspiration Windows**: For depth $d \\ge 3$, the engine restricts the alpha-beta search window to $[prevScore - 50, prevScore + 50]$ using the score from depth $d-1$. If the score falls outside this window (fails low/high), the search is repeated with a full window. This saves search time by pruning branches outside the expected score range.

### 3. Move Ordering

Move ordering is critical for alpha-beta efficiency. Moves at each node are prioritized using `order_moves` in the following sequence:

1. **Transposition Table Move**: The best move stored from a previous search of this position.
1. **Captures**: Winning/tactical moves.
1. **Killer Moves**: Quiet moves that caused a beta cutoff at the current depth in other branches (stored in a thread-local `KillerTable` tracking 2 slots per depth).
1. **Quiet Moves**: Remaining moves.

### 4. Principal Variation Search (PVS)

PVS assumes the first ordered move (the Principal Variation) is likely the best.

- The first move is searched with a full window $[\\alpha, \\beta]$.
- Subsequent moves are searched with a **null (zero) window** $[\\alpha, \\alpha + 1]$ (or $[-\\alpha-1, -\\alpha]$ depending on turn flips) to quickly check if they are better than the PV.
- Only if a scout search fails high (score $> \\alpha$) is the move re-searched with a full window.

### 5. Null-Move Pruning (NMP)

If the player can "pass" the turn (null move) and the opponent still cannot obtain a score $\\ge \\beta$ at a reduced depth ($depth - 1 - 2$), the engine assumes the position is strong enough to cause a beta cutoff, and prunes the node.

- **Zugzwang Prevention**: Disabled if the active player has $< 4$ pieces.
- **Capture Chains**: Disabled mid-capture chain.

### 6. Late Move Reductions (LMR)

Quiet moves ordered late in the list (move index $\\ge 3$) are unlikely to be the best.

- They are searched at a reduced depth (reduced by 1, or by 2 if depth $\\ge 6$ and move index $\\ge 6$).
- If the reduced search yields a score $> \\alpha$, it is re-searched at full depth.

### 7. Quiescence Search (Q-Search)

To avoid the **horizon effect** (where a player blunders because a bad capture sequence starts just beyond the search depth limit), leaf nodes ($depth \\le 0$) transition to a quiescence search.

- Only captures and turn-ending moves (`END_CHAIN_MOVE`) are evaluated.
- Uses "stand-pat" evaluation (the static evaluation of the current board state) as a lower bound.

______________________________________________________________________

## Zobrist Hashing

Zobrist hashing maps arbitrary board layouts to a 64-bit unsigned integer space. It is defined in [zobrist.hpp](../engine/include/kribu/zobrist.hpp).

### 1. The Concept

Two different sequences of moves can lead to the exact same board state (transposition). Recalculating identical states is redundant. Zobrist hashing provides a fast, unique 64-bit signature (hash) for each unique position.

### 2. Compile-Time Key Generation

The engine generates unique keys at compile-time to avoid initialization overhead:

- A custom, constexpr linear congruential/Xorshift pseudo-random number generator (`next_random`) generates keys using a seed value `ZOBRIST_SEED = 0x9e3779b97f4a7c15ULL`.
- **`ZobristKeys`**: Contains distinct key arrays mapping:
  - Active player piece positions (`me` array, size 37).
  - Opponent piece positions (`opp` array, size 37).
  - Active capture index positions (`activeCapture` array, size 38; includes a sentinel slot at index 37 for `-1`).

### 3. Incremental Hash Updates

Hashing is extremely fast because it is updated incrementally using the bitwise **XOR (`^`)** operator during move applications inside `apply_move`:

- Since $A \\oplus B \\oplus B = A$, applying or reversing a state change is a matter of XOR-ing the state keys:
  - When a piece moves from node $A$ to node $B$, the hash is updated with:
    `hash ^= KEYS.me[A] ^ KEYS.me[B]`
  - If a piece is captured at node $C$, the opponent's key is XOR-ed:
    `hash ^= KEYS.opp[C]`
  - Capture lock indexes are updated similarly.
- This avoids scanning all 37 board nodes on every search step. When needed (e.g. at initialization), a hash can be computed from scratch via `compute_hash`.

______________________________________________________________________

## Transposition Table (TT)

The transposition table acts as a global cache for search results. It is defined in [transposition_table.hpp](../engine/include/kribu/transposition_table.hpp).

### 1. Entry Structure (`TTEntry`)

Each entry in the table stores:

- `hash`: The full 64-bit Zobrist key of the board state.
- `score`: The evaluated score of the state.
- `moveId`: The best move identified from this state (used for move ordering).
- `depth`: The remaining depth of the search when this state was cached.
- `flag`: The bounding type of the score (`TTFlag`).

### 2. Transposition Flags (`TTFlag`)

Because alpha-beta pruning clips search trees, the exact score is not always known:

| Flag            | Name        | Meaning                                                                              |
| :-------------- | :---------- | :----------------------------------------------------------------------------------- |
| `TTFlag::EXACT` | Exact       | The exact evaluation score was computed (between alpha and beta).                    |
| `TTFlag::ALPHA` | Upper Bound | The search failed low (score $\\le \\alpha$). The true score is at most this value.  |
| `TTFlag::BETA`  | Lower Bound | The search failed high (score $\\ge \\beta$). The true score is at least this value. |

### 3. Probing & Bounds Verification

When `probe` is called:

1. Calculates the index via `hash % table.size()`.
1. Verifies that the stored entry matches the requested `hash`.
1. Verifies that the stored entry's depth is $\\ge$ the current search depth (ensuring the cached result is sufficiently searched).
1. Verifies bounding rules:
   - If `EXACT`, returns the score.
   - If `ALPHA` and the stored score is $\\le \\alpha$, returns $\\alpha$.
   - If `BETA` and the stored score is $\\ge \\beta$, returns $\\beta$.

### 4. Replacement Scheme

To maximize cache hit rates, the table uses a **depth-preferred replacement scheme**. During `store`:

- An entry is overwritten if the new evaluation was performed at a greater remaining depth than the stored depth (`depth >= entry.depth`).
- This prioritizes keeping deep search results (which are very expensive to compute) over shallow, near-leaf evaluations.

### 5. Thread-Local Cache Safety

Since each thread runs its own game and search thread-locally, the transposition table is completely thread-safe without needing locks or atomics. This design achieves maximum lookup speed and zero lock contention during deep searches.

______________________________________________________________________

## Recursive Search Control Flow

The flowchart below outlines the recursive `alpha_beta` search algorithm and its `evaluate_children` helper:

```mermaid
flowchart TD
    Start(["alpha_beta state, depth, alpha, beta, isRoot"]) --> TerminalCheck{"Is Terminal?<br/>Repetitions (if !isRoot) / 0 pieces"}
    TerminalCheck -- Yes --> ReturnTerminal[Return Win/Loss/Draw score]

    TerminalCheck -- No --> LeafCheck{"Is depth <= 0?"}
    LeafCheck -- Yes --> QSearch[Run quiescence_search]

    LeafCheck -- No --> TTOProbe{"TT Probe Match?"}
    TTOProbe -- Yes --> RootTTCheck{"IsRoot & moveId == -1?"}
    RootTTCheck -- No --> ReturnTT[Return TT Score & Move]
    RootTTCheck -- Yes --> NMPCheck

    TTOProbe -- No --> NMPCheck{"Null Move Pruning?<br/>!isRoot & depth >= 3 & pieces >= 4"}
    NMPCheck -- Yes --> NullSearch[Pass turn, search at reduced depth]
    NullSearch --> NullCutoff{"Score >= beta?"}
    NullCutoff -- Yes --> ReturnBeta[Return beta cutoff]
    NullCutoff -- No --> OrderMoves
    NMPCheck -- No --> OrderMoves

    OrderMoves["Order moves: TT -> Captures -> Killers -> Quiets"] --> EvalChildren[evaluate_children]

    EvalChildren --> LoopStart{"For each move i"}

    LoopStart -- No more moves / cutoff --> StoreTT[Store result in TT]
    StoreTT --> ReturnBest["Return bestScore & bestMoveId"]

    LoopStart -- Yes --> LMRCheck{"Apply LMR?<br/>i >= 3, quiet move, depth >= 3"}
    LMRCheck -- Yes --> SearchReduced[Search child at reduced depth]
    LMRCheck -- No --> SearchPVS[Search child via PVS]

    SearchReduced --> LMRCutoff{"Score > alpha?"}
    LMRCutoff -- Yes --> SearchPVS
    LMRCutoff -- No --> UpdateStats

    SearchPVS --> UpdateStats

    UpdateStats{"Is score > bestScore?"}
    UpdateStats -- Yes --> SetBest["bestScore = score, bestMoveId = moveId, alpha = max alpha, score"]
    UpdateStats -- No --> CutoffCheck
    SetBest --> CutoffCheck

    CutoffCheck{"alpha >= beta?"}
    CutoffCheck -- Yes --> StoreKiller[If quiet, store as Killer Move]
    StoreKiller --> StoreTT
    CutoffCheck -- No --> LoopStart

    style Start fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
    style TerminalCheck fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    style LeafCheck fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    style TTOProbe fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    style NMPCheck fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    style EvalChildren fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    style CutoffCheck fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    style ReturnBest fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
```
