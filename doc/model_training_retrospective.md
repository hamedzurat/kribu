# Model Training Retrospective

This document captures the full path taken to get the current supervised Sholo Guti models from weak, loop-prone behavior to the current state where they reliably beat `greedy`, crush `random`, and can often force draws against stronger search opponents.

It is meant to answer four questions:

1. What data sources exist and how good are they?
1. What bugs or mismatches were fixed along the way?
1. Which training runs and evaluation results mattered?
1. What should be tried next?

______________________________________________________________________

## Short Version

The biggest lessons so far are:

- The original training failures were not caused by one single bad dataset.
- Several engine, benchmark, trainer, and inference issues were amplifying each other.
- Larger `Minimax8` policy coverage helped a lot.
- Broad mixed-search supervision helped even more, but tends to steer the model toward draw-preserving play.
- The current strongest learned model is `search_blend_joint`, not the pure `Minimax8` policy model.
- The next bottleneck is inference and draw conversion, not dataset size alone.

______________________________________________________________________

## Raw Benchmark Datasets

The repository now contains multiple historical raw benchmark databases:

| File                                                                  | Notes                                                                                                                           |
| :-------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| [benchmark/dataset.duckdb](../benchmark/dataset.duckdb)               | Current clean generator output. Good structure, but draw-heavy in `Minimax8` vs `Mad` matchups.                                 |
| [benchmark/dataset1.duckdb](../benchmark/dataset1.duckdb)             | Largest historical dataset. Includes `ForcedRandom`, `MCTS`, and older benchmark fields. Surprisingly valuable after filtering. |
| [benchmark/dataset2.duckdb](../benchmark/dataset2.duckdb)             | Medium-sized dataset with useful `MCTS800` coverage.                                                                            |
| [benchmark/dataset3.duckdb](../benchmark/dataset3.duckdb)             | Older `Minimax8`-family dataset with many progress-rule draws.                                                                  |
| [benchmark/dataset_merged.duckdb](../benchmark/dataset_merged.duckdb) | New merged and deduplicated salvage database built from all of the above.                                                       |

### Why multiple raw datasets mattered

At first, the large old datasets looked suspicious because they contained:

- `ForcedRandom`
- older benchmark logic
- more players than the clean current generator
- different draw regimes

However, the supervised extractor in [scripts/create_training_duckdb.py](../scripts/create_training_duckdb.py) already filters out many dangerous rows for policy distillation:

- `INVALID_MOVE`
- `DRAW_MAX_TURNS`
- `DRAW_PROGRESS_RULE`
- `MadPlayer`
- non-`minimax` and non-`mcts` actors/opponents

That means the large legacy raw DB was much more salvageable than it first appeared.

______________________________________________________________________

## Important Data Profiles

### Current clean `Minimax8` extract

Extracting `Minimax8` from the current clean DB with:

```bash
PYTHONPATH=python/src uv run python scripts/create_training_duckdb.py \
  --source benchmark/dataset.duckdb \
  --output /tmp/current_mm8.duckdb \
  --teacher Minimax8 \
  --max-draw-only-visits 8
```

produced:

- `226,607` policy rows
- `744,484` value rows
- average policy support `0.9998`

This dataset was clean but still relatively small.

### Legacy `dataset1` `Minimax8` extract

Extracting `Minimax8` from [benchmark/dataset1.duckdb](../benchmark/dataset1.duckdb) produced:

- `3,853,095` policy rows
- `4,198,129` value rows
- average policy support `0.9976`

This was the first clear sign that the large legacy dataset was useful.

### `dataset2` `MCTS800` extract

Extracting `MCTS800` from [benchmark/dataset2.duckdb](../benchmark/dataset2.duckdb) produced:

- `49,158` policy rows
- `114,395` value rows
- average policy support `0.9987`

This is good enough to be interesting, but not yet strong enough to be the main teacher by itself.

______________________________________________________________________

## Merged Salvage Dataset

A new merge tool was added at [scripts/merge_benchmark_duckdb.py](../scripts/merge_benchmark_duckdb.py).

It:

- merges multiple raw benchmark DuckDB files
- preserves player/game/turn tables
- remaps `game_id`
- tracks provenance in `source_games`
- deduplicates exact repeated full-game transcripts
- normalizes old/new `games` schema differences such as `forced_random_turns`

The merged database was created with:

```bash
PYTHONPATH=python/src uv run python scripts/merge_benchmark_duckdb.py \
  --output benchmark/dataset_merged.duckdb \
  --source benchmark/dataset1.duckdb \
  --source benchmark/dataset2.duckdb \
  --source benchmark/dataset3.duckdb \
  --source benchmark/dataset.duckdb
```

Result:

- `128,738` games
- `35,953,468` turns
- `29,631` duplicate games removed
- `4,727,164` duplicate turns removed

This is now the main "big salvage" raw source.

______________________________________________________________________

## Extracted Training Datasets

Two major extracted training databases were built from the merged raw DB.

### 1. `training_mm8_legacy.duckdb`

Built from the merged raw DB with only `Minimax8` as teacher:

```bash
PYTHONPATH=python/src uv run python scripts/create_training_duckdb.py \
  --source benchmark/dataset_merged.duckdb \
  --output benchmark/training_mm8_legacy.duckdb \
  --teacher Minimax8 \
  --max-draw-only-visits 8
```

Profile:

- `4,426,070` policy rows
- `5,531,188` value rows
- average policy support `0.9977`

Purpose:

- best pure `Minimax8` imitation candidate

### 2. `training_search_blend.duckdb`

Built from the merged raw DB without teacher restriction:

```bash
PYTHONPATH=python/src uv run python scripts/create_training_duckdb.py \
  --source benchmark/dataset_merged.duckdb \
  --output benchmark/training_search_blend.duckdb \
  --max-draw-only-visits 8
```

Profile:

- `16,713,096` policy rows
- `19,205,627` value rows
- average policy support `0.9964`

Purpose:

- strongest broad supervised search-teacher mixture

______________________________________________________________________

## Important Fixes Made Along the Way

Several bugs and mismatches were fixed before the current results became meaningful.

### 1. Trainer weighting and dataset imbalance

Files:

- [python/src/trainer/dataset.py](../python/src/trainer/dataset.py)
- [python/src/trainer/train.py](../python/src/trainer/train.py)
- [python/src/trainer/config.py](../python/src/trainer/config.py)

Key fixes:

- use `visit_count` and `move_support` as policy weights
- down-weight mixed and draw-only value states
- stop oversampling tiny policy datasets because the value dataset is larger
- policy-only runs now choose `best_model.pt` using validation policy accuracy rather than only validation loss

### 2. Repetition feature normalization mismatch

Files:

- [python/src/trainer/dataset.py](../python/src/trainer/dataset.py)
- [python/src/arena/player.py](../python/src/arena/player.py)

Problem:

- training normalized repetition features by `64`
- inference had been using a different scale

Effect:

- models could behave strangely in arena despite reasonable training curves

### 3. Arena using an untrained value head for policy-only models

File:

- [python/src/arena/player.py](../python/src/arena/player.py)

Problem:

- policy-only checkpoints were still being evaluated with value-guided move selection

Effect:

- arena could become dramatically weaker even when policy accuracy improved

Fix:

- checkpoints now carry `use_value_guidance`
- policy-only runs disable value guidance at inference

### 4. Minimax draw modeling mismatch

File:

- [engine/include/kribu/player/minimax.hpp](../engine/include/kribu/player/minimax.hpp)

Problem:

- `DRAW_PROGRESS_RULE` was not modeled consistently as a terminal draw inside search
- cycles were treated as exactly neutral

Fix:

- progress draws treated as terminal
- mild draw contempt added

### 5. Benchmark invalid-move accounting

File:

- [engine/include/kribu/benchmark.hpp](../engine/include/kribu/benchmark.hpp)

Problem:

- invalid moves were previously logged as draws

Fix:

- invalid move now awards a win to the opponent with reason `INVALID_MOVE`

### 6. Benchmark schedule cleanup

Files:

- [engine/benchmark/config.hpp](../engine/benchmark/config.hpp)
- [engine/benchmark/players.hpp](../engine/benchmark/players.hpp)

Changes:

- added `Minimax4`
- reduced deterministic repeated matchups to one explicit game per start side
- left stochastic or broad-coverage matchups as the main data generators

______________________________________________________________________

## Trainer Presets

To avoid hand-editing config every run, named presets were added to [python/src/trainer/config.py](../python/src/trainer/config.py).

Current presets:

| Preset                  | Purpose                                       |
| :---------------------- | :-------------------------------------------- |
| `mm8_current_policy`    | Current clean `Minimax8` policy-only baseline |
| `mm8_legacy_policy`     | Large merged `Minimax8` policy-only run       |
| `search_blend_joint`    | Broad mixed-search joint policy+value run     |
| `mcts_bootstrap_policy` | Smaller experimental `MCTS` bootstrap preset  |

Example:

```bash
KRIBU_TRAIN_PRESET=mm8_legacy_policy PYTHONPATH=python/src uv run python -m trainer
```

______________________________________________________________________

## Fight Club Tuning

File:

- [python/src/fightclub/config.py](../python/src/fightclub/config.py)

Changes:

- reduced `MAX_TURNS` to `512`
- reduced `GAMES_PER_PAIR` to `20`
- capped workers and left one CPU core free

Purpose:

- keep human-facing comparison runs responsive
- avoid extremely long tournaments and painful lockups

______________________________________________________________________

## Major Training Runs

Three runs became the key comparison set.

### 1. `mm8_current_policy`

TensorBoard:

- best validation policy accuracy about `52.55%`

Arena:

- vs `greedy`: `15/20`
- vs `minimax4`: `0/20`
- vs `minimax8`: `0/20`

Conclusion:

- useful baseline
- clearly weaker than later runs

### 2. `mm8_legacy_policy`

TensorBoard:

- best validation policy accuracy about `67.49%`

Arena before shallow search:

- vs `greedy`: `20/20`
- vs `minimax4`: `0/20`
- vs `minimax8`: `0/19/1`

Fight Club:

- much stronger than the current-clean baseline
- beats all `mcts_*`
- still loses to `minimax4`

Conclusion:

- the large merged `Minimax8` policy dataset worked
- data scale was a real bottleneck up to this point

### 3. `search_blend_joint`

TensorBoard:

- best validation policy accuracy about `67.12%`
- best validation loss about `0.9901`
- best value MAE about `0.2878`

Arena before shallow search:

- vs `greedy`: `20/20`
- vs `minimax4`: `0 wins, 10 losses, 10 draws`
- vs `minimax8`: `0 wins, 10 losses, 10 draws`

Fight Club:

- strongest learned model of the three

Conclusion:

- the value head plus broad policy data made the model much harder to beat
- but also more draw-prone

______________________________________________________________________

## Shallow Search Inference Layer

File:

- [python/src/arena/player.py](../python/src/arena/player.py)

Later, a shallow policy-pruned search layer was added on top of the neural model.

Key ideas:

- root candidate limit
- reply candidate limit
- shallow negamax-style recursion
- mild draw contempt
- leaf blending between predicted value and draw score

This was intended to convert "safe draw" play into stronger practical decisions without retraining.

### Effect on `search_blend_joint`

Arena after shallow search:

- vs `greedy`: `20/20`
- vs `minimax4`: all tested games became `progress_rule` draws
- vs `minimax8`: all tested games became `progress_rule` draws

Interpretation:

- the model is now very good at not losing
- but still not yet converting those non-losing lines into wins

### Effect on `mm8_legacy_policy`

Arena after shallow search:

- still crushes `greedy`
- still loses to `minimax4`

Interpretation:

- shallow search only helps when the value head is strong enough to guide it

______________________________________________________________________

## Current Best Understanding

The current evidence says:

1. Dataset size and quality were real bottlenecks early on.
1. Those bottlenecks are no longer the main issue.
1. The strongest branch is now `search_blend_joint`.
1. The remaining problem is draw conversion, not basic competence.

This means the next gains are more likely to come from:

- stronger inference
- better anti-draw pressure
- slightly deeper or wider policy-pruned search

and less likely to come from:

- endlessly creating more random dataset variants

______________________________________________________________________

## Recommended Next Steps

### Primary branch

Treat `search_blend_joint` as the main branch.

Why:

- strongest Fight Club standing
- best practical survival vs `minimax4` and `minimax8`
- best use of the value head

### Best next engineering move

Improve the shallow search path in [python/src/arena/player.py](../python/src/arena/player.py):

- increase draw contempt slightly
- prefer materially favorable non-draw continuations more aggressively
- increase candidate breadth carefully

Goal:

- convert a fraction of the current `progress_rule` draws into wins

### Only then consider one extra training run

If search improvements plateau, the clean next training experiment is:

- use [benchmark/training_search_blend.duckdb](../benchmark/training_search_blend.duckdb)
- train it with `policy_only = True`

Purpose:

- isolate whether broad policy coverage alone is enough
- separate "value head helps survival" from "value head encourages draws"

______________________________________________________________________

## Useful Commands

### Train a preset

```bash
KRIBU_TRAIN_PRESET=mm8_current_policy PYTHONPATH=python/src uv run python -m trainer
```

```bash
KRIBU_TRAIN_PRESET=mm8_legacy_policy PYTHONPATH=python/src uv run python -m trainer
```

```bash
KRIBU_TRAIN_PRESET=search_blend_joint PYTHONPATH=python/src uv run python -m trainer
```

### Run arena

```bash
PYTHONPATH=python/src uv run python -m arena -m models/search_blend_joint.pt -o minimax4 -n 20 -c arena_search_blend_joint_minimax4.csv
```

### Run Fight Club

```bash
PYTHONPATH=python/src uv run python -m fightclub
```

### Rebuild merged raw DB

```bash
PYTHONPATH=python/src uv run python scripts/merge_benchmark_duckdb.py \
  --output benchmark/dataset_merged.duckdb \
  --source benchmark/dataset1.duckdb \
  --source benchmark/dataset2.duckdb \
  --source benchmark/dataset3.duckdb \
  --source benchmark/dataset.duckdb
```

______________________________________________________________________

## Final Takeaway

The project is no longer in the "bad data / broken trainer" phase.

It is now in the "strong enough to survive, not yet strong enough to convert" phase.

That is a much better problem to have.
