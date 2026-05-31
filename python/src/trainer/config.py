"""Configuration for the Sholo Guti PyTorch trainer."""

import dataclasses


@dataclasses.dataclass
class TrainerConfig:
    # Dataset

    duckdb_path: str = "benchmark/dataset.duckdb"
    # Remove exact duplicate training rows when reading DuckDB views.
    dedupe_dataset_rows: bool = True
    # # Batch Size
    # What it does: Dictates how many board states the network looks at before updating its weights.
    # How to choose: For Value Networks, the labels are inherently noisy (a single blunder on turn 40 changes the label for turn 5). You *need* massive batch sizes so the noise cancels out and the true gradient emerges.
    # When to adjust: Keep it as high as your GPU VRAM allows. `16384` is excellent. If you run out of memory, drop it to `8192`.
    batch_size: int = 16384
    num_workers: int = 0

    # Model Architecture

    # 37 bits for me, 37 for opp, 6 bits for active_capture
    input_features: int = 37 + 37 + 6
    # # Neurons per layer
    # What it does: Width determines the network's capacity to recognize complex, concurrent patterns (like identifying multiple trapping setups across the board at once).
    # How to choose: For bitboard inputs, standard sizes are `256`, `512`, or `1024`.
    # When to adjust: If your training loss refuses to go down (underfitting), increase the width. If your training loss drops near zero but your model plays terribly against new opponents (overfitting), decrease the width. `512` is a perfect starting point.
    hidden_dim: int = 512
    # # Number of Residual Blocks
    # What it does: Depth allows the network to perform multi-step "logical deductions." Think of each block as one step of lookahead intuition.
    # How to choose: AlphaZero used 19 blocks for Go and 40 for Chess. Sholo Guti is simpler, so `10` is a very smart, lightweight choice.
    # When to adjust: If you find the engine is missing deep tactical sequences, try increasing to `15` or `20`. Just keep in mind that deeper networks are slower to evaluate during MCTS search.
    num_residual_blocks: int = 10
    # # Action Space
    # What it does: The total number of possible moves the engine can make.
    action_space: int = 265  # TOTAL_MOVE_COUNT

    # Training Loop

    # # Learning Rate
    # What it does: Controls the size of the "steps" the network takes when updating its weights. Too high, and it overshoots the solution (exploding gradients). Too low, and it takes forever to train.
    # How to choose: `1e-3` (0.001) is a very standard starting point for AdamW.
    # When to adjust: If your loss suddenly spikes to NaN or Infinity, decrease the learning rate (e.g., to `3e-4`). If your loss barely moves, you can try increasing it slightly, but be careful.
    learning_rate: float = 1e-3
    # # Weight Decay
    # What it does: A technique to prevent overfitting. It penalizes large weights, encouraging the network to learn simpler patterns. `1e-4` is a robust, standard value.
    weight_decay: float = 1e-4
    # # Epochs
    # What it does: One epoch is one complete pass through the entire dataset. For a 100M dataset, 100 epochs means the network sees 10 Billion positions. This is a decent number for training a strong model from scratch.
    # When to adjust: Train until the validation loss stops improving (overfitting starts). For small datasets (like 1M), you might only need 50 epochs. For huge datasets, you might do 200+.
    epochs: int = 100
    save_dir: str = "checkpoints"
    # # Save Checkpoints
    # What it does: How often to save a snapshot of the model weights to disk. Setting this to `1` saves after every epoch. Setting it to `10` saves only every 10 epochs.
    # When to adjust: For long runs, setting this to a higher number (e.g., 10, 50) saves disk space and time. If you want to be able to resume training from *exactly* the 7th epoch, set it to `1`.
    save_every_epochs: int = 5
    # # Value Loss Weight
    # What it does: Balances how much the network cares about getting the "Move" right (Policy) vs getting the "Win Probability" right (Value). Policy loss is usually Cross-Entropy (values ~2.0 to 5.0). Value loss is usually Mean Squared Error (values ~0.1 to 0.3).
    # When to adjust: If you notice your network is amazing at predicting moves but terrible at evaluating who is winning, increase `value_loss_weight` to `2.0` or `5.0`.
    value_loss_weight: float = 1.0


config = TrainerConfig()
