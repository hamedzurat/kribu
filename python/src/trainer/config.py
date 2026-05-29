"""Configuration for the Sholo Guti PyTorch trainer."""

import dataclasses


@dataclasses.dataclass
class TrainerConfig:
    # Dataset
    duckdb_path: str = "benchmark/dataset.duckdb"
    batch_size: int = 1024
    num_workers: int = 0
    prefetch_factor: int | None = None

    # Model Architecture
    input_features: int = 37 + 37 + 6  # me, opp, active_capture
    hidden_dim: int = 256
    num_residual_blocks: int = 6
    action_space: int = 265  # TOTAL_MOVE_COUNT derived from engine/include/kribu/board.hpp (includes END_CHAIN_MOVE)

    # Training Loop
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    save_dir: str = "checkpoints"
    save_every_epochs: int = 5
    value_loss_weight: float = 1.0  # Weight for value loss relative to policy loss


config = TrainerConfig()
