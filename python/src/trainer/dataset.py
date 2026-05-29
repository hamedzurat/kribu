import duckdb
import torch
from torch.utils.data import IterableDataset
import numpy as np


class DuckDBDataset(IterableDataset):
    def __init__(self, db_path: str, view_name: str, batch_size: int = 10000):
        super().__init__()
        self.db_path = db_path
        self.view_name = view_name
        self.batch_size = batch_size

    def __iter__(self):
        # Create a new connection for each worker/iterator
        con = duckdb.connect(self.db_path, read_only=True)
        # We order by game_id and turn_idx just to have deterministic but we actually want shuffling.
        # But since it's an iterable dataset on a large DB, shuffling is hard.
        # We can just fetch randomly or sequentially and shuffle inside the batch.
        query = f"SELECT * FROM {self.view_name}"

        # Use fetchmany for streaming without PyArrow
        cursor = con.execute(query)

        while True:
            chunk = cursor.fetchmany(self.batch_size)
            if not chunk:
                break

            import pandas as pd

            cols = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(chunk, columns=cols)

            # 1. Decode me (uint64) to 37 bits
            me_vals = df["me"].to_numpy(dtype=np.uint64)
            me_bits = np.zeros((len(df), 37), dtype=np.float32)
            for i in range(37):
                me_bits[:, i] = (me_vals >> np.uint64(i)) & np.uint64(1)

            # 2. Decode opp (uint64) to 37 bits
            opp_vals = df["opp"].to_numpy(dtype=np.uint64)
            opp_bits = np.zeros((len(df), 37), dtype=np.float32)
            for i in range(37):
                opp_bits[:, i] = (opp_vals >> np.uint64(i)) & np.uint64(1)

            # 3. Decode active_capture_idx (-1 to 36) to 6 bits
            cap_vals = df["active_capture_idx"].to_numpy(dtype=np.int8) + np.int8(1)  # 0 to 37
            cap_vals = cap_vals.astype(np.uint8)
            cap_bits = np.zeros((len(df), 6), dtype=np.float32)
            for i in range(6):
                cap_bits[:, i] = (cap_vals >> np.uint8(i)) & np.uint8(1)

            # Concatenate features
            # shape: (batch_size, 80)
            X = np.concatenate([me_bits, opp_bits, cap_bits], axis=1)

            X_tensor = torch.from_numpy(X)

            # Policy target (chosen_move)
            if "chosen_move" in df.columns:
                policy_target = torch.from_numpy(df["chosen_move"].to_numpy(dtype=np.int64).copy())
            else:
                policy_target = torch.zeros(len(df), dtype=torch.int64)

            # Value target
            if "value_label" in df.columns:
                value_target = torch.from_numpy(df["value_label"].to_numpy(dtype=np.float32).copy())
            else:
                value_target = torch.zeros(len(df), dtype=torch.float32)  # Dummy if not present

            # Yield individual samples (DataLoader will re-batch them or we can yield batches directly)
            # Since we use DataLoader, yielding items is standard, but yielding batches is faster if we use batch_size=None
            for i in range(len(df)):
                yield X_tensor[i], policy_target[i], value_target[i]


def get_dataloaders(config):
    # For performance, we can let DataLoader do the batching,
    # but since DuckDBDataset reads in chunks, it's efficient.
    policy_dataset = DuckDBDataset(config.duckdb_path, "policy_data", batch_size=10000)
    value_dataset = DuckDBDataset(config.duckdb_path, "value_data", batch_size=10000)

    # Use standard DataLoader. Shuffle buffer could be added if we wanted,
    # but sequential reading + small random behavior is okay for now.
    policy_loader = torch.utils.data.DataLoader(
        policy_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    value_loader = torch.utils.data.DataLoader(
        value_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    return policy_loader, value_loader
