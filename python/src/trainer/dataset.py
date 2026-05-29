import duckdb
import torch
from torch.utils.data import Dataset
import numpy as np


class InMemoryDuckDBDataset(Dataset):
    def __init__(self, db_path: str, view_name: str, limit: int | None = None):
        super().__init__()

        # Load everything into memory ONCE
        con = duckdb.connect(db_path, read_only=True)
        query = f"SELECT * FROM {view_name}"
        if limit is not None:
            query += f" LIMIT {limit}"

        df = con.execute(query).df()

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
        cap_vals = df["active_capture_idx"].to_numpy(dtype=np.int8) + np.int8(1)
        cap_vals = cap_vals.astype(np.uint8)
        cap_bits = np.zeros((len(df), 6), dtype=np.float32)
        for i in range(6):
            cap_bits[:, i] = (cap_vals >> np.uint8(i)) & np.uint8(1)

        # Concatenate features -> shape: (N, 80)
        X = np.concatenate([me_bits, opp_bits, cap_bits], axis=1)
        self.X = torch.from_numpy(X)

        # Parse targets
        if "chosen_move" in df.columns:
            self.policy_target = torch.from_numpy(df["chosen_move"].to_numpy(dtype=np.int64).copy())
        else:
            self.policy_target = torch.zeros(len(df), dtype=torch.int64)

        if "value_label" in df.columns:
            self.value_target = torch.from_numpy(df["value_label"].to_numpy(dtype=np.float32).copy())
        else:
            self.value_target = torch.zeros(len(df), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.policy_target[index], self.value_target[index]


def get_dataloaders(config):
    # Load entire dataset directly into memory for insane speed
    # We can add a limit if we ever get >10M rows to protect RAM
    policy_dataset = InMemoryDuckDBDataset(config.duckdb_path, "policy_data")
    value_dataset = InMemoryDuckDBDataset(config.duckdb_path, "value_data")

    # Use standard DataLoader with shuffle=True so PyTorch handles batching optimally
    policy_loader = torch.utils.data.DataLoader(
        policy_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,  # Speeds up CPU to GPU transfer
    )

    value_loader = torch.utils.data.DataLoader(
        value_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True
    )

    return policy_loader, value_loader
