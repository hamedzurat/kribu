import duckdb
import numpy as np
import torch
from torch.utils.data import Dataset


BITBOARD_OFFSETS = np.arange(37, dtype=np.uint64)
CAPTURE_OFFSETS = np.arange(6, dtype=np.uint8)


class InMemoryDuckDBDataset(Dataset):
    def __init__(self, db_path: str, view_name: str, limit: int | None = None, dedupe_rows: bool = True):
        super().__init__()

        con = duckdb.connect(db_path, read_only=True)
        columns = "*" if not dedupe_rows else "DISTINCT *"
        query = f"SELECT {columns} FROM {view_name}"
        if limit is not None:
            query += f" LIMIT {limit}"

        data = con.execute(query).fetchnumpy()
        con.close()

        self.me = data["me"].astype(np.uint64, copy=False)
        self.opp = data["opp"].astype(np.uint64, copy=False)
        self.active_capture_idx = data["active_capture_idx"].astype(np.int8, copy=False)
        self.policy_target = data.get("chosen_move")
        self.value_target = data.get("value_label")

        if self.policy_target is not None:
            self.policy_target = self.policy_target.astype(np.int64, copy=False)
        if self.value_target is not None:
            self.value_target = self.value_target.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.me)

    def __getitem__(self, index: int) -> int:
        return index

    def collate(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = np.asarray(indices, dtype=np.int64)
        me_bits = ((self.me[idx, None] >> BITBOARD_OFFSETS) & np.uint64(1)).astype(np.float32, copy=False)
        opp_bits = ((self.opp[idx, None] >> BITBOARD_OFFSETS) & np.uint64(1)).astype(np.float32, copy=False)
        cap_vals = (self.active_capture_idx[idx] + np.int8(1)).astype(np.uint8, copy=False)
        cap_bits = ((cap_vals[:, None] >> CAPTURE_OFFSETS) & np.uint8(1)).astype(np.float32, copy=False)
        features = torch.from_numpy(np.concatenate([me_bits, opp_bits, cap_bits], axis=1))

        if self.policy_target is None:
            policy_target = torch.zeros(len(idx), dtype=torch.int64)
        else:
            policy_target = torch.from_numpy(self.policy_target[idx])

        if self.value_target is None:
            value_target = torch.zeros(len(idx), dtype=torch.float32)
        else:
            value_target = torch.from_numpy(self.value_target[idx])

        return features, policy_target, value_target


def get_dataloaders(config):
    policy_dataset = InMemoryDuckDBDataset(config.duckdb_path, "policy_data", dedupe_rows=config.dedupe_dataset_rows)
    value_dataset = InMemoryDuckDBDataset(config.duckdb_path, "value_data", dedupe_rows=config.dedupe_dataset_rows)

    # Use standard DataLoader with shuffle=True so PyTorch handles batching optimally
    policy_loader = torch.utils.data.DataLoader(
        policy_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,  # Speeds up CPU to GPU transfer
        collate_fn=policy_dataset.collate,
    )

    value_loader = torch.utils.data.DataLoader(
        value_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=value_dataset.collate,
    )

    return policy_loader, value_loader
