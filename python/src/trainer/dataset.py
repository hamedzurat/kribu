import duckdb
import kribu
import numpy as np
import torch
from torch.utils.data import Dataset


BITBOARD_OFFSETS = np.arange(37, dtype=np.uint64)
CAPTURE_OFFSETS = np.arange(6, dtype=np.uint8)


def legal_move_mask(me: int, opp: int, active_capture_idx: int, action_space: int) -> np.ndarray:
    """Return a boolean action mask for legal moves in one board state."""
    state = kribu.boardState()
    state.me = int(me)
    state.opp = int(opp)
    state.activeCaptureIdx = int(active_capture_idx)

    mask = np.zeros(action_space, dtype=np.bool_)
    for moveId in kribu.all_possible_moves(state):
        if 0 <= moveId < action_space:
            mask[moveId] = True
    return mask


class InMemoryDuckDBDataset(Dataset):
    """DuckDB-backed dataset kept in memory for fast repeated training passes."""

    def __init__(
        self,
        db_path: str,
        view_name: str,
        limit: int | None = None,
        *,
        include_legal_mask: bool = False,
        action_space: int = 265,
    ):
        super().__init__()

        con = duckdb.connect(db_path, read_only=True)
        query = f"SELECT * FROM {view_name}"
        if limit is not None:
            query += f" LIMIT {limit}"

        data = con.execute(query).fetchnumpy()
        con.close()

        self.me = data["me"].astype(np.uint64, copy=False)
        self.opp = data["opp"].astype(np.uint64, copy=False)
        self.active_capture_idx = data["active_capture_idx"].astype(np.int8, copy=False)
        self.policy_target = data.get("chosen_move")
        self.value_target = data.get("value_label")
        self.include_legal_mask = include_legal_mask
        self.action_space = action_space

        if self.policy_target is not None:
            self.policy_target = self.policy_target.astype(np.int64, copy=False)
        if self.value_target is not None:
            self.value_target = self.value_target.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.me)

    def __getitem__(self, index: int) -> int:
        return index

    def collate(self, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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

        legal_mask = None
        if self.include_legal_mask:
            masks = [
                legal_move_mask(self.me[row], self.opp[row], self.active_capture_idx[row], self.action_space)
                for row in idx
            ]
            legal_mask = torch.from_numpy(np.stack(masks, axis=0))
            if self.policy_target is not None:
                legal_mask.scatter_(1, policy_target[:, None], True)

        return features, policy_target, value_target, legal_mask


class ModuloSplitDataset(Dataset):
    """Map positions to original dataset indices using an every-Nth-row validation split."""

    def __init__(self, dataset: Dataset, validation_stride: int, split: str):
        super().__init__()
        if validation_stride < 2:
            raise ValueError("validation_stride must be at least 2")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")

        self.dataset = dataset
        self.validation_stride = validation_stride
        self.split = split
        self.validation_len = (len(dataset) + validation_stride - 1) // validation_stride
        self.train_len = len(dataset) - self.validation_len

    def __len__(self) -> int:
        if self.split == "validation":
            return self.validation_len
        return self.train_len

    def __getitem__(self, index: int) -> int:
        if self.split == "validation":
            original_index = index * self.validation_stride
        else:
            original_index = index + 1 + index // (self.validation_stride - 1)
        return self.dataset[original_index]


def validation_stride(validation_fraction: float) -> int | None:
    """Return split stride for the requested validation fraction, or None when disabled."""
    if validation_fraction < 0.0 or validation_fraction >= 0.5:
        raise ValueError("validation_fraction must be in [0.0, 0.5)")
    if validation_fraction == 0.0:
        return None
    return max(2, round(1.0 / validation_fraction))


def split_dataset(dataset: Dataset, validation_fraction: float) -> tuple[Dataset, Dataset | None]:
    """Split a dataset without materializing large index arrays."""
    stride = validation_stride(validation_fraction)
    if stride is None:
        return dataset, None
    return ModuloSplitDataset(dataset, stride, "train"), ModuloSplitDataset(dataset, stride, "validation")


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, pin_memory: bool, shuffle: bool, collate_fn):
    """Create a DataLoader using project trainer defaults."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def get_dataloaders(config):
    """Build policy/value train and validation dataloaders."""
    policy_dataset = InMemoryDuckDBDataset(
        config.duckdb_path,
        "policy_data",
        include_legal_mask=config.policy_legal_mask,
        action_space=config.action_space,
    )
    value_dataset = InMemoryDuckDBDataset(config.duckdb_path, "value_data")
    policy_train_dataset, policy_validation_dataset = split_dataset(policy_dataset, config.validation_fraction)
    value_train_dataset, value_validation_dataset = split_dataset(value_dataset, config.validation_fraction)

    policy_loader = make_loader(
        policy_train_dataset,
        config.batch_size,
        config.num_workers,
        torch.cuda.is_available(),
        True,
        policy_dataset.collate,
    )
    value_loader = make_loader(
        value_train_dataset,
        config.batch_size,
        config.num_workers,
        torch.cuda.is_available(),
        True,
        value_dataset.collate,
    )

    if policy_validation_dataset is None or value_validation_dataset is None:
        return policy_loader, value_loader, None, None

    policy_validation_loader = make_loader(
        policy_validation_dataset,
        config.batch_size,
        config.num_workers,
        torch.cuda.is_available(),
        False,
        policy_dataset.collate,
    )
    value_validation_loader = make_loader(
        value_validation_dataset,
        config.batch_size,
        config.num_workers,
        torch.cuda.is_available(),
        False,
        value_dataset.collate,
    )

    return policy_loader, value_loader, policy_validation_loader, value_validation_loader
