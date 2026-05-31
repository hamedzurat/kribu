import pytest
import torch
from torch.utils.data import Dataset

from trainer.dataset import split_dataset, validation_stride
from trainer.model import SholoGutiNet
from trainer.train import (
    dataset_passes,
    is_improved,
    loss_total,
    metric_points,
    render_line_chart,
    render_training_dashboard,
    resolve_steps_per_epoch,
)


class IndexDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return index


def test_resolve_steps_per_epoch_uses_larger_loader_by_default():
    assert resolve_steps_per_epoch(policy_batches=640, value_batches=647, requested_steps=None) == 647


def test_resolve_steps_per_epoch_keeps_explicit_override():
    assert resolve_steps_per_epoch(policy_batches=640, value_batches=647, requested_steps=1000) == 1000


def test_resolve_steps_per_epoch_rejects_invalid_values():
    with pytest.raises(ValueError):
        resolve_steps_per_epoch(policy_batches=640, value_batches=647, requested_steps=0)

    with pytest.raises(ValueError):
        resolve_steps_per_epoch(policy_batches=0, value_batches=647, requested_steps=None)


def test_dataset_passes_reports_epoch_coverage():
    assert dataset_passes(steps_per_epoch=647, batches_per_dataset=647) == 1.0
    assert dataset_passes(steps_per_epoch=1000, batches_per_dataset=640) == pytest.approx(1.5625)


def test_validation_stride_uses_requested_fraction():
    assert validation_stride(0.02) == 50
    assert validation_stride(0.0) is None

    with pytest.raises(ValueError):
        validation_stride(0.5)


def test_split_dataset_uses_every_nth_row_for_validation():
    train_dataset, validation_dataset = split_dataset(IndexDataset(), validation_fraction=0.2)

    assert [validation_dataset[i] for i in range(len(validation_dataset))] == [0, 5]
    assert [train_dataset[i] for i in range(len(train_dataset))] == [1, 2, 3, 4, 6, 7, 8, 9]


def test_loss_and_improvement_helpers():
    assert loss_total(policy_loss=2.0, value_loss=0.25, value_loss_weight=2.0) == 2.5
    assert is_improved(metric=0.9, best_metric=1.0, min_improvement=0.01)
    assert not is_improved(metric=0.995, best_metric=1.0, min_improvement=0.01)


def test_metric_points_skips_nan_values():
    history = [(1, 3.0, float("nan"), 0.1, 0.4), (2, 2.5, 2.6, 0.2, 0.3)]

    assert metric_points(history, 2) == [(2, 2.6)]


def test_render_line_chart_fits_requested_size():
    history = [(epoch, 3.0 - epoch * 0.1, 3.2 - epoch * 0.08, 0.1 + epoch * 0.01, 0.5) for epoch in range(1, 8)]

    lines = render_line_chart(history, [("train", 1, "*"), ("validation", 2, "o")], "Total loss", width=48, height=8)

    assert len(lines) == 8
    assert all(len(line) <= 48 for line in lines)
    assert "Total loss" in lines[0]


def test_render_training_dashboard_uses_available_height():
    history = [(epoch, 3.0 - epoch * 0.1, 3.2 - epoch * 0.08, 0.1 + epoch * 0.01, 0.5) for epoch in range(1, 8)]

    dashboard = render_training_dashboard(history, width=60, height=18)

    lines = dashboard.plain.splitlines()
    assert len(lines) == 18
    assert all(len(line) <= 60 for line in lines)


def test_model_outputs_policy_logits_and_bounded_value():
    model = SholoGutiNet(input_features=80, hidden_dim=32, num_residual_blocks=2, action_space=265)
    policy_logits, value = model(torch.zeros(4, 80))

    assert policy_logits.shape == (4, 265)
    assert value.shape == (4,)
    assert torch.all(value >= 0.0)
    assert torch.all(value <= 1.0)
