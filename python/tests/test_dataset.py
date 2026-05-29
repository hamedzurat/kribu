## @file test_dataset.py
#  @brief Unit tests for Sholo Guti dataset generation, loading, and neural network structure.

import os
import tempfile
import torch

from kribu.generate_dataset import generate_dataset, generate_game
from kribu.train import SholoGutiDataset, SholoGutiNet
import kribu


## @brief Tests simulating a single game of Sholo Guti using minimax.
def test_generate_game():
    samples = generate_game(minimax_depth=1)
    assert isinstance(samples, list)
    assert len(samples) > 0

    # Verify structure of a single sample
    sample = samples[0]
    assert "me" in sample
    assert "opp" in sample
    assert "activeCaptureIdx" in sample
    assert "policy" in sample
    assert "value" in sample

    assert isinstance(sample["me"], int)
    assert isinstance(sample["opp"], int)
    assert isinstance(sample["activeCaptureIdx"], int)
    assert isinstance(sample["policy"], list)
    assert isinstance(sample["value"], float)

    # Policy list size should match total move count
    assert len(sample["policy"]) == kribu.TOTAL_MOVE_COUNT
    # Value should be normalized between -1.0 and 1.0
    assert -1.0 <= sample["value"] <= 1.0


## @brief Tests writing and reading the dataset in DuckDB format.
def test_dataset_generation_and_loading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "test_dataset.duckdb")

        # Generate a dataset with 2 games
        generate_dataset(num_games=2, output_path=dataset_path, minimax_depth=1)
        assert os.path.exists(dataset_path)

        # Load using SholoGutiDataset
        ds = SholoGutiDataset(dataset_path)
        assert len(ds) > 0

        # Retrieve a sample
        state, policy, value = ds[0]

        # Verify shapes and types
        assert isinstance(state, torch.Tensor)
        assert isinstance(policy, torch.Tensor)
        assert isinstance(value, torch.Tensor)

        assert state.shape == (3, 37)
        assert policy.shape == (kribu.TOTAL_MOVE_COUNT,)
        assert value.shape == (1,)

        # Active player vector and Opponent player vector should not have pieces on same nodes
        me_mask = state[0]
        opp_mask = state[1]
        assert torch.sum(me_mask * opp_mask).item() == 0.0


## @brief Tests the forward pass of the policy-value network.
def test_shologuti_net_forward():
    model = SholoGutiNet()

    # Batch of size 4
    dummy_input = torch.zeros((4, 3, 37), dtype=torch.float32)
    policy_logits, value_pred = model(dummy_input)

    assert policy_logits.shape == (4, kribu.TOTAL_MOVE_COUNT)
    assert value_pred.shape == (4, 1)

    # Check value predictions are bounded in [-1, 1] due to tanh
    assert torch.all(value_pred >= -1.0)
    assert torch.all(value_pred <= 1.0)
