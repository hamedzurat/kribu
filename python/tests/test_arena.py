from kribu import (
    INITIAL_STATE,
    minimax_player_8,
    all_possible_moves,
)
import os
import torch
from arena.player import NeuralPlayer
from trainer.model import SholoGutiNet


def test_minimax_player_8():
    # Verify that minimax_player_8 is callable and returns a valid move index
    move_id = minimax_player_8(INITIAL_STATE)
    assert isinstance(move_id, int)

    valid_moves = all_possible_moves(INITIAL_STATE)
    assert move_id in valid_moves


def test_neural_player_dynamic_architecture(tmp_path):
    # Create a model with custom architecture (different hidden_dim and blocks)
    customModel = SholoGutiNet(input_features=80, hidden_dim=128, num_residual_blocks=3, action_space=265)

    # Save state dict
    modelPath = os.path.join(tmp_path, "custom_model.pt")
    torch.save(customModel.state_dict(), modelPath)

    # Load via NeuralPlayer
    player = NeuralPlayer(model_path=modelPath, device="cpu")

    # Assert model was instantiated with the auto-detected architecture
    assert player.model.input_proj[0].out_features == 128
    assert len(player.model.blocks) == 3
