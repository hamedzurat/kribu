from kribu import (
    INITIAL_STATE,
    minimax_player_4,
    minimax_player_8,
    minimax_player_8_mad2,
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


def test_additional_arena_opponents_return_legal_moves():
    valid_moves = all_possible_moves(INITIAL_STATE)

    assert minimax_player_4(INITIAL_STATE) in valid_moves
    assert minimax_player_8_mad2(INITIAL_STATE) in valid_moves


def test_neural_player_dynamic_architecture(tmp_path):
    # Create a model with custom architecture (different hidden_dim and blocks)
    customModel = SholoGutiNet(input_features=83, hidden_dim=128, num_residual_blocks=3, action_space=265)

    # Save state dict
    modelPath = os.path.join(tmp_path, "custom_model.pt")
    torch.save(customModel.state_dict(), modelPath)

    # Load via NeuralPlayer
    player = NeuralPlayer(model_path=modelPath, device="cpu")

    # Assert model was instantiated with the auto-detected architecture
    assert player.model.input_proj[0].out_features == 128
    assert len(player.model.blocks) == 3


def test_neural_player_selects_best_valid_move(tmp_path):
    model = SholoGutiNet(input_features=83, hidden_dim=32, num_residual_blocks=1, action_space=265)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy_head[3].bias[1] = 1.0
        model.policy_head[3].bias[3] = 5.0

    modelPath = os.path.join(tmp_path, "masked_model.pt")
    torch.save(model.state_dict(), modelPath)

    player = NeuralPlayer(model_path=modelPath, device="cpu")
    move_id, _ = player.get_move(
        INITIAL_STATE.me,
        INITIAL_STATE.opp,
        INITIAL_STATE.activeCaptureIdx,
        valid_moves=[1],
    )

    assert move_id == 1


def test_neural_player_uses_value_guidance_for_valid_moves(tmp_path):
    model = SholoGutiNet(input_features=83, hidden_dim=32, num_residual_blocks=1, action_space=265)
    modelPath = os.path.join(tmp_path, "value_guided_model.pt")
    torch.save({"model": model.state_dict(), "use_value_guidance": True}, modelPath)

    player = NeuralPlayer(model_path=modelPath, device="cpu")

    valid_moves = all_possible_moves(INITIAL_STATE)
    assert len(valid_moves) >= 2
    preferred_move = int(valid_moves[0])
    value_preferred_move = int(valid_moves[1])

    class CaptureModel(torch.nn.Module):
        def __init__(self, action_space: int, preferred_move: int, value_preferred_move: int):
            super().__init__()
            self.action_space = action_space
            self.preferred_move = preferred_move
            self.value_preferred_move = value_preferred_move

        def forward(self, x):
            batch = x.shape[0]
            policy = torch.zeros(batch, self.action_space)
            value = torch.zeros(batch)
            if batch == 1:
                policy[0, self.preferred_move] = 5.0
                policy[0, self.value_preferred_move] = 1.0
            else:
                value[0] = 0.1
                value[1] = 0.9
            return policy, value

    player.model = CaptureModel(265, preferred_move, value_preferred_move)
    player._successor_value_for_move = lambda _state, move_id, predicted_value: (
        predicted_value if move_id == value_preferred_move else 0.0
    )
    move_id, _ = player.get_move(
        INITIAL_STATE.me,
        INITIAL_STATE.opp,
        INITIAL_STATE.activeCaptureIdx,
        valid_moves=[preferred_move, value_preferred_move],
        state=INITIAL_STATE,
    )

    assert move_id == value_preferred_move


def test_neural_player_policy_only_skips_value_guidance(tmp_path):
    model = SholoGutiNet(input_features=83, hidden_dim=32, num_residual_blocks=1, action_space=265)
    modelPath = os.path.join(tmp_path, "policy_only_model.pt")
    torch.save({"model": model.state_dict(), "use_value_guidance": False}, modelPath)

    player = NeuralPlayer(model_path=modelPath, device="cpu")
    valid_moves = [7, 11, 19]

    class PolicyOnlyModel(torch.nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            policy = torch.full((batch, 265), -10.0)
            policy[:, 7] = 0.5
            policy[:, 11] = 3.0
            policy[:, 19] = 1.0
            return policy, torch.zeros(batch)

    player.model = PolicyOnlyModel()
    move_id, _ = player.get_move(
        INITIAL_STATE.me,
        INITIAL_STATE.opp,
        INITIAL_STATE.activeCaptureIdx,
        valid_moves=valid_moves,
        state=INITIAL_STATE,
    )

    assert move_id == 11


def test_neural_player_scales_repetition_features_like_trainer(tmp_path):
    model = SholoGutiNet(input_features=83, hidden_dim=32, num_residual_blocks=1, action_space=265)
    modelPath = os.path.join(tmp_path, "feature_scale_model.pt")
    torch.save(model.state_dict(), modelPath)

    player = NeuralPlayer(model_path=modelPath, device="cpu")

    class CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_input = None

        def forward(self, x):
            self.last_input = x.detach().cpu()
            batch = x.shape[0]
            return torch.zeros(batch, 265), torch.zeros(batch)

    capture_model = CaptureModel()
    player.model = capture_model

    from kribu import boardState

    state = boardState()
    state.me = INITIAL_STATE.me
    state.opp = INITIAL_STATE.opp
    state.activeCaptureIdx = INITIAL_STATE.activeCaptureIdx
    state.hash = INITIAL_STATE.hash
    state.historyCount = 64
    move_id, _ = player.get_move(
        state.me,
        state.opp,
        state.activeCaptureIdx,
        valid_moves=[1],
        state=state,
    )

    assert move_id == 1
    assert capture_model.last_input is not None
    assert capture_model.last_input[0, 80].item() == 1.0
