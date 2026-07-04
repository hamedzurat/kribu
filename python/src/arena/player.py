"""Neural Network Player wrapper for Sholo Guti in the Arena."""

import os
import numpy as np
import torch
import kribu

from trainer.config import config
from trainer.model import SholoGutiNet


MAX_REPETITION_HISTORY = 64.0
POLICY_LOGIT_WEIGHT = 0.15


class NeuralPlayer:
    """A reusable class that loads a trained PyTorch model and provides move predictions."""

    def __init__(self, model_path: str = "checkpoints/best_model.pt", device: str | None = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}. Train the model first!")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                try:
                    # Dummy operation to catch architecture mismatch (e.g., on GTX 1050 Ti)
                    _ = torch.nn.Linear(1, 1).to(device)(torch.zeros(1, 1).to(device))
                except RuntimeError:
                    device = "cpu"

        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.use_value_guidance = not config.policy_only
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            stateDict = checkpoint["model"]
            self.use_value_guidance = bool(checkpoint.get("use_value_guidance", True))
        else:
            stateDict = checkpoint

        # Auto-detect architecture from stateDict
        hiddenDim = config.hidden_dim
        inputFeatures = config.input_features
        numResidualBlocks = config.num_residual_blocks
        actionSpace = config.action_space

        if "input_proj.0.weight" in stateDict:
            hiddenDim = stateDict["input_proj.0.weight"].shape[0]
            inputFeatures = stateDict["input_proj.0.weight"].shape[1]

        blockIndices = []
        for key in stateDict.keys():
            if key.startswith("blocks.") and key.endswith(".fc1.weight"):
                try:
                    parts = key.split(".")
                    blockIndices.append(int(parts[1]))
                except (ValueError, IndexError):
                    pass
        if blockIndices:
            numResidualBlocks = max(blockIndices) + 1

        if "policy_head.3.weight" in stateDict:
            actionSpace = stateDict["policy_head.3.weight"].shape[0]

        self.model = SholoGutiNet(
            input_features=inputFeatures,
            hidden_dim=hiddenDim,
            num_residual_blocks=numResidualBlocks,
            action_space=actionSpace,
        ).to(self.device)
        self.input_features = inputFeatures

        self.model.load_state_dict(stateDict)
        self.model.eval()

    def _encode_state_features(self, state: kribu.boardState) -> np.ndarray:
        """Build one model input vector from a full board state."""
        me_bits = np.zeros(37, dtype=np.float32)
        for i in range(37):
            me_bits[i] = (state.me >> i) & 1

        opp_bits = np.zeros(37, dtype=np.float32)
        for i in range(37):
            opp_bits[i] = (state.opp >> i) & 1

        cap_val = state.activeCaptureIdx + 1
        cap_bits = np.zeros(6, dtype=np.float32)
        for i in range(6):
            cap_bits[i] = (cap_val >> i) & 1

        if self.input_features >= 83:
            historyCount, currentRepeatCount, currentFlipRepeatCount = kribu.repetition_features(state)
            extra_features = np.array(
                [
                    historyCount / MAX_REPETITION_HISTORY,
                    currentRepeatCount / MAX_REPETITION_HISTORY,
                    currentFlipRepeatCount / MAX_REPETITION_HISTORY,
                ],
                dtype=np.float32,
            )
            return np.concatenate([me_bits, opp_bits, cap_bits, extra_features])

        return np.concatenate([me_bits, opp_bits, cap_bits])

    def _infer_batch(self, feature_rows: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run batched model inference for one or more encoded states."""
        features = torch.from_numpy(np.stack(feature_rows, axis=0)).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(features)
        return policy_logits, value

    def _successor_value_for_move(self, state: kribu.boardState, move_id: int, predicted_value: float) -> float:
        """Convert successor-side value into current-player value for one candidate move."""
        next_state = kribu.apply_move(state, move_id)
        if next_state.activeCaptureIdx != -1:
            return predicted_value
        return 1.0 - predicted_value

    def _select_move_with_value_guidance(
        self,
        state: kribu.boardState,
        valid_moves: list[int],
        current_policy_logits: torch.Tensor,
    ) -> int:
        """Choose among valid moves using one-ply value guidance plus policy prior."""
        successor_states = []
        for move_id in valid_moves:
            next_state = kribu.apply_move(state, move_id)
            if next_state.activeCaptureIdx == -1:
                next_state = kribu.flip_board(next_state)
            successor_states.append(next_state)

        _, successor_values = self._infer_batch(
            [self._encode_state_features(next_state) for next_state in successor_states]
        )
        move_indices = torch.tensor(valid_moves, dtype=torch.long, device=self.device)
        valid_logits = current_policy_logits[move_indices]
        valid_log_probs = torch.log_softmax(valid_logits, dim=0)

        best_move = valid_moves[0]
        best_score = float("-inf")
        for idx, move_id in enumerate(valid_moves):
            move_value = self._successor_value_for_move(state, move_id, float(successor_values[idx].item()))
            score = move_value + POLICY_LOGIT_WEIGHT * float(valid_log_probs[idx].item())
            if score > best_score:
                best_score = score
                best_move = move_id
        return best_move

    def get_move(
        self,
        me_mask: int,
        opp_mask: int,
        active_capture_idx: int,
        valid_moves: list[int] | None = None,
        state: kribu.boardState | None = None,
    ) -> tuple[int, float]:
        """
        Runs inference on a given board state.

        Args:
            me_mask: The bitmask representing the active player's pieces.
            opp_mask: The bitmask representing the opponent's pieces.
            active_capture_idx: The node index of a piece currently in a capture sequence (-1 if none).
            valid_moves: Optional legal move IDs. When provided, policy selection is masked to these moves.
            state: Optional full board state, used to derive repetition-aware input features when the model expects them.

        Returns:
            A tuple containing (best_move_index, win_probability).
        """
        if state is None:
            state = kribu.boardState()
            state.me = int(me_mask)
            state.opp = int(opp_mask)
            state.activeCaptureIdx = int(active_capture_idx)

        features = self._encode_state_features(state)
        policy_logits, value = self._infer_batch([features])
        current_policy_logits = policy_logits[0]

        if valid_moves is not None:
            if not valid_moves:
                raise ValueError("valid_moves must not be empty when provided")
            if len(valid_moves) == 1:
                best_move_idx = int(valid_moves[0])
            elif self.use_value_guidance:
                best_move_idx = self._select_move_with_value_guidance(state, valid_moves, current_policy_logits)
            else:
                move_indices = torch.tensor(valid_moves, dtype=torch.long, device=self.device)
                valid_logits = current_policy_logits[move_indices]
                best_move_idx = int(valid_moves[int(torch.argmax(valid_logits, dim=0).item())])
        else:
            best_move_idx = int(torch.argmax(current_policy_logits, dim=-1).item())

        win_prob = float(value.item())

        return best_move_idx, win_prob
