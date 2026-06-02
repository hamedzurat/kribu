"""Neural Network Player wrapper for Sholo Guti in the Arena."""

import os
import numpy as np
import torch
import kribu

from trainer.config import config
from trainer.model import SholoGutiNet


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
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            stateDict = checkpoint["model"]
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
        # Decode me (uint64) to 37 bits
        me_bits = np.zeros(37, dtype=np.float32)
        for i in range(37):
            me_bits[i] = (me_mask >> i) & 1

        # Decode opp (uint64) to 37 bits
        opp_bits = np.zeros(37, dtype=np.float32)
        for i in range(37):
            opp_bits[i] = (opp_mask >> i) & 1

        # Decode active_capture_idx (-1 to 36) to 6 bits
        cap_val = active_capture_idx + 1
        cap_bits = np.zeros(6, dtype=np.float32)
        for i in range(6):
            cap_bits[i] = (cap_val >> i) & 1

        extra_features = np.zeros(3, dtype=np.float32)
        if self.input_features >= 83 and state is not None:
            historyCount, currentRepeatCount, currentFlipRepeatCount = kribu.repetition_features(state)
            extra_features[:] = np.array(
                [
                    historyCount / 24.0,
                    currentRepeatCount / 24.0,
                    currentFlipRepeatCount / 24.0,
                ],
                dtype=np.float32,
            )

        X = (
            np.concatenate([me_bits, opp_bits, cap_bits, extra_features])
            if self.input_features >= 83
            else np.concatenate([me_bits, opp_bits, cap_bits])
        )
        X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            policy_logits, value = self.model(X_tensor)

        if valid_moves is not None:
            if not valid_moves:
                raise ValueError("valid_moves must not be empty when provided")
            move_indices = torch.tensor(valid_moves, dtype=torch.long, device=self.device)
            valid_logits = policy_logits[0, move_indices]
            best_move_idx = int(move_indices[int(torch.argmax(valid_logits).item())].item())
        else:
            best_move_idx = int(torch.argmax(policy_logits, dim=-1).item())

        win_prob = float(value.item())

        return best_move_idx, win_prob
