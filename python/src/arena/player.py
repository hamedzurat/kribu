"""Neural Network Player wrapper for Sholo Guti in the Arena."""

import os
import torch
import numpy as np

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
        self.model = SholoGutiNet(
            input_features=config.input_features,
            hidden_dim=config.hidden_dim,
            num_residual_blocks=config.num_residual_blocks,
            action_space=config.action_space,
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def get_move(self, me_mask: int, opp_mask: int, active_capture_idx: int) -> tuple[int, float]:
        """
        Runs inference on a given board state.

        Args:
            me_mask: The bitmask representing the active player's pieces.
            opp_mask: The bitmask representing the opponent's pieces.
            active_capture_idx: The node index of a piece currently in a capture sequence (-1 if none).

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

        X = np.concatenate([me_bits, opp_bits, cap_bits])
        X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            policy_logits, value = self.model(X_tensor)

        best_move_idx = int(torch.argmax(policy_logits, dim=-1).item())
        win_prob = float(value.item())

        return best_move_idx, win_prob
