"""Neural Network Player wrapper for Sholo Guti in the Arena."""

import os
import numpy as np
import torch
import kribu

from trainer.config import config
from trainer.model import SholoGutiNet


MAX_REPETITION_HISTORY = 64.0
POLICY_LOGIT_WEIGHT = 0.15
ROOT_SEARCH_DEPTH = 2
ROOT_CANDIDATE_LIMIT = 8
REPLY_CANDIDATE_LIMIT = 6
DRAW_CONTEMPT_WEIGHT = 0.08
DRAW_HISTORY_PENALTY = 0.06
LEAF_DRAW_BLEND = 0.20


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

    def _legal_moves(self, state: kribu.boardState) -> list[int]:
        """Return legal move ids for one state."""
        return list(kribu.all_possible_moves(state))

    def _apply_move(self, state: kribu.boardState, move_id: int) -> kribu.boardState:
        """Return the successor state after applying one move."""
        return kribu.apply_move(state, move_id)

    def _flip_board(self, state: kribu.boardState) -> kribu.boardState:
        """Return the board with perspective swapped."""
        return kribu.flip_board(state)

    def _piece_balance(self, state: kribu.boardState) -> float:
        """Return a normalized material edge from the current player's perspective."""
        return max(-1.0, min(1.0, (int(state.me).bit_count() - int(state.opp).bit_count()) / 16.0))

    def _draw_score(self, state: kribu.boardState) -> float:
        """Return a draw value with mild contempt for sterile positions when ahead."""
        history_penalty = DRAW_HISTORY_PENALTY * min(1.0, float(state.historyCount) / MAX_REPETITION_HISTORY)
        score = 0.5 + DRAW_CONTEMPT_WEIGHT * self._piece_balance(state) - history_penalty
        return max(0.0, min(1.0, score))

    def _terminal_score(self, state: kribu.boardState) -> float | None:
        """Return a terminal score from the current player's perspective, or None if non-terminal."""
        if int(state.opp) == 0:
            return 1.0
        if int(state.me) == 0:
            return 0.0
        if int(state.historyCount) >= int(MAX_REPETITION_HISTORY):
            return self._draw_score(state)

        legal_moves = self._legal_moves(state)
        if not legal_moves:
            return 0.0
        if not self._legal_moves(self._flip_board(state)):
            return 1.0
        return None

    def _leaf_score(self, state: kribu.boardState, predicted_value: float) -> float:
        """Blend leaf value with draw contempt so neutral loop states are less attractive."""
        blended = (1.0 - LEAF_DRAW_BLEND) * predicted_value + LEAF_DRAW_BLEND * self._draw_score(state)
        return max(0.0, min(1.0, blended))

    def _candidate_moves(
        self,
        valid_moves: list[int],
        policy_logits: torch.Tensor,
        limit: int,
    ) -> list[int]:
        """Return the highest-priority legal moves according to the policy head."""
        if len(valid_moves) <= limit:
            return list(valid_moves)

        move_indices = torch.tensor(valid_moves, dtype=torch.long, device=self.device)
        valid_logits = policy_logits[move_indices]
        top_count = min(limit, len(valid_moves))
        top_indices = torch.topk(valid_logits, k=top_count, dim=0).indices.tolist()
        return [valid_moves[int(index)] for index in top_indices]

    def _search_score(self, state: kribu.boardState, depth: int) -> float:
        """Return a shallow search score from the current player's perspective."""
        terminal_score = self._terminal_score(state)
        if terminal_score is not None:
            return terminal_score

        policy_logits, value = self._infer_batch([self._encode_state_features(state)])
        predicted_value = self._leaf_score(state, float(value[0].item()))
        if depth <= 0:
            return predicted_value

        valid_moves = self._legal_moves(state)
        candidate_limit = ROOT_CANDIDATE_LIMIT if depth >= ROOT_SEARCH_DEPTH else REPLY_CANDIDATE_LIMIT
        candidate_moves = self._candidate_moves(valid_moves, policy_logits[0], candidate_limit)
        move_indices = torch.tensor(candidate_moves, dtype=torch.long, device=self.device)
        valid_logits = policy_logits[0][move_indices]
        valid_log_probs = torch.log_softmax(valid_logits, dim=0)

        best_score = float("-inf")
        for idx, move_id in enumerate(candidate_moves):
            next_state = self._apply_move(state, move_id)
            turn_flips = next_state.activeCaptureIdx == -1
            search_state = self._flip_board(next_state) if turn_flips else next_state
            child_depth = depth - 1 if turn_flips else depth
            child_score = self._search_score(search_state, child_depth)
            move_score = 1.0 - child_score if turn_flips else child_score
            score = move_score + POLICY_LOGIT_WEIGHT * float(valid_log_probs[idx].item())
            if score > best_score:
                best_score = score

        if best_score == float("-inf"):
            return predicted_value
        return max(0.0, min(1.0, best_score))

    def _select_move_with_value_guidance(
        self,
        state: kribu.boardState,
        valid_moves: list[int],
        current_policy_logits: torch.Tensor,
    ) -> int:
        """Choose among valid moves using shallow search plus policy prior."""
        candidate_moves = self._candidate_moves(valid_moves, current_policy_logits, ROOT_CANDIDATE_LIMIT)
        move_indices = torch.tensor(candidate_moves, dtype=torch.long, device=self.device)
        valid_logits = current_policy_logits[move_indices]
        valid_log_probs = torch.log_softmax(valid_logits, dim=0)

        best_move = candidate_moves[0]
        best_score = float("-inf")
        for idx, move_id in enumerate(candidate_moves):
            next_state = self._apply_move(state, move_id)
            turn_flips = next_state.activeCaptureIdx == -1
            search_state = self._flip_board(next_state) if turn_flips else next_state
            child_depth = ROOT_SEARCH_DEPTH - 1 if turn_flips else ROOT_SEARCH_DEPTH
            child_score = self._search_score(search_state, child_depth)
            move_score = 1.0 - child_score if turn_flips else child_score
            score = move_score + POLICY_LOGIT_WEIGHT * float(valid_log_probs[idx].item())
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
