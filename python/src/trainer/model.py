import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out += identity
        out = self.act(out)
        return out


class SholoGutiNet(nn.Module):
    def __init__(self, input_features: int, hidden_dim: int, num_residual_blocks: int, action_space: int):
        super().__init__()

        # Initial projection
        self.input_proj = nn.Sequential(nn.Linear(input_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        # Residual backbone
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)])

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_space),
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch_size, input_features)
        Returns:
            policy_logits: Tensor of shape (batch_size, action_space)
            value: Tensor of shape (batch_size, 1) bounded [0, 1]
        """
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)

        policy_logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1)  # shape: (batch_size,)

        return policy_logits, value
