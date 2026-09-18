import torch
import torch.nn as nn


class MLP(nn.Module):
    """Feed-Forward Network used inside a Transformer block.

    Input:
        (batch, time, n_embd)

    Output:
        (batch, time, n_embd)
    """

    def __init__(self, config):
        super().__init__()

        # Transformer FFN expands the embedding dimension.
        # Example:
        # n_embd = 384
        # ffn_mult = 4
        # hidden_dim = 1536
        hidden_dim = config.n_embd * config.ffn_mult

        self.fc1 = nn.Linear(config.n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Expand:
        # (B, T, C) -> (B, T, hidden_dim)
        x = self.fc1(x)

        # Non-linear activation.
        x = torch.nn.functional.gelu(x)

        # Project back:
        # (B, T, hidden_dim) -> (B, T, C)
        x = self.fc2(x)

        return self.dropout(x)