import torch.nn as nn

from src.attention import CausalSelfAttention
from src.mlp import MLP


class TransformerBlock(nn.Module):
    """Single Transformer decoder block.

    Structure:

        x
        │
        ├── LayerNorm
        │
        ├── Causal Self-Attention
        │
        └── Residual connection
        │
        ├── LayerNorm
        │
        ├── Feed-Forward Network
        │
        └── Residual connection

    Input:
        (batch, time, n_embd)

    Output:
        (batch, time, n_embd)
    """

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN + causal self-attention + residual connection
        x = x + self.attn(self.ln_1(x))

        # Pre-LN + feed-forward network + residual connection
        x = x + self.mlp(self.ln_2(x))

        return x