import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer_block import TransformerBlock


class OurLLM(nn.Module):
    """Small decoder-only Transformer language model.

    Architecture:

        Token Embedding
              +
        Position Embedding
              ↓
        Transformer Blocks × n_layer
              ↓
        Final LayerNorm
              ↓
        LM Head
              ↓
        Logits

    Input:
        token IDs with shape (B, T)

    Output:
        logits with shape (B, T, vocab_size)
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # --------------------------------------------------------
        # Token embedding
        # --------------------------------------------------------
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # --------------------------------------------------------
        # Positional embedding
        # --------------------------------------------------------
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.embedding_dropout = nn.Dropout(config.dropout)

        # --------------------------------------------------------
        # Transformer blocks
        # --------------------------------------------------------
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        # --------------------------------------------------------
        # Final normalization
        # --------------------------------------------------------
        self.ln_f = nn.LayerNorm(config.n_embd)

        # --------------------------------------------------------
        # Language Model Head
        # --------------------------------------------------------
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # --------------------------------------------------------
        # Weight tying
        # --------------------------------------------------------
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # --------------------------------------------------------
        # Initialize weights
        # --------------------------------------------------------
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, idx, targets=None):
        """
        Args:
            idx:
                Input token IDs.
                Shape: (B, T)

            targets:
                Target token IDs.
                Shape: (B, T)

        Returns:
            logits:
                Shape: (B, T, vocab_size)

            loss:
                Cross-entropy loss if targets are provided,
                otherwise None.
        """

        B, T = idx.shape

        # Sequence cannot exceed configured context size.
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"block_size {self.config.block_size}"
            )

        # --------------------------------------------------------
        # Token positions
        # --------------------------------------------------------
        positions = torch.arange(
            0,
            T,
            device=idx.device
        )

        # --------------------------------------------------------
        # Token + positional embeddings
        # --------------------------------------------------------
        token_embeddings = self.token_embedding(idx)

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        x = self.embedding_dropout(x)

        # --------------------------------------------------------
        # Transformer blocks
        # --------------------------------------------------------
        for block in self.blocks:
            x = block(x)

        # --------------------------------------------------------
        # Final LayerNorm
        # --------------------------------------------------------
        x = self.ln_f(x)

        # --------------------------------------------------------
        # Convert hidden states → vocabulary logits
        # --------------------------------------------------------
        logits = self.lm_head(x)

        # --------------------------------------------------------
        # Language modeling loss
        # --------------------------------------------------------
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    def num_parameters(self):
        """Return total number of trainable parameters."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None
    ):
        """Generate new tokens autoregressively."""

        for _ in range(max_new_tokens):

            # Keep only the latest context window.
            idx_cond = idx[:, -self.config.block_size:]

            # Get model predictions.
            logits, _ = self(idx_cond)

            # Take predictions for the last position.
            logits = logits[:, -1, :]

            # Temperature controls randomness.
            logits = logits / temperature

            # Optional Top-K filtering.
            if top_k is not None:
                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")

            # Convert logits to probabilities.
            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Sample next token.
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append token.
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx