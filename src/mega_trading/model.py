"""Decoder-only Transformer baseline for market event modeling."""

from __future__ import annotations

import math

import torch
from torch import nn


class TradingModel(nn.Module):
    """Small causal Transformer trained with next-token cross entropy."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        hidden_dim: int = 128,
        layers: int = 4,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(block_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.block_size:
            raise ValueError(f"sequence length {sequence_length} exceeds block_size {self.block_size}")
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = hidden * math.sqrt(self.token_embedding.embedding_dim)
        mask = _causal_mask(sequence_length, input_ids.device)
        hidden = self.blocks(hidden, mask=mask)
        return self.output(self.norm(hidden))

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, top_k: int = 16) -> torch.Tensor:
        self.eval()
        tokens = input_ids
        for _ in range(max_new_tokens):
            context = tokens[:, -self.block_size :]
            logits = self(context)[:, -1, :]
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens


def _causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device), diagonal=1)
