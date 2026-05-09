"""TradingFoundationModel neural architecture."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mega_trading.train.config import RETURN_LABELS, RISK_LABELS


class SwiGLU(nn.Module):
    """Gated feed-forward block used in modern transformer FFNs."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TradingFoundationModel(nn.Module):
    """Cross-attention model over price, fundamentals, and evidence streams."""

    def __init__(
        self,
        price_window_size: int,
        fundamental_size: int,
        evidence_size: int,
        hidden_dim: int,
        attention_heads: int = 4,
        use_price: bool = True,
        use_fundamentals: bool = True,
        use_evidence: bool = True,
    ) -> None:
        super().__init__()
        if not (use_price or use_fundamentals or use_evidence):
            raise ValueError("at least one modality must be enabled")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.price_proj = nn.Linear(2, hidden_dim)
        self.fundamental_proj = nn.Linear(1, hidden_dim)
        self.evidence_proj = nn.Linear(1, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=attention_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fusion_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_ffn = SwiGLU(hidden_dim, hidden_dim * 4)
        self.return_head = nn.Linear(hidden_dim, len(RETURN_LABELS))
        self.risk_head = nn.Linear(hidden_dim, len(RISK_LABELS))
        self.price_window_size = price_window_size
        self.fundamental_size = fundamental_size
        self.evidence_size = evidence_size
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.use_price = use_price
        self.use_fundamentals = use_fundamentals
        self.use_evidence = use_evidence

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch["price"].shape[0]
        device = batch["price"].device
        zero_repr = torch.zeros((batch_size, self.hidden_dim), dtype=batch["price"].dtype, device=device)

        fundamental_tokens = self.fundamental_proj(batch["fundamentals"].unsqueeze(-1)) if self.use_fundamentals else None
        evidence_tokens = self.evidence_proj(batch["evidence"].unsqueeze(-1)) if self.use_evidence else None
        context_parts = [tokens for tokens in (fundamental_tokens, evidence_tokens) if tokens is not None]

        if self.use_price:
            price_tokens = self.price_proj(batch["price"])
            if context_parts:
                context_tokens = torch.cat(context_parts, dim=1)
                attended_price, _ = self.cross_attention(price_tokens, context_tokens, context_tokens)
                price_repr = self.norm(price_tokens + attended_price).mean(dim=1)
            else:
                price_repr = price_tokens.mean(dim=1)
        else:
            price_repr = zero_repr

        fundamental_repr = fundamental_tokens.mean(dim=1) if fundamental_tokens is not None else zero_repr
        evidence_repr = evidence_tokens.mean(dim=1) if evidence_tokens is not None else zero_repr
        fused = self.fusion_proj(torch.cat([price_repr, fundamental_repr, evidence_repr], dim=1))
        fused = self.output_ffn(fused)
        return self.return_head(fused), self.risk_head(fused)
