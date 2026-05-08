"""MarketFusion neural architecture."""

from __future__ import annotations

import torch
from torch import nn

from marketfm.train.market_fusion.config import RETURN_LABELS, RISK_LABELS


class MarketFusionModel(nn.Module):
    """Cross-attention fusion model over price, fundamentals, and evidence streams."""

    def __init__(self, price_window_size: int, fundamental_size: int, evidence_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.price_proj = nn.Linear(2, hidden_dim)
        self.fundamental_proj = nn.Linear(1, hidden_dim)
        self.evidence_proj = nn.Linear(1, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.return_head = nn.Linear(hidden_dim, len(RETURN_LABELS))
        self.risk_head = nn.Linear(hidden_dim, len(RISK_LABELS))
        self.price_window_size = price_window_size
        self.fundamental_size = fundamental_size
        self.evidence_size = evidence_size

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        price_tokens = self.price_proj(batch["price"])
        fundamental_tokens = self.fundamental_proj(batch["fundamentals"].unsqueeze(-1))
        evidence_tokens = self.evidence_proj(batch["evidence"].unsqueeze(-1))
        context_tokens = torch.cat([fundamental_tokens, evidence_tokens], dim=1)
        attended_price, _ = self.cross_attention(price_tokens, context_tokens, context_tokens)
        price_repr = self.norm(price_tokens + attended_price).mean(dim=1)
        fundamental_repr = fundamental_tokens.mean(dim=1)
        evidence_repr = evidence_tokens.mean(dim=1)
        fused = self.fusion(torch.cat([price_repr, fundamental_repr, evidence_repr], dim=1))
        return self.return_head(fused), self.risk_head(fused)
