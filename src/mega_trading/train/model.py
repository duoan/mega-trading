"""TradingFoundationModel neural architecture."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from mega_trading.train.config import RETURN_LABELS, RISK_LABELS


def _transformer_block(hidden_dim: int, attention_heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=attention_heads,
        dim_feedforward=hidden_dim * 4,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)


class SwiGLU(nn.Module):
    """Gated feed-forward block used in modern transformer FFNs."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w3 = nn.Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class PriceEncoder(nn.Module):
    """Encode price return/level windows into temporal market tokens."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        self.temporal_encoder = _transformer_block(hidden_dim, attention_heads)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, price: torch.Tensor) -> torch.Tensor:
        return self.norm(self.temporal_encoder(self.input_proj(price)))


class MarketEventEncoder(nn.Module):
    """Encode scalar event streams such as fundamentals or evidence token ids."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.event_encoder = _transformer_block(hidden_dim, attention_heads)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(values.unsqueeze(-1))
        return self.norm(self.event_encoder(tokens))


class GatedCrossAttentionFusion(nn.Module):
    """Fuse context modalities into query tokens with a learned gate."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=attention_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.output_ffn = SwiGLU(hidden_dim, hidden_dim * 4)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_tokens: torch.Tensor, context_tokens: list[torch.Tensor]) -> torch.Tensor:
        if not context_tokens:
            return self.norm(query_tokens + self.output_ffn(query_tokens))
        context = torch.cat(context_tokens, dim=1)
        attended, _ = self.cross_attention(query_tokens, context, context)
        gate = self.gate(torch.cat([query_tokens, attended], dim=-1))
        fused_query = query_tokens + gate * attended
        return self.norm(fused_query + self.output_ffn(fused_query))


class SharedMarketMemory(nn.Module):
    """Compress fused modality tokens into shared market-memory tokens."""

    def __init__(self, hidden_dim: int, attention_heads: int, memory_tokens: int = 4) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.randn(memory_tokens, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=attention_heads, batch_first=True)
        self.output_ffn = SwiGLU(hidden_dim, hidden_dim * 4)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        memory_query = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attention(memory_query, tokens, tokens)
        memory = self.norm(memory_query + attended)
        return self.norm(memory + self.output_ffn(memory))


class TaskDecoder(nn.Module):
    """Decode a task-specific prediction from shared market memory."""

    def __init__(self, hidden_dim: int, attention_heads: int, output_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=attention_heads, batch_first=True)
        self.output_ffn = SwiGLU(hidden_dim, hidden_dim * 4)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch_size = memory.shape[0]
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attention(query, memory, memory)
        decoded = self.norm(query + attended)
        decoded = self.norm(decoded + self.output_ffn(decoded))
        return self.head(decoded.squeeze(1))


class TradingFoundationModel(nn.Module):
    """Modular market foundation model with gated fusion and shared memory."""

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
        self.price_encoder = PriceEncoder(hidden_dim, attention_heads)
        self.fundamental_encoder = MarketEventEncoder(hidden_dim, attention_heads)
        self.evidence_encoder = MarketEventEncoder(hidden_dim, attention_heads)
        self.query_seed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.fusion = GatedCrossAttentionFusion(hidden_dim, attention_heads)
        self.market_memory = SharedMarketMemory(hidden_dim, attention_heads)
        self.return_decoder = TaskDecoder(hidden_dim, attention_heads, len(RETURN_LABELS))
        self.risk_decoder = TaskDecoder(hidden_dim, attention_heads, len(RISK_LABELS))
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
        query_tokens = self.price_encoder(batch["price"]) if self.use_price else self._learned_query(batch_size, batch["price"])
        context_tokens = self._context_tokens(batch)
        fused_query = self.fusion(query_tokens, context_tokens)
        all_tokens = torch.cat([fused_query] + context_tokens, dim=1) if context_tokens else fused_query
        memory = self.market_memory(all_tokens)
        return self.return_decoder(memory), self.risk_decoder(memory)

    def _context_tokens(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        tokens: list[torch.Tensor] = []
        if self.use_fundamentals:
            tokens.append(self.fundamental_encoder(batch["fundamentals"]))
        if self.use_evidence:
            tokens.append(self.evidence_encoder(batch["evidence"]))
        return tokens

    def _learned_query(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        if self.use_fundamentals:
            length = self.fundamental_size
        elif self.use_evidence:
            length = self.evidence_size
        else:
            length = 1
        query = self.query_seed.to(dtype=reference.dtype, device=reference.device)
        return query.unsqueeze(0).expand(batch_size, length, -1)
