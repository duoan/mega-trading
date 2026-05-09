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


class MarketDataEncoder(nn.Module):
    """Encode market_data return/level windows into temporal market tokens."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        self.temporal_encoder = _transformer_block(hidden_dim, attention_heads)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, market_data: torch.Tensor) -> torch.Tensor:
        return self.norm(self.temporal_encoder(self.input_proj(market_data)))


class FeatureSequenceEncoder(nn.Module):
    """Encode a per-modality numeric feature sequence into tokens."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.event_encoder = _transformer_block(hidden_dim, attention_heads)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(values.unsqueeze(-1))
        return self.norm(self.event_encoder(tokens))


class NewsEncoder(FeatureSequenceEncoder):
    """Encode financial-news embedding features."""


class SecFilingEncoder(FeatureSequenceEncoder):
    """Encode SEC filing and company-fact features."""


class EarningsEncoder(FeatureSequenceEncoder):
    """Encode earnings-call and earnings-event features."""


class MacroEncoder(FeatureSequenceEncoder):
    """Encode macro and regime features."""


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


class ForwardReturnDecoder(nn.Module):
    """Predict forward-return bucket labels from shared market memory."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.return_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=attention_heads, batch_first=True)
        self.output_ffn = SwiGLU(hidden_dim, hidden_dim * 4)
        self.norm = nn.LayerNorm(hidden_dim)
        self.return_bucket_head = nn.Linear(hidden_dim, len(RETURN_LABELS))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch_size = memory.shape[0]
        query = self.return_query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attention(query, memory, memory)
        decoded = self.norm(query + attended)
        decoded = self.norm(decoded + self.output_ffn(decoded))
        return self.return_bucket_head(decoded.squeeze(1))


class RiskDecoder(nn.Module):
    """Predict risk bucket labels from shared market memory."""

    def __init__(self, hidden_dim: int, attention_heads: int) -> None:
        super().__init__()
        self.risk_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=attention_heads, batch_first=True)
        self.output_ffn = SwiGLU(hidden_dim, hidden_dim * 4)
        self.norm = nn.LayerNorm(hidden_dim)
        self.risk_bucket_head = nn.Linear(hidden_dim, len(RISK_LABELS))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch_size = memory.shape[0]
        query = self.risk_query.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attention(query, memory, memory)
        decoded = self.norm(query + attended)
        decoded = self.norm(decoded + self.output_ffn(decoded))
        return self.risk_bucket_head(decoded.squeeze(1))


class TradingFoundationModel(nn.Module):
    """Modular market foundation model with gated fusion and shared memory."""

    def __init__(
        self,
        market_window_size: int,
        news_size: int,
        sec_filing_size: int,
        earnings_size: int,
        macro_size: int,
        hidden_dim: int,
        attention_heads: int = 4,
        use_market_data: bool = True,
        use_news: bool = True,
        use_sec_filings: bool = True,
        use_earnings: bool = True,
        use_macro: bool = True,
    ) -> None:
        super().__init__()
        if not (use_market_data or use_news or use_sec_filings or use_earnings or use_macro):
            raise ValueError("at least one modality must be enabled")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.market_data_encoder = MarketDataEncoder(hidden_dim, attention_heads)
        self.news_encoder = NewsEncoder(hidden_dim, attention_heads)
        self.sec_filing_encoder = SecFilingEncoder(hidden_dim, attention_heads)
        self.earnings_encoder = EarningsEncoder(hidden_dim, attention_heads)
        self.macro_encoder = MacroEncoder(hidden_dim, attention_heads)
        self.query_seed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.fusion = GatedCrossAttentionFusion(hidden_dim, attention_heads)
        self.market_memory = SharedMarketMemory(hidden_dim, attention_heads)
        self.forward_return_decoder = ForwardReturnDecoder(hidden_dim, attention_heads)
        self.risk_decoder = RiskDecoder(hidden_dim, attention_heads)
        self.market_window_size = market_window_size
        self.news_size = news_size
        self.sec_filing_size = sec_filing_size
        self.earnings_size = earnings_size
        self.macro_size = macro_size
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.use_market_data = use_market_data
        self.use_news = use_news
        self.use_sec_filings = use_sec_filings
        self.use_earnings = use_earnings
        self.use_macro = use_macro

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch["market_data"].shape[0]
        query_tokens = self.market_data_encoder(batch["market_data"]) if self.use_market_data else self._learned_query(batch_size, batch["market_data"])
        context_tokens = self._context_tokens(batch)
        fused_query = self.fusion(query_tokens, context_tokens)
        all_tokens = torch.cat([fused_query] + context_tokens, dim=1) if context_tokens else fused_query
        memory = self.market_memory(all_tokens)
        return self.forward_return_decoder(memory), self.risk_decoder(memory)

    def _context_tokens(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        tokens: list[torch.Tensor] = []
        if self.use_news:
            tokens.append(self.news_encoder(batch["news"]))
        if self.use_sec_filings:
            tokens.append(self.sec_filing_encoder(batch["sec_filings"]))
        if self.use_earnings:
            tokens.append(self.earnings_encoder(batch["earnings"]))
        if self.use_macro:
            tokens.append(self.macro_encoder(batch["macro"]))
        return tokens

    def _learned_query(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        if self.use_news:
            length = self.news_size
        elif self.use_sec_filings:
            length = self.sec_filing_size
        elif self.use_earnings:
            length = self.earnings_size
        elif self.use_macro:
            length = self.macro_size
        else:
            length = 1
        query = self.query_seed.to(dtype=reference.dtype, device=reference.device)
        return query.unsqueeze(0).expand(batch_size, length, -1)
