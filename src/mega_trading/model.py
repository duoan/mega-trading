"""Llama-style decoder-only Transformer for order-flow event modeling."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mega_trading.kernels.triton_attention import triton_attention as _triton_attention
from mega_trading.kernels.triton_ops import (
    triton_apply_rope as _triton_apply_rope,
    triton_fused_linear_cross_entropy as _triton_fused_linear_cross_entropy,
    triton_rms_norm as _triton_rms_norm,
    triton_swiglu_gate as _triton_swiglu_gate,
)


class TradingModel(nn.Module):
    """Small Llama-style causal Transformer trained with next-token cross entropy."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        hidden_dim: int = 128,
        layers: int = 4,
        attention_heads: int = 4,
        kv_heads: int | None = None,
        intermediate_dim: int | None = None,
        dropout: float = 0.1,
        rope_theta: float = 500_000.0,
        norm_eps: float = 1e-5,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        kv_heads = kv_heads or attention_heads
        if attention_heads % kv_heads != 0:
            raise ValueError("attention_heads must be divisible by kv_heads")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.kv_heads = kv_heads
        # Operator backend is captured here so the loss head can route through the fused
        # Triton kernel without re-threading config into every call site.
        self.operator_backend = attention_backend
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                LlamaDecoderBlock(
                    hidden_dim=hidden_dim,
                    block_size=block_size,
                    attention_heads=attention_heads,
                    kv_heads=kv_heads,
                    intermediate_dim=intermediate_dim,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    norm_eps=norm_eps,
                    attention_backend=attention_backend,
                )
                for _ in range(layers)
            ]
        )
        self.norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.apply(_init_weights)
        self.output.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        _batch_size, sequence_length = input_ids.shape
        if sequence_length > self.block_size:
            raise ValueError(f"sequence length {sequence_length} exceeds block_size {self.block_size}")
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden))

    def forward_with_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        return_logits: bool,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the network and compute next-token cross-entropy in one call.

        When `operator_backend == "triton"` and logits are not requested, the loss is
        computed by the fused linear-cross-entropy kernel so the `[batch, time, vocab]`
        logits tensor is never materialized. On metric steps callers pass
        `return_logits=True` to get logits back for top-k accuracy reporting; that
        path falls back to the standard linear projection plus `F.cross_entropy`.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids shape")
        _batch_size, sequence_length = input_ids.shape
        if sequence_length > self.block_size:
            raise ValueError(f"sequence length {sequence_length} exceeds block_size {self.block_size}")

        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        normalized = self.norm(hidden)

        if self.operator_backend == "triton" and not return_logits:
            flat_hidden = normalized.reshape(-1, normalized.shape[-1])
            flat_labels = labels.reshape(-1)
            loss = _triton_fused_linear_cross_entropy(
                flat_hidden,
                self.output.weight,
                flat_labels,
                ignore_index=ignore_index,
            )
            return loss, None

        logits = self.output(normalized)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )
        # Drop logits when callers do not need them so the trainer can rely on
        # `None` to skip top-k accuracy work without an extra branch.
        return loss, (logits if return_logits else None)

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


class LlamaDecoderBlock(nn.Module):
    """Pre-norm decoder block with RoPE attention and SwiGLU MLP."""

    def __init__(
        self,
        hidden_dim: int,
        block_size: int,
        attention_heads: int,
        kv_heads: int,
        intermediate_dim: int | None,
        dropout: float,
        rope_theta: float,
        norm_eps: float,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.attention = LlamaAttention(
            hidden_dim=hidden_dim,
            block_size=block_size,
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            dropout=dropout,
            rope_theta=rope_theta,
            attention_backend=attention_backend,
        )
        self.ffn_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.feed_forward = SwiGLU(
            hidden_dim,
            intermediate_dim or _llama_intermediate_dim(hidden_dim),
            dropout,
            operator_backend=attention_backend,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class LlamaAttention(nn.Module):
    """Causal self-attention with rotary positions and optional grouped-query attention."""

    def __init__(
        self,
        hidden_dim: int,
        block_size: int,
        attention_heads: int,
        kv_heads: int,
        dropout: float,
        rope_theta: float,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if attention_backend not in {"auto", "flash", "efficient", "math", "triton"}:
            raise ValueError("attention_backend must be one of: auto, flash, efficient, math, triton")
        self.attention_heads = attention_heads
        self.kv_heads = kv_heads
        self.head_dim = hidden_dim // attention_heads
        self.repeats = attention_heads // kv_heads
        self.q_proj = nn.Linear(hidden_dim, attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, block_size, theta=rope_theta, operator_backend=attention_backend)
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.attention_backend = attention_backend

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden.shape
        query = self.q_proj(hidden).view(batch_size, sequence_length, self.attention_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden).view(batch_size, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden).view(batch_size, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)
        query, key = self.rope(query, key)
        dropout_p = self.dropout if self.training else 0.0
        if self.attention_backend == "triton" and dropout_p == 0.0:
            attended = _triton_attention(query, key, value, repeats=self.repeats)
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=True,
                enable_gqa=self.repeats > 1,
            )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_dim)
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float, operator_backend: str = "auto") -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.operator_backend = operator_backend

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = self.w1(hidden)
        up = self.w3(hidden)
        if self.operator_backend == "triton":
            gated = _triton_swiglu_gate(gate, up)
        else:
            gated = F.silu(gate) * up
        return self.w2(self.dropout(gated))


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5, operator_backend: str = "auto") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
        self.operator_backend = operator_backend

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.operator_backend == "triton":
            return _triton_rms_norm(hidden, self.weight, eps=self.eps)
        return F.rms_norm(hidden, (hidden.shape[-1],), self.weight, eps=self.eps)


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE tables so training avoids rebuilding trig tensors per block."""

    def __init__(self, head_dim: int, max_sequence_length: int, theta: float, operator_backend: str = "auto") -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.operator_backend = operator_backend
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos().view(1, 1, max_sequence_length, head_dim // 2), persistent=False)
        self.register_buffer("sin", freqs.sin().view(1, 1, max_sequence_length, head_dim // 2), persistent=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        if sequence_length > self.cos.shape[-2]:
            raise ValueError(f"sequence length {sequence_length} exceeds RoPE cache length {self.cos.shape[-2]}")
        cos = self.cos[:, :, :sequence_length].to(dtype=query.dtype)
        sin = self.sin[:, :, :sequence_length].to(dtype=query.dtype)
        if self.operator_backend == "triton":
            return _triton_apply_rope(query, key, cos, sin)
        return _rotate(query, cos, sin), _rotate(key, cos, sin)


def _apply_rope(query: torch.Tensor, key: torch.Tensor, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    sequence_length = query.shape[-2]
    device = query.device
    dtype = query.dtype
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype).view(1, 1, sequence_length, head_dim // 2)
    sin = freqs.sin().to(dtype).view(1, 1, sequence_length, head_dim // 2)
    return _rotate(query, cos, sin), _rotate(key, cos, sin)


def _rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(start_dim=-2)


def _init_weights(module: nn.Module) -> None:
    """Use small Transformer-style weights so tied logits start near log-vocab loss."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def _llama_intermediate_dim(hidden_dim: int) -> int:
    # Llama-family SwiGLU uses roughly 8/3 d_model, commonly rounded to a hardware-friendly multiple.
    return int(math.ceil((8 * hidden_dim / 3) / 16) * 16)
