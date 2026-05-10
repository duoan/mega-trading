"""Triton causal attention kernels."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def triton_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, repeats: int) -> torch.Tensor:
    """Causal grouped-query attention using local Triton kernels for the server shape."""
    if query.device.type != "cuda":
        raise RuntimeError("triton attention backend requires CUDA tensors")
    if query.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise RuntimeError("triton attention supports float16, bfloat16, and float32 tensors")
    if key.shape[1] * repeats != query.shape[1]:
        raise ValueError("query heads must equal key/value heads multiplied by repeats")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query, key, and value must agree on batch, sequence, and head dimensions")
    return _TritonCausalAttention.apply(query, key, value, repeats)


class _TritonCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, repeats: int) -> torch.Tensor:
        _ensure_triton_descriptor_allocator()

        batch_size, heads, sequence_length, head_dim = query.shape
        output = torch.empty_like(query)
        logsumexp = torch.empty((batch_size, heads, sequence_length), device=query.device, dtype=torch.float32)
        block_m, block_n = _attention_tile_shape(sequence_length, head_dim)
        block_d = triton.next_power_of_2(head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        grid = (triton.cdiv(sequence_length, block_m), batch_size * heads)
        kernel = _triton_attention_forward_server_kernel if _uses_server_attention_shape(sequence_length, head_dim, repeats) else _triton_attention_forward_kernel
        kernel[grid](
            query,
            key,
            value,
            output,
            logsumexp,
            scale,
            heads,
            repeats,
            sequence_length,
            head_dim,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            logsumexp.stride(0),
            logsumexp.stride(1),
            logsumexp.stride(2),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(query, key, value, output, logsumexp)
        ctx.repeats = repeats
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        query, key, value, output, logsumexp = ctx.saved_tensors
        repeats = int(ctx.repeats)
        scale = float(ctx.scale)
        if _uses_server_attention_backward_shape(query, key, repeats):
            return _triton_attention_server_backward(query, key, value, output, logsumexp, grad_output, scale, repeats)

        key_for_query = key.repeat_interleave(repeats, dim=1) if repeats > 1 else key
        value_for_query = value.repeat_interleave(repeats, dim=1) if repeats > 1 else value

        query_f = query.float()
        key_f = key_for_query.float()
        value_f = value_for_query.float()
        grad_output_f = grad_output.float()
        scores = torch.matmul(query_f, key_f.transpose(-2, -1)) * scale
        sequence_length = query.shape[-2]
        causal_mask = torch.ones((sequence_length, sequence_length), device=query.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)

        grad_probs = torch.matmul(grad_output_f, value_f.transpose(-2, -1))
        grad_scores = probs * (grad_probs - (grad_probs * probs).sum(dim=-1, keepdim=True))
        grad_query = torch.matmul(grad_scores, key_f) * scale
        grad_key = torch.matmul(grad_scores.transpose(-2, -1), query_f) * scale
        grad_value = torch.matmul(probs.transpose(-2, -1), grad_output_f)

        if repeats > 1:
            batch_size, kv_heads, sequence_length, head_dim = key.shape
            grad_key = grad_key.view(batch_size, kv_heads, repeats, sequence_length, head_dim).sum(dim=2)
            grad_value = grad_value.view(batch_size, kv_heads, repeats, sequence_length, head_dim).sum(dim=2)
        return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype), None


def _attention_tile_shape(sequence_length: int, head_dim: int) -> tuple[int, int]:
    if _uses_server_attention_shape(sequence_length, head_dim, repeats=4):
        return 64, 64
    return 32, 64


def _uses_server_attention_shape(sequence_length: int, head_dim: int, repeats: int) -> bool:
    return sequence_length == 512 and head_dim == 64 and repeats == 4


def _uses_server_attention_backward_shape(query: torch.Tensor, key: torch.Tensor, repeats: int) -> bool:
    return (
        _uses_server_attention_shape(query.shape[2], query.shape[3], repeats)
        and query.shape[1] == 16
        and key.shape[1] == 4
        and query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
    )


def _triton_attention_server_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
    repeats: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    del repeats
    block_m, block_n = 64, 64
    dkv_block_n = 64
    block_d = 64
    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    batch_size, heads, sequence_length, head_dim = query.shape
    kv_heads = key.shape[1]

    _triton_attention_backward_dq_server_kernel[(triton.cdiv(sequence_length, block_m), batch_size * heads)](
        query,
        key,
        value,
        output,
        logsumexp,
        grad_output,
        grad_query,
        scale,
        heads,
        4,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        logsumexp.stride(0),
        logsumexp.stride(1),
        logsumexp.stride(2),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_query.stride(0),
        grad_query.stride(1),
        grad_query.stride(2),
        grad_query.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    _triton_attention_backward_dkv_server_kernel[(triton.cdiv(sequence_length, dkv_block_n), batch_size * kv_heads)](
        query,
        key,
        value,
        output,
        logsumexp,
        grad_output,
        grad_key,
        grad_value,
        scale,
        heads,
        4,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        logsumexp.stride(0),
        logsumexp.stride(1),
        logsumexp.stride(2),
        grad_output.stride(0),
        grad_output.stride(1),
        grad_output.stride(2),
        grad_output.stride(3),
        grad_key.stride(0),
        grad_key.stride(1),
        grad_key.stride(2),
        grad_key.stride(3),
        grad_value.stride(0),
        grad_value.stride(1),
        grad_value.stride(2),
        grad_value.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=dkv_block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return grad_query, grad_key, grad_value, None


def _ensure_triton_descriptor_allocator() -> None:
    if getattr(_ensure_triton_descriptor_allocator, "_installed", False):
        return

    def alloc_fn(size: int, alignment: int, stream: int | None):
        del alignment, stream
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    _ensure_triton_descriptor_allocator._installed = True


@triton.jit
def _triton_attention_forward_kernel(
    query,
    key,
    value,
    output,
    logsumexp,
    scale: tl.constexpr,
    heads: tl.constexpr,
    repeats: tl.constexpr,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    query_stride_b: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_t: tl.constexpr,
    query_stride_d: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_t: tl.constexpr,
    key_stride_d: tl.constexpr,
    value_stride_b: tl.constexpr,
    value_stride_h: tl.constexpr,
    value_stride_t: tl.constexpr,
    value_stride_d: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_t: tl.constexpr,
    output_stride_d: tl.constexpr,
    logsumexp_stride_b: tl.constexpr,
    logsumexp_stride_h: tl.constexpr,
    logsumexp_stride_t: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_index = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    batch_index = batch_head_index // heads
    query_head_index = batch_head_index - batch_index * heads
    kv_head_index = query_head_index // repeats

    row_offsets = block_index * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)
    key_desc = tl.make_tensor_descriptor(
        key + batch_index * key_stride_b + kv_head_index * key_stride_h,
        shape=[sequence_length, head_dim],
        strides=[key_stride_t, key_stride_d],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    value_desc = tl.make_tensor_descriptor(
        value + batch_index * value_stride_b + kv_head_index * value_stride_h,
        shape=[sequence_length, head_dim],
        strides=[value_stride_t, value_stride_d],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    query_values = tl.load(
        query
        + batch_index * query_stride_b
        + query_head_index * query_stride_h
        + row_offsets[:, None] * query_stride_t
        + dim_offsets[None, :] * query_stride_d,
        mask=(row_offsets[:, None] < sequence_length) & (dim_offsets[None, :] < head_dim),
        other=0.0,
    )
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_col in tl.range(0, (block_index + 1) * BLOCK_M, BLOCK_N):
        cols = start_col + col_offsets
        key_values = key_desc.load([start_col, 0])
        scores = tl.dot(query_values, tl.trans(key_values)) * scale
        causal_mask = (row_offsets[:, None] < sequence_length) & (cols[None, :] <= row_offsets[:, None])
        scores = tl.where(causal_mask & (cols[None, :] < sequence_length), scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(probs, axis=1)
        value_values = value_desc.load([start_col, 0])
        acc = acc * alpha[:, None] + tl.dot(probs.to(value_values.dtype), value_values)
        l_i = l_new
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        logsumexp + batch_index * logsumexp_stride_b + query_head_index * logsumexp_stride_h + row_offsets * logsumexp_stride_t,
        m_i + tl.log(l_i),
        mask=row_offsets < sequence_length,
    )
    tl.store(
        output
        + batch_index * output_stride_b
        + query_head_index * output_stride_h
        + row_offsets[:, None] * output_stride_t
        + dim_offsets[None, :] * output_stride_d,
        acc,
        mask=(row_offsets[:, None] < sequence_length) & (dim_offsets[None, :] < head_dim),
    )


@triton.jit
def _triton_attention_forward_server_kernel(
    query,
    key,
    value,
    output,
    logsumexp,
    scale: tl.constexpr,
    heads: tl.constexpr,
    repeats: tl.constexpr,
    sequence_length: tl.constexpr,
    head_dim: tl.constexpr,
    query_stride_b: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_t: tl.constexpr,
    query_stride_d: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_t: tl.constexpr,
    key_stride_d: tl.constexpr,
    value_stride_b: tl.constexpr,
    value_stride_h: tl.constexpr,
    value_stride_t: tl.constexpr,
    value_stride_d: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_t: tl.constexpr,
    output_stride_d: tl.constexpr,
    logsumexp_stride_b: tl.constexpr,
    logsumexp_stride_h: tl.constexpr,
    logsumexp_stride_t: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_index = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    batch_index = batch_head_index // heads
    query_head_index = batch_head_index - batch_index * heads
    kv_head_index = query_head_index // repeats

    row_offsets = block_index * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)
    query_desc = tl.make_tensor_descriptor(
        query + batch_index * query_stride_b + query_head_index * query_stride_h,
        shape=[sequence_length, head_dim],
        strides=[query_stride_t, query_stride_d],
        block_shape=[BLOCK_M, BLOCK_D],
    )
    query_values = query_desc.load([block_index * BLOCK_M, 0])
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_col in tl.range(0, block_index * BLOCK_M, BLOCK_N):
        key_values = tl.load(
            key
            + batch_index * key_stride_b
            + kv_head_index * key_stride_h
            + (start_col + col_offsets)[:, None] * key_stride_t
            + dim_offsets[None, :] * key_stride_d,
        )
        scores = tl.dot(query_values, tl.trans(key_values)) * scale
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        probs = tl.exp(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(probs, axis=1)
        value_values = tl.load(
            value
            + batch_index * value_stride_b
            + kv_head_index * value_stride_h
            + (start_col + col_offsets)[:, None] * value_stride_t
            + dim_offsets[None, :] * value_stride_d,
        )
        acc = acc * alpha[:, None] + tl.dot(probs.to(value_values.dtype), value_values)
        l_i = l_new
        m_i = m_new

    diagonal_col = block_index * BLOCK_M
    cols = diagonal_col + col_offsets
    key_values = tl.load(
        key
        + batch_index * key_stride_b
        + kv_head_index * key_stride_h
        + (diagonal_col + col_offsets)[:, None] * key_stride_t
        + dim_offsets[None, :] * key_stride_d,
    )
    scores = tl.dot(query_values, tl.trans(key_values)) * scale
    scores = tl.where(cols[None, :] <= row_offsets[:, None], scores, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    alpha = tl.exp(m_i - m_new)
    probs = tl.exp(scores - m_new[:, None])
    l_new = l_i * alpha + tl.sum(probs, axis=1)
    value_values = tl.load(
        value
        + batch_index * value_stride_b
        + kv_head_index * value_stride_h
        + (diagonal_col + col_offsets)[:, None] * value_stride_t
        + dim_offsets[None, :] * value_stride_d,
    )
    acc = acc * alpha[:, None] + tl.dot(probs.to(value_values.dtype), value_values)
    l_i = l_new
    m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        logsumexp + batch_index * logsumexp_stride_b + query_head_index * logsumexp_stride_h + row_offsets * logsumexp_stride_t,
        m_i + tl.log(l_i),
    )
    tl.store(
        output
        + batch_index * output_stride_b
        + query_head_index * output_stride_h
        + row_offsets[:, None] * output_stride_t
        + dim_offsets[None, :] * output_stride_d,
        acc,
    )


@triton.jit
def _triton_attention_backward_dq_server_kernel(
    query,
    key,
    value,
    output,
    logsumexp,
    grad_output,
    grad_query,
    scale: tl.constexpr,
    heads: tl.constexpr,
    repeats: tl.constexpr,
    query_stride_b: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_t: tl.constexpr,
    query_stride_d: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_t: tl.constexpr,
    key_stride_d: tl.constexpr,
    value_stride_b: tl.constexpr,
    value_stride_h: tl.constexpr,
    value_stride_t: tl.constexpr,
    value_stride_d: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_t: tl.constexpr,
    output_stride_d: tl.constexpr,
    logsumexp_stride_b: tl.constexpr,
    logsumexp_stride_h: tl.constexpr,
    logsumexp_stride_t: tl.constexpr,
    grad_output_stride_b: tl.constexpr,
    grad_output_stride_h: tl.constexpr,
    grad_output_stride_t: tl.constexpr,
    grad_output_stride_d: tl.constexpr,
    grad_query_stride_b: tl.constexpr,
    grad_query_stride_h: tl.constexpr,
    grad_query_stride_t: tl.constexpr,
    grad_query_stride_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_index = tl.program_id(0)
    batch_head_index = tl.program_id(1)
    batch_index = batch_head_index // heads
    query_head_index = batch_head_index - batch_index * heads
    kv_head_index = query_head_index // repeats

    row_offsets = block_index * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)

    query_values = tl.load(
        query
        + batch_index * query_stride_b
        + query_head_index * query_stride_h
        + row_offsets[:, None] * query_stride_t
        + dim_offsets[None, :] * query_stride_d
    )
    output_values = tl.load(
        output
        + batch_index * output_stride_b
        + query_head_index * output_stride_h
        + row_offsets[:, None] * output_stride_t
        + dim_offsets[None, :] * output_stride_d
    )
    grad_output_values = tl.load(
        grad_output
        + batch_index * grad_output_stride_b
        + query_head_index * grad_output_stride_h
        + row_offsets[:, None] * grad_output_stride_t
        + dim_offsets[None, :] * grad_output_stride_d
    )
    lse_values = tl.load(
        logsumexp
        + batch_index * logsumexp_stride_b
        + query_head_index * logsumexp_stride_h
        + row_offsets * logsumexp_stride_t
    )
    delta = tl.sum(output_values.to(tl.float32) * grad_output_values.to(tl.float32), axis=1)
    grad_query_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_col in tl.range(0, block_index * BLOCK_M, BLOCK_N):
        key_values = tl.load(
            key
            + batch_index * key_stride_b
            + kv_head_index * key_stride_h
            + (start_col + col_offsets)[:, None] * key_stride_t
            + dim_offsets[None, :] * key_stride_d
        )
        value_values = tl.load(
            value
            + batch_index * value_stride_b
            + kv_head_index * value_stride_h
            + (start_col + col_offsets)[:, None] * value_stride_t
            + dim_offsets[None, :] * value_stride_d
        )
        scores = tl.dot(query_values, tl.trans(key_values)) * scale
        probs = tl.exp(scores - lse_values[:, None])
        grad_probs = tl.dot(grad_output_values, tl.trans(value_values))
        grad_scores = probs * (grad_probs - delta[:, None])
        grad_query_acc += tl.dot(grad_scores.to(key_values.dtype), key_values)

    diagonal_col = block_index * BLOCK_M
    cols = diagonal_col + col_offsets
    key_values = tl.load(
        key
        + batch_index * key_stride_b
        + kv_head_index * key_stride_h
        + cols[:, None] * key_stride_t
        + dim_offsets[None, :] * key_stride_d
    )
    value_values = tl.load(
        value
        + batch_index * value_stride_b
        + kv_head_index * value_stride_h
        + cols[:, None] * value_stride_t
        + dim_offsets[None, :] * value_stride_d
    )
    scores = tl.dot(query_values, tl.trans(key_values)) * scale
    causal_mask = cols[None, :] <= row_offsets[:, None]
    probs = tl.where(causal_mask, tl.exp(scores - lse_values[:, None]), 0.0)
    grad_probs = tl.dot(grad_output_values, tl.trans(value_values))
    grad_scores = probs * (grad_probs - delta[:, None])
    grad_query_acc += tl.dot(grad_scores.to(key_values.dtype), key_values)

    tl.store(
        grad_query
        + batch_index * grad_query_stride_b
        + query_head_index * grad_query_stride_h
        + row_offsets[:, None] * grad_query_stride_t
        + dim_offsets[None, :] * grad_query_stride_d,
        grad_query_acc * scale,
    )


@triton.jit
def _triton_attention_backward_dkv_server_kernel(
    query,
    key,
    value,
    output,
    logsumexp,
    grad_output,
    grad_key,
    grad_value,
    scale: tl.constexpr,
    heads: tl.constexpr,
    repeats: tl.constexpr,
    query_stride_b: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_t: tl.constexpr,
    query_stride_d: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_t: tl.constexpr,
    key_stride_d: tl.constexpr,
    value_stride_b: tl.constexpr,
    value_stride_h: tl.constexpr,
    value_stride_t: tl.constexpr,
    value_stride_d: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_t: tl.constexpr,
    output_stride_d: tl.constexpr,
    logsumexp_stride_b: tl.constexpr,
    logsumexp_stride_h: tl.constexpr,
    logsumexp_stride_t: tl.constexpr,
    grad_output_stride_b: tl.constexpr,
    grad_output_stride_h: tl.constexpr,
    grad_output_stride_t: tl.constexpr,
    grad_output_stride_d: tl.constexpr,
    grad_key_stride_b: tl.constexpr,
    grad_key_stride_h: tl.constexpr,
    grad_key_stride_t: tl.constexpr,
    grad_key_stride_d: tl.constexpr,
    grad_value_stride_b: tl.constexpr,
    grad_value_stride_h: tl.constexpr,
    grad_value_stride_t: tl.constexpr,
    grad_value_stride_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    col_block_index = tl.program_id(0)
    batch_kv_head_index = tl.program_id(1)
    batch_index = batch_kv_head_index // (heads // repeats)
    kv_head_index = batch_kv_head_index - batch_index * (heads // repeats)

    row_offsets = tl.arange(0, BLOCK_M)
    col_offsets = col_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)

    key_values = tl.load(
        key
        + batch_index * key_stride_b
        + kv_head_index * key_stride_h
        + col_offsets[:, None] * key_stride_t
        + dim_offsets[None, :] * key_stride_d
    )
    value_values = tl.load(
        value
        + batch_index * value_stride_b
        + kv_head_index * value_stride_h
        + col_offsets[:, None] * value_stride_t
        + dim_offsets[None, :] * value_stride_d
    )
    grad_key_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    grad_value_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    for repeat_index in tl.static_range(0, 4):
        query_head_index = kv_head_index * repeats + repeat_index
        for start_row in tl.range(col_block_index * BLOCK_N, 512, BLOCK_M):
            rows = start_row + row_offsets
            query_values = tl.load(
                query
                + batch_index * query_stride_b
                + query_head_index * query_stride_h
                + rows[:, None] * query_stride_t
                + dim_offsets[None, :] * query_stride_d
            )
            output_values = tl.load(
                output
                + batch_index * output_stride_b
                + query_head_index * output_stride_h
                + rows[:, None] * output_stride_t
                + dim_offsets[None, :] * output_stride_d
            )
            grad_output_values = tl.load(
                grad_output
                + batch_index * grad_output_stride_b
                + query_head_index * grad_output_stride_h
                + rows[:, None] * grad_output_stride_t
                + dim_offsets[None, :] * grad_output_stride_d
            )
            lse_values = tl.load(
                logsumexp
                + batch_index * logsumexp_stride_b
                + query_head_index * logsumexp_stride_h
                + rows * logsumexp_stride_t
            )
            scores = tl.dot(query_values, tl.trans(key_values)) * scale
            causal_mask = col_offsets[None, :] <= rows[:, None]
            probs = tl.where(causal_mask, tl.exp(scores - lse_values[:, None]), 0.0)
            delta = tl.sum(output_values.to(tl.float32) * grad_output_values.to(tl.float32), axis=1)
            grad_probs = tl.dot(grad_output_values, tl.trans(value_values))
            grad_scores = probs * (grad_probs - delta[:, None])
            grad_value_acc += tl.dot(tl.trans(probs.to(grad_output_values.dtype)), grad_output_values)
            grad_key_acc += tl.dot(tl.trans(grad_scores.to(query_values.dtype)), query_values)

    tl.store(
        grad_key
        + batch_index * grad_key_stride_b
        + kv_head_index * grad_key_stride_h
        + col_offsets[:, None] * grad_key_stride_t
        + dim_offsets[None, :] * grad_key_stride_d,
        grad_key_acc * scale,
    )
    tl.store(
        grad_value
        + batch_index * grad_value_stride_b
        + kv_head_index * grad_value_stride_h
        + col_offsets[:, None] * grad_value_stride_t
        + dim_offsets[None, :] * grad_value_stride_d,
        grad_value_acc,
    )

