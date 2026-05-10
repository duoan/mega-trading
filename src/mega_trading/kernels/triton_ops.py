"""Triton kernels for non-attention transformer operators."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def triton_rms_norm(hidden: torch.Tensor, weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    """RMSNorm with a Triton forward and backward path for CUDA tensors."""
    if not _can_use_triton(hidden) or hidden.shape[-1] != weight.numel() or hidden.stride(-1) != 1:
        return F.rms_norm(hidden, (hidden.shape[-1],), weight, eps=eps)
    return _TritonRMSNorm.apply(hidden, weight, eps)


def triton_apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    if not _can_use_triton(query) or not _can_use_triton(key) or query.shape[-1] != key.shape[-1]:
        return _torch_rope(query, key, cos, sin)
    return _TritonRoPE.apply(query, key, cos, sin)


def triton_swiglu_gate(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute the SwiGLU activation gate, `silu(gate) * up`."""
    if not _can_use_triton(gate) or gate.shape != up.shape or gate.stride(-1) != 1 or up.stride(-1) != 1:
        return F.silu(gate) * up
    return _TritonSwiGLUGate.apply(gate, up)


def _can_use_triton(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda" and tensor.dtype in {torch.float16, torch.bfloat16, torch.float32}


def _torch_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _torch_rotate(query, cos, sin), _torch_rotate(key, cos, sin)


def _torch_rotate(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor | None = None) -> torch.Tensor:
    sin = cos if sin is None else sin
    cos = cos[:, :, : value.shape[-2]].to(dtype=value.dtype, device=value.device)
    sin = sin[:, :, : value.shape[-2]].to(dtype=value.dtype, device=value.device)
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(start_dim=-2)


class _TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        hidden_contiguous = hidden.contiguous()
        weight_contiguous = weight.contiguous()
        hidden_dim = hidden_contiguous.shape[-1]
        rows = hidden_contiguous.numel() // hidden_dim
        output = torch.empty_like(hidden_contiguous)
        rstd = torch.empty((rows,), device=hidden.device, dtype=torch.float32)
        block_d = triton.next_power_of_2(hidden_dim)
        _rms_norm_forward_kernel[(rows,)](
            hidden_contiguous,
            weight_contiguous,
            output,
            rstd,
            eps,
            hidden_dim,
            BLOCK_D=block_d,
            num_warps=8,
        )
        ctx.save_for_backward(hidden_contiguous, weight_contiguous, rstd)
        ctx.hidden_shape = hidden.shape
        ctx.hidden_dim = hidden_dim
        return output.view(hidden.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        hidden, weight, rstd = ctx.saved_tensors
        grad_output_contiguous = grad_output.contiguous()
        hidden_dim = int(ctx.hidden_dim)
        rows = hidden.numel() // hidden_dim
        block_d = triton.next_power_of_2(hidden_dim)
        grad_hidden = torch.empty_like(hidden)
        sm_count = torch.cuda.get_device_properties(hidden.device).multi_processor_count
        # A small multiple of SM count improves dX parallelism without making dW reduction too large.
        reduction_blocks = min(rows, sm_count * 2)
        partial_weight = torch.empty((reduction_blocks, hidden_dim), device=hidden.device, dtype=torch.float32)
        _rms_norm_backward_combined_kernel[(reduction_blocks,)](
            hidden,
            weight,
            rstd,
            grad_output_contiguous,
            grad_hidden,
            partial_weight,
            rows,
            hidden_dim,
            triton.cdiv(rows, reduction_blocks),
            BLOCK_D=block_d,
            num_warps=8,
        )
        grad_weight = partial_weight.sum(dim=0).to(weight.dtype)
        return grad_hidden.view(ctx.hidden_shape), grad_weight, None


class _TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_out = torch.empty_like(query)
        key_out = torch.empty_like(key)
        _launch_rope_kernel(query, key, cos, sin, query_out, key_out, negate_sin=False)
        ctx.save_for_backward(cos, sin)
        ctx.query_meta = (query.shape, query.stride())
        ctx.key_meta = (key.shape, key.stride())
        return query_out, key_out

    @staticmethod
    def backward(
        ctx,
        grad_query: torch.Tensor,
        grad_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        cos, sin = ctx.saved_tensors
        grad_query_out = torch.empty_like(grad_query)
        grad_key_out = torch.empty_like(grad_key)
        _launch_rope_kernel(grad_query, grad_key, cos, sin, grad_query_out, grad_key_out, negate_sin=True)
        return grad_query_out, grad_key_out, None, None


class _TritonSwiGLUGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate_contiguous = gate.contiguous()
        up_contiguous = up.contiguous()
        output = torch.empty_like(gate_contiguous)
        hidden_dim = gate_contiguous.shape[-1]
        rows = gate_contiguous.numel() // hidden_dim
        block_d = triton.next_power_of_2(hidden_dim)
        num_warps = min(max(block_d // 256, 1), 8)
        _swiglu_gate_forward_row_kernel[(rows,)](
            gate_contiguous,
            up_contiguous,
            output,
            hidden_dim,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        ctx.save_for_backward(gate_contiguous, up_contiguous)
        ctx.gate_shape = gate.shape
        ctx.hidden_dim = hidden_dim
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        grad_output_contiguous = grad_output.contiguous()
        hidden_dim = int(ctx.hidden_dim)
        rows = gate.numel() // hidden_dim
        block_d = triton.next_power_of_2(hidden_dim)
        num_warps = min(max(block_d // 256, 1), 8)
        _swiglu_gate_backward_row_kernel[(rows,)](
            gate,
            up,
            grad_output_contiguous,
            grad_gate,
            grad_up,
            hidden_dim,
            BLOCK_D=block_d,
            num_warps=num_warps,
        )
        return grad_gate.view(ctx.gate_shape), grad_up.view(ctx.gate_shape)


def _launch_rope_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    *,
    negate_sin: bool,
) -> None:
    batch_size, query_heads, sequence_length, head_dim = query.shape
    key_heads = key.shape[1]
    half_dim = head_dim // 2
    total_pairs = batch_size * (query_heads + key_heads) * sequence_length * half_dim
    _rope_kernel[(triton.cdiv(total_pairs, 256),)](
        query,
        key,
        cos,
        sin,
        query_out,
        key_out,
        batch_size,
        query_heads,
        key_heads,
        sequence_length,
        half_dim,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        query_out.stride(0),
        query_out.stride(1),
        query_out.stride(2),
        query_out.stride(3),
        key_out.stride(0),
        key_out.stride(1),
        key_out.stride(2),
        key_out.stride(3),
        cos.stride(2),
        cos.stride(3),
        sin.stride(2),
        sin.stride(3),
        NEGATE_SIN=negate_sin,
        BLOCK=256,
    )


@triton.jit
def _rms_norm_forward_kernel(hidden, weight, output, rstd, eps: tl.constexpr, hidden_dim: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < hidden_dim
    values = tl.load(hidden + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / hidden_dim
    inv_rms = tl.rsqrt(mean_square + eps)
    weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + row * hidden_dim + offsets, values * inv_rms * weights, mask=mask)
    tl.store(rstd + row, inv_rms)


@triton.jit
def _rms_norm_backward_hidden_kernel(
    hidden,
    weight,
    rstd,
    grad_output,
    grad_hidden,
    hidden_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < hidden_dim
    values = tl.load(hidden + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_values = tl.load(grad_output + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(rstd + row).to(tl.float32)
    weighted_grad_dot = tl.sum(grad_values * weights * values, axis=0)
    grad_hidden_values = grad_values * weights * inv_rms - values * (inv_rms * inv_rms * inv_rms / hidden_dim) * weighted_grad_dot
    tl.store(grad_hidden + row * hidden_dim + offsets, grad_hidden_values, mask=mask)


@triton.jit
def _rms_norm_backward_combined_kernel(
    hidden,
    weight,
    rstd,
    grad_output,
    grad_hidden,
    partial_weight,
    rows: tl.constexpr,
    hidden_dim: tl.constexpr,
    rows_per_program: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    row_start = program_id * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, rows)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < hidden_dim
    weights = tl.load(weight + offsets, mask=mask, other=0.0)
    grad_weight_acc = tl.zeros((BLOCK_D,), tl.float32)

    for row in range(row_start, row_end):
        values = tl.load(hidden + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_values = tl.load(grad_output + row * hidden_dim + offsets, mask=mask, other=0.0)
        inv_rms = tl.load(rstd + row).to(tl.float32)
        weighted_grad = (grad_values * weights).to(tl.float32)
        normed_values = values * inv_rms
        correction = tl.sum(weighted_grad * values, axis=0)
        grad_hidden_values = inv_rms * (weighted_grad - values * (inv_rms * inv_rms / hidden_dim) * correction)
        grad_weight_acc += grad_values.to(tl.float32) * normed_values
        tl.store(grad_hidden + row * hidden_dim + offsets, grad_hidden_values, mask=mask)

    tl.store(partial_weight + program_id * hidden_dim + offsets, grad_weight_acc, mask=mask)


@triton.jit
def _rms_norm_backward_weight_kernel(
    hidden,
    rstd,
    grad_output,
    partial_weight,
    rows: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_row = tl.program_id(0)
    row_offsets = block_row * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    mask = (row_offsets[:, None] < rows) & (dim_offsets[None, :] < hidden_dim)
    values = tl.load(hidden + row_offsets[:, None] * hidden_dim + dim_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
    grad_values = tl.load(grad_output + row_offsets[:, None] * hidden_dim + dim_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(rstd + row_offsets, mask=row_offsets < rows, other=0.0).to(tl.float32)
    partial = tl.sum(values * grad_values * inv_rms[:, None], axis=0)
    tl.store(partial_weight + block_row * hidden_dim + dim_offsets, partial, mask=dim_offsets < hidden_dim)


@triton.jit
def _rope_kernel(
    query,
    key,
    cos,
    sin,
    query_out,
    key_out,
    batch_size: tl.constexpr,
    query_heads: tl.constexpr,
    key_heads: tl.constexpr,
    sequence_length: tl.constexpr,
    half_dim: tl.constexpr,
    query_stride_b: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_t: tl.constexpr,
    query_stride_d: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_t: tl.constexpr,
    key_stride_d: tl.constexpr,
    query_out_stride_b: tl.constexpr,
    query_out_stride_h: tl.constexpr,
    query_out_stride_t: tl.constexpr,
    query_out_stride_d: tl.constexpr,
    key_out_stride_b: tl.constexpr,
    key_out_stride_h: tl.constexpr,
    key_out_stride_t: tl.constexpr,
    key_out_stride_d: tl.constexpr,
    cos_stride_t: tl.constexpr,
    cos_stride_d: tl.constexpr,
    sin_stride_t: tl.constexpr,
    sin_stride_d: tl.constexpr,
    NEGATE_SIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    query_pairs = query_heads * sequence_length * half_dim
    key_pairs = key_heads * sequence_length * half_dim
    total_query_pairs = batch_size * query_pairs
    total_pairs = total_query_pairs + batch_size * key_pairs
    mask = offsets < total_pairs
    is_query = offsets < total_query_pairs
    local = tl.where(is_query, offsets, offsets - total_query_pairs)
    heads = tl.where(is_query, query_heads, key_heads)
    dim_pair = local % half_dim
    token = (local // half_dim) % sequence_length
    head = (local // (half_dim * sequence_length)) % heads
    batch = local // (heads * sequence_length * half_dim)
    sine = tl.load(sin + token * sin_stride_t + dim_pair * sin_stride_d, mask=mask, other=0.0)
    sine = tl.where(NEGATE_SIN, -sine, sine)
    cosine = tl.load(cos + token * cos_stride_t + dim_pair * cos_stride_d, mask=mask, other=0.0)
    query_even_ptr = query + batch * query_stride_b + head * query_stride_h + token * query_stride_t + (dim_pair * 2) * query_stride_d
    query_odd_ptr = query_even_ptr + query_stride_d
    key_even_ptr = key + batch * key_stride_b + head * key_stride_h + token * key_stride_t + (dim_pair * 2) * key_stride_d
    key_odd_ptr = key_even_ptr + key_stride_d
    query_even = tl.load(query_even_ptr, mask=mask & is_query, other=0.0)
    query_odd = tl.load(query_odd_ptr, mask=mask & is_query, other=0.0)
    key_even = tl.load(key_even_ptr, mask=mask & ~is_query, other=0.0)
    key_odd = tl.load(key_odd_ptr, mask=mask & ~is_query, other=0.0)
    even = tl.where(is_query, query_even, key_even)
    odd = tl.where(is_query, query_odd, key_odd)
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    query_out_even_ptr = query_out + batch * query_out_stride_b + head * query_out_stride_h + token * query_out_stride_t + (dim_pair * 2) * query_out_stride_d
    query_out_odd_ptr = query_out_even_ptr + query_out_stride_d
    key_out_even_ptr = key_out + batch * key_out_stride_b + head * key_out_stride_h + token * key_out_stride_t + (dim_pair * 2) * key_out_stride_d
    key_out_odd_ptr = key_out_even_ptr + key_out_stride_d
    tl.store(query_out_even_ptr, rotated_even, mask=mask & is_query)
    tl.store(query_out_odd_ptr, rotated_odd, mask=mask & is_query)
    tl.store(key_out_even_ptr, rotated_even, mask=mask & ~is_query)
    tl.store(key_out_odd_ptr, rotated_odd, mask=mask & ~is_query)


@triton.jit
def _swiglu_gate_forward_kernel(gate, up, output, total: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    gate_values = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    up_values = tl.load(up + offsets, mask=mask, other=0.0)
    sigmoid = tl.sigmoid(gate_values)
    tl.store(output + offsets, gate_values * sigmoid * up_values, mask=mask)


@triton.jit
def _swiglu_gate_forward_row_kernel(gate, up, output, hidden_dim: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < hidden_dim
    gate_values = tl.load(gate + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    up_values = tl.load(up + row * hidden_dim + offsets, mask=mask, other=0.0)
    silu = gate_values * tl.sigmoid(gate_values)
    tl.store(output + row * hidden_dim + offsets, silu.to(up_values.dtype) * up_values, mask=mask)


@triton.jit
def _swiglu_gate_backward_kernel(gate, up, grad_output, grad_gate, grad_up, total: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    gate_values = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    up_values = tl.load(up + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_values = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(gate_values)
    silu = gate_values * sigmoid
    silu_grad = sigmoid * (1.0 + gate_values * (1.0 - sigmoid))
    tl.store(grad_gate + offsets, grad_values * up_values * silu_grad, mask=mask)
    tl.store(grad_up + offsets, grad_values * silu, mask=mask)


@triton.jit
def _swiglu_gate_backward_row_kernel(
    gate,
    up,
    grad_output,
    grad_gate,
    grad_up,
    hidden_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < hidden_dim
    gate_values = tl.load(gate + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    up_values = tl.load(up + row * hidden_dim + offsets, mask=mask, other=0.0)
    grad_values = tl.load(grad_output + row * hidden_dim + offsets, mask=mask, other=0.0)
    sigmoid = tl.sigmoid(gate_values)
    silu = gate_values * sigmoid
    silu_grad = sigmoid * (1.0 + gate_values * (1.0 - sigmoid))
    tl.store(grad_gate + row * hidden_dim + offsets, grad_values * up_values * silu_grad, mask=mask)
    tl.store(grad_up + row * hidden_dim + offsets, grad_values * silu.to(grad_values.dtype), mask=mask)
