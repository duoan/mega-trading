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


def triton_add_rms_norm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMSNorm.

    Returns `(new_residual, normed)` where `new_residual = hidden + residual`
    and `normed = RMSNorm(new_residual, weight, eps)`. Falls back to PyTorch
    when the tensors are not on CUDA or shapes/strides are unsupported.
    """
    if (
        not _can_use_triton(hidden)
        or not _can_use_triton(residual)
        or hidden.shape != residual.shape
        or hidden.shape[-1] != weight.numel()
        or hidden.stride(-1) != 1
        or residual.stride(-1) != 1
    ):
        new_residual = hidden + residual
        normed = F.rms_norm(new_residual, (new_residual.shape[-1],), weight, eps=eps)
        return new_residual, normed
    return _TritonAddRMSNorm.apply(hidden, residual, weight, eps)


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


def triton_fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute mean cross-entropy of `linear(hidden, weight)` against `labels` without materializing logits.

    The fused path keeps each row's logits in registers/SRAM long enough to compute
    `loss` and the gradient w.r.t. logits, then immediately collapses that gradient
    back into `grad_hidden` and `grad_weight`. This avoids the `[rows, vocab]`
    activation that dominates HBM for vocab-heavy heads.

    `hidden` must be `[rows, hidden_dim]`-shaped (caller flattens batch/time);
    `weight` must be `[vocab, hidden_dim]` matching `nn.Linear(bias=False).weight`.
    Rows whose label equals `ignore_index` contribute zero loss and zero gradient.
    """
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [rows, hidden_dim]")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[1]:
        raise ValueError("weight must have shape [vocab, hidden_dim] matching hidden's last dim")
    if labels.ndim != 1 or labels.shape[0] != hidden.shape[0]:
        raise ValueError("labels must be a 1D tensor with one entry per hidden row")
    if labels.dtype not in {torch.int32, torch.int64}:
        raise ValueError("labels must be an integer tensor")
    if not _can_use_triton(hidden) or not _can_use_triton(weight):
        return _torch_fused_linear_cross_entropy(hidden, weight, labels, ignore_index=ignore_index)
    return _TritonFusedLinearCrossEntropy.apply(hidden, weight, labels, int(ignore_index))


def _can_use_triton(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda" and tensor.dtype in {torch.float16, torch.bfloat16, torch.float32}


def _torch_fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
) -> torch.Tensor:
    # Reference path used on CPU and as a Triton fallback; matches `F.linear` + `F.cross_entropy`.
    # `F.cross_entropy` returns NaN when every label is ignored, so collapse that case to a
    # zero loss (still on the autograd graph) to match the Triton kernel's behaviour.
    if int((labels != ignore_index).sum().item()) == 0:
        return (hidden.sum() + weight.sum()) * 0.0
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


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


class _TritonAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_contiguous = hidden.contiguous()
        residual_contiguous = residual.contiguous()
        weight_contiguous = weight.contiguous()
        hidden_dim = hidden_contiguous.shape[-1]
        rows = hidden_contiguous.numel() // hidden_dim
        new_residual = torch.empty_like(hidden_contiguous)
        normed = torch.empty_like(hidden_contiguous)
        rstd = torch.empty((rows,), device=hidden.device, dtype=torch.float32)
        block_d = triton.next_power_of_2(hidden_dim)
        _add_rms_norm_forward_kernel[(rows,)](
            hidden_contiguous,
            residual_contiguous,
            weight_contiguous,
            new_residual,
            normed,
            rstd,
            eps,
            hidden_dim,
            BLOCK_D=block_d,
            num_warps=8,
        )
        ctx.save_for_backward(new_residual, weight_contiguous, rstd)
        ctx.hidden_shape = hidden.shape
        ctx.hidden_dim = hidden_dim
        return new_residual.view(hidden.shape), normed.view(hidden.shape)

    @staticmethod
    def backward(
        ctx,
        grad_new_residual: torch.Tensor,
        grad_normed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        new_residual, weight, rstd = ctx.saved_tensors
        hidden_dim = int(ctx.hidden_dim)
        rows = new_residual.numel() // hidden_dim
        block_d = triton.next_power_of_2(hidden_dim)
        # grad_new_residual flows directly into both inputs of the add, so the
        # fused kernel can share a single dX buffer for hidden and residual.
        grad_input = torch.empty_like(new_residual)
        sm_count = torch.cuda.get_device_properties(new_residual.device).multi_processor_count
        # Match the row-blocked dW reduction strategy used by _TritonRMSNorm.
        reduction_blocks = min(rows, sm_count * 2)
        partial_weight = torch.empty(
            (reduction_blocks, hidden_dim), device=new_residual.device, dtype=torch.float32
        )
        grad_new_residual_contiguous = grad_new_residual.contiguous()
        grad_normed_contiguous = grad_normed.contiguous()
        _add_rms_norm_backward_combined_kernel[(reduction_blocks,)](
            new_residual,
            weight,
            rstd,
            grad_new_residual_contiguous,
            grad_normed_contiguous,
            grad_input,
            partial_weight,
            rows,
            hidden_dim,
            triton.cdiv(rows, reduction_blocks),
            BLOCK_D=block_d,
            num_warps=8,
        )
        grad_weight = partial_weight.sum(dim=0).to(weight.dtype)
        grad_input_view = grad_input.view(ctx.hidden_shape)
        # Both hidden and residual contributed identically to the add, so the
        # autograd contract returns the same gradient tensor twice.
        return grad_input_view, grad_input_view, grad_weight, None


class _TritonFusedLinearCrossEntropy(torch.autograd.Function):
    """Liger-style chunked fused linear-cross-entropy.

    Per-chunk recipe (Liger Kernel, Han et al.):
      1. `logits_chunk = hidden_chunk @ weight.T` — one cuBLAS GEMM (PyTorch picks the
         optimal tile shape; we never beat it with handwritten Triton matmul).
      2. `_fused_ce_row_kernel` — one program per row of the chunk. Reads logits, runs
         online softmax, writes the per-row loss to `loss_partial`, then *in-place*
         rewrites the chunk into `grad_logits` (mean-reduced and ignore-index aware).
      3. `grad_hidden_chunk = grad_logits @ weight` — second cuBLAS GEMM, written
         straight into the row slice of `grad_hidden`.
      4. `grad_weight += grad_logits.T @ hidden_chunk` — third cuBLAS GEMM, accumulated
         in fp32 across chunks for numerical stability.

    Memory: peak is `chunk_size * vocab` bf16 (the in-place logits buffer) instead of
    the full `[rows, vocab]` fp32 logits + bf16 grad_logits that PyTorch materialises.
    The chunk size is picked so the logits chunk has roughly the same footprint as
    `hidden`, i.e. `chunk_size = ceil(rows / ceil(vocab / hidden_dim))`.
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
    ) -> torch.Tensor:
        hidden_contiguous = hidden.contiguous()
        weight_contiguous = weight.contiguous()
        labels_contiguous = labels.contiguous().to(torch.int64)
        rows, hidden_dim = hidden_contiguous.shape
        vocab_size = weight_contiguous.shape[0]

        n_valid = int((labels_contiguous != ignore_index).sum().item())
        # Skip the backward GEMMs entirely when neither input requires grad — inference
        # and metric eval pay only for the chunked forward GEMM + the row-wise CE kernel.
        needs_grad = bool(hidden.requires_grad or weight.requires_grad)

        if n_valid == 0:
            grad_hidden = torch.zeros_like(hidden_contiguous) if needs_grad else None
            grad_weight = torch.zeros_like(weight_contiguous) if needs_grad else None
            loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        else:
            chunk_size = _fused_ce_chunk_size(rows, hidden_dim, vocab_size)
            block_v = triton.next_power_of_2(vocab_size)
            num_warps = _fused_ce_num_warps(block_v)

            loss_partial = torch.empty((rows,), device=hidden.device, dtype=torch.float32)
            grad_hidden = (
                torch.empty_like(hidden_contiguous)
                if needs_grad
                else None
            )
            # `grad_weight` accumulates in the caller's dtype to match Liger Kernel's
            # default. Profiling on the rtx shape showed fp32 accumulation costs ~1.3ms
            # per step (extra bf16->fp32 casts + fp32 matmul) while the chunk count is
            # small enough (~8 chunks) that bf16 accumulation stays numerically clean.
            # Callers that need fp32 stability can plumb an `accum_dtype` knob later.
            grad_weight = (
                torch.zeros((vocab_size, hidden_dim), device=hidden.device, dtype=weight.dtype)
                if needs_grad
                else None
            )

            for start in range(0, rows, chunk_size):
                end = min(start + chunk_size, rows)
                hidden_chunk = hidden_contiguous[start:end]
                labels_chunk = labels_contiguous[start:end]
                # Step 1: cuBLAS GEMM into a per-chunk logits buffer that we will reuse
                # in-place as grad_logits in step 2 — keeping peak memory at one chunk.
                logits_chunk = hidden_chunk @ weight_contiguous.t()

                # Step 2: row-parallel CE kernel computes the per-row loss and overwrites
                # `logits_chunk` with `grad_logits = (softmax - one_hot(label)) / n_valid`.
                # Ignored rows are zeroed and produce zero loss.
                n_rows = end - start
                _fused_ce_row_kernel[(n_rows,)](
                    logits_chunk,
                    logits_chunk.stride(0),
                    labels_chunk,
                    loss_partial[start:end],
                    vocab_size,
                    float(n_valid),
                    ignore_index,
                    HAS_GRADIENTS=needs_grad,
                    BLOCK_V=block_v,
                    num_warps=num_warps,
                )

                if needs_grad:
                    # Step 3: project grad_logits back into hidden-space via cuBLAS GEMM.
                    grad_hidden[start:end] = logits_chunk @ weight_contiguous
                    # Step 4: outer-product into grad_weight directly in the caller's
                    # dtype — this is the same recipe Liger uses by default.
                    grad_weight.addmm_(logits_chunk.t(), hidden_chunk)
            # Per-row losses already absorb the `1 / n_valid` mean-normalisation through
            # the kernel, so `sum` here recovers the scalar mean loss.
            loss = loss_partial.sum()

        if needs_grad:
            ctx.save_for_backward(grad_hidden, grad_weight)
        else:
            # Tell autograd this output is not differentiable so it skips backward.
            ctx.mark_non_differentiable(loss)
        ctx.hidden_shape = hidden.shape
        ctx.weight_shape = weight.shape
        return loss.to(hidden.dtype) if loss.dtype != hidden.dtype else loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        grad_hidden, grad_weight = ctx.saved_tensors
        # The fused kernel computed grads assuming an upstream scalar gradient of 1.0.
        # Scale by the actual upstream gradient before returning so the autograd
        # contract holds (the user could be calling `.backward(retain_graph=...)` with
        # a non-trivial loss reduction upstream).
        scale = grad_output.detach()
        return grad_hidden * scale, grad_weight * scale, None, None


def _fused_ce_chunk_size(rows: int, hidden_dim: int, vocab_size: int) -> int:
    # Pick the chunk size so the logits chunk has roughly the same footprint as the
    # full hidden tensor; this is the same trick Liger Kernel uses to bound peak HBM.
    inc_factor = triton.cdiv(vocab_size, hidden_dim)
    chunk_size = max(triton.next_power_of_2(triton.cdiv(rows, inc_factor)), 1)
    return min(chunk_size, rows)


def _fused_ce_num_warps(block_v: int) -> int:
    # Bigger vocab tiles need more warps to keep the row-wise softmax/grad loop pipelined.
    if block_v <= 1024:
        return 4
    if block_v <= 8192:
        return 8
    return 16


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
def _add_rms_norm_forward_kernel(
    hidden,
    residual,
    weight,
    new_residual,
    normed,
    rstd,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < hidden_dim
    # Fuse the residual add: load both inputs once, write the new residual once,
    # and keep the fp32 sum in registers for the RMS reduction below.
    hidden_values = tl.load(hidden + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    residual_values = tl.load(residual + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
    summed = hidden_values + residual_values
    tl.store(new_residual + row * hidden_dim + offsets, summed, mask=mask)
    mean_square = tl.sum(summed * summed, axis=0) / hidden_dim
    inv_rms = tl.rsqrt(mean_square + eps)
    weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(normed + row * hidden_dim + offsets, summed * inv_rms * weights, mask=mask)
    tl.store(rstd + row, inv_rms)


@triton.jit
def _add_rms_norm_backward_combined_kernel(
    new_residual,
    weight,
    rstd,
    grad_new_residual,
    grad_normed,
    grad_input,
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
        values = tl.load(new_residual + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_residual_values = tl.load(grad_new_residual + row * hidden_dim + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_normed_values = tl.load(grad_normed + row * hidden_dim + offsets, mask=mask, other=0.0)
        inv_rms = tl.load(rstd + row).to(tl.float32)
        weighted_grad = (grad_normed_values * weights).to(tl.float32)
        normed_values = values * inv_rms
        correction = tl.sum(weighted_grad * values, axis=0)
        grad_through_norm = inv_rms * (weighted_grad - values * (inv_rms * inv_rms / hidden_dim) * correction)
        # The forward node is `new_residual = hidden + residual` followed by
        # RMSNorm on `new_residual`. The gradient w.r.t. either add input is the
        # sum of the direct residual gradient and the gradient flowing back
        # through the norm.
        grad_input_values = grad_residual_values + grad_through_norm
        grad_weight_acc += grad_normed_values.to(tl.float32) * normed_values
        tl.store(grad_input + row * hidden_dim + offsets, grad_input_values, mask=mask)

    tl.store(partial_weight + program_id * hidden_dim + offsets, grad_weight_acc, mask=mask)


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


@triton.jit
def _fused_ce_row_kernel(
    logits_ptr,
    logits_row_stride,
    labels_ptr,
    loss_ptr,
    vocab_size: tl.constexpr,
    n_valid,
    ignore_index,
    HAS_GRADIENTS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # One program per row. Reads the full `[vocab_size]` logits row, runs an online
    # softmax (max + log-sum-exp), writes `loss = (lse - logit_y) / n_valid`, and — if
    # gradients are needed — overwrites the same row in place with the cross-entropy
    # gradient `(softmax - one_hot(y)) / n_valid`.
    #
    # When `BLOCK_V >= vocab_size` we run a single load/exp/store per row; otherwise we
    # tile across vocab and use the streaming online-softmax recurrence.
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * logits_row_stride

    label = tl.load(labels_ptr + row)
    if label == ignore_index:
        # Zero out the row's grad slot (the row stays in `grad_hidden` via the GEMM in
        # step 3, so we must explicitly zero the contribution here) and write a zero loss.
        if HAS_GRADIENTS:
            for off in tl.range(0, vocab_size, BLOCK_V):
                cols = off + tl.arange(0, BLOCK_V)
                mask = cols < vocab_size
                tl.store(base + cols, tl.zeros((BLOCK_V,), dtype=base.dtype.element_ty), mask=mask)
        tl.store(loss_ptr + row, 0.0)
        return

    inv_n_valid = 1.0 / n_valid.to(tl.float32)

    # Pass 1: online softmax — find row max and sum-of-exp without rematerialising logits.
    m = float("-inf")
    d = 0.0
    for off in tl.range(0, vocab_size, BLOCK_V):
        cols = off + tl.arange(0, BLOCK_V)
        mask = cols < vocab_size
        block = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
        block_max = tl.max(block, axis=0)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(block - m_new), axis=0)
        m = m_new

    lse = m + tl.log(d)
    # The captured logit at the true class is `logit_y`; loss for this row is
    # `(lse - logit_y) / n_valid`, summed across rows = mean cross-entropy.
    logit_y = tl.load(base + label).to(tl.float32)
    tl.store(loss_ptr + row, (lse - logit_y) * inv_n_valid)

    if HAS_GRADIENTS:
        # Pass 2: write `grad_logits = (softmax - one_hot(label)) / n_valid` back into
        # the same slab. We re-read the row instead of caching it to keep the kernel's
        # register footprint independent of vocab_size.
        for off in tl.range(0, vocab_size, BLOCK_V):
            cols = off + tl.arange(0, BLOCK_V)
            mask = cols < vocab_size
            block = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
            softmax = tl.exp(block - m) / d
            grad = softmax - tl.where(cols == label, 1.0, 0.0)
            grad = grad * inv_n_valid
            tl.store(base + cols, grad.to(base.dtype.element_ty), mask=mask)
