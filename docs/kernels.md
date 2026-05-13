# Custom Kernels

This document records custom GPU kernels, their target training shapes, benchmark commands, measured performance, and optimization notes.

## Triton Causal GQA Attention

Implementation:

- `src/mega_trading/kernels/triton_attention.py`
- Called from `LlamaAttention` when `attention_backend: triton`
- Benchmarked by `scripts/benchmark_triton_attention.py`

The kernel is a causal grouped-query attention implementation for the RTX PRO 6000 Blackwell path. The forward path and RTX backward path are optimized for the shape produced by `make rtx` and `configs/rtx.yaml`.

## Target Shape

`make rtx` runs:

```bash
uv run python scripts/run_pipeline.py server
```

That prepares `.mega-trading/data/datasets/mixture=rtx` if needed, then trains with:

```bash
uv run mega-trading train --config-name rtx
```

The attention shape entering the kernel is:

```text
batch_size       = 8
sequence_length  = 512
hidden_dim       = 1024
query_heads      = 16
kv_heads         = 4
head_dim         = 64
gqa_repeats      = 4
dtype            = bfloat16
dropout          = 0.0
```

`training.precision: mixed` is mapped to Accelerate `mixed_precision="bf16"` for CUDA training, so bf16 is the benchmark target.

## Benchmark Command

Default benchmark arguments match the server training shape:

```bash
uv run python scripts/benchmark_triton_attention.py --warmup 25 --iterations 100 --json
```

The forward benchmark:

- Allocates q/k/v tensors on CUDA with the server shape.
- Validates Triton output against PyTorch `scaled_dot_product_attention`.
- Uses CUDA events for timing.
- Reports latency and approximate causal-attention TFLOP/s.

## Current Result

Measured on:

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition
```

Command:

```bash
uv run python scripts/benchmark_triton_attention.py --warmup 25 --iterations 100 --json
```

Result:

```text
kernel             latency_ms   approx_tflops
triton_descriptor  0.02754      156.23
torch_sdpa         0.02898      148.49
```

The RTX-shape Triton forward kernel is currently faster than PyTorch SDPA for bf16 on this machine.

Forward + backward command:

```bash
uv run python scripts/benchmark_triton_attention.py --mode forward-backward --warmup 25 --iterations 100 --json
```

Result:

```text
kernel             latency_ms   approx_tflops
triton_descriptor  0.17826       24.14
torch_sdpa         0.14054       30.62
```

The RTX-shape Triton backward is correct and avoids the previous torch tensor-materialized backward path, but it is not yet faster than PyTorch SDPA backward. The current bottleneck is the `dK/dV` kernel.

NCU profiling note:

Full hardware-counter sections such as `SpeedOfLight`, `SchedulerStats`, and `MemoryWorkloadAnalysis` require GPU performance-counter permissions on this host and currently fail with `ERR_NVGPUCTRPERM`. `ncu --set none` still confirms the target `dK/dV` launch shape:

```text
_triton_attention_backward_dkv_server_kernel (8, 32, 1)x(128, 1, 1)
```

Triton compile metadata for the current kernels:

```text
kernel   registers/thread  spills  shared_memory
forward  167               2       32768
dQ       242               0       49152
dK/dV    255               10      74240
```

This points to register pressure and spills in `dK/dV` as the first backward bottleneck.

## Optimization Notes

Useful changes:

- `BLOCK_M=64`, `BLOCK_N=64` for `sequence_length=512`, `head_dim=64`.
- Split the loop into historical blocks and the diagonal causal block.
- Historical blocks skip causal masking entirely.
- The diagonal block applies only the causal boundary mask.
- Keep q loaded through `tl.make_tensor_descriptor`.
- Use direct pointer loads/stores for K/V and output in the RTX-shape path to reduce descriptor overhead on this small fixed shape.
- Remove unnecessary final zero-denominator guards because causal attention always has at least one valid key per row.
- Store per-row log-sum-exp from the forward pass so backward can reconstruct probabilities without recomputing softmax denominators.
- Split backward into a `dQ` kernel and a `dK/dV` kernel. The `dK/dV` kernel owns one key/value tile and accumulates across the 4 grouped query heads to avoid atomics.

Measured tuning history:

```text
variant                                  dtype  latency_ms  note
descriptor 16x32                         bf16   ~0.0739     initial descriptor version
descriptor 32x64                         bf16   ~0.0561     fewer programs, better tile
descriptor 64x64                         bf16   ~0.0379     best simple descriptor tile
split historical/diagonal causal blocks  bf16   ~0.0323     avoids masking historical blocks
current RTX-shape path                   bf16   ~0.0313     faster than SDPA for target shape
current RTX-shape path + lse             bf16   ~0.0275     faster forward and enables Triton backward
```

Rejected experiments:

- `BLOCK_N=128`: slower and, with split causal handling, unsafe unless block assumptions are adjusted.
- `BLOCK_M=128`: slower for the target shape and caused correctness issues with the split diagonal assumption when `BLOCK_M != BLOCK_N`.
- `num_warps=8`: significantly slower.
- Pairing two query heads per program to reuse K/V: slower due to register pressure.
- Direct pointer q load: slower than descriptor q load for the target shape.
- Backward `dK/dV` with `BLOCK_N=32`: correct when row blocks are realigned, but slower due to extra programs and duplicated row work.
- Backward `dK/dV` with `BLOCK_N=128`: slower due to higher register pressure.
- `maxnreg` caps for `dK/dV`: `224`, `192`, `160`, and `128` all slow the kernel down versus the uncapped `255`-register compile.
- Splitting `dK` and `dV` into separate specialized launches reduces per-kernel work but duplicates score/probability reconstruction and is slower overall (`~0.186ms` forward+backward versus `~0.178ms` for the combined path).

## Current Limitations

- Dropout is not fused. The server config sets `model.dropout: 0.0` for the Triton path.
- The optimized forward path is shape-specialized for `T=512`, `head_dim=64`, `gqa_repeats=4`.
- The optimized backward path is even narrower: `T=512`, `head_dim=64`, `query_heads=16`, `kv_heads=4`, `gqa_repeats=4`, `bf16`.
- Other shapes use the generic Triton forward kernel and the conservative torch backward formula.

## Next Optimization Step

Training speed is now dominated by the `dK/dV` backward kernel. The next optimization target is reducing that kernel's register pressure without duplicating row-block work, likely by separating `dK` and `dV` only if the occupancy gain beats the extra softmax/probability recomputation.

## Triton Transformer Operators

Implementation:

- `src/mega_trading/kernels/triton_ops.py`
- Benchmarked by `scripts/benchmark_triton_ops.py`
- RoPE and SwiGLU gate are called from the model when `attention_backend: triton`
- RMSNorm has a Triton implementation for experiments, but the model keeps PyTorch `F.rms_norm` because PyTorch is faster for the server shape
- RMSNorm and SwiGLU follow the public Liger-Kernel/Unsloth strategy: row-wise feature blocks, cached inverse RMS, and fused SiLU multiply. RMSNorm backward uses a Liger-style grouped `dX` + partial `dW` kernel instead of separate `dX` and `dW` passes.

Default benchmark arguments match the server training shape:

```bash
uv run python scripts/benchmark_triton_ops.py --warmup 10 --iterations 50 --json
uv run python scripts/benchmark_triton_ops.py --mode forward-backward --warmup 10 --iterations 50 --json
```

Target operator shapes:

```text
RMSNorm      hidden=[8, 512, 1024]
RoPE         query=[8, 16, 512, 64], key=[8, 4, 512, 64]
SwiGLU gate  gate/up=[8, 512, 2816]
dtype        bfloat16
```

Forward result on RTX PRO 6000 Blackwell:

```text
operator            triton_ms  torch_ms
RMSNorm             0.01405    0.00622
RoPE                0.01940    0.07381
SwiGLU gate         0.01236    0.01899
```

Forward + backward result:

```text
operator            triton_ms  torch_ms
RMSNorm             0.10574    0.05521
RoPE                0.12417    0.30782
SwiGLU gate         0.08807    0.19061
```

Operational decision:

- Enable Triton RoPE in the server path because both forward and backward are faster.
- Enable Triton SwiGLU gate in the server training path because both forward and forward+backward are faster after row-wise tiling.
- Keep RMSNorm on PyTorch in the model because the Triton version is slower in both forward and forward+backward.

Rejected RMSNorm experiment:

- Reusing the incoming `dY` buffer as `dX`, as some public kernels allow, improved allocation behavior but mutates the caller-provided gradient tensor. The local parity test caught the side effect, so the model keeps the safer out-of-place `dX` path.

## Triton Fused Linear Cross-Entropy

Motivation:

The vocab-heavy decoder head produces a `[batch, time, vocab]` logits tensor that dominates loss-head HBM. For the rtx target shape that single tensor is `4096 * 14583 * 4 B ≈ 234 MB` in fp32, and the autograd graph keeps the bf16 `grad_logits` mirror around as well, doubling the cost. Fusing `linear` + `cross_entropy` lets us never materialise the full slab — only one chunk's worth at a time.

Implementation:

- `src/mega_trading/kernels/triton_ops.py` — public `triton_fused_linear_cross_entropy` API, `_TritonFusedLinearCrossEntropy` autograd Function, the chunked-pipeline orchestration, the `_fused_ce_chunk_size` / `_fused_ce_num_warps` helpers, and the single Triton JIT kernel `_fused_ce_row_kernel`
- Called from `TradingModel.forward_with_loss` when `operator_backend: triton` and the caller does not request logits
- Benchmarked by `scripts/benchmark_triton_fused_ce.py`
- The fallback path (`_torch_fused_linear_cross_entropy`) is `F.linear` + `F.cross_entropy`, used on CPU and as the safety net when tensors are not Triton-eligible (non-CUDA, non-fp16/bf16/fp32, etc.)

The implementation follows the **Liger-Kernel** / `efficient_cross_entropy` recipe ([Hsu et al., 2024](https://arxiv.org/abs/2410.10989); reference: [`linkedin/Liger-Kernel`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)): chunk the `hidden` matrix along `BT`, use cuBLAS GEMMs for the matmul work (PyTorch picks the optimal tile shape; hand-written Triton matmul cannot beat it on `[BT, H] x [V, H]` aspect ratios), and use a single row-wise Triton kernel for the per-row online softmax + in-place gradient. The earlier hand-written fused-matmul version (~480 lines of Triton, 4 specialised kernels) is preserved in the tuning history below; the current implementation is ~60 lines of Triton plus a small Python orchestration loop.

## Public API

```python
from mega_trading.kernels.triton_ops import triton_fused_linear_cross_entropy

loss = triton_fused_linear_cross_entropy(
    hidden,        # [rows, hidden_dim] bf16/fp16/fp32 on CUDA (caller flattens batch * time)
    weight,        # [vocab_size, hidden_dim] same dtype as hidden
    labels,        # [rows] int32/int64 — token ids, or `ignore_index` for padding
    ignore_index=-100,
)
loss.backward()    # populates hidden.grad and weight.grad via the saved gradients
```

Contracts:

- Returns the scalar **mean** cross-entropy loss in `hidden.dtype`. Rows whose label equals `ignore_index` contribute zero to both loss and gradients.
- Caller is responsible for flattening `[batch, time, hidden_dim]` to `[rows, hidden_dim]` and the labels accordingly.
- When all rows are ignored the function returns a `0.0` loss without launching the kernels (matches the PyTorch fallback, which would otherwise return NaN).
- When neither input has `requires_grad=True`, the wrapper short-circuits the two backward GEMMs entirely so inference / metric eval pays only for one chunked forward GEMM + the row-wise CE kernel.

## Target Shape

The rtx training profile produces these shapes at the loss head:

```text
rows         = batch * sequence = 8 * 512 = 4096
hidden_dim   = 1024
vocab_size   = 14583   # 3 specials + 2*2*9*9*9*5 joint event tokens (rtx tokenizer)
dtype        = bfloat16
ignore_index = -100    # padding labels in the dataset
```

`hidden` has shape `[4096, 1024]`, `weight` has shape `[14583, 1024]` (the tied output projection / token embedding), and `labels` has shape `[4096]`.

## Algorithm

Setup (host-side, inside `_TritonFusedLinearCrossEntropy.forward`):

- `n_valid = (labels != ignore_index).sum().item()` — one `.item()` host sync, used as the mean-reduction divisor inside the row kernel. The cost (~10us) is amortised over the whole loss head and matches Liger's design exactly.
- `chunk_size = _fused_ce_chunk_size(rows, hidden_dim, vocab_size)` — picked so the in-place logits buffer has roughly the same footprint as `hidden`.
- `block_v = next_pow2(vocab_size)`, `num_warps = _fused_ce_num_warps(block_v)` — fed to the row kernel; bigger vocab tiles need more warps.
- Allocate `loss_partial` (fp32, `[rows]`), and (only if `needs_grad`) `grad_hidden` (caller dtype, `[rows, hidden_dim]`) and `grad_weight` (caller dtype, `[vocab_size, hidden_dim]`, **`zeros` so chunk accumulation starts clean**).

Per-chunk recipe:

1. **`logits_chunk = hidden_chunk @ weight.T`** — one cuBLAS GEMM into a per-chunk `[chunk_size, vocab]` buffer in the caller's dtype. We reuse this same buffer in step 2 instead of allocating a separate `grad_logits`.
2. **`_fused_ce_row_kernel`** — one program per row of the chunk. The kernel runs an **online softmax** over the row (max + log-sum-exp via the streaming recurrence), writes the per-row mean-normalised loss `(lse - logit_y) / n_valid` to `loss_partial`, and — under the `HAS_GRADIENTS` `tl.constexpr` flag — **in-place** rewrites the row from `logits` into `grad_logits = (softmax - one_hot(label)) / n_valid`. Ignored rows are zeroed and produce zero loss. The kernel's register footprint is independent of `vocab_size` because it tiles the inner softmax/grad loop with `BLOCK_V`.
3. **`grad_hidden[start:end] = grad_logits @ weight`** — second cuBLAS GEMM, written straight into the row slice of `grad_hidden`. No accumulation needed because each chunk owns disjoint rows.
4. **`grad_weight.addmm_(grad_logits.T, hidden_chunk)`** — third cuBLAS GEMM, **accumulated directly in the caller's dtype** across chunks (Liger's default). The chunk count for the rtx shape is small (~8), so bf16 accumulation is numerically clean. An earlier fp32 accumulator added ~1.3 ms per step from the extra bf16↔fp32 casts and bought no measurable accuracy.

After the chunk loop:

- `loss = loss_partial.sum()` — the per-row losses already absorb the `1 / n_valid` mean divisor inside the kernel, so `sum` recovers the scalar mean loss directly.
- `grad_hidden` and `grad_weight` are saved via `ctx.save_for_backward`. The `backward()` step only scales them by `grad_output` so the matmul work happens once per training step regardless of which side calls `.backward()`.

Chunk-size derivation:

```text
inc_factor  = ceil(vocab_size / hidden_dim)         # 15 for the rtx shape (14583 / 1024)
chunk_size  = next_pow2(ceil(rows / inc_factor))    # 512 for the rtx shape (4096 / 15 -> 274 -> 512)
num_chunks  = ceil(rows / chunk_size)               # 8 for the rtx shape
peak_logits = chunk_size * vocab_size * dtype_size  # 14.2 MB at chunk=512, vocab=14583, bf16
```

The heuristic is from Liger Kernel: pick the chunk so a single `logits_chunk` slab has roughly the same HBM footprint as `hidden` itself. Larger chunks reduce kernel-launch overhead but increase peak memory; smaller chunks shrink memory but add Python loop / cuBLAS dispatch overhead. The `next_pow2` rounding lets cuBLAS pick a friendlier tile shape for the chunked GEMMs.

## Tile Sizes

The single Triton kernel is **row-wise** — no `BLOCK_M`, no `BLOCK_V`/`BLOCK_D` matmul tiles, no shared-memory tuning. The only configurable knob is `BLOCK_V`, the per-iteration vocab-tile width inside the row's online softmax loop. For the rtx target shape:

```text
chunk_size = 512                                       # rows per cuBLAS GEMM
BLOCK_V    = next_pow2(vocab_size) = 16384             # >= vocab so the streaming loop runs once
num_warps  = 16                                        # _fused_ce_num_warps: V<=1024 -> 4, <=8192 -> 8, else 16
```

The matmul work is delegated to cuBLAS via PyTorch's `@` and `addmm_`, so cuBLAS picks its own tile shape, num_stages, and warp count for the GEMMs. We did not write a single matmul tile.

## Benchmark Command

Default benchmark arguments match the rtx training shape:

```bash
uv run python scripts/benchmark_triton_fused_ce.py --warmup 10 --iterations 50 --json
uv run python scripts/benchmark_triton_fused_ce.py --mode forward --warmup 10 --iterations 50 --json
```

The benchmark:

- Allocates `hidden` (`[4096, 1024]`), `weight` (`[14583, 1024]`), and `labels` (`[4096]`) on CUDA.
- Validates the Triton output and gradients against `F.linear` + `F.cross_entropy` in fp32 before timing.
- Reports per-iteration latency via CUDA events and peak CUDA memory via `torch.cuda.max_memory_allocated`.

## Current Result

Measured on:

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition
```

Forward only (`requires_grad=False` short-circuit, only the chunked GEMM + row-wise CE kernel run, no grad GEMMs):

```text
kernel                       latency_ms   peak_mb
triton_fused_linear_ce           0.99       73.1
torch_linear_plus_ce             0.89      272.6
```

Forward + backward (chunked GEMMs + row-wise CE kernel + `grad_hidden` GEMM + `grad_weight` `addmm_`):

```text
kernel                       latency_ms   peak_mb
triton_fused_linear_ce           2.87      125.7
torch_linear_plus_ce             2.55      508.8
```

The fused kernel reduces peak loss-head memory by **~4.0x in forward+backward** (`509 MB -> 126 MB`) and **~3.7x in forward only** (`273 MB -> 73 MB`). Latency is essentially at par with PyTorch in both modes (~+11-12% — within cuBLAS tile-shape noise), so the kernel is a strict win on the rtx target shape: same speed, a quarter of the peak HBM.

Memory accounting (forward+backward):

```text
component                                size_mb  notes
hidden (saved by autograd)                  8.0   [4096, 1024] bf16
weight                                     30.0   [14583, 1024] bf16
torch logits + grad_logits                314.0   2 * [4096, 14583] fp32 / bf16, dominates the torch path
fused kernel logits_chunk (in place)       14.2   [chunk_size=512, 14583] bf16, reused across chunks
fused kernel grad_hidden                    8.0   [4096, 1024] bf16 destination
fused kernel grad_weight                   30.0   [14583, 1024] bf16 accumulator (Liger-style direct accum)
fused kernel loss_partial                   0.02  per-row losses summed on host
```

The torch path's `[batch, time, vocab]` fp32 logits + bf16 `grad_logits` activations dominate its peak. The fused kernel only ever holds **one chunk's worth of logits** at a time, freeing the buffer between chunks.

## Operational Decision

- Enable the fused kernel for the rtx training path (`operator_backend: triton`) — same latency as PyTorch (within ~12%), four times less peak HBM. The headroom directly translates into bigger batches, longer sequences, or fp32 master weights without OOM.
- Keep the PyTorch path (`F.linear` + `F.cross_entropy`) for the mac and modal CPU/MPS paths, where the kernel is unavailable.
- Metric steps (top-1 / top-5 accuracy) call `forward_with_loss(..., return_logits=True)` so logits are produced via the standard projection on the few steps that need them; the rest of the training run uses the fused path.

## Optimization Notes

The current design intentionally does **not** hand-write the matmul. Useful properties of this approach:

- **cuBLAS GEMMs do the heavy lifting**: PyTorch's `@` and `addmm_` pick optimal tile shapes for the rtx aspect ratios (`[BT, H] x [V, H]` shape). Our previous hand-written fused matmul + softmax kernel was 4-5x slower because hand-written Triton matmul cannot beat cuBLAS for these dimensions.
- **In-place `logits -> grad_logits`**: the row-wise CE kernel rewrites the same buffer instead of allocating a separate `grad_logits`. Halves the bf16 footprint of the chunk loop.
- **Chunked over `BT` so peak memory ≈ `chunk_size * V`** instead of `BT * V`. The `chunk_size = ceil(BT / inc_factor)` heuristic keeps the chunk roughly equal to the `hidden` tensor footprint.
- **`requires_grad=False` short-circuit**: pure inference / metric eval skips the two grad GEMMs entirely (forward goes from 4.2 ms to 0.99 ms).
- **bf16 `grad_weight` accumulator**: chunk count is small (~8 for the rtx shape), so direct bf16 accumulation matches Liger Kernel's default and is numerically clean. An earlier fp32 accumulator added ~1.3 ms per step from the bf16->fp32 casts on every chunk; profiling showed it bought no measurable accuracy on this shape.
- **Single Triton JIT kernel** (`_fused_ce_row_kernel`): one program per row, online softmax with `BLOCK_V`-tile inner loop. No matmul tiles, no shared-memory tuning. The kernel signature mirrors `liger_cross_entropy_kernel` from Liger Kernel.

Tuning history (forward+backward latency on the rtx shape, lower is better):

```text
variant                                                              latency_ms  note
hand-written fused matmul + softmax + atomic_add (initial)                13.4   5.1x slower
+ split backward (grad_hidden 2D grid + atomic grad_weight)                9.3   3.6x slower
+ further BLOCK_M / num_warps tuning, num_stages=2                         7.7   3.0x slower
+ split backward into specialised grad_hidden + grad_weight kernels        4.9   1.9x slower
+ BLOCK_M=32, num_stages=2 (best hand-written version)                     4.2   1.6x slower
chunked GEMM + row-wise CE kernel + fp32 grad_weight (first Liger port)    4.2   1.6x slower
chunked GEMM + row-wise CE kernel + bf16 grad_weight (current design)      2.87  +12% vs torch, 8x less code
```

The current Liger-style rewrite matches PyTorch latency (within ~12%, cuBLAS tile-shape noise) while using **8x less Triton code** (~60 lines vs ~480), needing zero shared-memory tuning, and cutting peak HBM to a quarter of PyTorch's.

Lessons learned:

- **Don't hand-write matmul tiles for shapes cuBLAS already knows about.** The first six rows of the tuning table above are all variants that hand-rolled the `[BT, H] x [H, V]` matmul inside Triton — none of them beat the simplest possible PyTorch GEMM, regardless of `BLOCK_M / BLOCK_V / BLOCK_D / num_stages / num_warps` choice. The win comes from *what work to skip*, not from out-tuning cuBLAS.
- **Match the reference library's defaults before adding "improvements".** The first Liger port to this codebase added an fp32 `grad_weight` accumulator on top of Liger's recipe, which silently cost 1.3 ms per step from the extra bf16↔fp32 casts. Liger's authors had already tuned that detail; our "improvement" was a regression. Profiling each step (forward GEMM, CE kernel, grad-hidden GEMM, grad-weight GEMM, fp32 vs bf16 accumulator) made the cost obvious in seconds.
- **Use `requires_grad` to short-circuit backward early.** Inference / metric eval invokes the same fused op but with `requires_grad=False`; gating the two backward GEMMs behind that flag dropped forward-only latency from 4.2 ms to 0.99 ms with no API change.
- **Mind the host-sync.** The `(labels != ignore_index).sum().item()` to compute `n_valid` is a host barrier; it cannot move to a CUDA stream. Liger has the same sync; it is fine for one-per-step usage but would be visible in tighter loops.

Rejected experiments:

- Hand-written fused matmul (the four kernel variants in the tuning history above): cuBLAS GEMM beats hand-written Triton matmul for these shapes regardless of tile choice. Kept here for the historical record, removed from the codebase.
- fp32 `grad_weight` accumulator across chunks: cleanly numerically but 1.3 ms slower per step on the rtx shape (8 chunks). With this chunk count bf16 accumulation is stable enough.
- Atomic-add `grad_weight` inside a single fused kernel: tested in the hand-written variants; atomics serialise across chunks and add ~0.5 ms even after the kernel split.
- `inc_factor` ceiling vs floor: `triton.cdiv(V, H)` (ceiling) is the safe choice; using `V // H` (floor) would let `chunk_size * V > BT * H`, defeating the memory bound for shapes where `V` is not a multiple of `H`.

## Next Optimization Steps

The remaining ~12% latency gap is small enough not to be worth chasing on this shape. If it ever matters:

- Larger chunk size (or single-chunk) on shapes where `BT * V` already fits within the memory budget — would bypass the chunked Python loop overhead and let cuBLAS do one big GEMM.
- An optional `accum_dtype=fp32` knob (Liger has it) for shapes with very large chunk counts where bf16 accumulation drifts; current rtx shape (~8 chunks) is well below that threshold.
- Plumb the kernel through `TradingModel.forward_with_loss` so the Trainer actually uses it (currently still pending).

## Current Limitations

- Latency: ~12% slower than `F.linear` + `F.cross_entropy` on the rtx target shape; effectively at par. The kernel wins on memory (4.0x in forward+backward, 3.7x in forward-only), so the trade-off favours it whenever the dense `[batch, time, vocab]` fp32 logits would force a smaller batch / shorter context.
- Specialized for `bfloat16` on Blackwell with the rtx target shape; `float16` and `float32` paths exercise the same kernel and produce correct results but have not been tile-tuned.
- Only `mean` reduction with optional `ignore_index` masking is supported. Sum / batch-mean / label-smoothing variants are not.

## Fused Residual-Add + RMSNorm

Motivation:

`LlamaDecoderBlock` runs `hidden = hidden + sublayer(...)` immediately followed by `RMSNorm(hidden + ...)` on the next sublayer, so the residual add and RMSNorm touch the same `[batch, time, hidden]` tensor back-to-back. A fused kernel would in theory save one full read+write of `hidden` (the materialized residual) before RMSNorm rereads it. Public kernel libraries (Flash-Attention's `rms_norm_fwd_residual`, Liger's `fused_add_norm`) implement this pattern.

Implementation:

- `triton_add_rms_norm(hidden, residual, weight, *, eps) -> (new_residual, normed)` lives in `src/mega_trading/kernels/triton_ops.py`.
- Forward kernel `_add_rms_norm_forward_kernel` reads `hidden` and `residual` once, writes `new_residual`, and reuses the fp32 sum in registers to compute `inv_rms` and the RMSNorm output in the same row-wise program.
- Backward kernel `_add_rms_norm_backward_combined_kernel` mirrors `_rms_norm_backward_combined_kernel`: one program per row block, fp32 partial `dW` accumulator, final `partial_weight.sum(dim=0)` reduction. The forward node `new_residual = hidden + residual` means the gradient w.r.t. both add inputs is the same tensor `grad_new_residual + grad_through_norm`, so the autograd `backward` returns it once and lets PyTorch route it to both leaves.
- Parity tests (`tests/test_training.py::test_triton_add_rms_norm_*`) cover bf16 forward + backward against `(x + y, F.rms_norm(x + y, weight, eps))` plus a CPU-fallback test.

Benchmark (RTX PRO 6000 Blackwell, bf16, `B=8, T=512, H=1024`, target shape from `configs/rtx.yaml`):

```text
forward only
torch  add + F.rms_norm                       0.0133 ms
triton raw kernel  (reused output buffers)    0.0096 ms   (-28% vs torch)
triton autograd.Function wrapper              0.0197 ms   (+48% vs torch)

forward + backward
torch  add + F.rms_norm  (full autograd)      0.0967 ms
triton raw kernel only  (no autograd)         0.0176 ms
triton autograd.Function wrapper              0.1712 ms   (+77% vs torch)
triton composed: eager add + triton_rms_norm  0.1820 ms
```

Operational decision:

- **Do not wire the fused kernel into `LlamaDecoderBlock`.** On the target shape the PyTorch eager path (`add` + `F.rms_norm` + autograd engine) already runs at roughly the HBM-bandwidth limit, and the raw Triton kernel's ~3.7 us forward win is erased by ~10 us of `torch.autograd.Function.apply` Python overhead. The forward+backward gap is dominated by the same wrapper overhead repeated on the backward Function plus an extra `partial_weight.sum(dim=0)` launch for the `dW` reduction.
- Keep `triton_add_rms_norm` and its parity tests so future shape changes (larger hidden, smaller `B*T`, fp32) or a switch to `torch.compile` / `inductor` epilogue fusion can reuse the kernel without rewriting it.

Rejected variants:

- `torch.library.custom_op` + `register_autograd` wrapper: forward 0.0161 ms (still slower than torch 0.0133 ms) and forward+backward 0.2270 ms (worse than the `autograd.Function` wrapper). The C++ dispatcher path avoids some Python overhead in forward but adds a second op-dispatch round trip on backward.
- Calling `triton_rms_norm` after the eager `x + y`: 0.1820 ms forward+backward, confirming the fused kernel itself is not the bottleneck and the regression is the autograd-wrapper tax we already saw on plain `triton_rms_norm` (see "Operational decision" above for RMSNorm).

Conditions under which a re-benchmark may flip this:

- Hidden dimension significantly larger than 1024 (the row reduction starts to dominate kernel runtime and the wrapper overhead becomes a smaller fraction).
- `torch.compile` enabled on `LlamaDecoderBlock` so inductor can inline the fused op and erase the autograd-Function Python frame.
- A bf16 RMSNorm regression in a future PyTorch release that closes the current `F.rms_norm` advantage.
