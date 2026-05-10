# Custom Kernels

This document records custom GPU kernels, their target training shapes, benchmark commands, measured performance, and optimization notes.

## Triton Causal GQA Attention

Implementation:

- `src/mega_trading/kernels/triton_attention.py`
- Called from `LlamaAttention` when `attention_backend: triton`
- Benchmarked by `scripts/benchmark_triton_attention.py`

The kernel is a causal grouped-query attention implementation for the RTX PRO 6000 Blackwell server path. The forward path and a server-only backward path are optimized for the shape produced by `make server` and `configs/server-rtx6000.yaml`.

## Target Shape

`make server` runs:

```bash
uv run python scripts/run_pipeline.py server
```

That prepares `.mega-trading/binance-modal` if needed, then trains with:

```bash
uv run mega-trading train --config-name server-rtx6000
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

The server-shape Triton forward kernel is currently faster than PyTorch SDPA for bf16 on this machine.

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

The server-shape Triton backward is correct and avoids the previous torch tensor-materialized backward path, but it is not yet faster than PyTorch SDPA backward. The current bottleneck is the `dK/dV` kernel.

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
- Use direct pointer loads/stores for K/V and output in the server-shape path to reduce descriptor overhead on this small fixed shape.
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
current server-shape path                bf16   ~0.0313     faster than SDPA for target shape
current server-shape path + lse          bf16   ~0.0275     faster forward and enables Triton backward
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
