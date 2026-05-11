# Performance Profiling

This document records PyTorch profiler runs for the RTX training path and the current bottleneck analysis.

## Executive Summary

We investigated intermittent `tokens_per_gpu_second` and GPU utilization drops in the RTX training run. The initial profile showed that normal gradient-accumulation micro-steps were stable around 35-37 ms, but every optimizer boundary jumped to about 198-200 ms when using `MuonAdamW` with `gradient_accumulation_steps=8`.

The original bottleneck was a mix of CUDA scalar synchronization and fragmented per-parameter Muon work:

```text
baseline Muon:
  max profiler step              199.561 ms
  CPU Optimizer.step             323.645 ms / 2 optimizer updates
  GPU Optimizer.step annotation  309.636 ms / 2 optimizer updates
  aten::item/_local_scalar_dense 190+ ms / 280 calls
  cudaStreamSynchronize          284.439 ms / 312 calls
```

After the optimization pass, the final best smoke-run setting is shape-bucketed Muon with `training.muon_ns_steps=3`:

```text
final Muon candidate:
  max profiler step              100.900 ms
  CPU Optimizer.step              20.018 ms / 2 optimizer updates
  GPU Optimizer.step annotation  148.919 ms / 2 optimizer updates
  aten::item/_local_scalar_dense absent from trace
  validation_loss                6.254144 at step 104
```

Compared with the baseline, the profiler-window max step dropped from about 200 ms to about 101 ms, and the CPU optimizer window dropped from about 324 ms to about 20 ms per two optimizer updates. The remaining sawtooth is now mostly real GPU Newton-Schulz work at accumulation boundaries, not Python scalar sync or per-tensor launch fragmentation.

Techniques used:

- PyTorch profiler with CPU/CUDA activities and TensorBoard trace output.
- NVTX ranges around `train/data_wait`, `train/forward`, `train/backward`, `train/optimizer`, `train/eval`, and checkpoint sections.
- AdamW control run to separate Muon-specific optimizer cost from forward/backward and data-loading cost.
- Device-side Muon norm guard via `x.norm().clamp_min(1e-7)` to remove CUDA scalar synchronization.
- Shape-bucketed Muon updates grouped by `(shape, dtype, device)`.
- Batched Newton-Schulz orthogonalization over stacked same-shape tensors.
- `muon_ns_steps` sweep from `5` to `3` to measure speed/convergence tradeoff.

## 2026-05-10 RTX PyTorch Profiler Smoke Run

The goal was to explain intermittent `tokens_per_gpu_second` and GPU utilization drops observed during `configs/rtx.yaml` training.

The profiling runs used the production RTX model shape and data pipeline, but shortened training and disabled MLflow/progress/checkpoint interval work so the active trace focused on training step execution.

Common profile settings:

```bash
training.max_steps=72
training.profiler_enabled=true
training.nvtx_enabled=true
training.profiler_wait_steps=40
training.profiler_warmup_steps=8
training.profiler_active_steps=16
training.profiler_repeat=1
training.metric_interval=100000
training.eval_interval=100000
training.max_eval_batches=1
training.checkpoint_interval=null
training.mlflow_enabled=false
training.progress_bar=false
```

Baseline Muon run:

```bash
uv run mega-trading train --config-name rtx \
  run.run_id=rtx-profiler-smoke \
  training.max_steps=72 \
  training.profiler_enabled=true \
  training.nvtx_enabled=true \
  training.profiler_wait_steps=40 \
  training.profiler_warmup_steps=8 \
  training.profiler_active_steps=16 \
  training.profiler_repeat=1 \
  training.metric_interval=100000 \
  training.eval_interval=100000 \
  training.max_eval_batches=1 \
  training.checkpoint_interval=null \
  training.mlflow_enabled=false \
  training.progress_bar=false
```

AdamW comparison:

```bash
uv run mega-trading train --config-name rtx \
  run.run_id=rtx-profiler-adamw-smoke \
  training.optimizer=adamw \
  training.max_steps=72 \
  training.profiler_enabled=true \
  training.nvtx_enabled=true \
  training.profiler_wait_steps=40 \
  training.profiler_warmup_steps=8 \
  training.profiler_active_steps=16 \
  training.profiler_repeat=1 \
  training.metric_interval=100000 \
  training.eval_interval=100000 \
  training.max_eval_batches=1 \
  training.checkpoint_interval=null \
  training.mlflow_enabled=false \
  training.progress_bar=false
```

No-sync Muon validation:

```bash
uv run mega-trading train --config-name rtx \
  run.run_id=rtx-profiler-muon-nosync \
  training.max_steps=72 \
  training.profiler_enabled=true \
  training.nvtx_enabled=true \
  training.profiler_wait_steps=40 \
  training.profiler_warmup_steps=8 \
  training.profiler_active_steps=16 \
  training.profiler_repeat=1 \
  training.metric_interval=100000 \
  training.eval_interval=100000 \
  training.max_eval_batches=1 \
  training.checkpoint_interval=null \
  training.mlflow_enabled=false \
  training.progress_bar=false
```

Batched Muon validation:

```bash
uv run mega-trading train --config-name rtx \
  run.run_id=rtx-profiler-muon-batched \
  training.max_steps=72 \
  training.profiler_enabled=true \
  training.nvtx_enabled=true \
  training.profiler_wait_steps=40 \
  training.profiler_warmup_steps=8 \
  training.profiler_active_steps=16 \
  training.profiler_repeat=1 \
  training.metric_interval=100000 \
  training.eval_interval=100000 \
  training.max_eval_batches=1 \
  training.checkpoint_interval=null \
  training.mlflow_enabled=false \
  training.progress_bar=false
```

Batched Muon steady-state validation:

```bash
uv run mega-trading train --config-name rtx \
  run.run_id=rtx-profiler-muon-batched-steady \
  training.max_steps=104 \
  training.profiler_enabled=true \
  training.nvtx_enabled=true \
  training.profiler_wait_steps=72 \
  training.profiler_warmup_steps=8 \
  training.profiler_active_steps=16 \
  training.profiler_repeat=1 \
  training.metric_interval=100000 \
  training.eval_interval=100000 \
  training.max_eval_batches=1 \
  training.checkpoint_interval=null \
  training.mlflow_enabled=false \
  training.progress_bar=false
```

Muon `ns_steps=3` validation:

```bash
uv run mega-trading train --config-name rtx \
  run.run_id=rtx-profiler-muon-batched-ns3 \
  training.muon_ns_steps=3 \
  training.max_steps=104 \
  training.profiler_enabled=true \
  training.nvtx_enabled=true \
  training.profiler_wait_steps=72 \
  training.profiler_warmup_steps=8 \
  training.profiler_active_steps=16 \
  training.profiler_repeat=1 \
  training.metric_interval=100000 \
  training.eval_interval=100000 \
  training.max_eval_batches=1 \
  training.checkpoint_interval=null \
  training.mlflow_enabled=false \
  training.progress_bar=false
```

Trace artifacts:

- `.mega-trading/data/runs/rtx-profiler-smoke/profiler/rank-0.1778468055759283466.pt.trace.json`
- `.mega-trading/data/runs/rtx-profiler-adamw-smoke/profiler/rank-0.1778468179630843731.pt.trace.json`
- `.mega-trading/data/runs/rtx-profiler-muon-nosync/profiler/rank-0.1778468553251020186.pt.trace.json`
- `.mega-trading/data/runs/rtx-profiler-muon-batched/profiler/rank-0.1778468771164019197.pt.trace.json`
- `.mega-trading/data/runs/rtx-profiler-muon-batched-steady/profiler/rank-0.1778468850029291272.pt.trace.json`
- `.mega-trading/data/runs/rtx-profiler-muon-batched-ns3/profiler/rank-0.1778469166165035259.pt.trace.json`

Open the profiler directory with TensorBoard:

```bash
uv run tensorboard --logdir .mega-trading/data/runs/rtx-profiler-smoke/profiler
```

## Summary

The utilization dips are mainly optimizer-boundary spikes from `MuonAdamW`, not data loading.

The baseline Muon profile captured active `ProfilerStep#48` through `ProfilerStep#63`. Normal micro-steps were stable around 35-37 ms, but steps `55` and `63` jumped to about 198-200 ms. Those two slow steps line up with gradient accumulation boundaries where the optimizer update actually runs.

Muon active window:

```text
active steps: 48-63
count:        16
total:        889.865 ms
mean:         55.617 ms
p50:          36.262 ms
max:          199.561 ms

step durations:
48: 28.3 ms
49: 35.9 ms
50: 35.3 ms
51: 36.3 ms
52: 36.6 ms
53: 36.2 ms
54: 37.0 ms
55: 197.6 ms
56: 28.5 ms
57: 36.2 ms
58: 34.8 ms
59: 36.0 ms
60: 38.5 ms
61: 36.5 ms
62: 36.6 ms
63: 199.6 ms
```

AdamW comparison:

```text
active steps: 48-63
count:        16
total:        600.697 ms
mean:         37.544 ms
p50:          37.329 ms
max:          42.090 ms

step durations:
48: 42.1 ms
49: 37.5 ms
50: 34.3 ms
51: 36.0 ms
52: 36.8 ms
53: 37.3 ms
54: 37.5 ms
55: 38.2 ms
56: 39.4 ms
57: 39.1 ms
58: 35.5 ms
59: 37.3 ms
60: 37.2 ms
61: 37.4 ms
62: 37.3 ms
63: 37.8 ms
```

AdamW removes the periodic 200 ms spike, so the discontinuity is not an inherent forward/backward or data-pipeline issue.

After replacing the scalar zero-norm branch with a device-side `clamp_min`, the Muon scalar synchronizations disappeared, but the optimizer-boundary spikes remained:

```text
active steps: 48-63
count:        16
total:        882.980 ms
mean:         55.186 ms
p50:          37.172 ms
max:          179.823 ms

step durations:
48: 40.9 ms
49: 36.7 ms
50: 35.7 ms
51: 37.0 ms
52: 37.1 ms
53: 37.0 ms
54: 37.2 ms
55: 179.8 ms
56: 40.9 ms
57: 36.8 ms
58: 34.7 ms
59: 37.2 ms
60: 38.9 ms
61: 37.4 ms
62: 37.1 ms
63: 178.4 ms
```

After shape-bucketing same-shaped Muon tensors, optimizer work became much less fragmented. The CPU `Optimizer.step#MuonAdamW.step` window dropped to about 9-10 ms per optimizer update. The remaining cost is mostly asynchronous GPU work that is observed at later synchronization points, so the large profiler steps move to the micro-step after an optimizer update.

First batched window:

```text
active steps: 48-63
count:        16
total:        820.095 ms
mean:         51.256 ms
p50:          36.883 ms
max:          142.693 ms

step durations:
48: 142.7 ms
49: 36.4 ms
50: 35.4 ms
51: 36.9 ms
52: 37.1 ms
53: 36.8 ms
54: 36.8 ms
55: 48.9 ms
56: 141.9 ms
57: 36.6 ms
58: 35.5 ms
59: 36.8 ms
60: 36.5 ms
61: 37.2 ms
62: 37.0 ms
63: 47.5 ms
```

Later steady-state batched window:

```text
active steps: 80-95
count:        16
total:        811.293 ms
mean:         50.706 ms
p50:          36.562 ms
max:          141.941 ms

step durations:
80: 141.3 ms
81: 36.9 ms
82: 34.4 ms
83: 36.0 ms
84: 36.5 ms
85: 36.4 ms
86: 37.1 ms
87: 47.5 ms
88: 141.9 ms
89: 36.6 ms
90: 34.4 ms
91: 36.4 ms
92: 36.6 ms
93: 36.6 ms
94: 36.5 ms
95: 46.2 ms
```

Reducing Muon's Newton-Schulz iterations from `5` to `3` lowered the remaining optimizer burst without obvious short-run loss regression:

```text
active steps: 80-95
count:        16
total:        721.149 ms
mean:         45.072 ms
p50:          35.858 ms
max:          100.900 ms

step durations:
80: 100.9 ms
81: 35.7 ms
82: 34.5 ms
83: 36.6 ms
84: 35.1 ms
85: 35.7 ms
86: 35.8 ms
87: 48.0 ms
88: 99.7 ms
89: 35.6 ms
90: 33.8 ms
91: 35.7 ms
92: 35.9 ms
93: 36.3 ms
94: 36.0 ms
95: 45.9 ms
```

## Bottleneck Details

Top Muon active-window events:

```text
Optimizer.step#MuonAdamW.step     323.645 ms total, 2 calls, 162.461 ms max
cudaStreamSynchronize             284.439 ms total, 312 calls, 7.416 ms max
aten::is_nonzero                  190.850 ms total, 280 calls, 6.827 ms max
aten::item                        190.718 ms total, 280 calls, 6.825 ms max
aten::_local_scalar_dense         190.561 ms total, 280 calls, 6.824 ms max
```

Per slow step:

```text
step 55:
  Optimizer.step#MuonAdamW.step   161.2 ms CPU
  Optimizer.step#MuonAdamW.step   154.0 ms GPU annotation
  cudaStreamSynchronize           102.7 ms
  aten::item                       96.7 ms
  aten::_local_scalar_dense        96.6 ms

step 63:
  Optimizer.step#MuonAdamW.step   162.5 ms CPU
  Optimizer.step#MuonAdamW.step   155.6 ms GPU annotation
  cudaStreamSynchronize           100.1 ms
  aten::item                       94.0 ms
  aten::_local_scalar_dense        93.9 ms
```

The `aten::item` / `_local_scalar_dense` calls come from converting CUDA tensors to Python booleans or scalars inside the optimizer. The immediate suspect is the zero-norm guard in `_muon_orthogonalize`:

```text
norm = x.norm()
if norm == 0:
    return torch.zeros_like(update)
```

For the RTX model shape, there are 140 Muon tensors. The trace shows 280 `aten::is_nonzero`/`aten::item` calls across two optimizer updates, matching one scalar synchronization per Muon tensor per update.

The no-sync Muon validation removed those scalar conversion events:

```text
event                         baseline Muon        no-sync Muon
aten::is_nonzero              190.850 ms / 280     absent from trace
aten::item                    190.718 ms / 280     absent from trace
aten::_local_scalar_dense     190.561 ms / 280     absent from trace
cudaStreamSynchronize         284.439 ms / 312     130.256 ms / 32
Optimizer.step#MuonAdamW.step 323.645 ms / 2       283.477 ms / 2
max profiler step             199.561 ms           179.823 ms
```

Remaining no-sync optimizer-window cost:

```text
Optimizer.step#MuonAdamW.step 283.477 ms total, 2 calls, 142.423 ms max
aten::matmul                  117.038 ms total, 4200 calls
aten::mm                      114.497 ms total, 4200 calls
cudaLaunchKernel              110.747 ms total, 10276 calls
aten::mul                      75.355 ms total, 4480 calls
```

The remaining bottleneck is now structural: `MuonAdamW` loops over 140 Muon tensors and launches many small Newton-Schulz matmul and elementwise operations per optimizer update.

The shape-bucketed Muon implementation reduces that fragmentation substantially:

```text
event                          no-sync Muon        batched Muon
CPU Optimizer.step             283.477 ms / 2      19.364 ms / 2
GPU Optimizer.step annotation  293.243 ms / 2      232.716 ms / 2
aten::matmul in optimizer      117.038 ms / 4200   1.845 ms / 120
aten::bmm in optimizer         not used            1.041 ms / 120
cudaLaunchKernel in optimizer  110.747 ms / 10276  7.326 ms / 2396
max profiler step              179.823 ms          142.693 ms
```

The steady-state batched window confirms this is not only a first-active-window artifact:

```text
window       ns_steps  active steps  CPU Optimizer.step  GPU Optimizer.step  max step
batched      5         48-63         19.364 ms / 2       232.716 ms / 2      142.693 ms
steady       5         80-95         18.674 ms / 2       231.747 ms / 2      141.941 ms
ns3          3         80-95         20.018 ms / 2       148.919 ms / 2      100.900 ms
```

The current remaining sawtooth is therefore GPU Muon math, not Python scalar sync and not per-tensor launch fragmentation. The long `aten::to` / `aten::copy_` rows in the step table are synchronization sites waiting for asynchronous batched Muon work from the prior optimizer update.

Short-run metric comparison at step 104:

```text
run                   ns_steps  train_loss  validation_loss  validation_top5
batched steady        5         6.594489    6.253544         0.009521
batched ns3           3         6.594965    6.254144         0.009521
```

This is only a 104-step smoke run, so it does not prove final convergence parity. It is enough to show that `ns_steps=3` is a promising speed setting to test in a longer training run.

Data loading is not the bottleneck in this profile:

```text
enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__
total: 17.117 ms across 16 active steps
mean:  about 1.1 ms per step
```

The model compute path looks steady. Regular non-boundary steps are dominated by compiled forward/backward regions and GEMM kernels, with no periodic spike comparable to the Muon optimizer boundary.

## Interpretation

With `gradient_accumulation_steps=8`, most iterations only accumulate gradients. Every accumulation boundary performs the actual optimizer update. For AdamW this boundary remains near normal step latency, but for Muon it still introduces asynchronous GPU optimizer work. The shape-bucketed implementation reduces the spike from about 178-180 ms to about 141-143 ms in the profiled windows.

This explains the observed pattern:

- Token throughput appears high on accumulation-only micro-steps.
- Throughput drops periodically on optimizer-boundary steps.
- GPU utilization can look discontinuous because Newton-Schulz optimizer work still runs in bursts at accumulation boundaries.

## Recommended Fixes

Completed first fix: keep the Muon zero-norm guard on device.

```python
norm = x.norm().clamp_min(1e-7)
x = x / norm
```

This keeps the zero-update case on device and avoids the Python `if norm == 0` CUDA sync. The validation trace confirms `aten::item`, `aten::is_nonzero`, and `_local_scalar_dense` disappear from the active window.

Completed second fix: reduce Muon Python-loop and launch overhead with shape buckets.

- Group same-shaped Muon tensors into shape buckets.
- Stack each bucket into `[bucket_size, rows, cols]` and run batched Newton-Schulz with `torch.bmm` / batched matmul.
- Keep different shapes in separate buckets so reshape/transposition stays simple and explicit.

Remaining options:

- Promote `training.muon_ns_steps=3` to an RTX experiment candidate and run a longer training comparison against `5`.
- Consider skipping Muon for small projection matrices if orthogonalization overhead outweighs benefit.
- Try mixed precision inside Muon orthogonalization if stability remains acceptable.
- Consider a Triton implementation only for bucketed Newton-Schulz if PyTorch batched matmul remains too expensive.
- Keep the AdamW comparison trace as a control whenever modifying optimizer internals.

DataLoader tuning is lower priority based on this run. It may still help at larger batch sizes or multi-worker CPU pressure, but this trace does not show data wait as the source of the current utilization sawtooth.

## Follow-Up Validation

After the next Muon change, rerun:

```bash
uv run mega-trading train --config-name rtx \
  run.run_id=rtx-profiler-muon-next \
  training.max_steps=72 \
  training.profiler_enabled=true \
  training.nvtx_enabled=true \
  training.profiler_wait_steps=40 \
  training.profiler_warmup_steps=8 \
  training.profiler_active_steps=16 \
  training.profiler_repeat=1 \
  training.metric_interval=100000 \
  training.eval_interval=100000 \
  training.max_eval_batches=1 \
  training.checkpoint_interval=null \
  training.mlflow_enabled=false \
  training.progress_bar=false
```

Success criteria:

- Max batched Muon profiler step moves materially below the current 141-143 ms range.
- GPU `Optimizer.step#MuonAdamW.step` annotation moves materially below the current 231-233 ms per two optimizer updates.
- Perplexity/loss does not regress relative to the same short profiler smoke run.
- AdamW remains a stable control with no periodic spike.
