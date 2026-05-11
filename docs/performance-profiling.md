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

This was an end-to-end training-step investigation. The profiler covered data wait, host-side batch construction, forward, backward, optimizer, metrics, evaluation, and checkpoint hooks. Data loading was measured rather than assumed: the smoke active trace showed `DataLoader` iteration around 17 ms across 16 active steps, or about 1.1 ms per step. That rules out DataLoader as the bottleneck in that profiler window, but it does not rule out longer-run DataLoader tail latency from page-cache misses, CPU contention, worker startup, or mmap behavior outside the sampled window.

Techniques used:

- PyTorch profiler with CPU/CUDA activities and TensorBoard trace output.
- NVTX ranges around `train/data_wait`, `train/forward`, `train/backward`, `train/optimizer`, `train/eval`, and checkpoint sections.
- AdamW control run to separate Muon-specific optimizer cost from forward/backward and data-loading cost.
- Device-side Muon norm guard via `x.norm().clamp_min(1e-7)` to remove CUDA scalar synchronization.
- Shape-bucketed Muon updates grouped by `(shape, dtype, device)`.
- Batched Newton-Schulz orthogonalization over stacked same-shape tensors.
- `muon_ns_steps` sweep from `5` to `3` to measure speed/convergence tradeoff.
- Explicit live-run DataLoader diagnostics in `metrics.json` and MLflow: `data_wait_seconds`, `train_compute_seconds`, `train_step_seconds`, `tokens_per_second_e2e`, `tokens_per_gpu_second_e2e`, `train/dataloader_wait_seconds`, and `train/dataloader_wait_fraction`.
- Optional eager NumPy preload via `training.preload_numpy_arrays`, enabled in `configs/rtx.yaml`, so prepared `.npy` token shards are loaded into RAM instead of being faulted through mmap during training.

## End-to-End Scope

The profiling target was full training-step performance, not only kernel speed. Each trace should be read as a timeline of the whole loop:

```text
DataLoader -> forward -> loss -> backward -> optimizer -> metrics/eval/checkpoint hooks
```

The current smoke-profile finding is that optimizer work dominates the sampled intermittent stalls. The data path is tracked with `train/data_wait`, the built-in `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` annotation, and live metrics emitted every recorded training row. If the user-facing live run still shows instantaneous throughput drops, compare `tokens_per_second` with `tokens_per_second_e2e` and inspect `data_wait_seconds` for spikes before changing loader settings.

Important metric semantics:

```text
tokens_per_second       = forward/backward/optimizer throughput only
tokens_per_second_e2e   = batch wait + forward/backward/optimizer throughput
data_wait_seconds       = time spent waiting for the next batch
train_compute_seconds   = time spent after batch delivery
train_step_seconds      = data_wait_seconds + train_compute_seconds plus tiny loop overhead
```

Before these fields were added, a drop in `tokens_per_second` alone could not prove a DataLoader bottleneck because that timer started after `batch = next(iterator)`. A DataLoader stall should now show up as a spike in `data_wait_seconds` and a larger drop in `tokens_per_second_e2e` than in compute-only `tokens_per_second`.

MLflow receives the same signals through the training tracker payload:

```text
train/data_wait_seconds
train/dataloader_wait_seconds
train/dataloader_wait_fraction
train/compute_seconds
train/step_seconds
train/tokens_per_second_e2e
train/tokens_per_gpu_second_e2e
```

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

Data loading is part of the end-to-end profile. In this sampled smoke window it was small:

```text
enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__
total: 17.117 ms across 16 active steps
mean:  about 1.1 ms per step
```

This is why the first optimization pass targeted Muon instead of DataLoader workers or pinned memory. Those settings may still matter for longer live runs, especially if `data_wait_seconds` develops spikes that were not present in the profiler window.

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

DataLoader tuning is evidence-gated. If a live run shows `data_wait_seconds` spikes aligned with `tokens_per_second_e2e` drops, prioritize loader work next. Do not enable `num_workers>0` blindly yet: the current `NumpyTickerTimeDataset` is an `IterableDataset`, so multi-worker loading needs explicit worker sharding to avoid duplicated samples.

The first DataLoader-side mitigation is now available:

```yaml
training:
  preload_numpy_arrays: true
```

When enabled, `NumpyTickerTimeDataset` reads prepared NumPy token shards into memory during dataset construction and shares the same in-memory array cache between train and validation datasets. This avoids relying on mmap page faults during the training loop. `configs/rtx.yaml` enables it by default; memory-constrained configs keep it disabled.

### DataLoader Preload A/B

Command shape:

```bash
uv run mega-trading train --config-name rtx \
  run.run_id=rtx-dataloader-mmap-1100 \
  training.preload_numpy_arrays=false \
  training.optimizer=adamw \
  training.max_steps=1100 \
  training.metric_interval=1 \
  training.eval_interval=100000 \
  training.checkpoint_interval=null \
  training.mlflow_enabled=false \
  training.progress_bar=false \
  training.profiler_enabled=false \
  training.nvtx_enabled=false

uv run mega-trading train --config-name rtx \
  run.run_id=rtx-dataloader-preload-1100 \
  training.preload_numpy_arrays=true \
  training.optimizer=adamw \
  training.max_steps=1100 \
  training.metric_interval=1 \
  training.eval_interval=100000 \
  training.checkpoint_interval=null \
  training.mlflow_enabled=false \
  training.progress_bar=false \
  training.profiler_enabled=false \
  training.nvtx_enabled=false
```

The RTX token shards are about 59.1 GiB on disk. The test machine had about 163 GiB available RAM before preload, so this was safe for the local RTX server.

Measured after warmup (`step > 10`):

```text
run                 data_wait mean  p50     p95     p99     max     e2e tok/s mean  p50
mmap                0.966 ms        0.960   1.002   1.105   1.569   209,155.8       210,336.7
preload             0.607 ms        0.603   0.637   0.663   0.770   223,333.9       228,658.2
```

Preload reduced mean DataLoader wait by about 37%, p99 by about 40%, and max observed DataLoader wait by about 51% in this 1100-step controlled run. That validates eager preload as useful for smoothing batch delivery.

The largest instantaneous `tokens_per_second_e2e` drop in this controlled run was not DataLoader-bound:

```text
run       slowest step  data_wait  compute      e2e tok/s
mmap      642           0.968 ms   177.113 ms   22,999.2
preload   642           0.596 ms   172.671 ms   23,638.2
```

So preload improves the DataLoader component, but if a live run still has a single-step tok/s collapse with flat `data_wait_seconds`, the next bottleneck is compute/optimizer/compile rather than file loading. If the live run shows `data_wait_seconds` spikes aligned with the collapse, then the DataLoader remains the active bottleneck.

### MLflow Throughput Sawtooth

The `make rtx` run still showed throughput sawtooth after preload. The live metrics confirmed the pattern was not a file-loading spike:

```text
slow rows: step % 8 == 0
gradient_accumulation_steps: 8
optimizer: Muon
data_wait_seconds: flat around the same level as neighboring rows
train_compute_seconds: higher on optimizer-boundary rows
```

The previous MLflow chart used per-step throughput for the logged step. With `metric_interval=10` and `gradient_accumulation_steps=8`, the logged samples alternated between accumulation-only steps and optimizer-boundary steps, producing a persistent chart sawtooth.

The training loop now records two throughput families:

```text
train/tokens_per_second              window average used for default monitoring
train/tokens_per_gpu_second          window average used for default monitoring
train/tokens_per_second_e2e          window average including DataLoader wait
train/step_tokens_per_second         raw single logged step throughput
train/step_tokens_per_gpu_second     raw single logged step throughput
train/step_tokens_per_second_e2e     raw single logged step throughput including DataLoader wait
```

`configs/rtx.yaml` used `metric_interval: 40` for the `batch_size=8`, `gradient_accumulation_steps=8` setup. After increasing the RTX micro-batch to `batch_size=64`, the equivalent token-span window is `metric_interval: 5` with `gradient_accumulation_steps=1`. Each logged throughput window contains a stable number of optimizer boundaries, so the default MLflow throughput charts track sustained training throughput rather than sampling artifacts.

Smoke validation:

```text
run                         metric_interval  window min/max ratio  raw min/max ratio
rtx-window-metrics-smoke     10               0.376                 0.295
rtx-window-metrics-aligned   40               0.780                 0.997
```

The first aligned window includes startup/warmup overhead. Subsequent aligned windows were stable around 201k-202k tokens/GPU/s in the smoke run. Raw step throughput remains available for diagnosing real optimizer-boundary cost.

### GPU Utilization Headroom

The live RTX run still showed GPU-utilization dips even after DataLoader preload. The process was using about 13.7 GiB out of 95.9 GiB of GPU memory with `batch_size=8`, while CPU/host activity was high. That indicates the Blackwell GPU is under-filled by the micro-batch.

`configs/rtx.yaml` now keeps the effective batch unchanged while increasing the GPU work per micro-step:

```text
before: batch_size=8,  gradient_accumulation_steps=8  -> effective 64 sequences/update
after:  batch_size=64, gradient_accumulation_steps=1  -> effective 64 sequences/update
```

Because `max_steps` is counted in micro-steps, the schedule and training length need token-equivalent scaling. Validation and checkpointing are intentionally kept less frequent than strict token equivalence because they create visible GPU idle periods:

```text
max_steps:            20000 -> 2500
lr_warmup_steps:       1000 -> 125
eval_interval:          500 -> 250
checkpoint_interval:   1000 -> 500
metric_interval:         40 -> 10
max_eval_batches:        64 -> 8
```

This keeps total token exposure, optimizer updates, and warmup position approximately equivalent to the old effective-batch-64 run. Validation uses fewer batches per pass and runs less often because the first compiled validation pass can create a large cold-start stall. Checkpoints are less frequent because they protect recoverability, not overfit detection, and full model serialization can idle the GPU.

The training loop now also emits:

```text
train/eval_seconds
train/checkpoint_seconds
```

Use these with GPU-utilization drops to distinguish validation pauses from checkpoint serialization pauses. The current already-running `make rtx` process will not pick this up; restart the run to use the new config.

### TorchDynamo Recompile From Torchinfo

The RTX path uses `torch.compile`. `torchinfo.summary()` registers forward hooks to inspect modules, and those hooks include Python object identity checks such as `id(module)`. Running torchinfo against a compiled model caused TorchDynamo recompilation warnings like:

```text
torch._dynamo hit config.recompile_limit
function: 'hook' (.../torchinfo/torchinfo.py)
last reason: tensor 'outputs' dtype mismatch
```

The model summary is only an MLflow convenience artifact, not part of training. The training path now logs torchinfo summary before `torch.compile` is applied. This keeps the MLflow model summary artifact while preventing torchinfo hooks from entering the compiled training graph.

### GPU Drops After Compile Warmup

The latest `rtx` metrics show a different bottleneck from the earlier optimizer and TorchDynamo issues:

- Step 1 is a one-time `torch.compile`/warmup outlier.
- Early steady-state windows show `eval_seconds=0` and `checkpoint_seconds=0`, so the repeated short GPU-utilization drops before scheduled validation/checkpointing are not primarily validation or serialization.
- `metric_window_data_wait_seconds` is about `0.85s` per 10 steps while `metric_window_compute_seconds` is about `0.67s` per 10 steps, so the GPU is waiting on the synchronous DataLoader path.
- Raising workers from 4 to 8 did not materially change the repeated drops; the remaining `next(iterator)` time likely includes synchronous pinned host-to-device transfer and/or waiting for the previous CUDA work at the step boundary.

The RTX config now enables worker-backed loading:

```yaml
training:
  preload_numpy_arrays: true
  dataloader_num_workers: 8
  dataloader_prefetch_factor: 4
  dataloader_pin_memory: true
  dataloader_persistent_workers: true
  dataloader_non_blocking: true
```

The iterable NumPy dataset is worker-sharded, so multiple DataLoader workers split rows instead of duplicating them. This overlaps Python slicing/tensor creation and pinned, non-blocking host-to-device transfer with GPU compute. Re-run `make rtx` after this change and watch `train/dataloader_wait_seconds` and `train/dataloader_wait_fraction`; they should fall if the GPU drops were from CPU-side batch production or blocking batch transfer.

Metrics logging is split into two parts:

- GPU metric materialization, such as `loss.detach().cpu()` and top-k accuracy, still synchronizes because the values must leave the GPU.
- MLflow logging and `metrics.json` writes now run through a background single-thread queue and are flushed before final checkpoint/manifest creation. This removes monitoring I/O from the hot training step without dropping metrics.

Periodic checkpointing follows the same pattern:

- The training thread still snapshots model/optimizer/scheduler state first so the background writer never reads live tensors while training continues.
- The expensive `torch.save`/filesystem serialization for `checkpoints/step-*.pt` runs in a background single-thread queue.
- The final `checkpoint.pt` remains synchronous so the manifest only references a fully written terminal checkpoint.

New live-run check:

```bash
python3 - <<'PY'
import json, pathlib
path = pathlib.Path(".mega-trading/data/runs/<run-id>/metrics.json")
rows = json.loads(path.read_text())["metrics"]
for row in rows[-20:]:
    print(
        row["step"],
        "data_ms=", round(row.get("data_wait_seconds", 0.0) * 1000, 2),
        "compute_ms=", round(row.get("train_compute_seconds", 0.0) * 1000, 2),
        "tok_s=", round(row.get("tokens_per_second", 0.0), 1),
        "e2e_tok_s=", round(row.get("tokens_per_second_e2e", 0.0), 1),
    )
PY
```

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
