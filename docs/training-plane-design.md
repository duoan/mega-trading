# Training Plane Design

## Purpose

The Training Plane turns versioned event-token financial streams into model capability. It is the module that proves the infra-model co-design loop: data contracts, tokenization, model architecture, training metrics, checkpointing, and run manifests all agree on the same sequence schema.

## Training Workload

The primary workload is `TradingModel`:

- event-like streams derived from public market data for the no-credential MVP.
- scale-invariant token buckets for side/action proxy, return, range, gap, volume, and calendar time.
- fixed-length autoregressive token blocks.
- decoder-only Transformer trained with next-token cross entropy.
- evaluation with loss, perplexity, top-k accuracy, and stylized facts.

## Inputs

Each token shard row should include:

- `sequence_id`
- `ticker`
- `start_date`
- `end_date`
- `tokens`

The profile artifact `stage=05_shards/mixture=<name>/tokens-profile.json` records the stream contract, sequence counts, block size, event size, and vocabulary size.

## Outputs

Each training run writes:

- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/checkpoint.pt`
- `manifests/training/<run_id>.json`

The manifest records the shard path, token profile, stream contract, requested/effective device and precision, training backend, and sequence counts.

## Metrics

Training metrics should make both model progress and infrastructure efficiency visible:

- next-token loss.
- perplexity.
- top-1 token accuracy.
- top-5 token accuracy.
- validation loss/perplexity.
- tokens/sec.
- checkpoint path and manifest path.
- W&B run ID and URL when tracking is enabled.
- main-process progress bar with current step, loss, perplexity, accuracy, and tokens/sec.

The local trainer writes the same metrics to `runs/<run_id>/metrics.jsonl` and to Weights & Biases. The default W&B mode is `offline`, so reviewer runs do not require credentials and can be synced later with `wandb sync`. The trainer uses each ticker's earlier sequences for training and later sequences for validation. Future runs should add data-loader wait time, checkpoint duration, GPU utilization, and distributed training efficiency metrics.

## Local And GPU Paths

Training uses Hydra config from `configs/default.yaml` and Hugging Face Accelerate for model, optimizer, dataloader, autocast, backward, and checkpoint-state handling. By default `training.device=auto` selects `cuda` first, then Apple Silicon `mps`, then `cpu`. `training.precision=auto` enables mixed precision on CUDA and keeps MPS/CPU in fp32.

Step progress is visible by default through a main-process progress bar. Disable it for log-only CI runs with `training.progress_bar=false`.

W&B tracking is enabled by default in offline mode:

```bash
uv run mega-trading train training.wandb_mode=offline
```

To stream metrics to the hosted W&B dashboard:

```bash
wandb login
uv run mega-trading train training.wandb_mode=online
```

Local smoke training can pin CPU for deterministic reviewer runs:

```bash
uv run mega-trading train \
  run.run_id=public \
  training.max_steps=200 \
  training.device=cpu \
  model.hidden_dim=64 \
  training.batch_size=32
```

A GPU run can rely on automatic device and precision selection:

```bash
uv run mega-trading train \
  run.run_id=public-gpu \
  training.device=auto \
  training.precision=auto \
  training.batch_size=128
```

Ablations should use Hydra overrides so runs remain reproducible and easy to compare:

```bash
uv run mega-trading train model.hidden_dim=32 training.learning_rate=0.001
uv run mega-trading train model.hidden_dim=64 training.learning_rate=0.0005
```

GPU jobs should call the same trainer with the same shard contract, only changing device, batch size, worker count, and checkpoint storage.

## Design Rules

- Keep the training surface focused on event-token streams.
- Do not generate text-only training artifacts in the default ingestion path.
- Keep model modules under `mega_trading`.
- Preserve run lineage through manifests.
- Prefer small, fast smoke runs that prove the contract before scaling compute.
