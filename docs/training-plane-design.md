# Training Plane Design

## Purpose

The Training Plane turns versioned multi-stream financial samples into model capability. It is the module that proves the infra-model co-design loop: data contracts, model architecture, training metrics, checkpointing, and run manifests all agree on the same sample schema.

## Training Workload

The primary workload is `TradingFoundationModel`:

- market_data-window encoder over trailing adjusted-close returns and levels.
- news encoder over precomputed financial-news embedding features.
- SEC filing encoder over filing features visible at `as_of_time`.
- earnings encoder over earnings-event features visible at `as_of_time`.
- macro encoder over macro and regime features.
- cross-attention fusion block.
- prediction heads for forward-return and risk buckets.

The training path intentionally consumes `stage=05_shards/mixture=<name>/samples.jsonl`. It does not consume text-only training files.

## Inputs

Each stream shard row should include:

- `sample_id`
- `ticker`
- `as_of_time`
- `market_returns`
- `market_levels`
- `news_embeddings`
- `sec_filing_features`
- `earnings_features`
- `macro_features`
- `return_label`
- `risk_label`
- `source_ids`

Labels must be generated from windows that begin strictly after `as_of_time`.

## Outputs

Each training run writes:

- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/checkpoint.pt`
- `manifests/runs/<run_id>-trading-foundation-model.json`

The manifest records the shard path, stream contract, config hash, requested/effective device, requested/effective precision, training backend, and step count.

## Metrics

Training metrics should make both model progress and infrastructure efficiency visible:

- loss.
- return-bucket accuracy.
- risk-bucket accuracy.
- validation loss.
- validation return-bucket accuracy.
- validation risk-bucket accuracy.
- examples/sec.
- checkpoint path and manifest path.
- W&B run ID and URL when tracking is enabled.

The local trainer writes the same metrics to `runs/<run_id>/metrics.jsonl` and to Weights & Biases. The default W&B mode is `offline`, so reviewer runs do not require credentials and can be synced later with `wandb sync`. The trainer uses a time-ordered validation tail from the sample shard, so validation metrics measure later `as_of_time` samples instead of a random split that can hide temporal leakage. Future runs should add calibration, rank correlation, data-loader wait time, checkpoint duration, and GPU utilization.

## Local And GPU Paths

Training uses Hydra config from `configs/train/default.yaml` and Hugging Face Accelerate for model, optimizer, dataloader, autocast, backward, and checkpoint-state handling. By default `training.device=auto` selects `cuda` first, then Apple Silicon `mps`, then `cpu`. `training.precision=auto` enables mixed precision on CUDA and keeps MPS/CPU in fp32.

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
  run.run_id=public-tfm \
  training.max_steps=200 \
  training.device=cpu \
  model.hidden_dim=64 \
  training.batch_size=32
```

A GPU run can rely on automatic device and precision selection:

```bash
uv run mega-trading train \
  run.run_id=public-tfm-gpu \
  training.device=auto \
  training.precision=auto \
  training.batch_size=128
```

Ablations should use Hydra overrides so runs remain reproducible and easy to compare:

```bash
uv run mega-trading train model.hidden_dim=32 training.learning_rate=0.001
uv run mega-trading train model.hidden_dim=64 training.learning_rate=0.0005
```

The default ablation suite lives in `configs/ablation/public.yaml`:

```bash
uv run mega-trading ablate
```

It compares market_data-only, sec_filings-only, market_data+sec_filings, and all-modality runs, then writes `.mega-trading/public/reports/ablation-summary.json`.

For a faster local smoke run, override the shared run settings:

```bash
uv run mega-trading ablate 'base_overrides=["training.max_steps=1","training.batch_size=2","model.hidden_dim=8"]'
```

GPU jobs should call the same trainer with the same shard contract, only changing device, batch size, worker count, and checkpoint storage.

## Design Rules

- Keep the training surface focused on multi-stream samples.
- Do not generate text-only training artifacts in the default ingestion path.
- Keep model modules directly under `mega_trading.train`.
- Preserve run lineage through manifests.
- Prefer small, fast smoke runs that prove the contract before scaling compute.
