# Training Plane Design

## Purpose

The Training Plane turns versioned multi-stream financial samples into model capability. It is the module that proves the infra-model co-design loop: data contracts, model architecture, training metrics, checkpointing, and run manifests all agree on the same sample schema.

## Training Workload

The primary workload is `TradingFoundationModel`:

- price-window encoder over trailing adjusted-close returns and levels.
- fundamental encoder over point-in-time SEC facts.
- text/evidence encoder over evidence tokens visible at `as_of_time`.
- cross-attention fusion block.
- prediction heads for forward-return and risk buckets.

The training path intentionally consumes `stage=05_shards/mixture=<name>/samples.jsonl`. It does not consume text-only training files.

## Inputs

Each stream shard row should include:

- `sample_id`
- `ticker`
- `as_of_time`
- `price_returns`
- `price_levels`
- `fundamental_values`
- `evidence_token_ids`
- `return_label`
- `risk_label`
- `source_ids`
- `evidence_ids`

Labels must be generated from windows that begin strictly after `as_of_time`.

## Outputs

Each training run writes:

- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/checkpoint.pt`
- `manifests/runs/<run_id>-trading-foundation-model.json`

The manifest records the shard path, stream contract, config hash, device, and step count.

## Metrics

Training metrics should make both model progress and infrastructure efficiency visible:

- loss.
- return-bucket accuracy.
- risk-bucket accuracy.
- examples/sec.
- checkpoint path and manifest path.

Future runs should add calibration, rank correlation, data-loader wait time, checkpoint duration, and GPU utilization.

## Local And GPU Paths

Local CPU training is the reviewer-friendly smoke path:

```bash
uv run mega-trading train \
  run.run_id=public-tfm \
  training.max_steps=100 \
  model.hidden_dim=64 \
  training.batch_size=16
```

Training config lives in `configs/train/default.yaml`. Ablations should use Hydra overrides so runs remain reproducible and easy to compare:

```bash
uv run mega-trading train model.hidden_dim=32 training.learning_rate=0.001
uv run mega-trading train model.hidden_dim=64 training.learning_rate=0.0005
```

The default ablation suite lives in `configs/ablation/public.yaml`:

```bash
uv run mega-trading ablate
```

It compares price-only, fundamentals-only, evidence-only, price+fundamentals, and all-modality runs, then writes `.mega-trading/public/reports/ablation-summary.json`.

For a faster local smoke run, override the shared run settings:

```bash
uv run mega-trading ablate 'base_overrides=["training.max_steps=1","training.batch_size=2","model.hidden_dim=8"]'
```

Modal or Kubernetes jobs should call the same trainer with the same shard contract, only changing device, batch size, worker count, and checkpoint storage.

## Design Rules

- Keep the training surface focused on multi-stream samples.
- Do not generate text-only training artifacts in the default ingestion path.
- Keep model modules directly under `mega_trading.train`.
- Preserve run lineage through manifests.
- Prefer small, fast smoke runs that prove the contract before scaling compute.
