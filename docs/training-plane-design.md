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
uv run mega-trading train-trading-foundation-model \
  --data-dir .mega-trading/public \
  --mixture public \
  --run-id public-tfm \
  --steps 100 \
  --hidden-dim 64 \
  --batch-size 16
```

Modal or Kubernetes jobs should call the same trainer with the same shard contract, only changing device, batch size, worker count, and checkpoint storage.

## Design Rules

- Keep the training surface focused on multi-stream samples.
- Do not generate text-only training artifacts in the default ingestion path.
- Keep model modules directly under `mega_trading.train`.
- Preserve run lineage through manifests.
- Prefer small, fast smoke runs that prove the contract before scaling compute.
