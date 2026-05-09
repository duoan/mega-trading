# Training Plane Design

## Purpose

The Training Plane turns versioned order-flow token streams into model capability. The key contract is that data processing, tokenization, model input, training metrics, checkpoints, and evaluation all agree on the same paper-style sequence schema.

## Training Workload

The primary workload is `TradingModel`:

- paper-style event streams with action, side, relative price, price depth, relative size, and interarrival time.
- fitted composite event tokens, one token per event.
- fixed-length autoregressive token blocks.
- Llama 3-style decoder trained with next-token cross entropy.
- evaluation with loss, perplexity, top-k accuracy, and generated event-feature distributions.

## Inputs

Each token shard row includes:

- `sequence_id`
- `ticker`
- `start_time`
- `end_time`
- `tokens`

The profile artifact records the stream contract, feature order, sequence counts, block size, event size, vocabulary size, binning method, and tokenizer path.

## Outputs

Each training run writes:

- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/checkpoint.pt`
- `manifests/training/<run_id>.json`

## Metrics

Training metrics include next-token loss, perplexity, top-1/top-5 token accuracy, validation loss/perplexity, tokens/sec, checkpoint path, and manifest path.
