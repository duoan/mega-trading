# End-to-End Platform Design

## Objective

Mega-Trading is a narrow reproduction-oriented market microstructure foundation model prototype. The active system learns from paper-style order-flow event features: action, side, relative price, price depth, relative size, and interarrival time.

## Current MVP

- Config-driven ingestion for deterministic fixtures and `hf_ohlcv_1m` public smoke data.
- Normalized `order_flow` records with a single paper feature contract.
- Data readiness checks over order-flow records.
- `EventBuilder` for event corpora and token shards.
- `MarketEventTokenizer` for fitted composite event tokens.
- `TradingModel`, a Llama 3-style decoder trained with next-token cross entropy.
- Per-ticker time-aware train/validation split, metrics, checkpoints, and manifests.
- `eval` checks generated-vs-real price-depth feature distributions.

## Target Flow

```text
order-flow source rows
  -> mid-price estimation
  -> scale-invariant order_flow records
  -> fitted composite tokenizer
  -> token sequence shards
  -> Llama 3-style trading model
  -> generated feature-distribution evaluation
```

## Artifact Contracts

- `stage=02_normalized/family=order_flow/source=<source>.jsonl`
- `stage=04_corpus/mixture=<name>/events.jsonl`
- `stage=05_shards/mixture=<name>/tokenizer.json`
- `stage=05_shards/mixture=<name>/tokens.jsonl`
- `stage=05_shards/mixture=<name>/tokens-profile.json`
- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/checkpoint.pt`
- `evals/<run_id>/report.json`

The token profile and fitted tokenizer are the training contract. They record the stream version, feature order, event size, vocabulary size, binning method, bucket counts, block size, sequence counts, ticker counts, and source mixture.
