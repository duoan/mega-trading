# End-to-End Platform Design

## Objective

Mega-Trading is a trading foundation model prototype. The current system learns reusable market-event representations from public market data by building scale-normalized event streams, tokenizing them, training decoder-only next-token models, and evaluating generated market behavior.

Exact microstructure modeling requires event-level L3/TAQ/ITCH-style data. The public MVP reproduces the data, tokenization, training, and evaluation mechanics with OHLCV-derived proxy events while preserving extension points for real trade/order-flow feeds.

## Current MVP

The repository implements the runnable offline training path:

- Public universe ingestion from `configs/universes/sp500.txt`.
- Stage-based raw, normalized, event, token-shard, profile, run, and eval artifacts.
- Data readiness checks and deterministic enrichment before event construction.
- `EventBuilder` for public OHLCV-to-event proxy streams.
- `MarketEventTokenizer` for side/action, return, range, gap, volume, and calendar-time tokens.
- `TradingModel`, a decoder-only Transformer trained with next-token cross entropy.
- Per-ticker time-aware train/validation split, W&B metrics, checkpoints, and manifests.
- `eval` generated rollout checks for heavy tails, volatility clustering, and return autocorrelation.

## Target Flow

```text
public market data
  -> normalized market records
  -> event-like stream builder
  -> scale-invariant tokenizer
  -> token sequence shards
  -> decoder-only trading model
  -> perplexity and stylized-fact evaluation
```

## Artifact Contracts

Primary artifacts:

- `stage=02_normalized/family=market_data/source=<source>.jsonl`
- `stage=04_corpus/mixture=<name>/events.jsonl`
- `stage=05_shards/mixture=<name>/tokens.jsonl`
- `stage=05_shards/mixture=<name>/tokens-profile.json`
- `runs/<run_id>/metrics.jsonl`
- `runs/<run_id>/checkpoint.pt`
- `evals/<run_id>/report.json`
- `manifests/build/<mixture>.json`
- `manifests/training/<run_id>.json`

The token profile is the training contract. It records the stream version, event size, vocabulary size, block size, sequence counts, ticker counts, and source mixture.

## Scaling Path

The implementation is local-first but designed to scale by replacing only the backing systems:

- Replace local object storage with shared object storage.
- Add real L3/TAQ/ITCH adapters that emit the same event contract.
- Stream token shards instead of loading full datasets.
- Use Accelerate as the local API surface before adding FSDP/DeepSpeed.
- Track data freshness, shard throughput, tokens/sec, checkpoint time, and cost per token.

## Future Multimodal Adapters

News, SEC filings, earnings, and macro data should enter as context events or side-channel tokens only after their point-in-time contracts are explicit. The first production-quality extension should preserve the same sequence-first contract.

## Success Criteria

A credible MVP should show:

- Token shard construction over a broad public universe.
- Decreasing train loss in smoke training.
- Validation perplexity emitted in metrics.
- Generated proxy returns with plausible heavy tails and low raw return autocorrelation.
- Fully reproducible runs through configs, manifests, profiles, and tests.
