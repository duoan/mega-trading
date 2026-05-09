# Data Plane Design

## Purpose

The Data Plane turns public market data into the exact feature contract consumed by `TradingFoundationModel`. The code path is intentionally narrow:

```text
normalized records
  -> point-in-time feature samples
  -> numeric feature shards
  -> offline training
```

No reasoning runtime, evidence corpus, or text-only tokenization path is part of the training data plane.

## Implemented Sources

Current real public sources:

- Yahoo chart data for market_data history.
- SEC EDGAR company facts for filing-style structured features.

Target modality slots already exist in the feature contract:

- `news_window` from normalized `family=news` records.
- `macro_window` from normalized `family=macro` records.

If a target source is not present, the sample builder writes an empty window. The shard builder then emits an empty numeric vector, and the dataset pads it to the configured size. Missing data is explicit; it is not replaced by evidence text or mock tokens.

## Feature Construction

`MultiStreamSampleBuilder` reads normalized records and builds one point-in-time sample per ticker/date label:

- `market_data_window`: trailing market_data records ending at `as_of_time`.
- `news_window`: normalized news events with `timestamp <= as_of_time`.
- `sec_filing_window`: SEC company facts with `as_of_time <= sample.as_of_time`.
- `macro_window`: normalized macro records with `timestamp <= as_of_time`.
- `labels`: forward-return and risk labels generated from future market_data.
- `source_ids`: lineage from every included feature record.

Leakage rule:

```text
feature_time <= sample.as_of_time < label_window_start
```

## Shard Contract

`StreamShardBuilder` packs samples into numeric JSONL rows for training:

- `sample_id`
- `ticker`
- `as_of_time`
- `market_returns`
- `market_levels`
- `news_embeddings`
- `sec_filing_features`
- `macro_features`
- `return_label`
- `risk_label`
- `forward_return`
- `source_ids`

The trainer reads only this shard contract. Any future feature must first be added here and covered by tests before the model consumes it.

## Real Data Run

Small public-data smoke run:

```bash
uv run mega-trading ingest-public \
  --tickers AAPL,MSFT \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --out .mega-trading/public-smoke \
  --sec-user-agent "Mega-Trading your-email@example.com"
```

Full configured run:

```bash
uv run mega-trading ingest --config configs/ingest-public.toml
```

Both commands run:

```text
ingest -> quality -> enrichment -> samples -> shards
```

The expected training shard is:

```text
stage=05_shards/mixture=public/samples.jsonl
```

## Tests

The feature construction path is covered by:

- `tests/test_samples.py`: point-in-time sample windows and leakage filtering.
- `tests/test_shards.py`: numeric feature shard packing.
- `tests/test_cli.py`: config-driven ingest and training CLI integration.
