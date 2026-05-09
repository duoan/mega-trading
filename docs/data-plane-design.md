# Data Plane Design

## Purpose

The Data Plane turns public market data into the event/token contract consumed by the training path:

```text
normalized market_data records
  -> event-like stream rows
  -> scale-normalized token blocks
  -> decoder-only next-token training
```

No reasoning runtime, evidence corpus, or text-only tokenization path is part of the primary training data plane.

## Implemented Sources

Current real public sources:

- Yahoo/Stooq chart data for market_data history.
- SEC EDGAR company facts for future multimodal context adapters.

The current no-credential MVP uses market data only. Future adapters can add real L3/TAQ/ITCH-style trade/order-flow events and multimodal context from news, SEC filings, earnings, and macro data.

## Event Construction

`EventBuilder` reads normalized `family=market_data` records and builds one event-like row per ticker/date:

- side/action proxy from signed daily return.
- signed return bps.
- intraday range proxy.
- open gap proxy.
- log volume ratio.
- calendar/interarrival-time fields.
- source IDs from current and previous market records.

Leakage rule:

```text
event_date is derived only from current and prior market records
```

## Shard Contract

The builder packs token blocks into JSONL rows for training:

- `sequence_id`
- `ticker`
- `start_date`
- `end_date`
- `tokens`

The expected training shard is:

```text
stage=05_shards/mixture=public/tokens.jsonl
```

The matching profile is:

```text
stage=05_shards/mixture=public/tokens-profile.json
```

Any future real event feed or multimodal adapter must first map into this contract and be covered by tests before the model consumes it.

## Real Data Run

Full configured public run:

```bash
uv run mega-trading ingest --config configs/ingest-public.toml
uv run mega-trading build data.data_dir=.mega-trading/public
```

## Tests

The primary path is covered by:

- `tests/test_training.py`: event building, tokenizer determinism, model shape, small training, and stylized-fact evaluation.
- `tests/test_cli.py`: config-driven ingest plus build/train/eval CLI integration.
