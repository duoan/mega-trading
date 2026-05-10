# Data Plane Design

## Purpose

The Data Plane now has one job: produce paper-style order-flow events for next-token training.

```text
raw source rows
  -> mid-price estimation
  -> scale-invariant order_flow events
  -> fitted composite tokenizer
  -> autoregressive token blocks
```

## Implemented Sources

- `fixture`: deterministic local order-flow events for smoke tests.
- `binance_trades`: Binance public spot trades downloaded from `data.binance.vision`, mapped into event-level execution proxies.
- `hf_ohlcv_1m`: Hugging Face `mito0o852/OHLCV-1m` loaded through `datasets.load_dataset`, then mapped into the same order-flow feature contract for public no-credential runs.

No non-order-flow data family is part of the active data plane.

## Event Contract

`stage=02_normalized/family=order_flow/source=<source>.jsonl` contains:

- `event_id`
- `ticker`
- `timestamp`
- `date`
- `action`
- `side`
- `midprice`
- `relative_price_bps`
- `price_depth_bps`
- `size`
- `interarrival_seconds`
- `provider`
- `source_ids`
- optional `midprice_return_bps` for diagnostics only

The model-visible feature contract is exactly:

```text
action, side, relative_price, price_depth, size, time
```

## Mid-Price Estimation

The paper defines mid-price as the midpoint of the best bid and ask. For real L3 data, the adapter should compute:

```text
midprice = (best_bid + best_ask) / 2
```

The public `binance_trades` adapter has no best bid/ask stream in the monthly trade files, so it uses the previous trade price as a causal midpoint proxy. The public `hf_ohlcv_1m` adapter has no order book either, so it uses a conservative bar proxy:

```text
estimated_midprice = (minute_high + minute_low) / 2
```

and falls back to close only when high/low are unavailable.

## Scale-Invariant Features

The normalized event contract stores features in scale-stable units:

- `relative_price_bps = 10000 * log(order_price / midprice)`.
- `price_depth_bps = abs(relative_price_bps)`.
- `size = current_volume / causal_median(previous_volume)`.
- `interarrival_seconds = timestamp - previous_ticker_timestamp`.

For the Binance trades proxy, `order_price` is the execution price, side is inferred from `isBuyerMaker`, and action is `delete` because an execution removes resting maker liquidity. For the OHLCV proxy, `order_price` is the minute close and the volume baseline is computed only from previous rows for that ticker to avoid future leakage.

## Build Artifacts

```text
stage=04_corpus/mixture=<name>/events.jsonl
stage=05_shards/mixture=<name>/tokenizer.json
stage=05_shards/mixture=<name>/tokens.jsonl
stage=05_shards/mixture=<name>/tokens-profile.json
```

`tokenizer.json` stores fitted bin edges for relative price, price depth, log relative size, and interarrival time.

## Run

```bash
uv run mega-trading ingest --config configs/ingest-hf-ohlcv-1m.toml
uv run mega-trading build data.data_dir=.mega-trading/hf-1m data.source=hf_ohlcv_1m
```
