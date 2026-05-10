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

No non-order-flow data family is part of the active data plane.

## Binance Preparation

Binance preparation is staged for server-scale runs:

```text
binance-datatool aria2 download
  -> stage=01_raw/source=binance_trades/data/spot/<freq>/trades/<symbol>/*.zip
  -> multi-process parse/convert from cached ZIPs
  -> partitioned NumPy shards
```

The raw cache keeps download progress visible on disk and allows retries without re-downloading. `binance-datatool` lists the remote archive and downloads existing ZIPs with aria2, while Mega-Trading keeps the date-window selection and downstream order-flow conversion. Missing symbol/month files are skipped because some Binance spot pairs launch late or migrate; a symbol with no files in the configured window is treated as a configuration error. Processing uses all CPU cores when `process_workers = 0`.

## Event Contract

In-memory order-flow events contain:

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

The public `binance_trades` adapter has no best bid/ask stream in the monthly trade files, so it uses the previous trade price as a causal midpoint proxy.

## Scale-Invariant Features

The normalized event contract stores features in scale-stable units:

- `relative_price_bps = 10000 * log(order_price / midprice)`.
- `price_depth_bps = abs(relative_price_bps)`.
- `size = current_volume / causal_median(previous_volume)`.
- `interarrival_seconds = timestamp - previous_ticker_timestamp`.

For the Binance trades proxy, `order_price` is the execution price, side is inferred from `isBuyerMaker`, and action is `delete` because an execution removes resting maker liquidity.

## Build Artifacts

```text
stage=05_shards/mixture=<name>/tokenizer.json
stage=05_shards/mixture=<name>/tokens-profile.json
stage=05_shards/mixture=<name>/tokens-numpy.json
stage=05_shards/mixture=<name>/numpy/part-*/tokens.npy
stage=05_shards/mixture=<name>/numpy/part-*/ticker_ids.npy
```

`tokenizer.json` stores fitted bin edges for relative price, price depth, log relative size, and interarrival time.
`tokens-numpy.json` also stores per-ticker chronological split counts. The order is train, validation, then backtest, so the backtest rows are held-out future windows for each ticker.

## Run

```bash
make local
```
