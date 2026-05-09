# Mega-Trading

Mega-Trading is a reproduction-oriented trading foundation model prototype. The active path is now intentionally narrow: order-flow style events are represented with the paper-aligned feature contract, tokenized as one composite token per event, trained with a Llama 3-style decoder, and evaluated on generated feature distributions.

The target paper uses participant-observable trade-flow messages with action, side, price depth, volume, and interarrival time. This repo keeps that contract as the only modeling interface. The checked-in public source, `hf_ohlcv_1m`, is a minute-OHLCV adapter that maps bars into this paper-style schema for smoke runs; it is not treated as a separate market-data modeling path.

## Quickstart

```bash
uv sync
make test
make demo
```

`make demo` is the no-network order-flow fixture path. The public minute-bar smoke path is:

```bash
uv run mega-trading ingest --config configs/ingest-hf-ohlcv-1m.toml
uv run mega-trading build data.data_dir=.mega-trading/hf-1m data.source=hf_ohlcv_1m
uv run mega-trading train data.data_dir=.mega-trading/hf-1m run.run_id=hf-1m training.max_steps=100
uv run mega-trading eval data.data_dir=.mega-trading/hf-1m --run-id hf-1m
```

Training uses Hydra config from `configs/default.yaml`, Hugging Face Accelerate for device placement and mixed precision, a main-process progress bar, and Weights & Biases for metric tracking. By default W&B runs in offline mode under `runs/<run_id>/wandb`.

## Paper Feature Contract

Normalized records live at:

```text
stage=02_normalized/family=order_flow/source=<source>.jsonl
```

Each event contains only the paper-style feature fields:

- `action`: `add` or `delete`.
- `side`: `buy` or `sell`.
- `midprice`: estimated midpoint price; real L3 adapters should use `(best_bid + best_ask) / 2`.
- `relative_price_bps`: `10000 * log(order_price / midprice)`.
- `price_depth_bps`: `abs(relative_price_bps)`.
- `size`: scale-invariant relative size, normalized by a causal ticker-level volume baseline.
- `interarrival_seconds`: elapsed time since the previous event for the ticker.

The build stage writes `events.jsonl`, `tokenizer.json`, `tokens.jsonl`, and `tokens-profile.json`. `tokenizer.json` stores fitted quantile or histogram bins and the composite vocabulary metadata.

## Documents

- [End-to-End Platform Design](docs/end-to-end-platform-design.md)
- [Model Design](docs/model-design.md)
- [Data Plane Design](docs/data-plane-design.md)
- [Training Plane Design](docs/training-plane-design.md)
- [Project Structure](docs/project-structure.md)
- [Training Systems Alignment](docs/training-systems-alignment.md)

## Goal

The submission demonstrates the infra-model contract for a market microstructure foundation model: deterministic ingestion, data quality, fitted event tokenization, autoregressive shards, Llama 3-style training, checkpointing, and generated event-feature evaluation.
