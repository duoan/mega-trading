# Training Configs

The active path is intentionally small:

- `configs/binance-local.yaml`: local Mac training on a modest multi-symbol Binance slice.
- `configs/binance-modal-prep.yaml`: local preprocessing config for the larger Modal dataset.
- `configs/modal-binance.yaml`: Modal GPU training config for uploaded Binance NumPy shards.
- `configs/default.yaml`: CLI fallback and fixture/demo defaults.

The matching source TOML files are:

- `configs/ingest-binance-local.toml`
- `configs/ingest-binance-modal-prep.toml`
- `configs/ingest-demo.toml`

## Local

```bash
make local
```

This downloads two days of public trades for BTC, ETH, BNB, and SOL, converts events in memory, writes partitioned NumPy shards, and runs a 300-step local train by default. Use `uv run python scripts/run_pipeline.py local --local-steps 50` when you only need a quick wiring check.

## Remote

```bash
make remote
```

This prepares `.mega-trading/binance-modal` locally with mixture `binance_public`, uploads the prepared `stage=05_shards` directory to the `mega-trading-artifacts` Modal Volume, syncs W&B credentials, and launches `modal-binance` training on Modal.

Modal training mounts the uploaded data at `/data/binance-trades`. The checked-in Modal Binance model is about 41M parameters; under the `tokens ~= 20 * params` rule of thumb, it wants roughly 0.8B training tokens.
