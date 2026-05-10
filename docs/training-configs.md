# Training Configs

The active path is intentionally small:

- `configs/binance-local.yaml`: local Mac training on a modest multi-symbol Binance slice.
- `configs/binance-modal-prep.yaml`: local preprocessing config for the larger Modal dataset.
- `configs/server-rtx6000.yaml`: single-server RTX 6000 Pro CUDA training config.
- `configs/modal-binance.yaml`: Modal GPU training config for uploaded Binance NumPy shards.
- `configs/default.yaml`: CLI fallback and fixture/demo defaults.

The matching source TOML files are:

- `configs/ingest-binance-local.toml`
- `configs/ingest-binance-modal-prep.toml`
- `configs/ingest-demo.toml`

The remote/server Binance ingest uses `configs/binance-usdt-liquid-universe.txt`, a 100+ symbol liquid USDT spot universe. Local ingest stays intentionally small for fast smoke tests.

## Local

```bash
make local
```

This downloads two days of public trades for BTC, ETH, BNB, and SOL, converts events in memory, writes partitioned NumPy shards, pre-splits each ticker by time into train/validation/backtest, and runs a 300-step local train by default. Use `uv run python scripts/run_pipeline.py local --local-steps 50` when you only need a quick wiring check.

## Remote

```bash
make remote
```

This prepares `.mega-trading/binance-modal` locally with mixture `binance_public`, uploads only the prepared `stage=05_shards` directory to the `mega-trading-artifacts` Modal Volume, syncs W&B credentials, and launches `modal-binance` training on Modal. Binance raw ZIPs are cached locally under `stage=01_raw` but are not uploaded to Modal. The Modal image uses a CUDA devel base and installs `flash-attn` during image build. The prepared metadata keeps the last 10% of each ticker as a held-out backtest split and uses a small validation split before it.

Modal training mounts the uploaded data at `/data/binance-trades`. The checked-in Modal Binance model is about 41M parameters; under the `tokens ~= 20 * params` rule of thumb, it wants roughly 0.8B training tokens.

## RTX 6000 Server

```bash
make server
```

This prepares `.mega-trading/binance-modal` locally on the server, including the train/validation/backtest split metadata, then trains with `configs/server-rtx6000.yaml`. The prepare path uses `binance-datatool` plus `aria2c` to download only the configured Binance ZIP date window into `stage=01_raw`, then runs a bounded-memory streaming build: sampled tokenizer fitting, parallel per-ticker ZIP scans, and incremental NumPy partition writes. The config targets a single large CUDA GPU with mixed precision, FlashAttention, `torch.compile`, batch size 32, and gradient accumulation 2. `make train-server-rtx6000` runs `scripts/install_flash_attn.py --require-cuda` first, so CUDA servers install `flash-attn` automatically and CPU/Mac paths stay clean. If the prepared shards already exist, use:

```bash
make train-server-rtx6000
```
