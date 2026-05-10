# Training Configs

The active environment surface is intentionally small:

- `configs/mac.yaml`: Mac/MPS training on a modest multi-symbol Binance slice.
- `configs/rtx.yaml`: single-server RTX PRO 6000 CUDA training config.
- `configs/modal.yaml`: Modal GPU training config for uploaded NumPy shards.
- `configs/default.yaml`: CLI fallback and fixture/demo defaults.

The matching source TOML files are:

- `configs/ingest-mac.toml`
- `configs/ingest-rtx.toml`
- `configs/ingest-modal.toml`
- `configs/ingest-demo.toml`

The RTX ingest uses `configs/binance-usdt-rtx-universe.txt`, a smaller liquid subset sized for the roughly 234M-parameter RTX model. Modal uses `configs/binance-usdt-liquid-universe.txt`, a 100+ symbol liquid USDT spot universe for the larger 1.1B-parameter model. Both use a recent complete 36-month monthly archive window, while Mac ingest stays intentionally small for fast smoke tests.

All three environment ingest configs share the same raw Binance ZIP cache at `.mega-trading/raw` and the same processed data root at `.mega-trading/data`. The environment name is carried by `mixture=mac|rtx|modal` for datasets and `run_id=mac|rtx|modal` for training artifacts.

## Mac

```bash
make mac
```

This downloads two days of public trades for BTC, ETH, BNB, and SOL, converts events in memory, writes partitioned NumPy shards under `.mega-trading/data/datasets/mixture=mac`, pre-splits each ticker by time into train/validation/backtest, trains, backtests, and renders the dashboard. Use `uv run python scripts/run_pipeline.py mac --mac-steps 50` when you only need a quick wiring check.

## RTX

```bash
make rtx
```

This prepares `.mega-trading/data/datasets/mixture=rtx` locally, including the train/validation/backtest split metadata, trains with `configs/rtx.yaml`, backtests, and renders the report. The prepare path uses `binance-datatool` plus `aria2c` to download only the configured Binance ZIP date window into the shared `.mega-trading/raw` cache, then runs a bounded-memory streaming build: sampled tokenizer fitting, parallel per-ticker ZIP scans, and incremental NumPy partition writes. The config targets a single large CUDA GPU with bf16 mixed precision, the local Triton attention backend, `torch.compile`, batch size 8, and gradient accumulation 8.

The checked-in RTX config is a convergence/backtest run around 234M parameters (`hidden_dim=1024`, `layers=20`) with batch size 8 and gradient accumulation 8.

## Modal

```bash
make modal
```

This prepares `.mega-trading/data/datasets/mixture=modal` locally, uploads only the shared `datasets` directory to the `mega-trading-artifacts` Modal Volume at `/shared/datasets`, and launches `modal` training on Modal against `/data/shared`. Binance raw ZIPs are cached locally under `.mega-trading/raw` but are not uploaded to Modal. The Modal image uses a CUDA devel base and installs `flash-attn` during image build. The prepared metadata keeps the last 10% of each ticker as a held-out backtest split and uses a small validation split before it. As of the checked-in config, the ingest window is 2023-05-01 through 2026-04-30, avoiding the incomplete current month.

Modal artifacts are synced back into `.mega-trading/data/runs` and `.mega-trading/data/manifests`, then the local backtest/report steps score the checkpoint on the chronological backtest split and compare generated rollout stylized facts against held-out real token streams. This is the public-data MVP analogue of the TradeFM paper's simulator-based closed-loop evaluation; a full LOB simulator would be needed for market-impact and optimal-execution backtests.
