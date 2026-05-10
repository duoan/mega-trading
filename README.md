# Mega-Trading

Mega-Trading is a reproduction-oriented trading foundation model prototype. The active path is now intentionally narrow: order-flow style events are represented with the paper-aligned feature contract, tokenized as one composite token per event, trained with a Llama 3-style decoder, and evaluated on generated feature distributions.

The target paper uses participant-observable trade-flow messages with action, side, price depth, volume, and interarrival time. This repo keeps that contract as the modeling interface. The active no-credential path is `binance_trades`, an event-level public trades adapter.

## Quickstart

```bash
uv sync
make test
make demo
```

`make demo` is the no-network order-flow fixture path. The three real environment entrypoints are:

```bash
make mac
make rtx
make modal
```

`make mac` prepares a small multi-symbol Binance public-trades slice directly into partitioned NumPy shards, trains, backtests, and renders a report under `.mega-trading/mac/`. `make rtx` runs the same end-to-end flow for the RTX CUDA profile under `.mega-trading/rtx/`. `make modal` prepares `.mega-trading/modal/`, uploads `datasets/` to Modal Volume, launches Modal training, downloads `runs/` and `manifests/`, then runs local backtest/report against the synced artifacts. All three environments share raw Binance ZIPs under `.mega-trading/raw/`; environment roots hold only derived artifacts. If the processed NumPy shards already exist, prepare is skipped.

Training uses Hydra config from `configs/default.yaml`, Hugging Face Accelerate for device placement and mixed precision, a main-process progress bar, and MLflow for metric tracking. By default MLflow writes to a local SQLite backend under `<data_dir>/runs/mlflow/mlflow.db`; set `training.mlflow_tracking_uri` to point at a remote MLflow server when needed.

`make mac` writes a backtest JSON and dashboard:

```text
.mega-trading/mac/evals/mac/backtest.json
.mega-trading/mac/reports/mac/backtest.html
```

Each environment also has standalone report targets:

```bash
make report-mac
make report-rtx
make report-modal
```

The report is written to `reports/<run_id>/backtest.html` under the configured data directory and visualizes loss curves, split counts, held-out backtest metrics, real-vs-generated stylized facts, sampled ticker K-line charts, model-implied forecast paths, and artifact paths.

`make rtx` and `make modal` use separate prepared datasets under `.mega-trading/rtx/datasets` and `.mega-trading/modal/datasets` so each environment has a stable artifact root.

## Distributed Training

The same `mega-trading train` entry point supports local CPU smoke runs, single-node GPU runs, and distributed launches. Accelerate handles device placement, DDP, optional FSDP wrapping, gradient accumulation, mixed precision, and main-process checkpoint writing.

```bash
uv run torchrun --nproc-per-node=8 -m mega_trading.cli train \
  --config-name rtx \
  run.run_id=rtx-ddp \
  training.device=cuda \
  training.precision=mixed \
  training.distributed_strategy=ddp \
  training.compile=true \
  training.attention_backend=flash
```

FSDP uses the same command with `training.distributed_strategy=fsdp`. Modal launchers live in `modal_train.py`:

```bash
make modal
```

Data processing happens before upload. Modal only sees uploaded `datasets` artifacts and runs training against `/data/modal`. Every training manifest records distributed strategy, world size, gradient accumulation, compile mode, attention backend, precision, and checkpoint/resume settings. `configs/modal.yaml` is the public-data Modal path.

## Paper Feature Contract

Each in-memory event contains only the paper-style feature fields:

- `action`: `add` or `delete`.
- `side`: `buy` or `sell`.
- `midprice`: estimated midpoint price; real L3 adapters should use `(best_bid + best_ask) / 2`.
- `relative_price_bps`: `10000 * log(order_price / midprice)`.
- `price_depth_bps`: `abs(relative_price_bps)`.
- `size`: scale-invariant relative size, normalized by a causal ticker-level volume baseline.
- `interarrival_seconds`: elapsed time since the previous event for the ticker.

The active prepare path writes prepared NumPy token datasets plus small metadata files: `tokenizer.json`, `tokens-profile.json`, and `tokens-numpy.json`. `tokenizer.json` stores fitted quantile or histogram bins and the composite vocabulary metadata.

For direct experiments:

```python
import json
import numpy as np

root = ".mega-trading/mac"
meta = json.load(open(f"{root}/datasets/mixture=mac/tokens-numpy.json"))
part = meta["partitions"][0]
tokens = np.load(f"{root}/{part['tokens_path']}", mmap_mode="r")
offset = 0
window = tokens[offset : offset + meta["sequence_length"]]

input_ids = window[:-1]
labels = window[1:]
```

## Documents

- [End-to-End Platform Design](docs/end-to-end-platform-design.md)
- [Model Design](docs/model-design.md)
- [Data Plane Design](docs/data-plane-design.md)
- [Training Plane Design](docs/training-plane-design.md)
- [Training Configs](docs/training-configs.md)
- [Custom Kernels](docs/kernels.md)
- [Project Structure](docs/project-structure.md)
- [Training Systems Alignment](docs/training-systems-alignment.md)

## Goal

The submission demonstrates the infra-model contract for a market microstructure foundation model: deterministic source preparation, fitted event tokenization, partitioned NumPy shards, Llama 3-style training, checkpointing, and generated event-feature evaluation.
