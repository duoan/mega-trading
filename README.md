# Mega-Trading

Mega-Trading is a reproduction-oriented trading foundation model prototype. The active path is now intentionally narrow: order-flow style events are represented with the paper-aligned feature contract, tokenized as one composite token per event, trained with a Llama 3-style decoder, and evaluated on generated feature distributions.

The target paper uses participant-observable trade-flow messages with action, side, price depth, volume, and interarrival time. This repo keeps that contract as the modeling interface. The active no-credential path is `binance_trades`, an event-level public trades adapter.

## Quickstart

```bash
uv sync
make test
make demo
```

`make demo` is the no-network order-flow fixture path. The two real entrypoints are:

```bash
make local
make remote
```

`make local` prepares a multi-symbol Binance public-trades slice directly into partitioned NumPy shards if needed, then runs a local train long enough to show a loss curve. `make remote` prepares the larger local dataset if needed, uploads it to Modal Volume, syncs the local W&B login into a Modal Secret, and launches Modal training. If the processed NumPy shards already exist, prepare is skipped.

Training uses Hydra config from `configs/default.yaml`, Hugging Face Accelerate for device placement and mixed precision, a main-process progress bar, and Weights & Biases for metric tracking. By default W&B runs in offline mode under `runs/<run_id>/wandb`.

## Distributed Training

The same `mega-trading train` entry point supports local CPU smoke runs, single-node GPU runs, and distributed launches. Accelerate handles device placement, DDP, optional FSDP wrapping, gradient accumulation, mixed precision, and main-process checkpoint writing.

```bash
uv run torchrun --nproc-per-node=8 -m mega_trading.cli train \
  data.data_dir=.mega-trading/binance-modal \
  run.run_id=ddp-h100 \
  training.device=cuda \
  training.precision=mixed \
  training.distributed_strategy=ddp \
  training.compile=true \
  training.attention_backend=flash
```

FSDP uses the same command with `training.distributed_strategy=fsdp`. Modal launchers live in `modal_train.py`:

```bash
make remote
```

Data processing happens locally. Modal only sees uploaded `datasets` artifacts and runs training against `/data/binance-trades`. Every training manifest records distributed strategy, world size, gradient accumulation, compile mode, attention backend, precision, and checkpoint/resume settings. `configs/modal-binance.yaml` is the public-data Modal path.

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

root = ".mega-trading/binance-local"
meta = json.load(open(f"{root}/datasets/mixture=binance_local/tokens-numpy.json"))
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
- [Project Structure](docs/project-structure.md)
- [Training Systems Alignment](docs/training-systems-alignment.md)

## Goal

The submission demonstrates the infra-model contract for a market microstructure foundation model: deterministic source preparation, fitted event tokenization, partitioned NumPy shards, Llama 3-style training, checkpointing, and generated event-feature evaluation.
