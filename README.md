# Mega-Trading

Mega-Trading is an attempt to build an open trading foundation model: a transparent, reproducible sequence-modeling stack for market microstructure data. The long-term goal is to learn from venue-grade order-flow streams, generate realistic market-event scenarios, and support downstream research in execution, risk, and trading agents.

The current repository is the public-data baseline for that goal. It converts Binance public trade archives into a paper-aligned order-flow event contract, tokenizes each event as one composite token, trains a Llama-style decoder with next-token prediction, and evaluates the checkpoint with chronological backtests plus generated-vs-real event statistics.

This baseline intentionally keeps the modeling interface close to participant-observable trade-flow messages: action, side, price depth, volume, and interarrival time. The active no-credential path is `binance_trades`, an event-level public trades adapter. The roadmap is to replace this proxy with L3 market data such as IEX DEEP/HIST and then add closed-loop simulator evaluation.

## Key Links

- [Hugging Face Space](https://huggingface.co/spaces/duoan/mega-trading): hosted static project poster.
- [Technical Report PDF](https://github.com/duoan/mega-trading/blob/main/huggingface-space/technical-report.pdf): reviewer-facing report with methods, architecture, ablations, and roadmap.
- [Poster Source](huggingface-space/index.html): static HTML page deployed to the Space.
- [Backtest Dashboard](huggingface-space/backtest-report-rtx.html): RTX run report with training curves, backtest metrics, and generated-vs-real charts.
- [Architecture Figure](huggingface-space/pipeline-figure.pdf): paper-style data-to-model-to-output pipeline diagram.
- [Ablation Results](docs/ablation-results.md): capacity, data-size, and training-budget probe results.
- [Performance Profiling](docs/performance-profiling.md): PyTorch profiler notes for optimizer and throughput bottlenecks.
- [Custom Kernels](docs/kernels.md): Triton causal GQA attention benchmark and kernel notes.

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

`make mac`, `make rtx`, and `make modal` prepare environment-specific mixtures under the shared `.mega-trading/data/` root, then train, backtest, and render reports keyed by `run_id`. All three environments share raw Binance ZIPs under `.mega-trading/raw/` and processed NumPy shards under `.mega-trading/data/datasets/`; `mixture=mac|rtx|modal` keeps the data budgets distinct. If the processed NumPy shards already exist, prepare is skipped.

Training uses Hydra config from `configs/default.yaml`, Hugging Face Accelerate for device placement and mixed precision, a main-process progress bar, and MLflow for metric tracking. Backtest metrics, decoded backtest examples, and HTML reports are also logged as MLflow evaluation artifacts when `training.mlflow_enabled=true`. The `make mac`, `make rtx`, `make train-mac`, and `make train-rtx` targets first ensure a local MLflow server is available at `http://127.0.0.1:5000`, reusing it when it is already healthy. Server state lives under `.mega-trading/mlflow/`.

```bash
make mlflow-ui      # start or reuse the local MLflow server, then print the UI URL
make mlflow-stop    # stop the process recorded in .mega-trading/mlflow/mlflow-server.pid
```

Direct `mega-trading train`, `backtest`, `backtest-examples`, and `report` commands still fall back to a local SQLite backend under `<data_dir>/runs/mlflow/mlflow.db` unless `training.mlflow_tracking_uri` is set explicitly.
Each MLflow training run also logs a `torchinfo` model summary artifact at `model/model_summary.txt`.

`make mac` writes a backtest JSON and dashboard:

```text
.mega-trading/data/evals/mac/backtest.json
.mega-trading/data/evals/mac/backtest-examples.json
.mega-trading/data/reports/mac/backtest.html
```

For model-size and data-size ablations, edit the YAML files under `configs/ablations/` and run:

```bash
make ablation-demo
make ablation-rtx-dry-run
make ablation-rtx
```

The ablation runner expands `data_sizes x model_sizes` into prepare, train, backtest, decoded examples, and report commands. Run IDs follow `<name>__data_<data_size>__model_<model_size>`, and MLflow tags include `ablation.name`, `ablation.data_size`, and `ablation.model_size`.

Each environment also has standalone report targets:

```bash
make report-mac
make report-rtx
make report-modal
```

The report is written to `reports/<run_id>/backtest.html` under the configured data directory and visualizes loss curves, split counts, held-out backtest metrics, real-vs-generated stylized facts, sampled ticker K-line charts, model-implied forecast paths, and artifact paths.

`make rtx` and `make modal` use separate mixtures under `.mega-trading/data/datasets`, so the processed data root is shared while each environment keeps a stable dataset contract.

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

Data processing happens before upload. Modal only sees uploaded `datasets` artifacts and runs training against `/data/shared`. Every training manifest records distributed strategy, world size, gradient accumulation, compile mode, attention backend, precision, and checkpoint/resume settings. `configs/modal.yaml` is the public-data Modal path.

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

root = ".mega-trading/data"
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
- [Ablation Results](docs/ablation-results.md)
- [Custom Kernels](docs/kernels.md)
- [Performance Profiling](docs/performance-profiling.md)
- [Project Structure](docs/project-structure.md)
- [Training Systems Alignment](docs/training-systems-alignment.md)

## Goal

The submission demonstrates the infra-model contract for a market microstructure foundation model: deterministic source preparation, fitted event tokenization, partitioned NumPy shards, Llama 3-style training, checkpointing, and generated event-feature evaluation.
