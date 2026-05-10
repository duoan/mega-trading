# Training Configs

## Local Mac

This machine is an Apple M4 Pro MacBook Pro with 24GB unified memory and 14 CPU cores. The local config is intentionally small and optimized for iteration:

- data config: `configs/ingest-local-mac.toml`
- training config: `configs/local-mac.yaml`
- model: about 5.2M parameters
- sequence length: 128 event tokens
- device: MPS, fp32
- data scope: 8 liquid tickers, about one month of minute bars

Recommended commands:

```bash
uv run mega-trading ingest --config configs/ingest-local-mac.toml
uv run mega-trading build --config-name local-mac
uv run mega-trading train --config-name local-mac
```

This is a fast correctness path, not a scaling-law-optimal training run. With one month of 1-minute bars, the data is too small to judge model quality; use it to test ingestion, numpy dataset export, checkpoints, metrics, and the training loop.

## Binance Public Trades

Binance public trades are the practical no-credential data path. They are event-level executions, so they have much higher token density than 1-minute OHLCV bars. They still are not full L3 order messages: there are no order IDs for resting book add/cancel events. The adapter maps each trade execution as a liquidity-removal event (`action=delete`) and uses `isBuyerMaker` to infer taker buy/sell side.

Local commands:

```bash
make local
```

Modal commands:

```bash
make remote
```

This boundary is intentional: local machines do ingestion, feature construction, tokenizer fitting, and numpy shard export; Modal only trains from uploaded artifacts. `make remote` writes `.mega-trading/binance-modal` with mixture `binance_public`, skips that work if numpy shards already exist, copies the directory to the `mega-trading-artifacts` Modal Volume at `/binance-trades`, syncs the W&B secret, and starts Modal training. The directory is mounted in training containers as `/data/binance-trades`.

Modal training mounts the `wandb-secret` Modal Secret. The remote pipeline reads the local W&B login from `WANDB_API_KEY` or `~/.netrc` and creates/updates that Modal Secret without printing the key. The training configs set `training.wandb_enabled=true` and `training.wandb_mode=online`, so W&B starts automatically when `WANDB_API_KEY` is present.

The checked-in Modal Binance model is about 41.3M parameters. Under the `tokens ~= 20 * params` rule of thumb, it wants roughly 0.83B training tokens. Binance trades can plausibly reach that order of magnitude with enough liquid pairs and history, unlike minute OHLCV.

## Paper-Scale Target

The paper-scale target follows TradeFM's reported setup: a 524M-parameter decoder-only Transformer trained on more than 10B tokens from more than 9K US equities. Appendix C reports scaling runs around 125M, 250M, and 500M parameters, with the 500M model used as the main evaluation target.

Files:

- data config: `configs/ingest-modal-paper.toml`
- training config: `configs/modal-paper.yaml`
- launcher: `modal_train.py`
- model: about 525.8M parameters with the current Llama-style block
- sequence length: 1,024 event tokens
- tokenizer: 8/8/8/4 bins, giving about 14.6K composite event tokens, close to the paper's 16,384-token vocabulary target
- runtime: Modal H100, mixed precision, FSDP, flash SDPA, `torch.compile`

Recommended commands:

```bash
uv run mega-trading ingest --config configs/ingest-modal-paper.toml
uv run mega-trading build --config-name modal-paper
uv run modal run modal_train.py --mode cluster --run-id modal-paper-500m --strategy fsdp --max-steps 80000
```

The Modal launcher defaults to `modal-binance` because the paper-scale equities data is not available in this no-credential project path. Use `--config-name modal-paper` only when a true L3/order-flow dataset is available.

## Public OHLCV Proxy Run

The public OHLCV data path is a proxy, not a full paper reproduction. With one event token per minute bar, raw token count is roughly:

```text
tokens ~= tickers * years * 252 trading_days * 390 minutes
```

For compute-optimal sizing, use the Chinchilla-style rule of thumb:

```text
training_tokens ~= 20 * parameters
parameters ~= training_tokens / 20
```

For a 524M model, compute-optimal training wants roughly 10.5B tokens. Minute bars cannot realistically provide that unless the dataset has broad all-equity coverage across many years:

```text
524M params * 20 ~= 10.5B tokens
9,000 tickers * 3 years * 252 days * 390 minutes ~= 2.65B OHLCV tokens
```

That is why `configs/modal-proxy.yaml` remains intentionally small. It is useful for testing the distributed training system on public data, but should not be described as the full TradeFM reproduction.

Proxy files:

- data config: `configs/ingest-modal-proxy.toml`
- training config: `configs/modal-proxy.yaml`
- model: about 3.4M parameters
- sequence length: 512 event tokens

```bash
uv run mega-trading ingest --config configs/ingest-modal-proxy.toml
uv run mega-trading build --config-name modal-proxy
uv run modal run modal_train.py --mode cluster --run-id modal-proxy --config-name modal-proxy --strategy fsdp --max-steps 20000
```

## Scaling Notes

The paper's 10B+ tokens come from trade-flow events, not one-minute OHLCV bars. True order-flow data can produce many events per instrument per day, making 10B+ tokens plausible. Public OHLCV is lower frequency by construction. The repository therefore has three honest targets:

- `modal-paper`: paper-scale model configuration for the 500M reproduction target.
- `modal-binance`: practical public event-level training path using crypto trades.
- `modal-proxy`: lower-frequency OHLCV distributed systems test that exercises the same infrastructure without claiming paper-scale data.

Do not scale the model to billions of parameters on minute OHLCV bars alone. A 1B parameter model wants about 20B training tokens, which would require far more than public minute bars can provide. Billion-scale training only makes sense after switching the source from OHLCV proxy events to true L3/order-flow messages.
