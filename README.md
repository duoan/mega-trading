# Mega-Trading

Mega-Trading is a foundation model of trading. The primary path uses generative event modeling: public market data is converted into scale-normalized event-like streams, tokenized into autoregressive blocks, trained with a decoder-only Transformer, and evaluated with perplexity plus market stylized facts.

Exact paper reproduction requires event-level L3/TAQ/ITCH-style data. The no-credential MVP uses public OHLCV proxy events while keeping clean extension points for real trade/order-flow feeds and future multimodal trading FM inputs.

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environment and command execution.

```bash
uv sync
make test
make demo
```

`make demo` is the no-network smoke path. The public path starts from normalized market data:

Public ingestion writes replayable stage-based JSONL/manifests, LanceDB normalized tables, a data-readiness report, and staged training artifacts under the output directory:

```bash
uv run mega-trading ingest --config configs/ingest-public.toml
uv run mega-trading build data.data_dir=.mega-trading/public
uv run mega-trading train run.run_id=public training.max_steps=100
uv run mega-trading eval --run-id public
```

Training uses Hydra config from `configs/default.yaml`, Hugging Face Accelerate for device placement and mixed precision, a main-process progress bar for step-level visibility, and Weights & Biases for metric tracking. By default W&B runs in offline mode under `runs/<run_id>/wandb`; use `training.wandb_mode=online` after `wandb login` to stream metrics to the dashboard. Standard overrides include `model.hidden_dim=64 training.batch_size=16 training.max_steps=10`.

The config-driven API is the preferred path: a reviewer can inspect one TOML file and know exactly which sources, tickers, date windows, quality gates, enrichment steps, and output location will be used. The older flag-based shortcut is still available for quick SEC + Yahoo runs:

```bash
uv run mega-trading ingest-public \
  --tickers AAPL,MSFT,NVDA \
  --start 2015-01-01 \
  --end 2026-05-08 \
  --out .mega-trading/public \
  --sec-user-agent "your-name your-email@example.com"
```

The default checked-in ingest config uses `configs/universes/sp500.txt` and pulls a 10-year S&P 500 market_data window. Use the shortcut only for quick targeted runs.

## Documents

- [End-to-End Platform Design](docs/end-to-end-platform-design.md): target platform architecture for event modeling, training, evaluation, and multimodal extension.
- [Model Design](docs/model-design.md): event tokenization and decoder-only next-token modeling.
- [Project Structure](docs/project-structure.md): source tree conventions that map code packages to data, training, and evaluation.
- [Training Systems Alignment](docs/training-systems-alignment.md): mapping from the AI Infrastructure Engineer JD to project design choices, training efficiency metrics, checkpointing, failure modes, and learning-per-compute goals.

### Module Designs

- [Data Plane Design](docs/data-plane-design.md): ingestion, normalization, quality checks, leakage controls, corpus construction, lineage, and data health metrics.
- [Training Plane Design](docs/training-plane-design.md): next-token training, metrics, checkpointing, and learning per unit of compute.

## Goals

### Short-Term Goals

Within the 72-hour submission window, the goal is to build a credible end-to-end prototype that proves the core contracts of a finance foundation model lab for long-term investment research:

- Consume both offline and continuously updated market data sources into a unified object-store-backed data plane.
- Convert public market data into versioned event streams, token shards, manifests, profiles, and replayable lineage.
- Train a decoder-only next-token model over scale-normalized side/action, return, range, gap, volume, and calendar-time proxy tokens.
- Evaluate with loss, perplexity, top-k accuracy, and generated-vs-real stylized facts.
- Preserve an extension path for news, filings, earnings, macro, and real event-level trade/order-flow data.
- Keep evaluation focused on generative market-sequence quality, including perplexity and stylized facts.
- Run local smoke tests quickly and keep the training path ready for scalable GPU execution.
- Emit data and training metrics that show data health, checkpointing, and end-to-end time-to-result.
- Make the repository stand alone: a reviewer can run the fixture demo, inspect generated artifacts, and understand the scaling path without private data or credentials.

### Long-Term Goals

If extended into a real Deeter-scale system, the prototype should evolve into a continuous finance foundation model platform:

- Scale ingestion across proprietary research feeds, sec_filings, news, transcripts, macro releases, sec_filings, market_data, ownership data, options, credit data, real estate data, and alternative datasets.
- Maintain a governed market data lake with dataset versioning, entitlements, quality scoring, lineage, leakage controls, and reproducible corpus mixtures.
- Support large-scale distributed training for multi-stream models with streaming shards, async prefetch, elastic workers, FSDP/DeepSpeed, checkpoint orchestration, and automated failure recovery.
- Enable rapid research iteration across generative event modeling, evaluations, ablations, and model/data experiments.
- Build a robust evaluation stack for prediction quality, temporal robustness, and downstream portfolio research signals.
- Support traceable predictions where every output can be audited back to source IDs, source timestamps, market data windows, feature snapshots, and model/data versions.
- Integrate realistic event-level evaluation workflows that account for liquidity, timestamp quality, event ordering, and time-aware leakage prevention.
- Operate the platform with production-grade service metrics, training metrics, data drift, freshness SLOs, cost dashboards, and incident workflows.
- Minimize time from new information to model feedback, so researchers can evaluate new data, new objectives, and new model variants in hours rather than weeks.
- Provide a secure path to shared object storage and GPU training clusters, with least-privilege access, auditability, and private-data isolation.

## Evaluation

Mega-Trading evaluates whether the model can learn useful market microstructure proxies without leaking future information.

### Model Output Contract

Each run should include:

- `checkpoint`: decoder-only model weights.
- `metrics`: next-token loss, perplexity, top-1 accuracy, and top-5 accuracy.
- `lineage`: source data, token profile, run manifest, and evaluation run ID.
- `stylized_facts`: generated-vs-real return distribution checks.

### Evaluation Metrics

The evaluation stack should include:

- Language-model metrics: train/validation loss, perplexity, top-k token accuracy, and tokens/sec.
- Stylized facts: heavy tails, volatility clustering, return autocorrelation, and distribution distance between real and generated proxy returns.
- Robustness metrics: ticker split, market regime split, event-type split once real event feeds are connected, and time-aware train/test leakage checks.

