# Mega-Trading

Mega-Trading is a foundation model of trading. It is an end-to-end infra-model co-design prototype that designs the data contracts, training system, model architecture, evaluation loop, and operations layer together, because the model can only learn useful market behavior if the infrastructure gives it time-correct multi-stream inputs, labels, lineage, and fast feedback.

The target model is a multi-input market foundation model for long-term investment research. It is not a next-tick trading predictor and it is not a pure LLM over finance text. The core model follows the target data modalities: market time series, financial news, SEC filings/earnings, and macro data, each through a dedicated encoder before gated cross-attention fusion and task decoders.

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environment and command execution.

```bash
uv sync
make test
make demo
```

`make demo` is the no-network Day 1 path: it ingests deterministic fixture market and SEC filing data, runs readiness checks, builds samples/shards, and trains one CPU step.

Public ingestion writes replayable stage-based JSONL/manifests, LanceDB normalized tables, a data-readiness report, deterministic enrichment artifacts, and multi-stream prediction samples/shards under the output directory:

```bash
uv run mega-trading ingest --config configs/ingest-public.toml
uv run mega-trading train run.run_id=public-tfm
```

Training uses Hydra config from `configs/train/default.yaml`, including a time-ordered train/validation split, validation metrics, automatic device selection, mixed precision on CUDA, checkpoints, manifests, and model registry records. Ablations are standard overrides such as `model.hidden_dim=64 training.batch_size=16`.

The target-platform vertical slice then reuses the registered model version for replay inference, delayed labels, backtesting, online adaptation metadata, and serving-compatible local inference:

```bash
MODEL_VERSION_ID="<model_version_id_from_training_manifest>"

uv run mega-trading replay \
  --model-version-id "$MODEL_VERSION_ID" \
  --replay-id public-replay \
  --max-predictions 1000

uv run mega-trading materialize-labels \
  --prediction-path predictions/public-replay/predictions.jsonl \
  --label-run-id public-labels

uv run mega-trading backtest \
  --labeled-prediction-path labels/public-labels/labeled-predictions.jsonl \
  --backtest-id public-backtest

uv run mega-trading online-update \
  --labeled-prediction-path labels/public-labels/labeled-predictions.jsonl \
  --base-model-version "base-$MODEL_VERSION_ID" \
  --update-id public-online-update

uv run mega-trading serve-smoke \
  --model-version-id "$MODEL_VERSION_ID"
```

Run the default modality ablation suite and write a summary report:

```bash
uv run mega-trading ablate
```

The config-driven API is the preferred path: a reviewer can inspect one TOML file and know exactly which sources, tickers, date windows, quality gates, enrichment steps, training sample settings, and output location will be used. The older flag-based shortcut is still available for quick SEC + Yahoo runs:

```bash
uv run mega-trading ingest-public \
  --tickers AAPL,MSFT,NVDA \
  --start 2015-01-01 \
  --end 2026-05-08 \
  --out .mega-trading/public \
  --sec-user-agent "your-name your-email@example.com"
```

The default checked-in ingest config uses `configs/universes/sp500.txt`, pulls a 10-year S&P 500 market_data window, and uses multiprocessing for sample/shard construction. Use the shortcut only for quick targeted runs.

## Documents

- [End-to-End Platform Design](docs/end-to-end-platform-design.md): target platform architecture for realtime inference, delayed labels, online adaptation, replay training, multimodal fusion, and model registry.
- [Model Design](docs/model-design.md): multi-stream market foundation model, cross-attention fusion, prediction heads, labels, and MVP architecture.
- [Project Structure](docs/project-structure.md): source tree conventions that map code packages to data, training, evaluation, and serving.
- [Training Systems Alignment](docs/training-systems-alignment.md): mapping from the AI Infrastructure Engineer JD to project design choices, training efficiency metrics, checkpointing, failure modes, and learning-per-compute goals.

### Module Designs

- [Data Plane Design](docs/data-plane-design.md): ingestion, normalization, quality checks, leakage controls, corpus construction, lineage, and data health metrics.
- [Training Plane Design](docs/training-plane-design.md): multi-stream supervised training, dataloading, checkpointing, training efficiency, and learning per unit of compute.

## Goals

### Short-Term Goals

Within the 72-hour submission window, the goal is to build a credible end-to-end prototype that proves the core contracts of a finance foundation model lab for long-term investment research:

- Consume both offline and continuously updated market data sources into a unified object-store-backed data plane.
- Convert raw market, news, filing/earnings, macro, and market_data records into versioned multi-stream training examples with data quality checks, deduplication, entity mapping, label manifests, and replayable lineage.
- Produce trainable shards for market_data windows, news embeddings, filing features, macro features, forward-return labels, and risk labels.
- Demonstrate a small fusion model path over the same artifact contracts: modality encoders, cross-attention fusion, prediction heads, checkpointing, and metrics.
- Build the model interface around prediction contracts: forward return bucket, risk bucket, confidence, source IDs, model version, and as-of timestamp.
- Evaluate model outputs with long-horizon investing and portfolio backtesting metrics, not only NLP metrics.
- Run local smoke tests quickly and keep the training path ready for scalable GPU execution.
- Emit data and training metrics that show data health, checkpointing, and end-to-end time-to-result.
- Make the repository stand alone: a reviewer can run the fixture demo, inspect generated artifacts, and understand the scaling path without private data or credentials.

### Long-Term Goals

If extended into a real Deeter-scale system, the prototype should evolve into a continuous finance foundation model platform:

- Scale ingestion across proprietary research feeds, sec_filings, news, transcripts, macro releases, sec_filings, market_data, ownership data, options, credit data, real estate data, and alternative datasets.
- Maintain a governed market data lake with dataset versioning, entitlements, quality scoring, lineage, leakage controls, and reproducible corpus mixtures.
- Support large-scale distributed training for multi-stream models with streaming shards, async prefetch, elastic workers, FSDP/DeepSpeed, checkpoint orchestration, and automated failure recovery.
- Enable rapid research iteration across supervised model training, evaluations, ablations, and model/data experiments.
- Build a robust evaluation stack for prediction quality, temporal robustness, and downstream portfolio research signals.
- Support traceable predictions where every output can be audited back to source IDs, source timestamps, market data windows, feature snapshots, and model/data versions.
- Integrate realistic backtesting workflows that account for transaction costs, slippage, turnover, exposure, liquidity constraints, and time-aware data leakage prevention.
- Operate the platform with production-grade service metrics, training metrics, data drift, freshness SLOs, cost dashboards, and incident workflows.
- Minimize time from new information to model feedback, so researchers can evaluate new data, new objectives, and new model variants in hours rather than weeks.
- Provide a secure path to shared object storage and GPU training clusters, with least-privilege access, auditability, and private-data isolation.

## Prediction And Backtesting

Mega-Trading should evaluate whether the model can learn useful long-horizon market structure from multiple data streams without leaking future information.

### Model Output Contract

Each prediction should include:

- `prediction`: forward return bucket, ranking score, risk bucket, volatility estimate, or drawdown estimate.
- `confidence`: calibrated confidence or uncertainty score.
- `as_of_time`: the timestamp boundary proving the model did not use future information.
- `source_ids`: source record IDs used by the feature sample.
- `lineage`: model version, data mixture version, shard IDs, and evaluation run ID.

### Backtesting Metrics

The evaluation stack should include real finance and long-horizon portfolio metrics where applicable:

- Return metrics: cumulative return, annualized return, alpha, beta, and benchmark-relative return.
- Risk metrics: volatility, Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, and downside deviation.
- Portfolio metrics: hit rate, precision@k for ranked ideas, turnover, average holding period, exposure, concentration, capacity proxy, transaction costs, and slippage.
- Signal metrics: information coefficient, rank IC, IC decay, long/short spread, calibration, and bucketed 3-month/6-month/12-month forward-return analysis.
- Value-investing metrics: thesis hit rate, downside capture, upside/downside ratio, drawdown recovery, valuation multiple change, earnings revision alignment, and catalyst realization.
- Robustness metrics: walk-forward performance, market regime split, sector split, market-cap split, event-type split, and time-aware train/test leakage checks.

