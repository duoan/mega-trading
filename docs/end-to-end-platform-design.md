# End-to-End Platform Design

## Objective

Mega-Trading is a trading foundation model platform prototype. The target system learns reusable market representations from multimodal financial data, supports offline foundation training, replay-based evaluation, realtime inference, delayed-label learning, and risk-aware signal generation.

The goal is not simply to predict whether one ticker goes up or down. The goal is to build the data and model infrastructure needed to learn market state, event impact, risk, and regime behavior in a reproducible way.

## Current MVP Versus Target Platform

The current repository implements the offline foundation-training path:

- S&P 500 public universe from `configs/universes/sp500.txt`.
- Ten-year Yahoo market data history and SEC EDGAR filings.
- Stage-based raw, normalized, enriched, sample, and shard artifacts.
- Data readiness checks with leakage-aware sample construction.
- Partitioned sample corpus and streaming shard training for million-sample datasets.
- A multi-stream `TradingFoundationModel` with market_data, news, filing, and macro modality inputs.
- Time-ordered train/validation split, metrics, checkpoint, and manifests.

The target platform extends this into:

- Realtime market inference.
- News, filing, earnings, and macro modalities.
- Online feature store and prediction logging.
- Delayed label materialization.
- Replay training and online adapter updates.
- Backtesting and risk-filtered signal generation.
- Model registry for base model, adapter, decoder, feature, and label versions.

This separation is intentional: the MVP proves the core data and training contracts, while the target design shows the production scaling path.

## Implemented Platform Slice

The repository now implements a runnable target-platform vertical slice on top of the offline training path:

- [x] Model registry records under `models/<model_version_id>/model-version.json`.
- [x] Replay inference from registered checkpoints over stream shards.
- [x] Append-only prediction logs under `predictions/<replay_id>/predictions.jsonl`.
- [x] Delayed label materialization into separate `labels/<label_run_id>/...` artifacts.
- [x] Cost-aware deterministic backtest reports under `evals/<backtest_id>/backtest-report.json`.
- [x] Replay-buffer sampling for online adaptation.
- [x] Online adapter/head/calibrator update artifact interface with frozen-backbone metadata.
- [x] Serving-compatible local `ModelServer` request/response wrapper and `serve-smoke` CLI.
- [ ] Production realtime feature service and HTTP model serving.
- [ ] Real news, filing document, earnings transcript, and macro ingestion streams.
- [ ] Actual LoRA or adapter optimization against live delayed labels.

## System Goals

### Functional Goals

- Realtime market inference over a large symbol universe.
- Multimodal event understanding across market_data, news, sec_filings, earnings, and macro data.
- Delayed label generation for future return, direction, volatility, drawdown, and tail risk.
- Offline foundation training over replayable historical samples.
- Online adaptation through lightweight adapters, decoders, calibrators, and regime embeddings.
- Backtesting through historical replay with the same feature and prediction contracts used online.
- Reproducibility across data versions, feature versions, model versions, labels, predictions, and reports.
- Model versioning for base model, adapter, task head, feature contract, and label contract.

### Non-Functional Goals

- Inference latency target: below 50 ms for the online path.
- Throughput target: 1000+ symbols in realtime service mode.
- Scalability: multi-asset, multi-task, and multi-modality.
- Training: seamless offline foundation training plus online adaptation.
- Safety: zero future leakage and explicit lookahead-bias controls.
- Observability: metrics for ingestion, freshness, label maturity, training, latency, and prediction quality.

## High-Level Architecture

```text
External Data
  Market data
  News
  SEC filings
  Earnings
  Macro data
      |
      v
Data Plane
  ingestion
  normalization
  feature generation
  event embedding
  label generation
      |
      +--------------------+---------------------+
      |                    |                     |
      v                    v                     v
Offline Feature Store   Online Feature Store   Prediction Log + Label Store
      |                    |                     |
      v                    v                     v
Offline Foundation      Realtime Inference     Online Learning
Training                Service                Adapter Updates
      |                    |                     |
      +--------------------+---------------------+
                           |
                           v
                    Model Registry
                    base / adapter / head
```

## Data Modalities

### Market Time Series

Sources include Yahoo Finance for the MVP, and Polygon, Alpaca, Tiingo, or exchange feeds for the target realtime path.

Core fields:

- `timestamp`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `vwap`
- `spread`
- `trade_count`

Derived features:

- returns over multiple horizons
- rolling volatility
- momentum
- volume z-score
- volume imbalance
- drawdown features
- market beta and sector-relative features

### Financial News

Target sources include public financial news datasets, Reuters-style headlines, Reddit finance, Twitter/X finance feeds, and vendor news feeds.

Text should be embedded before model training or online serving using a stable embedding model and feature version. Candidate embedding sources include FinBERT, SentenceTransformer, or a hosted embedding model.

Core fields:

- `event_id`
- `timestamp`
- `ticker`
- `headline`
- `body`
- `source`
- `embedding`
- `sentiment`
- `importance_score`

### SEC SecFilings And Earnings

The MVP already ingests SEC company facts. The target path extends this to filing documents, sections, 8-K events, earnings transcripts, guidance, and EPS surprise.

Core fields:

- `event_id`
- `timestamp`
- `ticker`
- `filing_type`
- `section`
- `embedding`
- `sentiment`
- `importance_score`
- `source_uri`

### Macro Data

Target sources include FRED and market-derived macro proxies.

Core fields:

- `timestamp`
- `vix`
- `interest_rate`
- `cpi`
- `yield_curve`
- `oil`
- `dxy`
- `regime_features`

## Data Plane

### Raw Ingestion

The raw ingestion layer writes append-only event logs. Each record must preserve source ID, event time, ingestion time, provider, and raw payload reference.

Responsibilities:

- ingest market data
- ingest SEC filings and sec_filings
- ingest news and earnings events
- ingest macro updates
- normalize timestamps
- preserve raw lineage

### Feature Builder

The feature builder creates point-in-time model-visible features. Data Plane output is the contract consumed by Model Architecture and Offline Training, so field names must stay stable across sample building, shard building, training, replay, and serving.

Every model-visible feature must satisfy:

```text
feature_time <= prediction_time
```

Current implemented sample contract:

- `market_data_window`: trailing market time-series records ending at `as_of_time`.
- `news_window`: financial-news records visible at `as_of_time`; currently empty until news ingestion is implemented.
- `sec_filing_window`: SEC filing features visible at as_of_time visible at `as_of_time`.
- `macro_window`: macro and regime records visible at `as_of_time`; currently empty until macro ingestion is implemented.
- `labels`: forward-return bucket and risk bucket generated from future market_data windows.
- `source_ids`: lineage for market_data and filing records used by the sample.

Current implemented shard contract:

- `market_returns`
- `market_levels`
- `news_embeddings`
- `sec_filing_features`
- `macro_features`
- `return_label`
- `risk_label`
- `forward_return`
- `source_ids`

Empty modality windows are intentional placeholders, not evidence fallbacks. If a modality has no current data source, the pipeline writes an empty list and the trainer pads it to a fixed-size zero vector.

### Online Feature Store

The online feature store serves the latest feature snapshot for realtime inference. It should optimize low-latency lookup by ticker, timestamp, feature version, and modality.

Example schema:

```text
online_market_features
  timestamp
  ticker
  feature_vector
  feature_version
  source_watermark
```

### Offline Feature Store

The offline feature store is the system of record for training, replay simulation, and backtesting. It must preserve the same feature contract as online serving.

Example schema:

```text
offline_training_samples
  sample_time
  ticker
  market_data_window
  news_window
  sec_filing_window
  macro_window
  labels.forward_return_bucket
  labels.risk_bucket
  labels.forward_return
  feature_version
  source_ids
```

## Delayed Label Generation

Trading labels are delayed because actual outcomes are only known after the prediction horizon matures.

### Current Labels

```text
forward_return = market_data[label_end] / market_data[label_start] - 1
return_label = bucket(forward_return)
risk_label = bucket(max_drawdown_or_forward_risk)
```

Current schema:

```text
labels
  sample_time
  ticker
  forward_return
  forward_return_bucket
  risk_bucket
  label_ready_time
```

The label contract must satisfy:

```text
label_time > prediction_time
label_ready_time >= label_time
```

Future label extensions can add direction, volatility, drawdown, event impact, market regime, and calibration targets after those labels are materialized into the shard contract.

## Prediction Logging

Every prediction must be logged. Prediction logs are required for online learning, debugging, reproducibility, backtesting, and monitoring.

Example schema:

```text
prediction_log
  prediction_id
  prediction_time
  ticker
  sample_id
  model_version_id
  base_model_version
  adapter_version
  head_version
  feature_version
  label_version
  pred_return_bucket
  pred_risk_bucket
  confidence
  actual_return_bucket
  actual_risk_bucket
  actual_forward_return
  label_status
```

## Model Architecture

The model should not flatten all modalities into one unstructured vector. Each modality should have a dedicated encoder, followed by cross-attention fusion and task-specific decoders.

```text
MarketData Encoder      News Encoder      Filing Encoder      Macro Encoder
      |                 |                  |                  |
      +-----------------+------------------+------------------+
                               |
                               v
                 Gated Cross-Attention Fusion
                               |
                               v
                     Shared Market Memory
                               |
                               v
                         Task Decoders
```

### Encoders

- MarketData encoder consumes `market_returns` and `market_levels` from `market_data_window`.
- News encoder consumes `news_embeddings` from `news_window`; current public ingest writes no news, so this stream is zero-padded unless explicitly enabled with data.
- Filing encoder consumes `sec_filing_features` from `sec_filing_window`; the current public path maps SEC company facts into this stream.
- Macro encoder consumes `macro_features` from `macro_window`; current public ingest writes no macro records, so this stream is zero-padded unless explicitly enabled with data.

### Fusion

Fusion should be query-based because the importance of external events changes by market state.

```text
news_context = CrossAttention(
  query = market_data_tokens,
  key = news_tokens,
  value = news_tokens
)

gate = sigmoid(MLP(concat(market_data_tokens, news_context)))
fused = market_data_tokens + gate * news_context
```

The same pattern can extend to sec_filings and macro tokens.

### Task Decoders

Task decoders must match materialized labels. The current supervised implementation only trains decoders for labels that exist in the public shard contract:

- `ForwardReturnDecoder` for future-return buckets.
- `RiskDecoder` for risk buckets.

Future decoders should only be added after their labels exist in Data Plane artifacts:

- future return
- direction
- volatility
- drawdown
- event impact
- market regime
- confidence calibration

## Offline Training

Offline training learns the foundation backbone and the currently materialized supervised tasks from `stage=05_shards/mixture=<name>/samples.jsonl`.

Current training batch contract:

- `market_data`: tensor from `market_returns` and `market_levels`.
- `news`: tensor from `news_embeddings`; zero-padded when `news_window` is empty.
- `sec_filings`: tensor from `sec_filing_features`; populated from SEC company facts in the public MVP.
- `macro`: tensor from `macro_features`; zero-padded when `macro_window` is empty.
- `return_label`: class ID for `ForwardReturnDecoder`.
- `risk_label`: class ID for `RiskDecoder`.

Current supervised loss:

```text
loss =
  cross_entropy(forward_return_logits, return_label)
  + cross_entropy(risk_logits, risk_label)
```

Dataset rules:

```text
feature_time <= prediction_time
label_time > prediction_time
```

The current MVP implements this path with S&P 500 market_data history and SEC company facts mapped into the filing stream. News and macro streams are part of the contract but remain empty until corresponding ingestion adapters produce real records. They should not be replaced by evidence text or generic fallback data.

Future multitask losses can add direction, volatility, drawdown, event-impact, regime, or calibration terms only after the Data Plane materializes those labels.

## Realtime Inference

Target online path:

```text
Market Stream
  -> Feature Builder
  -> Online Feature Store
  -> Model Inference
  -> Signal Generation
  -> Risk Filter
  -> Prediction Log
```

Model output is not directly tradable. It must pass through confidence, transaction cost, risk, and sizing controls.

```python
if pred_return > (fee + slippage) and confidence > 0.65:
    signal = "long"
else:
    signal = "no_trade"
```

## Online Learning

The full foundation model should not be retrained online.

Frozen components:

- market_data encoder
- news encoder
- filing encoder
- macro encoder
- shared backbone

Trainable online components:

- LoRA adapters
- task decoders
- calibrators
- regime embeddings

Replay buffer mixture:

```text
batch = concat(
  recent(70%),
  same_regime(20%),
  random(10%)
)
```

Update cadence:

- realtime inference: every second or minute
- calibrator update: every 5 to 15 minutes
- adapter or head update: hourly
- full retraining: daily or weekly

## Backtesting

Backtesting should replay historical timelines exactly as they would have occurred online.

```python
for current_time in timeline:
    build_features(current_time)
    run_inference(current_time)
    log_prediction(current_time)
    if labels_matured(current_time):
        update_online_components(current_time)
```

Metrics:

- Sharpe ratio
- maximum drawdown
- hit rate
- PnL
- latency
- turnover
- slippage-adjusted return
- calibration error
- label maturity lag

## Model Registry

Every prediction should be reproducible from component versions:

```text
base_model_version
adapter_version
decoder_version
feature_version
label_version
data_snapshot_version
```

The registry must support rollback, replay, audit, and A/B evaluation.

## Serving Layout

```text
Feature Service
      |
      v
Model Server
  base + adapter + head
      |
      v
Signal Service
  risk + sizing
      |
      v
Prediction Logger
```

## Minimal Demo Scope

### Day 1

- OHLCV data integration.
- SEC filings integration.
- Multimodal sample contract.
- Offline training pipeline.
- Data readiness and leakage checks.

### Day 2

- Replay inference loop.
- Delayed label generation.
- Prediction logging schema.
- Baseline backtest report.
- Online adapter update stub.

### Day 3

- News or filing embedding extension.
- Model registry metadata.
- Observability report.
- Final demo script and architecture walkthrough.

## Final Philosophy

The foundation model learns long-term market representation. Online learning adapts to short-term regime shifts. Realtime inference produces low-latency model outputs. The risk layer decides whether model outputs are tradable. The data plane guarantees reproducibility and prevents lookahead bias.
