# End-to-End Platform Design

## Objective

Mega-Trading is a trading foundation model platform prototype. The target system learns reusable market representations from multimodal financial data, supports offline foundation training, replay-based evaluation, realtime inference, delayed-label learning, and risk-aware signal generation.

The goal is not simply to predict whether one ticker goes up or down. The goal is to build the data and model infrastructure needed to learn market state, event impact, risk, and regime behavior in a reproducible way.

## Current MVP Versus Target Platform

The current repository implements the offline foundation-training path:

- S&P 500 public universe from `configs/universes/sp500.txt`.
- Ten-year Yahoo price history and SEC EDGAR fundamentals.
- Stage-based raw, normalized, enriched, sample, and shard artifacts.
- Data readiness checks with leakage-aware sample construction.
- Partitioned sample corpus and streaming shard training for million-sample datasets.
- A multi-stream `TradingFoundationModel` with price, fundamentals, and evidence inputs.
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
- Multimodal event understanding across prices, news, filings, earnings, and macro data.
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

### SEC Filings And Earnings

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
- ingest SEC filings and fundamentals
- ingest news and earnings events
- ingest macro updates
- normalize timestamps
- preserve raw lineage

### Feature Builder

The feature builder creates point-in-time model-visible features. It must enforce that every feature satisfies:

```text
feature_time <= prediction_time
```

Generated features include:

- market windows
- rolling returns
- rolling volatility
- event embeddings
- macro regime features
- fundamentals visible as of prediction time

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
  price_window
  news_window
  filing_window
  macro_window
  labels
  feature_version
  source_ids
```

## Delayed Label Generation

Trading labels are delayed because actual outcomes are only known after the prediction horizon matures.

### Return Labels

```text
future_return_5m = log(price[t + 5m] / price[t])
future_return_30m = log(price[t + 30m] / price[t])
```

### Direction Labels

Direction labels must include transaction costs. A positive future return is not enough.

```text
direction = future_return > fee + slippage
```

### Risk Labels

Risk labels include:

- future volatility
- future drawdown
- tail risk
- regime stress

Example schema:

```text
labels
  sample_time
  ticker
  label_return_5m
  label_return_30m
  label_direction
  label_volatility
  label_drawdown
  label_ready_time
```

The label contract must satisfy:

```text
label_time > prediction_time
label_ready_time >= label_time
```

## Prediction Logging

Every prediction must be logged. Prediction logs are required for online learning, debugging, reproducibility, backtesting, and monitoring.

Example schema:

```text
prediction_log
  prediction_id
  prediction_time
  ticker
  base_model_version
  adapter_version
  head_version
  feature_version
  label_version
  pred_return_5m
  pred_return_30m
  pred_direction
  pred_volatility
  confidence
  actual_return_5m
  actual_return_30m
  actual_volatility
  label_status
```

## Model Architecture

The model should not flatten all modalities into one unstructured vector. Each modality should have a dedicated encoder, followed by cross-attention fusion and task-specific decoders.

```text
Price Encoder      News Encoder      Filing Encoder      Macro Encoder
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

- Price encoder: OHLCV windows and market features, using temporal transformer blocks or sequence encoders.
- News encoder: pre-computed event embeddings, using a transformer event encoder.
- Filing encoder: filing or section embeddings, using transformer or MLP blocks.
- Macro encoder: macro feature vectors and regime tokens, using MLP or transformer blocks.

### Fusion

Fusion should be query-based because the importance of external events changes by market state.

```text
news_context = CrossAttention(
  query = price_tokens,
  key = news_tokens,
  value = news_tokens
)

gate = sigmoid(MLP(concat(price_tokens, news_context)))
fused = price_tokens + gate * news_context
```

The same pattern can extend to filings and macro tokens.

### Task Decoders

Task decoders can be implemented as query decoders followed by MLP heads.

Target tasks:

- future return
- direction
- volatility
- drawdown
- event impact
- market regime
- confidence calibration

## Offline Training

Offline training learns the foundation backbone, multimodal representations, and long-term market structure.

A target multitask loss can combine return, direction, volatility, and drawdown objectives:

```text
loss =
  loss_return
  + 0.5 * loss_direction
  + 0.2 * loss_volatility
  + 0.2 * loss_drawdown
```

Dataset rules:

```text
feature_time <= prediction_time
label_time > prediction_time
```

The current MVP implements the first version of this path through forward-return and risk labels over S&P 500 price and fundamentals data.

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

- price encoder
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
- SEC fundamentals integration.
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
