# Model Design

## Purpose

Mega-Trading is not a pure LLM project and not a next-tick trading model. The model target is a multi-stream foundation model of trading for long-term investment research. Infrastructure and model architecture are designed together: the data plane decides what information is visible at `as_of_time`, the model consumes that information through modality-aware encoders, and the evaluation plane measures whether predictions and explanations are useful without leaking future data.

The key thesis is that finance data is not naturally a single text sequence. Prices are time series, fundamentals are structured point-in-time facts, filings and news are text evidence, and labels come from future market outcomes. A credible model should preserve those structures instead of forcing everything into weak text summaries.

## Architecture

```text
Price window encoder
  OHLCV, returns, volatility, drawdown, liquidity

Fundamental encoder
  SEC facts, ratios, reporting period, accepted_at timestamp

Text/evidence encoder
  filing snippets, company facts, news, transcripts, retrieved context

        ↓

Fusion transformer
  cross-attention over price, fundamental, and text tokens

        ↓

Prediction heads
  forward return bucket, risk bucket, volatility, drawdown, confidence

        ↓

Explanation layer
  cites evidence, summarizes drivers, states uncertainty
```

The core predictor is the fusion model. The explanation layer is downstream: it explains model-visible signals and retrieved evidence, but it is not expected to discover market structure from text-only artifacts.

## Inputs

### Price Stream

Price tokens represent a trailing window ending at `as_of_time`.

Initial fields:

- date offset within the window.
- adjusted close.
- daily return.
- rolling volatility.
- rolling drawdown.
- volume or dollar volume when available.

### Fundamental Stream

Fundamental tokens represent point-in-time SEC facts visible by `as_of_time`.

Initial fields:

- concept.
- value.
- unit.
- period end.
- SEC accepted timestamp.
- source ID.

### Text And Evidence Stream

Text tokens represent evidence available at `as_of_time`.

Initial sources:

- SEC company facts summaries.
- filing snippets when available.
- news or earnings snippets when adapters are added.
- retrieved evidence IDs from LanceDB.

## Labels

The MVP should avoid next-tick prediction. Next-tick labels are noisy, unstable, and make the project look like a fragile trading-alpha demo. The first supervised targets should be long-horizon and research-aligned.

Initial labels:

- `forward_return_20d_bucket`: underperform, neutral, outperform.
- `forward_return_60d_bucket`: underperform, neutral, outperform.
- `risk_bucket`: low, medium, high realized volatility or drawdown.
- `max_drawdown_60d`: regression or bucketed target.

Each label must carry:

- label window start and end.
- benchmark or universe reference when applicable.
- source price path.
- leakage assertion proving the label begins after `as_of_time`.

## Training Objectives

### Primary Objective

Train the fusion model on supervised market prediction:

- classify forward return buckets.
- classify or regress risk targets.
- calibrate confidence.

### Auxiliary Objectives

Auxiliary tasks improve representation quality and auditability:

- masked reconstruction of price/fundamental tokens.
- contrastive alignment between text evidence and structured facts.
- evidence-grounded explanation checks over model-visible evidence and prediction outputs.

### Role Of Explanations

Explanations are a product layer, not the primary training target in the MVP. They should cite model-visible evidence and stay faithful to prediction outputs.

The main model capability comes from multi-stream supervised training and careful labels.

## MVP Model

The 72-hour version should be small and honest:

- price MLP or 1D transformer encoder.
- fundamental embedding encoder over normalized SEC facts.
- text encoder using a tiny local tokenizer/model or lightweight bag-of-evidence embedding.
- one fusion transformer block with cross-attention.
- classification heads for 20D or 60D forward return bucket and risk bucket.
- explanation formatter that references evidence IDs.

This is enough to prove the co-design:

- data pipeline creates time-correct multi-stream samples.
- training loop consumes the same sample contract.
- metrics report prediction quality, leakage checks, and throughput.
- reasoning output is grounded in model-visible evidence.

## Evaluation

Model quality should be reported separately from explanation quality.

Prediction metrics:

- accuracy and macro F1 for return buckets.
- rank IC when scores are available.
- calibration by confidence bucket.
- long/short spread by predicted bucket.
- drawdown and volatility error.

Explanation metrics:

- evidence citation rate.
- citation accuracy.
- temporal correctness.
- schema compliance.
- faithfulness to prediction outputs and evidence.

Infrastructure metrics:

- examples/sec.
- tokens/sec or stream records/sec.
- data loading wait ratio.
- checkpoint time.
- end-to-end ingest-to-metric latency.

## Non-Goals

- Predicting the next tick.
- Claiming tradable alpha from public MVP data.
- Replacing all structured market data with text.
- Treating text-only training as sufficient for financial reasoning.
- Optimizing a production portfolio.
