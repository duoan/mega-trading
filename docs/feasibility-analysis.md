# Feasibility Analysis

## Executive Summary

Mega-Trading is feasible as a 72-hour technical submission if the claim is scoped correctly: it is not claiming production alpha, it is proving the infrastructure and model contract for a foundation model of trading.

The realistic MVP is:

- public SEC fundamentals and public prices become versioned, traceable multi-stream samples.
- quality checks and enrichment run before training artifacts are produced.
- labels are generated from future windows that begin after `as_of_time`.
- `TradingFoundationModel` consumes price, fundamental, and evidence streams.
- metrics, checkpoints, and manifests make the run auditable.

## Feasible In 72 Hours

- Build a local-first data plane that consumes a small public universe.
- Produce readiness reports, enriched snapshots, samples, stream shards, and run manifests.
- Run a PyTorch model locally over the same shard contract used by future GPU jobs.
- Report loss, return-bucket accuracy, risk-bucket accuracy, and throughput.
- Keep public data limitations explicit.

## Not Feasible To Claim

- Stable alpha from public data.
- A frontier-scale finance model.
- A production portfolio manager.
- A low-latency trading system.
- Outperformance without long-horizon, bias-controlled evaluation.

## Public Data Sources

- SEC EDGAR APIs for company facts and filing metadata.
- Yahoo/Stooq-style price history for adjusted daily prices.
- Optional public evidence datasets for evaluation and retrieval experiments.

## System Shape

The project should have three working planes in the MVP:

- **Data plane**: ingestion, normalization, quality, enrichment, labels, samples, stream shards.
- **Training plane**: dataset loading, model training, checkpointing, metrics, manifests.
- **Evaluation and operations plane**: leakage checks, prediction diagnostics, backtesting hooks, alarms, and deployment skeletons.

## Main Risk

The biggest risk is overbuilding text-only artifacts that do not help the core model. The safer path is to keep the training data contract concrete: every row should be a model sample with time-correct inputs and future labels.

## Recommended MVP

1. Keep ingestion config-driven.
2. Keep generated artifacts stage-based.
3. Produce only `samples.jsonl` for training data.
4. Train only `TradingFoundationModel` in the default path.
5. Explain results through evidence and lineage rather than separate text-training artifacts.
