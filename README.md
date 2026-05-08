# MarketFM Forge

MarketFM Forge is an end-to-end infra-model co-design prototype for a finance foundation model lab. The project designs the data contracts, training system, model architecture, evaluation loop, and operations layer together, because the model can only learn useful market behavior if the infrastructure gives it time-correct multi-stream inputs, labels, lineage, and fast feedback.

The target model is a multi-input market foundation model for long-term investment research. It is not a next-tick trading predictor and it is not a pure LLM over finance text. The core model consumes price windows, fundamentals, and text/evidence streams through modality-specific encoders and a fusion transformer, predicts forward return and risk targets, and then uses an explanation layer to trace the view back to filings, historical fundamentals, news, macro context, and prices.

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environment and command execution.

```bash
uv sync
make test
make demo
```

Public ingestion writes replayable stage-based JSONL/manifests, LanceDB normalized tables, a data-readiness report, deterministic enrichment artifacts, CPT shards, and evidence-grounded SFT examples under the output directory:

```bash
uv run marketfm ingest --config configs/ingest-public.toml
```

The config-driven API is the preferred path: a reviewer can inspect one TOML file and know exactly which sources, tickers, date windows, quality gates, enrichment steps, training corpus settings, and output location will be used. The older flag-based shortcut is still available for quick SEC + Yahoo runs:

CPT records are domain-adaptation text: they teach the base model the local finance vocabulary and schema. Reasoning behavior comes from SFT records, which ask for an investment thesis as of a specific time and require the answer to cite normalized SEC fundamental and historical price evidence IDs.

```bash
uv run marketfm ingest-public \
  --tickers AAPL,MSFT \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --out .marketfm/public \
  --sec-user-agent "your-name your-email@example.com"
```

## Documents

- [Feasibility Analysis](docs/feasibility-analysis.md): deep feasibility study covering public data, related work, system design, model training, evaluation, risks, and the recommended MVP.
- [Agent Purpose](docs/agent-purpose.md): purpose and operating contract for the long-term value investing reasoning agent.
- [High-Level Design](docs/high-level-design.md): system architecture, component boundaries, data flow, training flow, deployment design, artifact contracts, and MVP scope.
- [Model Design](docs/model-design.md): multi-stream market foundation model, cross-attention fusion, prediction heads, explanation layer, labels, and MVP architecture.
- [Project Structure](docs/project-structure.md): source tree conventions that map code packages to data, training, reasoning, evaluation, and operations planes.
- [Training Systems Alignment](docs/training-systems-alignment.md): mapping from the AI Infrastructure Engineer JD to project design choices, training efficiency metrics, checkpointing, failure modes, and learning-per-compute goals.

### Module Designs

- [Data Plane Design](docs/data-plane-design.md): ingestion, normalization, quality checks, leakage controls, corpus construction, lineage, and data health metrics.
- [Training Plane Design](docs/training-plane-design.md): multi-stream supervised training, CPT/DAPT and SFT side paths, dataloading, checkpointing, Modal GPU jobs, training efficiency, and learning per unit of compute.
- [Reasoning And Evidence Design](docs/reasoning-evidence-design.md): evidence retrieval, `as_of_time` constraints, investment thesis output schema, citation validation, and reasoning lineage.
- [Evaluation And Backtesting Design](docs/evaluation-backtesting-design.md): reasoning metrics, citation checks, temporal correctness, long-horizon backtesting, baselines, and leakage-aware reporting.
- [Observability And Deployment Design](docs/observability-deployment-design.md): metrics, alarm-as-code, failure injection, ops reports, Docker, Kubernetes, Terraform/IaC, and Modal integration.

## Goals

### Short-Term Goals

Within the 72-hour submission window, the goal is to build a credible end-to-end prototype that proves the core contracts of a finance foundation model lab for long-term investment research:

- Consume both offline and continuously updated market data sources into a unified object-store-backed data plane.
- Convert raw market documents, fundamentals, and prices into versioned multi-stream training examples with data quality checks, deduplication, entity mapping, label manifests, and replayable lineage.
- Produce trainable shards for price windows, fundamental features, text/evidence tokens, forward-return labels, risk labels, and evidence-grounded explanation examples.
- Demonstrate a small fusion model path over the same artifact contracts: modality encoders, cross-attention fusion, prediction heads, checkpointing, and metrics.
- Keep CPT/SFT as supporting paths for domain text adaptation and explanation formatting, not as the only model architecture.
- Build the model interface around prediction plus explanation: forward return bucket, risk/reward view, confidence, rationale, cited source IDs, and as-of timestamp.
- Evaluate model outputs with long-horizon investing and portfolio backtesting metrics, not only NLP metrics.
- Run local smoke tests quickly and provide Modal GPU entry points for scalable training.
- Containerize every runtime path and provide Kubernetes and infrastructure-as-code definitions for deployability.
- Emit metrics, alarms, and an ops report that show data health, training health, checkpoint recovery, and end-to-end time-to-result.
- Make the repository stand alone: a reviewer can run the fixture demo, inspect generated artifacts, and understand the scaling path without private data or credentials.

### Long-Term Goals

If extended into a real Deeter-scale system, the prototype should evolve into a continuous finance foundation model platform:

- Scale ingestion across proprietary research feeds, filings, news, transcripts, macro releases, fundamentals, prices, ownership data, options, credit data, real estate data, and alternative datasets.
- Maintain a governed market data lake with dataset versioning, entitlements, quality scoring, lineage, leakage controls, and reproducible corpus mixtures.
- Support large-scale distributed training for multi-stream models with streaming shards, async prefetch, elastic workers, FSDP/DeepSpeed, checkpoint orchestration, and automated failure recovery.
- Enable rapid research iteration across pretraining, supervised fine-tuning, RL/preference optimization, evaluations, ablations, and model/data mixture experiments.
- Build a robust evaluation stack for investment reasoning, evidence grounding, hallucination risk, business quality analysis, valuation reasoning, temporal robustness, and downstream portfolio research signals.
- Support traceable model reasoning where every prediction can be audited back to source documents, source timestamps, market data windows, retrieval queries, feature snapshots, and model/data versions.
- Integrate realistic backtesting workflows that account for transaction costs, slippage, turnover, exposure, liquidity constraints, and time-aware data leakage prevention.
- Operate the platform with production-grade observability: service metrics, training metrics, data drift, freshness SLOs, alerting, runbooks, cost dashboards, and incident workflows.
- Minimize time from new information to model feedback, so researchers can evaluate new data, new objectives, and new model variants in hours rather than weeks.
- Provide a secure deployment path across S3/EKS/Ray or internal GPU clusters, with least-privilege access, auditability, and private-data isolation.

## Prediction, Reasoning, And Backtesting

MarketFM Forge should evaluate whether the model can learn useful long-horizon market structure from multiple data streams, and whether its explanations remain faithful to the evidence and model-visible context.

### Model Output Contract

Each prediction should include:

- `prediction`: forward return bucket, ranking score, risk bucket, volatility estimate, or drawdown estimate.
- `investment_view`: long-term thesis, rating, or risk/reward view derived from the prediction heads.
- `confidence`: calibrated confidence or uncertainty score.
- `rationale`: concise explanation of the investment thesis, including business quality, valuation, catalyst, risk, and uncertainty.
- `evidence`: source document IDs, URLs or local object paths, timestamps, quoted snippets, and retrieved context.
- `as_of_time`: the timestamp boundary proving the model did not use future information.
- `lineage`: model version, data mixture version, shard IDs, prompt/template version, and evaluation run ID.

### Backtesting Metrics

The evaluation stack should include real finance and long-horizon portfolio metrics where applicable:

- Return metrics: cumulative return, annualized return, alpha, beta, and benchmark-relative return.
- Risk metrics: volatility, Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio, and downside deviation.
- Portfolio metrics: hit rate, precision@k for ranked ideas, turnover, average holding period, exposure, concentration, capacity proxy, transaction costs, and slippage.
- Signal metrics: information coefficient, rank IC, IC decay, long/short spread, calibration, and bucketed 3-month/6-month/12-month forward-return analysis.
- Value-investing metrics: thesis hit rate, downside capture, upside/downside ratio, drawdown recovery, valuation multiple change, earnings revision alignment, and catalyst realization.
- Robustness metrics: walk-forward performance, market regime split, sector split, market-cap split, event-type split, and time-aware train/test leakage checks.

### Reasoning Metrics

The reasoning layer should be evaluated separately from raw trading performance:

- Evidence coverage: whether the rationale cites the relevant news, filing, price window, or report.
- Citation accuracy: whether cited sources actually support the claim.
- Faithfulness: whether the rationale matches the retrieved evidence and model-visible context.
- Temporal correctness: whether the explanation only uses information available at `as_of_time`.
- Uncertainty quality: whether the model expresses uncertainty when evidence is weak or conflicting.
- Thesis quality: whether the model separates durable business fundamentals from short-term market noise.
