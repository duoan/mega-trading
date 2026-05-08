# High-Level Design

## Overview

MarketFM Forge is a prototype data and training platform for a long-term value investing reasoning model. The system turns public financial data into auditable training signal, trains small foundation-model variants through multiple stages, and evaluates outputs through reasoning quality, long-horizon backtesting, and infrastructure health metrics.

The design is intentionally infrastructure-first. The model is important, but the main artifact is the set of contracts around the model:

- What data was available at a given `as_of_time`.
- How raw documents became training corpora and tokenized shards.
- Which model stage consumed which data mixture.
- Whether the model can cite evidence for an investment thesis.
- Whether outputs can be evaluated through realistic long-horizon portfolio metrics.
- Whether the system is observable, recoverable, and fast enough for research iteration.

## Design Goals

### Product Goals

- Build a reasoning-oriented model for long-term value investing research.
- Produce evidence-grounded investment theses, not opaque short-term predictions.
- Evaluate model outputs with both reasoning metrics and long-horizon finance metrics.
- Preserve source lineage so every answer can be audited back to data, model, and config.

### Infrastructure Goals

- Support offline and continuously updated public market data.
- Use S3-compatible object storage as the primary artifact layer.
- Provide deterministic local runs and cloud/GPU training paths.
- Containerize all runtimes.
- Provide Kubernetes and infrastructure-as-code deployment definitions.
- Emit metrics, alarms, and end-to-end timing reports.
- Support checkpoint/resume and failure-injection tests.

### Research Velocity Goals

- Minimize time from new data to first evaluated model output.
- Make corpus mixtures, training stages, and eval runs reproducible from configs.
- Let researchers run small local smoke tests before launching Modal GPU jobs.
- Make bottlenecks visible: ingestion, quality, tokenization, dataloading, training, checkpointing, and evaluation.

## Non-Goals

MarketFM Forge is not:

- A live trading system.
- A low-latency signal engine.
- An autonomous portfolio manager.
- A claim of tradable alpha.
- A frontier-scale model training run.
- A replacement for human investment judgment.

The prototype should demonstrate system discipline and model-training feasibility, not production investing performance.

## System Context

```mermaid
flowchart LR
    Researcher[Researcher] --> CLI[CLI and API]
    CLI --> DataPlane[Data Plane]
    CLI --> TrainPlane[Training Plane]
    CLI --> EvalPlane[Evaluation Plane]
    DataPlane --> ObjectStore[S3 compatible object store]
    TrainPlane --> ObjectStore
    EvalPlane --> ObjectStore
    TrainPlane --> Modal[Modal GPU jobs]
    EvalPlane --> Reports[Reports and dashboards]
    ObjectStore --> Agent[Investment Reasoning Agent]
    Agent --> Researcher
```

The researcher interacts with the system through a CLI, API, and generated reports. The object store is the system of record for source data, derived corpora, training shards, checkpoints, model artifacts, evaluation outputs, and lineage manifests.

## Architecture

MarketFM Forge is organized into five major planes:

- **Data plane**: ingestion, normalization, quality, corpus construction, and tokenization.
- **Training plane**: CPT/DAPT, SFT, DPO/preference tuning, checkpointing, and metrics.
- **Reasoning plane**: retrieval, evidence packaging, model inference, and structured thesis generation.
- **Evaluation plane**: reasoning evaluation, long-horizon backtesting, leakage checks, and baselines.
- **Operations plane**: metrics, alarms, runbooks, deployment, failure injection, and end-to-end efficiency tracking.

```mermaid
flowchart TB
    subgraph dataPlane [Data Plane]
        Ingest[Ingest adapters]
        Normalize[Normalize and validate]
        Quality[Quality gates]
        Corpus[Corpus builder]
        Tokenize[Tokenizer and packer]
    end

    subgraph trainPlane [Training Plane]
        CPT[CPT or DAPT]
        SFT[SFT]
        DPO[DPO preference tuning]
        Checkpoint[Checkpoint manager]
    end

    subgraph reasoningPlane [Reasoning Plane]
        Retrieve[Evidence retrieval]
        Prompt[Evidence pack builder]
        Agent[Reasoning agent]
    end

    subgraph evalPlane [Evaluation Plane]
        ReasonEval[Reasoning eval]
        Backtest[Long horizon backtest]
        Leakage[Leakage checks]
        Baselines[Baselines]
    end

    subgraph opsPlane [Operations Plane]
        Metrics[Metrics]
        Alarms[Alarms]
        Reports[Ops reports]
        Deploy[K8s and IaC]
    end

    Ingest --> Normalize --> Quality --> Corpus --> Tokenize
    Tokenize --> CPT --> SFT --> DPO --> Checkpoint
    Corpus --> Retrieve --> Prompt --> Agent
    Checkpoint --> Agent
    Agent --> ReasonEval
    Agent --> Backtest
    Corpus --> Leakage
    Backtest --> Reports
    ReasonEval --> Reports
    Metrics --> Alarms
    Deploy --> Ingest
    Deploy --> CPT
```

## Component Design

### CLI And API

The CLI is the primary interface for the prototype. The API can expose the same workflows for local services and Kubernetes jobs.

Key commands:

- `marketfm ingest`
  - Fetch or load public financial data.
- `marketfm build-corpus`
  - Convert normalized records into foundation-model corpus records.
- `marketfm tokenize`
  - Tokenize and pack corpus records into shards.
- `marketfm train cpt`
  - Run continual/domain-adaptive pretraining.
- `marketfm train sft`
  - Run supervised fine-tuning.
- `marketfm train dpo`
  - Run preference tuning.
- `marketfm reason`
  - Generate evidence-grounded investment theses.
- `marketfm eval`
  - Run reasoning evaluation and backtesting.
- `marketfm ops-report`
  - Emit metrics, alarms, lineage, and end-to-end timing.
- `marketfm demo`
  - Run the deterministic end-to-end fixture path.

### Data Ingest Adapters

The ingestion layer should support both fixture data and live public data.

Initial adapters:

- SEC EDGAR submissions and company facts.
- Historical prices from `yfinance` or Stooq.
- FinanceBench open-source sample.
- FinQA/TAT-QA-style financial reasoning examples.
- Optional GDELT/RSS/news adapter for continuous updates.

Each adapter writes raw records to `bronze/` and emits an ingestion manifest:

- Source name.
- Fetch timestamp.
- Source timestamp.
- Request parameters.
- Record count.
- Content hash.
- Schema version.
- Known limitations.

### Normalization And Quality

Normalization converts heterogeneous raw data into typed records:

- `DocumentRecord`
- `FundamentalRecord`
- `PriceRecord`
- `QuestionAnswerRecord`
- `EventRecord`
- `EvidenceRecord`

Quality checks should include:

- Required fields.
- Valid timestamps.
- Duplicate detection.
- Ticker/CIK/entity mapping.
- Missing fundamentals.
- Price coverage.
- Source freshness.
- Language and text-length filters.
- Future leakage checks where labels are present.

Invalid records should be quarantined instead of silently dropped.

### Corpus Builder

The corpus builder creates foundation-model training text from normalized records.

Corpus types:

- Filing sections.
- Fundamentals summaries.
- Earnings-call snippets.
- Evidence-grounded QA.
- Investment-thesis SFT examples.
- Preference-pair examples.

Every corpus record must include:

- `corpus_id`
- `source_ids`
- `ticker`
- `company`
- `document_type`
- `as_of_time`
- `text`
- `task_type`
- `mixture_name`
- `quality_score`
- `metadata`

Corpus mixtures are config-driven. A mixture may combine filings, earnings QA, FinanceBench-style evidence QA, and synthetic thesis examples with explicit weights.

### Tokenizer And Shard Builder

The shard builder turns corpus records into packed language-model training examples.

Responsibilities:

- Load tokenizer config.
- Tokenize corpus records.
- Pack sequences to improve training efficiency.
- Preserve sample boundaries and source IDs.
- Write shard files and shard manifests.
- Track sequence packing efficiency.
- Support streaming reads during training.

Shard manifests should include:

- Tokenizer name and version.
- Sequence length.
- Number of examples.
- Number of tokens.
- Source corpus IDs.
- Content hash.
- Mixture config hash.
- Creation timestamp.

### Training Orchestrator

The training orchestrator launches local smoke tests and Modal GPU jobs.

Training stages:

- **CPT/DAPT**
  - Continue pretraining on finance-domain corpus.
  - Metric: held-out finance perplexity and tokens/sec.

- **SFT**
  - Fine-tune on evidence-grounded investment reasoning examples.
  - Metric: schema compliance, citation inclusion, reasoning task score.

- **DPO/preference**
  - Align toward grounded, cautious, temporally valid reasoning.
  - Metric: preference win rate, unsupported-claim reduction, temporal correctness.

The orchestrator should share core components across stages:

- Config loading.
- Dataset/shard loading.
- Checkpoint manager.
- Metrics emitter.
- Artifact writer.
- Resume validation.

### Checkpoint Manager

The checkpoint manager is responsible for reliability.

It should track:

- Model weights or adapter weights.
- Optimizer state where applicable.
- Scheduler state.
- Step number.
- Random seed state.
- Training config.
- Shard cursor.
- Metrics history.

The demo should include a checkpoint/resume validation path:

1. Train for a small number of steps.
2. Save checkpoint.
3. Simulate interruption.
4. Resume from checkpoint.
5. Verify step count, config hash, and eval continuity.

### Reasoning Agent Runtime

The reasoning runtime generates investment thesis outputs.

Flow:

1. Receive ticker, `as_of_time`, and horizon.
2. Retrieve only evidence available before `as_of_time`.
3. Build an evidence pack with source IDs and snippets.
4. Run model inference using the selected checkpoint or adapter.
5. Validate output schema.
6. Attach lineage.
7. Store output for evaluation.

The model output must be auditable. If evidence is missing or weak, the agent should lower confidence or refuse to make a strong claim.

### Evaluation Engine

The evaluation engine has three roles:

- Check whether the reasoning is faithful and well-formed.
- Check whether generated rankings have long-horizon signal.
- Check whether infrastructure behaved correctly.

Reasoning evaluation:

- Schema compliance.
- Evidence coverage.
- Citation accuracy.
- Faithfulness.
- Temporal correctness.
- Numerical reasoning accuracy.
- Uncertainty quality.

Backtesting evaluation:

- 3/6/12-month forward returns.
- Rank IC.
- Precision@k.
- Bucketed forward returns.
- Long/short spread.
- Benchmark-relative return.
- Sharpe, Sortino, max drawdown.
- Turnover and transaction-cost sensitivity.

Infrastructure evaluation:

- Raw-to-corpus time.
- Corpus-to-shards time.
- Shards-to-first-checkpoint time.
- Checkpoint-to-eval time.
- Total raw-to-eval time.
- Tokens/sec.
- Dataloader wait ratio.
- Checkpoint save and resume time.
- Alerts triggered.

### Observability And Alarms

Observability is part of the product, not a stretch goal.

Metrics should be emitted as JSONL and optionally exposed in Prometheus format.

Data alarms:

- Stale source feed.
- Duplicate spike.
- Low entity mapping coverage.
- Missing prices or fundamentals.
- High quarantine rate.
- Future leakage detected.

Training alarms:

- Low tokens/sec.
- High dataloader wait ratio.
- NaN loss.
- Stalled loss.
- Checkpoint failure.
- Resume validation failure.
- Modal job failure.
- GPU OOM.

Evaluation alarms:

- Citation accuracy regression.
- Schema compliance regression.
- Temporal correctness failure.
- Backtest leakage detected.
- E2E cycle-time regression.

The demo should include failure injection for at least two alarm paths.

## Data Flow

```mermaid
sequenceDiagram
    participant User as Researcher
    participant CLI as CLI
    participant Ingest as Ingest Worker
    participant Store as Object Store
    participant Corpus as Corpus Builder
    participant Tok as Tokenizer
    participant Train as Trainer
    participant Eval as Evaluator

    User->>CLI: marketfm demo
    CLI->>Ingest: fetch fixture and public data
    Ingest->>Store: write bronze records
    Ingest->>Store: write ingest manifest
    CLI->>Corpus: normalize and build corpus
    Corpus->>Store: write silver records
    Corpus->>Store: write corpus records and mixture manifest
    CLI->>Tok: tokenize and pack
    Tok->>Store: write training shards and shard manifest
    CLI->>Train: run CPT, SFT, DPO smoke stages
    Train->>Store: write checkpoints, metrics, model artifacts
    CLI->>Eval: run reasoning and backtest evaluation
    Eval->>Store: write eval report and ops report
    CLI->>User: print summary and artifact paths
```

## Training Flow

```mermaid
flowchart LR
    Corpus["Versioned corpus"] --> Shards["Tokenized packed shards"]
    Shards --> TinyGPT["Tiny GPT from scratch"]
    Shards --> BaseLM["Small open causal LM"]
    TinyGPT --> CPTMetrics["CPT smoke metrics"]
    BaseLM --> CPTLoRA["CPT or DAPT LoRA"]
    CPTLoRA --> SFT["Evidence grounded SFT"]
    SFT --> DPO["DPO preference tuning"]
    DPO --> ReasoningModel["Reasoning model artifact"]
    ReasoningModel --> Eval["Reasoning and backtest eval"]
```

The tiny GPT path proves the training stack can train a language model from tokenized shards. The small open causal LM path proves that the platform can adapt a pretrained model into a value-investing reasoning agent.

## Artifact Contracts

### Ingest Manifest

Required fields:

- `manifest_id`
- `source`
- `source_uri`
- `fetch_time`
- `record_count`
- `content_hash`
- `schema_version`
- `status`
- `errors`

### Corpus Manifest

Required fields:

- `corpus_version`
- `mixture_name`
- `mixture_config_hash`
- `source_manifest_ids`
- `record_count`
- `token_estimate`
- `as_of_min`
- `as_of_max`
- `quality_summary`

### Shard Manifest

Required fields:

- `shard_id`
- `corpus_version`
- `tokenizer`
- `sequence_length`
- `num_sequences`
- `num_tokens`
- `content_hash`
- `source_corpus_ids`

### Training Run Manifest

Required fields:

- `run_id`
- `stage`
- `base_model`
- `adapter_type`
- `train_config_hash`
- `data_mixture_version`
- `shard_ids`
- `checkpoint_paths`
- `metrics_path`
- `status`

### Model Artifact Manifest

Required fields:

- `model_version`
- `base_model`
- `adapter_paths`
- `training_run_ids`
- `data_mixture_version`
- `eval_run_ids`
- `created_at`
- `limitations`

### Reasoning Output Manifest

Required fields:

- `output_id`
- `ticker`
- `as_of_time`
- `horizon`
- `model_version`
- `evidence_ids`
- `prompt_version`
- `lineage`
- `schema_validation_status`

## Deployment Design

### Local Development

Local development should run through Docker Compose:

- API or CLI container.
- Worker container.
- MinIO object store.
- Optional Prometheus/Grafana stack.

The local path must be deterministic and runnable without private credentials.

### Kubernetes

Kubernetes should host the data/control plane:

- Ingest worker Deployment or CronJob.
- Corpus builder Job.
- Tokenizer Job.
- Training launcher Job.
- API Deployment.
- ConfigMaps for non-secret configs.
- Secrets templates for credentials.
- Persistent or object-store-backed artifact storage.
- Prometheus scrape configuration.

### Modal

Modal should host GPU training jobs:

- CPT job.
- SFT job.
- DPO job.

Modal jobs should read from and write to the same artifact contracts as local jobs. The local CPU path remains the reviewer-friendly fallback.

### Infrastructure As Code

Terraform and Kubernetes manifests should define:

- Buckets or local object-store equivalents.
- Service configuration.
- Secret templates.
- Namespace and service account layout.
- Alert rules.
- Optional AWS mapping for S3/EKS/Ray.

## MVP Scope

### Must Have

- Deterministic fixture demo.
- Data ingestion and normalized records.
- Corpus builder with `as_of_time` and source IDs.
- Tokenized packed shards.
- Tiny GPT CPT smoke test.
- Small causal LM SFT or LoRA path.
- Lightweight DPO/preference path or simulated preference trainer.
- Evidence-grounded reasoning output contract.
- 3/6/12-month forward-return evaluation for a small universe or fixture.
- Metrics and ops report.
- At least two alert simulations.
- Dockerized runtime.
- Kubernetes/IaC skeleton.

### Should Have

- Modal GPU entry points.
- FinanceBench or FinQA-derived examples.
- Price/fundamentals live adapter.
- Checkpoint/resume validation.
- Baselines: random, momentum, simple value, prompt-only, SFT-only.

### Can Defer

- Full S&P 500 ingestion.
- Large-scale distributed training.
- Production-grade vector database.
- Rich UI dashboard.
- Real proprietary data connectors.
- Multi-node FSDP/DeepSpeed implementation.

## Scaling Path

The prototype should scale along clear dimensions:

- More data sources: proprietary research, paid fundamentals, transcripts, macro, credit, options, real estate.
- More compute: Modal to internal GPU clusters, Ray, EKS, FSDP/DeepSpeed.
- Better corpora: higher-quality extraction, deduplication, entity resolution, and leakage controls.
- Better training: larger models, longer context, better SFT data, richer preference labels.
- Better evaluation: walk-forward experiments, broader universes, factor controls, regime splits.
- Better operations: SLOs, incident workflows, cost dashboards, and model/data governance.

## Key Design Trade-Offs

### Small Model Versus Real Model

The demo uses small models because the submission window is short. This is acceptable because the project is about proving system contracts. The design keeps model size independent from data and artifact contracts, so larger models can replace the demo model later.

### Fixture Data Versus Live Data

The demo needs deterministic fixture data for reliability. Live public adapters are still valuable, but they should not be required for the reviewer to run the project.

### Backtesting Signal Versus Reasoning Quality

The model may not produce strong investment signal in 72 hours. The evaluation stack should still compute real finance metrics, but the project should frame them as diagnostics. Reasoning quality, lineage, and leakage control are equally important.

### K8s Completeness Versus Implementation Depth

Kubernetes and IaC should show deployability, but the implementation should prioritize a working end-to-end local demo. A strong local demo plus clear K8s manifests is better than a broad but broken platform.

## Open Questions

- Which small causal LM should be the default: Qwen2.5-0.5B-Instruct, TinyLlama-1.1B, or another lightweight model?
- Should the initial universe be hand-picked large-cap companies or dynamically selected from S&P 500 fixtures?
- How much of the DPO stage should be real training versus a lightweight preference-loss smoke test?
- Should retrieval be simple lexical search in the MVP, or should a vector index be included?
- Which backtesting horizon should be the headline metric: 3-month, 6-month, or 12-month?

## Recommended Implementation Order

1. Define artifact schemas and configs.
2. Build deterministic fixture data.
3. Implement object-store abstraction and manifests.
4. Implement ingestion and normalization.
5. Implement corpus builder.
6. Implement tokenizer and shard builder.
7. Implement tiny GPT CPT smoke test.
8. Implement SFT path and reasoning output schema.
9. Implement DPO/preference path.
10. Implement evaluation and backtesting.
11. Implement metrics, alarms, and ops report.
12. Add Docker, K8s, IaC, Modal entry points.
13. Polish README and demo output.

## Success Criteria

The high-level design is successful if a reviewer can understand:

- What problem the system solves.
- Why it is not just a FinGPT clone.
- How public data becomes training signal.
- How the model is trained through CPT, SFT, and preference tuning.
- How outputs are audited back to evidence.
- How long-horizon evaluation is performed without obvious leakage.
- How the system is operated, monitored, and deployed.
- How the prototype scales to a real finance foundation model lab.
