# Execution Plan

## Principle

Build one narrow end-to-end path that is real: public data ingestion, quality checks, enrichment, point-in-time sample construction, stream shards, `TradingFoundationModel` training, metrics, and checkpoints.

Development rules:

- Use `uv` for Python execution.
- Write tests before production changes when adding behavior.
- Keep one step per commit.
- Do not add generated training data to git.

## Phase 1: Project And Contracts

- Initialize the Python project.
- Define stage-based artifact paths.
- Add schemas for entities, documents, fundamentals, prices, evidence, and manifests.
- Add deterministic hashing and local object-store helpers.

## Phase 2: Ingestion

- Add fixture ingestion for deterministic tests.
- Add SEC company facts ingestion.
- Add public price ingestion.
- Write raw and normalized manifests.
- Keep ingestion config-driven.

## Phase 3: Data Quality And Enrichment

- Run required-field checks, duplicate checks, date parsing, price sanity checks, and leakage checks.
- Write `reports/data-readiness.json`.
- Quarantine invalid records.
- Build enriched company snapshots from normalized SEC facts and price coverage.

## Phase 4: Labels And Samples

- Generate forward-return and risk labels from future price windows.
- Build point-in-time samples with trailing prices, visible fundamentals, visible evidence, and labels.
- Preserve `source_ids`, `evidence_ids`, and `as_of_time`.
- Write `stage=04_corpus/mixture=<name>/samples.jsonl`.

## Phase 5: Stream Shards

- Convert samples into compact stream rows.
- Preserve price returns, price levels, fundamental values, evidence token IDs, labels, and lineage.
- Write `stage=05_shards/mixture=<name>/samples.jsonl`.

## Phase 6: Model Training

- Implement `TradingFoundationDataset`.
- Implement `TradingFoundationModel`.
- Implement `TradingFoundationTrainer`.
- Write metrics and checkpoints.
- Validate with local CPU smoke runs.

## Phase 7: Evaluation And Ops

- Add prediction metrics, backtesting metrics, and leakage-aware reports.
- Add metrics and alarms for ingestion freshness, quality score, training throughput, and checkpoint health.
- Add Docker/Kubernetes/Modal entry points after the local path is stable.

## Current MVP Success Criteria

- `uv run mega-trading ingest --config configs/ingest-public.toml` produces readiness, enrichment, samples, and stream shards.
- `uv run mega-trading train training.max_steps=100` trains from Hydra config and writes metrics/checkpoint artifacts.
- No text-only training artifacts are produced by default.
