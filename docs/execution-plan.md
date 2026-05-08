# Execution Plan

## Operating Principles

Development should optimize for a credible 72-hour submission, not a broad unfinished platform.

The implementation process must follow:

- **TDD first**: write a failing test or executable contract before each production change.
- **SOLID design**: keep modules small, dependency-injected where useful, and isolated behind clear interfaces.
- **No over-engineering**: prefer simple local implementations that preserve the scaling contract.
- **One step, one commit**: each step should leave the repo in a working state and be committed separately.
- **Reviewer-first demo**: the deterministic local path must work without private credentials.
- **Infrastructure-first evidence**: each feature should produce artifacts, metrics, or manifests that show system behavior.
- **UV-based execution**: use `uv run` and `uv sync` for local Python commands and environment setup.

## Definition Of Done For Each Step

Each implementation step is complete only when:

- Tests for the step exist and pass.
- The smallest useful implementation is in place.
- Public interfaces are documented through config, schema, or README snippets.
- Generated artifacts are deterministic or clearly ignored.
- The working tree is clean after a git commit.
- The commit message explains the purpose of the step.

## Commit Strategy

Use one commit per step. Do not batch unrelated modules together.

Commit message style:

- `test: add artifact schema contracts`
- `feat: implement object store manifests`
- `feat: build fixture ingestion path`
- `feat: add corpus builder`
- `feat: add tokenizer shard builder`
- `feat: add tiny cpt smoke test`
- `feat: add reasoning output validator`
- `feat: add backtest diagnostics`
- `feat: add ops report and alarms`
- `chore: add docker runtime`

Do not add `Co-authored-by` trailers unless explicitly requested.

## Phase 0: Project Skeleton

Goal: create a minimal Python project that can run tests and expose a CLI.

Steps:

1. Add project metadata, package layout, test runner, and formatting/linting config.
2. Add a minimal `mega-trading` CLI with `--help`.
3. Add CI-like local commands in `Makefile` backed by `uv run`.

Tests first:

- CLI import test.
- `mega-trading --help` smoke test.

Commit after:

- project skeleton and test harness pass.

## Phase 1: Artifact Schemas And Configs

Goal: define the contracts that all modules share.

Steps:

1. Add typed schemas for entities, documents, fundamentals, prices, evidence, corpus records, manifests, training runs, reasoning outputs, and eval reports.
2. Add YAML config models for data mixtures, training stages, evals, and alarms.
3. Add validation helpers and deterministic hashing.

Tests first:

- valid schema fixtures parse.
- invalid timestamps fail.
- manifest hashes are deterministic.
- `as_of_time` is required for model-visible records.

Commit after:

- schema and config tests pass.

## Phase 2: Deterministic Fixture Data

Goal: build the guaranteed demo data path.

Steps:

1. Add small fixture universe.
2. Add filing excerpts, fundamentals, prices, evidence QA, and preference examples.
3. Add known bad records for stale feed, future leakage, and missing entity tests.

Tests first:

- fixture loader returns expected record counts.
- fixture timestamps are deterministic.
- bad fixtures are classified by expected failure reason.

Commit after:

- fixture loader and validation tests pass.

## Phase 3: Object Store And Manifests

Goal: write artifacts through a local object-store abstraction that can later target MinIO/S3.

Steps:

1. Implement local object store interface.
2. Add path conventions for stage-based raw, normalized, enriched, corpus, shards, runs, evals, and registry.
3. Add manifest writer and reader.

Tests first:

- object write/read round trip.
- manifest content hash is stable.
- missing artifacts fail with clear error.

Commit after:

- object store and manifest tests pass.

## Phase 4: Data Ingestion And Normalization

Goal: convert fixture and small public data into normalized records.

Steps:

1. Implement fixture adapter.
2. Implement minimal SEC company facts adapter or adapter interface with fixture-backed mode.
3. Implement price adapter interface with fixture-backed mode.
4. Add normalization and quarantine.

Tests first:

- fixture adapter writes raw-stage manifest.
- normalized records match schemas.
- invalid records go to quarantine.
- duplicate records are detected.

Commit after:

- data ingestion tests pass.

## Phase 5: Label And Multi-Stream Sample Builder

Goal: convert normalized financial records into model-facing samples for the fusion model and support corpora for text/explanation paths.

Steps:

1. Generate point-in-time price windows ending at `as_of_time`.
2. Generate forward-return and risk labels whose windows begin after `as_of_time`.
3. Attach visible fundamentals and text/evidence records.
4. Build CPT corpus records as a text-adaptation support path.
5. Build SFT instruction records as an explanation support path.
6. Build preference pairs for explanation alignment.
7. Write sample/corpus manifests with source lineage, label windows, and mixture config hash.

Tests first:

- samples preserve source IDs.
- samples preserve `as_of_time`.
- label windows start after `as_of_time`.
- mixture hash changes when weights change.
- future evidence is blocked.

Commit after:

- corpus builder tests pass.

## Phase 6: Tokenizer And Stream Shard Builder

Goal: convert samples and corpus records into packed training shards.

Steps:

1. Add tokenizer wrapper for text/evidence streams.
2. Implement sequence packing for language-model support paths.
3. Implement stream packing for price, fundamental, text/evidence, and label tensors.
4. Write shard files and shard manifests.
5. Track token counts, stream counts, and packing efficiency.

Tests first:

- tokenization is deterministic.
- shard manifests include tokenizer and source corpus IDs.
- sequence packing reduces padding on fixture examples.
- malformed corpus records fail cleanly.

Commit after:

- tokenizer and shard tests pass.

## Phase 7: Tiny Fusion Model Smoke Test

Goal: prove the core market foundation model path from multi-stream shards.

Steps:

1. Add tiny price encoder.
2. Add tiny fundamental encoder.
3. Add text/evidence encoder.
4. Add one fusion block with cross-stream attention.
5. Add forward-return and risk prediction heads.
6. Emit prediction, throughput, and checkpoint metrics.
7. Save and resume checkpoint.

Tests first:

- training consumes multi-stream shard dataset.
- one training step records prediction loss without crashing.
- checkpoint save and resume restore step count.
- metrics file is written.

Commit after:

- TradingFoundationModel smoke tests pass.

## Phase 8: CPT And SFT Support Paths

Goal: train or simulate text adaptation and explanation paths over CPT/SFT examples.

Steps:

1. Add CPT text dataset format for domain adaptation.
2. Add SFT dataset format for evidence-grounded explanations.
3. Add SFT training entry point.
4. Add output schema target.
5. Add local smoke mode and Modal-ready config.

Tests first:

- SFT examples format correctly.
- CPT examples remain separate from model prediction labels.
- output target validates against reasoning schema.
- SFT run writes manifest and metrics.

Commit after:

- SFT path tests pass.

## Phase 9: Preference / DPO Path

Goal: add preference tuning contract without overbuilding RL.

Steps:

1. Add preference pair schema.
2. Add DPO or preference-loss smoke trainer.
3. Add metrics for preference loss and chosen/rejected pairs.

Tests first:

- preference pairs validate.
- chosen/rejected examples preserve evidence IDs.
- preference training smoke run writes metrics.

Commit after:

- preference path tests pass.

## Phase 10: Reasoning Runtime

Goal: generate auditable long-term investment thesis outputs.

Steps:

1. Add evidence catalog and simple retrieval.
2. Add evidence pack builder.
3. Add prompt versioning.
4. Add reasoning output validator.
5. Add low-confidence/insufficient-evidence path.

Tests first:

- retrieval excludes future evidence.
- evidence pack includes source IDs.
- validator rejects missing citations.
- insufficient evidence path is deterministic.

Commit after:

- reasoning runtime tests pass.

## Phase 11: Evaluation And Backtesting

Goal: evaluate prediction quality, explanation quality, and long-horizon investment diagnostics.

Steps:

1. Add prediction metrics.
2. Add explanation metrics.
3. Add temporal correctness checks.
4. Reuse forward-return labels from the sample builder.
5. Add rank IC, bucketed return, precision@k, and simple top-k backtest.
6. Add random, momentum, value, price-only, and fundamentals-only baselines.

Tests first:

- future evidence invalidates eval.
- forward returns start after `as_of_time`.
- metrics match hand-computed fixture examples.
- baselines use same universe and dates.

Commit after:

- evaluation tests pass.

## Phase 12: Observability And Alarms

Goal: make the system measurable and operable.

Steps:

1. Add JSONL metrics emitter.
2. Add ops report generator.
3. Add alarm config and evaluator.
4. Add failure injection for stale feed and slow dataloader or future leakage.

Tests first:

- metrics are written in expected schema.
- alarm thresholds trigger on fixture metrics.
- ops report includes e2e timing and checkpoint status.
- failure injection triggers expected alarm.

Commit after:

- observability tests pass.

## Phase 13: Modal, Docker, Kubernetes, IaC

Goal: show deployability without making deployment the only working path.

Steps:

1. Add Dockerfile and Docker Compose.
2. Add Modal app entry points for training stages.
3. Add Kubernetes manifests for data/control plane jobs.
4. Add Terraform/IaC skeleton for object-store and deployment intent.

Tests first:

- Docker command can run CLI help or demo smoke.
- Modal entry points import.
- K8s YAML parses.
- Terraform files validate syntactically if tool is available.

Commit after:

- deployment smoke checks pass.

## Phase 14: Final Demo And Polish

Goal: make the repository stand alone for review.

Steps:

1. Add `make demo`.
2. Generate sample reports from fixture run.
3. Update README with quickstart, expected outputs, and limitations.
4. Add final runbook notes.

Tests first:

- `make test` passes.
- `make demo` passes or documented subset passes.
- generated sample artifacts match expected structure.

Commit after:

- final demo is reproducible.

## Scope Control

Prefer the simplest implementation that preserves the contract.

Defer:

- full S&P 500 ingestion.
- production-grade vector database.
- multi-node FSDP/DeepSpeed.
- rich UI dashboard.
- real trading/execution logic.
- complex portfolio optimizer.

Do not defer:

- `as_of_time` lineage.
- schema validation.
- deterministic fixtures.
- checkpoint/resume test.
- leakage checks.
- metrics and ops report.

## Daily Execution Target

### Day 1

- Project skeleton.
- schemas/configs.
- fixtures.
- object store/manifests.
- fixture ingestion.
- corpus builder.

### Day 2

- tokenizer/shards.
- tiny CPT smoke test.
- checkpoint/resume.
- SFT path.
- preference path.
- reasoning runtime.

### Day 3

- evaluation/backtesting.
- metrics/alarms.
- failure injection.
- Docker/K8s/IaC/Modal entry points.
- README/demo polish.

## Decision Rules

- If a live data source is unstable, fall back to deterministic fixture data and document the adapter path.
- If model quality is weak, focus evaluation on schema, citation, temporal correctness, and infrastructure metrics.
- If DPO is too slow, implement a preference-loss smoke test and document the full DPO scaling path.
- If deployment time is tight, keep Docker working and provide K8s/IaC skeletons.
- If a feature does not improve the demo, contracts, metrics, or reviewer clarity, defer it.
