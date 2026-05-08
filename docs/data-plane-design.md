# Data Plane Design

## Purpose

The Data Plane turns messy public financial data into reliable, replayable, and auditable training signal for a multi-stream market foundation model.

This module owns the path from source systems to model-ready corpora:

- Ingest offline and continuously updated financial data.
- Preserve raw source payloads for auditability.
- Normalize records into typed schemas.
- Enforce quality checks and time-aware leakage controls.
- Build model-facing samples with source lineage, price windows, fundamentals, text/evidence, and labels.
- Produce stream shards for supervised fusion-model training plus tokenizable records for CPT/DAPT, SFT, and preference support paths.
- Emit data health metrics and alarms.

The Data Plane is not a feature engineering notebook. It is the system of record for what the model was allowed to see, what future labels were generated after the as-of boundary, and which sample contract each model consumed.

## Design Goals

### Correctness

- Every record must have source provenance.
- Every record used for training or evaluation must have an `as_of_time`.
- Invalid or suspicious records should be quarantined, not silently dropped.
- Derived artifacts should be reproducible from manifests and configs.

### Training Readiness

- Multi-stream samples should be directly consumable by tokenization, stream packing, and training jobs.
- Data mixtures, label horizons, and stream windows should be config-driven and versioned.
- Records should preserve enough metadata for training, evaluation, and evidence retrieval.

### Time Discipline

- The system must prevent future information from leaking into historical reasoning examples.
- Filing accepted dates, publication times, price dates, and label windows must be tracked separately.
- The same data contracts should support training and backtesting.

### Operability

- Data freshness, coverage, duplicate rate, quality score, and leakage checks should be measurable.
- Failures should be visible through metrics and alarms.
- The deterministic fixture path should work without external credentials.

## Data Sources

### Required MVP Sources

## Config-Driven Ingestion API

Ingestion is configuration-driven. The CLI, future schedulers, and ops workflows should run from an ingest config instead of encoding source-specific arguments in orchestration code.

Example:

```toml
[ingest]
output_dir = ".marketfm/public"
sec_user_agent = "MarketFM Forge your-email@example.com"

[ingest.quality]
enabled = true
fail_on_error = false

[ingest.enrichment]
enabled = true

[ingest.training_data]
enabled = true
mixture_name = "public"
sequence_length = 32
input_window_observations = 20
horizon_observations = 20
return_threshold = 0.02

[[ingest.sources]]
name = "sec_companyfacts"
tickers = ["AAPL", "MSFT"]

[[ingest.sources]]
name = "yahoo_prices"
tickers = ["AAPL", "MSFT"]
start = "2024-01-01"
end = "2024-03-31"
```

Supported source names:

- `sec_companyfacts`
- `yahoo_prices`
- `stooq_prices`

The config is the ingestion API contract: by reading it, an operator should know which data will be fetched, which quality gates and enrichment steps will run, which trainable corpus/shards will be produced, where artifacts will be written, and which source-specific requirements apply.

Config-driven ingestion should produce:

- `stage=01_raw/...` and `stage=02_normalized/...` artifacts.
- LanceDB normalized tables.
- `reports/data-readiness.json`.
- `metrics/data/quality.jsonl`.
- `quarantine/quality/<run_id>.jsonl`.
- `stage=03_enriched/company_snapshots.jsonl`.
- `stage=04_corpus/mixture=<mixture_name>/cpt.jsonl`.
- `stage=04_corpus/mixture=<mixture_name>/sft.jsonl`.
- `stage=04_corpus/mixture=<mixture_name>/samples.jsonl` for multi-stream prediction examples.
- `stage=05_shards/mixture=<mixture_name>/samples.jsonl`.
- `stage=05_shards/mixture=<mixture_name>/cpt.jsonl`.

## Data Readiness

Normalized records are not considered training-ready just because ingestion succeeded. The Data Plane must first run quality gates and write a readiness report.

Initial quality gates:

- required field checks by record family.
- duplicate record ID detection.
- price sanity checks.
- date and datetime parsing.
- evidence timestamp versus `as_of_time` leakage checks.

Initial enrichment:

- join SEC entities, SEC fundamentals, and price coverage by ticker.
- produce deterministic company snapshots with latest fundamental availability and price observation windows.
- keep enrichment output as a staged artifact with a manifest so it can be replayed.

## Training Handoff

After readiness passes, public enriched snapshots are converted into model-facing corpus records today, and should evolve into multi-stream prediction samples next. This makes the Data Plane directly consumable by the Training Plane:

```text
reports/data-readiness.json
stage=03_enriched/company_snapshots.jsonl
  -> stage=04_corpus/mixture=public/cpt.jsonl
  -> stage=04_corpus/mixture=public/sft.jsonl
  -> stage=04_corpus/mixture=public/samples.jsonl
  -> stage=05_shards/mixture=public/samples.jsonl
  -> stage=05_shards/mixture=public/cpt.jsonl
  -> TorchFusionTrainer.train("stage=05_shards/mixture=public/samples.jsonl")
```

The MVP public CPT/SFT corpus proves the first data contract from real public ingestion to trainable artifacts. The next model-facing contract is stronger: each sample should preserve separate price, fundamental, and text/evidence streams with forward-return and risk labels. That contract is what enables a fusion model to learn market structure instead of asking a text-only LLM to infer everything from weak summaries.

The initial sample builder uses observation counts rather than market calendars: `input_window_observations` controls the trailing price window and `horizon_observations` controls the future label window. This keeps the MVP deterministic across sparse fixtures and public daily prices while preserving the no-future-input invariant.

#### Fixture Data

Purpose:

- Guarantee that `make demo` works without network access or credentials.
- Provide deterministic examples for corpus building, training smoke tests, reasoning outputs, and evaluation.

Contents:

- Small set of company metadata.
- Filing excerpts.
- Fundamental facts.
- Price history.
- Evidence QA examples.
- Preference examples.
- Known bad records for alarm/failure injection.

#### SEC EDGAR

Purpose:

- Provide authoritative filings, metadata, and XBRL fundamentals.

Initial endpoints:

- `company_tickers.json` for ticker-to-CIK mapping.
- `submissions/CIK##########.json` for filing history.
- `api/xbrl/companyfacts/CIK##########.json` for fundamentals.

Important fields:

- CIK.
- ticker.
- company name.
- filing form.
- filing date.
- accepted date.
- fiscal period.
- accession number.
- XBRL concept.
- unit.
- value.
- period start/end.

Constraints:

- SEC requires a proper `User-Agent`.
- Rate limits must be respected.
- XBRL tags vary across companies.
- Accepted date should be preferred over fiscal period end for availability.

#### Historical Prices

Purpose:

- Provide historical input windows and forward-return evaluation labels.

Initial providers:

- `yfinance` for demo convenience.
- Stooq as an alternative fallback.

Required fields:

- ticker.
- date.
- adjusted close.
- open/high/low/close/volume when available.
- provider.
- fetch timestamp.

Constraints:

- Public price APIs are not institutional-grade.
- Corporate action handling must be disclosed.
- Backtest labels should be generated only after model outputs are frozen.

#### FinanceBench / Financial QA Fixtures

Purpose:

- Provide evidence-grounded finance reasoning examples.

Useful fields:

- question.
- answer.
- evidence text.
- evidence document name.
- page number or source location.
- justification.
- company.
- reasoning type.

Use cases:

- SFT examples.
- reasoning evaluation.
- citation accuracy checks.

### Optional Sources

#### FinQA / ConvFinQA / TAT-QA

Purpose:

- Improve numerical reasoning over financial reports.

Use cases:

- SFT examples.
- reasoning eval.
- preference pair generation.

#### Earnings Call Datasets

Purpose:

- Add management commentary, Q&A, guidance, and qualitative signals.

Use cases:

- evidence retrieval.
- thesis generation.
- catalyst/risk reasoning.

#### Continuously Updated News

Purpose:

- Demonstrate freshness and continuous corpus updates.

Initial providers:

- GDELT.
- company RSS feeds.
- SEC latest filings feed.

Important clarification:

- For this project, continuous news updates improve research corpus freshness. They are not used for low-latency trading.

## Storage Layout

The Data Plane uses two complementary storage layers:

- Object-store artifacts hold raw payloads, manifests, quarantine records, corpus files, training shards, and replayable audit trails. The same paths should work locally, in MinIO, or in S3.
- LanceDB tables hold normalized records and retrieval-ready corpus/evidence records for fast local querying, future vector search, and reasoning-time evidence lookup.

LanceDB is the primary table/index layer, not the only source of truth. Raw staged artifacts and manifests remain in the object store so every table can be regenerated and audited.

```text
data/
  stage=01_raw/
    source=sec/
    source=yahoo/
    source=fixture/
  stage=02_normalized/
    family=documents/
    family=fundamentals/
    family=prices/
    family=qa/
    family=events/
    family=entities/
  stage=03_enriched/
    company_snapshots.jsonl
  stage=04_corpus/
    mixture=public/
    mixture=fixture/
  stage=05_shards/
    mixture=public/
    mixture=fixture/
  manifests/
    ingest/
    normalization/
    corpus/
    quality/
  quarantine/
    invalid_schema/
    duplicate/
    missing_entity/
    leakage/
  metrics/
    data/
  lancedb/
    stage_02_normalized_entities_sec.lance/
    stage_02_normalized_fundamentals_sec.lance/
    stage_02_normalized_prices_yahoo.lance/
    stage_02_normalized_evidence_fixture.lance/
```

Initial LanceDB table naming:

- `stage_02_normalized_entities_<source>`
- `stage_02_normalized_fundamentals_<source>`
- `stage_02_normalized_prices_<source>`
- `stage_02_normalized_evidence_<source>`
- `stage_04_corpus_<mixture>_<name>`

## Artifact Stages

### Stage 01 Raw

Stage 01 stores raw source payloads with minimal transformation.

Rules:

- Preserve source response as-is when practical.
- Attach fetch metadata.
- Do not discard source records at this layer.
- Raw records should be content-addressable or hash-tracked.

Raw record metadata:

- `source`
- `source_uri`
- `fetch_time`
- `source_time`
- `request_params`
- `content_hash`
- `schema_version`

### Stage 02 Normalized

Stage 02 stores normalized typed records.

Rules:

- Convert source-specific fields into canonical schemas.
- Attach entity IDs and timestamps.
- Run validation and quality checks.
- Quarantine invalid records.

Normalized record families:

- `EntityRecord`
- `DocumentRecord`
- `FundamentalRecord`
- `PriceRecord`
- `QuestionAnswerRecord`
- `EventRecord`
- `EvidenceRecord`

### Stage 04 Corpus

Stage 04 stores model-facing text records.

Rules:

- Every corpus record must have provenance.
- Every corpus record must have `as_of_time`.
- Corpus mixtures must be versioned.
- Records should be suitable for tokenization, retrieval, and SFT.

Corpus families:

- CPT/DAPT corpus.
- SFT instruction corpus.
- Preference pair corpus.
- Evidence retrieval corpus.

## Core Schemas

### EntityRecord

Purpose:

- Resolve tickers, CIKs, company names, and provider-specific IDs.

Required fields:

- `entity_id`
- `ticker`
- `cik`
- `company_name`
- `exchange`
- `sector`
- `industry`
- `valid_from`
- `valid_to`
- `source_ids`

### DocumentRecord

Purpose:

- Represent filings, reports, transcripts, and news documents.

Required fields:

- `document_id`
- `entity_id`
- `ticker`
- `source_type`
- `title`
- `text`
- `source_uri`
- `published_at`
- `accepted_at`
- `period_start`
- `period_end`
- `as_of_time`
- `source_ids`
- `content_hash`
- `metadata`

Availability rule:

- For filings, `as_of_time` should use accepted/publication time, not fiscal period end.

### FundamentalRecord

Purpose:

- Represent XBRL and derived fundamental facts.

Required fields:

- `fundamental_id`
- `entity_id`
- `ticker`
- `concept`
- `label`
- `value`
- `unit`
- `period_start`
- `period_end`
- `filing_date`
- `accepted_at`
- `as_of_time`
- `form`
- `source_ids`

### PriceRecord

Purpose:

- Represent historical price and volume data.

Required fields:

- `price_id`
- `ticker`
- `date`
- `open`
- `high`
- `low`
- `close`
- `adjusted_close`
- `volume`
- `provider`
- `fetch_time`
- `source_ids`

### EvidenceRecord

Purpose:

- Provide auditable evidence snippets for reasoning and evaluation.

Required fields:

- `evidence_id`
- `entity_id`
- `ticker`
- `source_type`
- `document_id`
- `timestamp`
- `as_of_time`
- `text`
- `uri`
- `page`
- `section`
- `content_hash`

### CorpusRecord

Purpose:

- Model-facing text unit for training or retrieval.

Required fields:

- `corpus_id`
- `task_type`
- `mixture_name`
- `entity_id`
- `ticker`
- `as_of_time`
- `text`
- `source_ids`
- `evidence_ids`
- `quality_score`
- `metadata`

Task types:

- `cpt_text`
- `sft_instruction`
- `preference_chosen`
- `preference_rejected`
- `retrieval_evidence`

## Ingestion Adapters

Each adapter should implement the same interface:

```python
class IngestAdapter:
    name: str

    def fetch(self, request: IngestRequest) -> Iterable[RawRecord]:
        ...

    def normalize(self, raw: RawRecord) -> Iterable[NormalizedRecord]:
        ...
```

### Adapter Responsibilities

- Fetch data from source or fixture.
- Write raw payloads to `stage=01_raw`.
- Emit ingest manifest.
- Normalize to typed records.
- Emit errors as structured records.
- Avoid embedding source-specific logic downstream.

### Initial Adapters

- `FixtureAdapter`
- `SecSubmissionsAdapter`
- `SecCompanyFactsAdapter`
- `PriceHistoryAdapter`
- `FinanceBenchAdapter`
- `NewsRssAdapter` or `GdeltAdapter` as optional.

## Manifests

### IngestManifest

Fields:

- `manifest_id`
- `source`
- `source_version`
- `fetch_time`
- `request`
- `raw_paths`
- `record_count`
- `content_hash`
- `status`
- `errors`

### NormalizationManifest

Fields:

- `manifest_id`
- `source_manifest_id`
- `schema_version`
- `normalized_paths`
- `record_counts_by_type`
- `quarantine_counts_by_reason`
- `quality_summary`

### CorpusManifest

Fields:

- `corpus_version`
- `mixture_name`
- `mixture_config_hash`
- `source_manifest_ids`
- `record_count`
- `token_estimate`
- `as_of_min`
- `as_of_max`
- `quality_summary`
- `leakage_check_status`

## Quality Checks

### Required Checks

- Required fields present.
- Timestamps parse correctly.
- `as_of_time` exists.
- Source IDs exist.
- Ticker/CIK mapping exists.
- Text length above minimum.
- Price series coverage meets minimum threshold.
- Fundamental concept values have valid units.
- Duplicate records detected by content hash and source keys.

### Financial Checks

- Filing accepted time is not after example `as_of_time`.
- Price input window ends before label window begins.
- Fundamental period end is not treated as availability time.
- Forward-return labels are not present in training input text.

### Quality Scores

Each normalized document or corpus record can receive a quality score based on:

- source reliability.
- timestamp confidence.
- text completeness.
- entity mapping confidence.
- duplicate status.
- evidence usefulness.
- language and readability.

Quality scores should be stored for filtering and metrics, not hidden inside code.

## Leakage Controls

Leakage control is a first-class Data Plane responsibility.

Rules:

- All model-visible data must satisfy `record.as_of_time <= example.as_of_time`.
- Future return labels are stored separately from model input records.
- Backtesting labels are generated after reasoning outputs are frozen.
- Corpus builder should support train/eval time windows.
- Records with ambiguous availability should be flagged.

Leakage report fields:

- `checked_records`
- `violations`
- `max_future_delta`
- `ambiguous_records`
- `status`

Policy:

- Hard fail for known future evidence in evaluation.
- Quarantine for ambiguous records in demo mode.
- Allow configurable policy for research experiments.

## Corpus Construction

### CPT/DAPT Corpus

Goal:

- Adapt model to financial language.

Sources:

- filing sections.
- fundamentals summaries.
- earnings call snippets.
- news snippets.

Record format:

- plain text with metadata sidecar.

### SFT Corpus

Goal:

- Teach evidence-grounded investment reasoning.

Sources:

- FinanceBench QA.
- FinQA/TAT-QA-style examples.
- synthetic thesis examples from filings and fundamentals.

Record format:

- instruction.
- input evidence pack.
- target structured answer.

### Preference Corpus

Goal:

- Align model toward cautious, cited, temporally valid reasoning.

Pair generation rules:

- Chosen cites evidence; rejected does not.
- Chosen respects `as_of_time`; rejected uses future evidence.
- Chosen states uncertainty; rejected overclaims.
- Chosen separates durable fundamentals from noise; rejected gives shallow sentiment.

## CLI And API

### CLI

Primary commands:

- `marketfm ingest --source fixtures`
- `marketfm ingest --source sec --tickers AAPL,MSFT --forms 10-K,10-Q`
- `marketfm ingest --source prices --tickers AAPL,MSFT --start 2018-01-01 --end 2024-12-31`
- `marketfm normalize --manifest <ingest_manifest>`
- `marketfm build-corpus --mixture configs/data/mixture.yaml`
- `marketfm data-quality --corpus-version <version>`
- `marketfm leakage-check --as-of <date>`

### API

Optional endpoints:

- `POST /data/ingest`
- `GET /data/ingest/{manifest_id}`
- `POST /data/corpus`
- `GET /data/corpus/{corpus_version}`
- `GET /data/quality/{run_id}`

## Metrics And Alarms

### Metrics

- `ingest_records_total`
- `ingest_records_per_second`
- `ingest_errors_total`
- `source_freshness_seconds`
- `duplicate_rate`
- `quarantine_rate`
- `entity_mapping_coverage`
- `missing_price_coverage`
- `missing_fundamental_coverage`
- `corpus_records_total`
- `corpus_token_estimate`
- `corpus_build_seconds`
- `leakage_violations_total`

### Alarms

- Source feed is stale.
- Duplicate rate exceeds threshold.
- Entity mapping coverage drops below threshold.
- Quarantine rate spikes.
- Price coverage is missing for evaluation universe.
- Leakage violations detected.
- Corpus build time regresses.

## MVP Scope

### Must Have

- Fixture adapter.
- SEC company/ticker mapping fixture or live fetch.
- SEC companyfacts ingestion for a small universe.
- Price history ingestion for the same universe.
- FinanceBench-style evidence QA fixture.
- Stage-based raw/normalized/enriched/corpus/shard layout.
- Core schemas and manifests.
- Corpus builder for CPT and SFT.
- Basic preference pair generation.
- Data quality report.
- Leakage check report.

### Should Have

- Live SEC submissions adapter.
- Live `yfinance` or Stooq price adapter.
- FinanceBench open-source loader.
- Simple GDELT/RSS freshness adapter.
- Config-driven corpus mixtures.

### Can Defer

- Full S&P 500 universe.
- Production-grade entity resolution.
- Paid data connectors.
- Complex XBRL taxonomy normalization.
- Vectorized distributed ingestion.

## Scaling Path

### Data Scale

- Move from fixture and small universe to broad U.S. equities.
- Add transcripts, macro releases, alternative data, credit data, and ownership data.
- Introduce batch and streaming ingestion workers.
- Partition object storage by source, date, ticker, and artifact type.

### Quality Scale

- Add richer deduplication.
- Add source reliability scoring.
- Add document section extraction.
- Add entity linking across subsidiaries and renamed tickers.
- Add schema evolution support.

### Training Scale

- Produce larger packed shards.
- Support streaming reads from S3.
- Add async prefetch and local caching.
- Track tokens and bytes per source mixture.

### Governance Scale

- Add dataset version registry.
- Add data entitlement checks.
- Add audit logs.
- Add retention policies.
- Add reproducibility reports for every model.

## Open Questions

- Which ticker universe should be used for the first live demo?
- Should `as_of_time` use filing accepted date or next trading day for filings?
- How much XBRL normalization is necessary for the first 72-hour build?
- Should corpus mixtures be stored as YAML only or registered as versioned artifacts?
- Should news ingestion be included in MVP or left as a freshness demo?
- How strict should leakage policy be for training data versus evaluation data?

## Success Criteria

The Data Plane is successful if:

- A reviewer can see exactly what data the model consumed.
- Every corpus record has source lineage and `as_of_time`.
- Invalid and suspicious records are visible, not hidden.
- Training shards can be traced back to source documents.
- Evaluation can enforce temporal correctness.
- Data health metrics and alarms are emitted.
- The same design can scale from local fixtures to S3-backed production data.
