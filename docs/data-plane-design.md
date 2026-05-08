# Data Plane Design

## Purpose

The Data Plane turns messy public financial data into reliable, replayable, and auditable training signal for a long-term value investing reasoning model.

This module owns the path from source systems to model-ready corpora:

- Ingest offline and continuously updated financial data.
- Preserve raw source payloads for auditability.
- Normalize records into typed schemas.
- Enforce quality checks and time-aware leakage controls.
- Build foundation-model corpora with source lineage.
- Produce tokenizable records for CPT/DAPT, SFT, and preference training.
- Emit data health metrics and alarms.

The Data Plane is not a feature engineering notebook. It is the system of record for what the model was allowed to see.

## Design Goals

### Correctness

- Every record must have source provenance.
- Every record used for training or evaluation must have an `as_of_time`.
- Invalid or suspicious records should be quarantined, not silently dropped.
- Derived artifacts should be reproducible from manifests and configs.

### Training Readiness

- Corpora should be directly consumable by tokenization and training jobs.
- Corpus mixtures should be config-driven and versioned.
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

The Data Plane uses an S3-compatible object layout. The same paths should work locally, in MinIO, or in S3.

```text
data/
  bronze/
    sec/
    prices/
    financebench/
    fixtures/
    news/
  silver/
    documents/
    fundamentals/
    prices/
    qa/
    events/
    entities/
  corpus/
    cpt/
    sft/
    preference/
    evidence/
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
```

## Artifact Stages

### Bronze

Bronze stores raw source payloads with minimal transformation.

Rules:

- Preserve source response as-is when practical.
- Attach fetch metadata.
- Do not discard source records at this layer.
- Raw records should be content-addressable or hash-tracked.

Bronze record metadata:

- `source`
- `source_uri`
- `fetch_time`
- `source_time`
- `request_params`
- `content_hash`
- `schema_version`

### Silver

Silver stores normalized typed records.

Rules:

- Convert source-specific fields into canonical schemas.
- Attach entity IDs and timestamps.
- Run validation and quality checks.
- Quarantine invalid records.

Silver record families:

- `EntityRecord`
- `DocumentRecord`
- `FundamentalRecord`
- `PriceRecord`
- `QuestionAnswerRecord`
- `EventRecord`
- `EvidenceRecord`

### Corpus

Corpus stores model-facing text records.

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

    def normalize(self, raw: RawRecord) -> Iterable[SilverRecord]:
        ...
```

### Adapter Responsibilities

- Fetch data from source or fixture.
- Write raw payloads to bronze.
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
- `silver_paths`
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
- Bronze/silver/corpus layout.
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
