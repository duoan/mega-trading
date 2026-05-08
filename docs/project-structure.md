# Project Structure

MarketFM Forge follows the same module boundaries described in the high-level design. Source code is organized by system plane, with a small compatibility layer at the package root for early imports.

```text
src/marketfm/
  cli.py                 # CLI entry point
  __main__.py            # python -m marketfm entry point

  core/                  # Shared contracts and infrastructure
    config.py            # Config dataclasses
    hashing.py           # Deterministic hashes
    schemas.py           # Artifact schemas
    store.py             # Local/S3-style artifact store

  data/                  # Data plane
    enrich.py            # Deterministic enrichment and company snapshots
    fixtures.py          # Deterministic demo data
    ingest.py            # Ingestion contracts and fixture ingestor
    ingest_config.py     # Config-driven ingestion API
    lance_store.py       # LanceDB table/index store
    quality.py           # Quality gates and data-readiness reports
    corpus.py            # Fixture/public text corpora and future sample builders
    labels.py            # Forward-return and risk label generation
    samples.py           # Multi-stream price/fundamental/text sample builder
    tokenize.py          # Tokenizer and stream shard builder
    public/              # Real public data adapters
      sec.py             # SEC EDGAR company facts
      prices.py          # Yahoo/Stooq price adapters

  train/                 # Training plane
    fusion.py            # Multi-stream fusion smoke trainer
    trading_foundation_model/  # TradingFoundationModel config, dataset, model, and trainer
    cpt.py               # CPT/DAPT smoke trainer
    sft.py               # Explanation SFT smoke path
    preference.py        # Explanation preference/DPO-style smoke path

  reasoning/             # Reasoning plane
    runtime.py           # Evidence retrieval, pack building, thesis output

  eval/                  # Evaluation plane
    __init__.py          # Placeholder for backtesting/eval modules

  ops/                   # Operations plane
    __init__.py          # Placeholder for metrics/alarms/deployment modules
```

New production code should import from the plane-specific packages, for example:

- `marketfm.core.schemas`
- `marketfm.data.corpus`
- `marketfm.data.public.sec`
- `marketfm.train.cpt`
- `marketfm.reasoning.runtime`

## Rules

- Put shared contracts in `core/`.
- Put data ingestion, label generation, sample/corpus construction, tokenization, and public data adapters in `data/`.
- Treat ingestion config as the public ingestion API; CLI and schedulers should dispatch from config rather than hard-coded source arguments.
- Run quality and enrichment between ingestion and corpus construction; corpus builders should consume data that has a readiness report.
- Use `marketfm.data.lance_store` for normalized query tables and evidence/corpus indexes.
- Put fusion-model and support training stages in `train/`.
- Put evidence-grounded prediction explanation runtime code in `reasoning/`.
- Put backtesting and evaluation code in `eval/`.
- Put metrics, alarms, failure injection, deployment helpers, and ops reports in `ops/`.
- Avoid adding production modules at the package root; keep ownership in `core/`, `data/`, `train/`, `reasoning/`, `eval/`, or `ops/`.
