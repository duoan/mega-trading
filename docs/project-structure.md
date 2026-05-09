# Project Structure

Mega-Trading follows the same module boundaries described in the high-level design. Source code is organized by system plane, with the TradingFoundationModel modules directly under `train/`.

```text
src/mega_trading/
  cli.py                 # CLI entry point
  __main__.py            # python -m mega_trading entry point

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
    samples.py           # Multi-stream market_data/news/filing/macro sample builder
    shards.py            # Numeric feature shard builder
    public/              # Real public data adapters
      sec.py             # SEC EDGAR company facts
      market.py          # Yahoo/Stooq market data adapters

  train/                 # Training plane
    config.py            # TradingFoundationModel training config
    dataset.py           # Stream shard dataset
    model.py             # TradingFoundationModel architecture
    trainer.py           # TradingFoundationModel trainer

  eval/                  # Evaluation plane
    __init__.py          # Placeholder for backtesting/eval modules
```

New production code should import from the plane-specific packages, for example:

- `mega_trading.core.schemas`
- `mega_trading.data.corpus`
- `mega_trading.data.public.sec`
- `mega_trading.train.trainer`

## Rules

- Put shared contracts in `core/`.
- Put data ingestion, label generation, sample construction, feature shard building, and public data adapters in `data/`.
- Treat ingestion config as the public ingestion API; CLI and schedulers should dispatch from config rather than hard-coded source arguments.
- Run quality and enrichment between ingestion and corpus construction; corpus builders should consume data that has a readiness report.
- Use `mega_trading.data.lance_store` for normalized query tables.
- Put foundation-model and support training stages in `train/`.
- Put backtesting and evaluation code in `eval/`.
- Avoid adding production modules at the package root; keep ownership in `core/`, `data/`, `train/`, `eval/`, or `serving/`.
