# Project Structure

Mega-Trading keeps the core modeling path directly under `mega_trading/`, because the project itself is the trading foundation model.

```text
src/mega_trading/
  cli.py                 # CLI entry point
  __main__.py            # python -m mega_trading entry point
  config.py              # Build/train config dataclasses
  dataset.py             # Autoregressive token datasets
  events.py              # Public OHLCV-to-event builder
  eval.py                # Stylized-fact evaluation
  model.py               # Decoder-only Transformer
  tokenizer.py           # Scale-invariant event tokenizer
  trainer.py             # Next-token trainer

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
    public/              # Real public data adapters
      sec.py             # SEC EDGAR company facts
      market.py          # Yahoo/Stooq market data adapters

```

New production code should import from the plane-specific packages, for example:

- `mega_trading.core.schemas`
- `mega_trading.data.public.sec`
- `mega_trading.trainer`

## Rules

- Put shared contracts in `core/`.
- Put data ingestion, quality checks, enrichment, and public data adapters in `data/`.
- Treat ingestion config as the public ingestion API; CLI and schedulers should dispatch from config rather than hard-coded source arguments.
- Run quality and enrichment before event construction; event builders should consume data that has a readiness report.
- Use `mega_trading.data.lance_store` for normalized query tables.
- Put shared infrastructure in `core/` and ingestion code in `data/`.
- Keep the main model, tokenizer, trainer, and evaluator as first-class project modules.
