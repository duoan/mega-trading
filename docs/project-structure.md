# Project Structure

Mega-Trading keeps the paper-style order-flow modeling path directly under `mega_trading/`.

```text
src/mega_trading/
  cli.py                 # CLI entry point
  config.py              # Build/train config dataclasses
  dataset.py             # Autoregressive token datasets
  events.py              # Order-flow corpus and shard builder
  eval.py                # Generated event-feature evaluation
  model.py               # Llama 3-style decoder
  tokenizer.py           # Fitted composite order-flow tokenizer
  trainer.py             # Next-token trainer

  core/
    hashing.py           # Deterministic hashes
    schemas.py           # Order-flow and manifest schemas
    store.py             # Local artifact store

  data/
    fixtures.py          # Deterministic order-flow demo data
    ingest.py            # Source request contracts
    ingest_config.py     # Source TOML parser
    public/binance.py    # Binance public trades to event-level order-flow proxy
```

Non-paper data planes are intentionally absent from the active source tree.
