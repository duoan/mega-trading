# Model Design

## Purpose

The primary model path is a generative event model. It consumes tokenized market event streams and learns next-token prediction over scale-normalized trading proxies. This is the core modeling path because the project is about learning market event structure, not maintaining a separate classifier.

Exact microstructure reproduction needs event-level L3/TAQ/ITCH-style messages. The public MVP derives event-like rows from OHLCV market data so the repo can demonstrate the tokenizer, shard contract, decoder-only model, trainer, and evaluation loop without private credentials.

## Inputs

`EventBuilder` writes two main artifacts:

- `stage=04_corpus/mixture=<name>/events.jsonl`: event-like rows with ticker, date, side proxy, return, range, gap, volume ratio, and calendar-time fields.
- `stage=05_shards/mixture=<name>/tokens.jsonl`: fixed-length autoregressive token blocks with `tokens`.

The tokenizer uses stable buckets for:

- side/action proxy.
- signed return bps.
- intraday range proxy.
- open gap proxy.
- log volume ratio.
- interarrival/calendar time.

## Architecture

```text
public market data
  -> event-like stream builder
  -> scale-invariant tokenizer
  -> autoregressive token blocks
  -> decoder-only Transformer
  -> perplexity + stylized-fact evaluation
```

`TradingModel` is a causal Transformer with token embeddings, learned positional embeddings, Transformer blocks with a causal mask, layer norm, and a vocabulary projection head. The training loss is next-token cross entropy.

## Extension Path

Future multimodal work should bridge news, filings, earnings, macro, and real trade-flow adapters into the event/token contract first. New objectives should only be added after the Data Plane materializes the required event fields and tests cover the token contract.
