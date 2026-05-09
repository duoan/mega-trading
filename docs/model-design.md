# Model Design

## Purpose

The model is a generative order-flow model. It learns the conditional distribution of the next paper-style event token from prior event tokens.

## Inputs

`EventBuilder` consumes normalized `order_flow` rows and writes:

- `events.jsonl`: event rows containing action, side, relative price, price depth, relative size, and interarrival time.
- `tokenizer.json`: fitted composite-token vocabulary metadata.
- `tokens.jsonl`: fixed-length autoregressive token blocks.

The tokenizer emits one joint token per event over:

```text
action x side x relative_price_bucket x price_depth_bucket x size_bucket x time_bucket
```

Relative price, price depth, log relative size, and interarrival time bins are fitted during build with clipping and quantile or histogram binning.

## Architecture

```text
paper-style order_flow events
  -> fitted composite tokenizer
  -> autoregressive token blocks
  -> Llama 3-style decoder
  -> generated event-feature evaluation
```

`TradingModel` is a Llama 3-style causal decoder: token embeddings, RoPE positional encoding, pre-norm decoder blocks, RMSNorm, grouped-query capable causal self-attention, SwiGLU feed-forward layers, tied output embeddings, and next-token cross entropy.
