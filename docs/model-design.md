# Model Design

## Purpose

`TradingFoundationModel` is a supervised multi-stream market model. It consumes the numeric feature shard emitted by the Data Plane and predicts the labels that are actually materialized today:

- forward-return bucket.
- risk bucket.

No text-only reasoning path, evidence runtime, or generic task decoder is part of the model implementation.

## Inputs

The model input contract is:

- `market_data`: padded tensor from `market_returns` and `market_levels`.
- `news`: padded tensor from `news_embeddings`.
- `sec_filings`: padded tensor from `sec_filing_features`.
- `macro`: padded tensor from `macro_features`.

`news` and `macro` are enabled by default. If the current public ingest has no real records for those families, the Data Plane emits empty vectors and the dataset pads them to zeros.

## Architecture

```text
MarketDataEncoder    NewsEncoder    SecFilingEncoder    MacroEncoder
      \             |              |              /
       \            |              |             /
        +------ GatedCrossAttentionFusion ------+
                          |
                  SharedMarketMemory
                          |
          +---------------+---------------+
          |                               |
 ForwardReturnDecoder              RiskDecoder
```

Encoders:

- `MarketDataEncoder`: temporal transformer over market_data return/level tokens.
- `NewsEncoder`: feature-sequence encoder over precomputed news features.
- `SecFilingEncoder`: feature-sequence encoder over SEC facts and SEC filing features.
- `MacroEncoder`: feature-sequence encoder over macro/regime features.

Fusion:

- Query-based cross-attention starts from market_data tokens when market_data is enabled.
- Other modalities are fused as context tokens through a learned gate.
- The fused token set is compressed into `SharedMarketMemory`.

Decoders:

- `ForwardReturnDecoder` maps shared memory to `RETURN_LABELS`.
- `RiskDecoder` maps shared memory to `RISK_LABELS`.

Future tasks should only be added after the Data Plane materializes labels for them.
