# Ablation Results

This note records the first practical ablation pass for model size and data size. The goal was not to produce a final leaderboard, but to validate the new ablation workflow, confirm MLflow/report artifact capture, and identify which dimensions are worth scaling next.

## Setup

The experiment used a small real-data RTX probe config:

- Config: `configs/ablations/rtx-probe.yaml`
- Ingest slice: `configs/ingest-ablation-probe.toml`
- Source data: Binance monthly trades for `BTCUSDT` and `ETHUSDT`, April 2026
- Device: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Training budget: 20 steps for the main grid, plus one 100-step follow-up
- Context/data shape: `block_size=128`, `stride=4096`
- Evaluation: 16 backtest batches, 5 decoded backtest examples, HTML report per run
- Tracking: MLflow at `http://127.0.0.1:5000`

The initial attempt used the full RTX ingest universe with `max_tickers=1`. That selected a much larger source window and created a million-sequence prepare job, so it was stopped and replaced with the smaller April 2026 probe slice above.

## Ablation Matrix

Data-size variants:

- `one_ticker`: BTCUSDT only, 22,393 sequences
- `two_tickers`: BTCUSDT + ETHUSDT, 45,985 sequences

Model-size variants:

- `micro`: hidden 128, 2 layers, 4 query heads, 2 KV heads, intermediate 352
- `small`: hidden 256, 4 layers, 8 query heads, 4 KV heads, intermediate 704

Follow-up:

- `two_tickers + small_100`: same `small` model and `two_tickers` data, but 100 training steps

## Results

| Run | Data | Model | Steps | Train loss | Val loss | Backtest loss | Backtest ppl | Top-1 | Top-5 | Price-depth L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rtx_probe__data_one_ticker__model_micro` | 1 ticker | micro | 20 | 3.410 | 2.915 | 3.060 | 21.33 | 0.619 | 0.790 | 0.632 |
| `rtx_probe__data_one_ticker__model_small` | 1 ticker | small | 20 | 2.580 | 1.922 | 2.114 | 8.28 | 0.621 | 0.802 | 0.146 |
| `rtx_probe__data_two_tickers__model_micro` | 2 tickers | micro | 20 | 3.675 | 4.773 | 4.771 | 117.98 | 0.304 | 0.597 | 1.229 |
| `rtx_probe__data_two_tickers__model_small` | 2 tickers | small | 20 | 2.797 | 4.990 | 4.943 | 140.22 | 0.004 | 0.111 | 0.141 |
| `rtx_probe__data_two_tickers__model_small_100` | 2 tickers | small | 100 | 1.457 | 4.152 | 4.166 | 64.48 | 0.051 | 0.161 | 0.801 |

Artifacts are written under:

- `runs/<run_id>/metrics.json`
- `evals/<run_id>/backtest.json`
- `evals/<run_id>/backtest-examples.json`
- `reports/<run_id>/backtest.html`

The same run IDs are logged in MLflow with ablation tags.

## Analysis

The cleanest result is the one-ticker model-size ablation. Moving from `micro` to `small` reduced validation loss from 2.915 to 1.922, backtest loss from 3.060 to 2.114, and price-depth L1 from 0.632 to 0.146. For this slice, extra capacity helped both likelihood and rollout distribution.

The data-size ablation shows that simply adding a second ticker makes the problem much harder at the same 20-step budget. `two_tickers + micro` degraded sharply on both backtest loss and L1. The `small` model recovered rollout distribution distance at 20 steps, but its token accuracy collapsed. That mismatch means the generated marginal price-depth distribution can look plausible even while next-token classification is poor.

The 100-step follow-up confirms that more steps helped likelihood for `two_tickers + small` but did not monotonically improve rollout quality. Backtest loss improved from 4.943 to 4.166, but price-depth L1 worsened from 0.141 to 0.801. This is a strong signal that optimization should monitor both open-loop token metrics and rollout stylized facts.

## Recommendations

For the next serious RTX ablation:

1. Keep `small` as the minimum viable model size for multi-ticker data.
2. Use at least two metrics as selection gates: backtest loss and price-depth distribution L1.
3. Add a medium model only after the two-ticker `small` run is stable across a longer schedule.
4. Increase evaluation coverage beyond 16 backtest batches before making final claims.
5. For data-size scaling, prefer controlled short windows first, then expand time range before expanding to many tickers.

Suggested next matrix:

| Axis | Values |
|---|---|
| Data | 1 ticker, 2 tickers, 4 tickers over the same one-month window |
| Model | small, medium |
| Steps | 100, 500 |
| Selection metrics | backtest loss, top-5 accuracy, price-depth L1, return autocorrelation gap |

Avoid comparing full RTX runs against this probe directly: the probe uses a shorter time window, smaller block size, larger stride, different tokenizer fit, and a much shorter training budget.
