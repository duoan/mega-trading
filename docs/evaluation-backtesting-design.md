# Evaluation And Backtesting Design

## Purpose

The Evaluation and Backtesting module determines whether Mega-Trading produces useful, auditable research artifacts.

It evaluates three different things that must not be confused:

- **Reasoning quality**: Does the model produce structured, evidence-grounded, temporally valid investment reasoning?
- **Investment signal diagnostics**: Do model-generated scores or rankings have any relationship with future long-horizon returns?
- **Infrastructure health**: Did the evaluation run use reproducible inputs, valid manifests, and leakage-safe joins?

Backtesting results are research diagnostics. They are not proof of tradable alpha.

## Design Goals

### Reasoning Evaluation Goals

- Validate output schema.
- Validate citation resolvability.
- Check evidence support.
- Detect temporal violations.
- Measure uncertainty and risk awareness.

### Backtesting Goals

- Evaluate long-horizon model rankings across 3/6/12-month windows.
- Compare model outputs against simple baselines.
- Report risk, return, drawdown, turnover, and cost sensitivity.
- Surface whether the model's ranking score has any diagnostic signal.

### Leakage And Rigor Goals

- Enforce `as_of_time`.
- Separate model output generation from future label computation.
- Prevent future documents, labels, or prices from entering evidence.
- Report limitations clearly.

## Inputs

### Reasoning Outputs

Required fields:

- `ticker`
- `as_of_time`
- `horizon`
- `investment_view`
- `ranking_score`
- `confidence`
- `thesis`
- `evidence`
- `lineage`

### Evidence Catalog

Used for:

- citation resolvability.
- citation timestamp validation.
- support checks.
- source coverage.

### Price Data

Used for:

- forward returns.
- benchmark-relative returns.
- drawdown.
- volatility.
- turnover and portfolio time series.

### Evaluation Config

Required fields:

- universe.
- rebalance dates.
- horizons.
- benchmark.
- cost assumptions.
- portfolio construction rules.
- baselines.
- leakage policy.

## Output Artifacts

Evaluation outputs should be written under:

```text
evals/
  <eval_run_id>/
    eval_run_manifest.json
    reasoning_metrics.json
    backtest_report.json
    leakage_report.json
    baselines_comparison.json
    eval_events.jsonl
    summary.md
```

### EvalRunManifest

Fields:

- `eval_run_id`
- `model_version`
- `reasoning_output_ids`
- `data_snapshot_id`
- `eval_config_hash`
- `universe`
- `date_range`
- `horizons`
- `status`
- `created_at`
- `limitations`

## Reasoning Evaluation

### Schema Compliance

Metrics:

- JSON parse rate.
- required field pass rate.
- field type pass rate.
- score range pass rate.
- allowed enum pass rate.

Failure examples:

- missing `as_of_time`.
- invalid `ranking_score`.
- unparseable evidence list.
- missing lineage.

### Evidence Coverage

Metrics:

- percentage of material claims with citations.
- evidence items per output.
- distinct source types cited.
- evidence concentration by source.
- unresolved citation rate.

Material claim categories:

- thesis.
- business quality.
- valuation.
- catalyst.
- risk.
- uncertainty.

### Citation Accuracy

MVP checks:

- cited evidence ID exists.
- cited evidence belongs to the correct company or allowed context.
- cited evidence timestamp is valid.
- cited snippet is present in evidence pack.

Scaling checks:

- claim extraction.
- entailment model.
- LLM-as-judge with fixed rubric.
- human audit sampling.

### Faithfulness

Faithfulness asks whether the model's claims are supported by the evidence pack.

MVP approach:

- rule-based support checks for fixture examples.
- FinanceBench-style gold evidence overlap where available.
- unsupported-claim count based on missing citations.

Scaling approach:

- natural language inference model.
- citation-level entailment.
- human reviewer workflow.

### Temporal Correctness

Checks:

- every evidence timestamp is `<= as_of_time`.
- retrieval filters exclude future evidence.
- generated output does not mention post-`as_of_time` outcomes.
- forward returns are not in the evidence pack.
- model output is frozen before labels are computed.

Metrics:

- temporal violation rate.
- number of future evidence items blocked.
- max future timestamp delta.
- leakage test pass/fail.

Policy:

- temporal violations should mark the eval run invalid.
- known leakage examples should be included in fixture tests.

### Uncertainty Quality

Metrics:

- low-confidence rate when evidence is sparse.
- refusal/insufficient-evidence rate.
- confidence distribution.
- confidence versus evidence coverage.

MVP:

- rule-based checks on fixture examples.

Scaling:

- calibration curves.
- confidence versus forward outcome.
- confidence versus human audit labels.

## Long-Horizon Backtesting

### Label Generation

Forward returns should be generated after reasoning outputs are frozen.

Supported horizons:

- 3 months.
- 6 months.
- 12 months.

Label fields:

- `forward_return_3m`
- `forward_return_6m`
- `forward_return_12m`
- `benchmark_return`
- `excess_return`
- `label_start_date`
- `label_end_date`

Rules:

- label window starts after `as_of_time`.
- use adjusted prices where available.
- missing prices should be reported.
- corporate action limitations should be disclosed.

### Signal Metrics

Metrics:

- Pearson IC.
- Spearman rank IC.
- IC decay by horizon.
- precision@k.
- hit rate.
- bucketed forward returns.
- long/short spread.
- calibration by score bucket.

Interpretation:

- These measure whether model rankings contain information.
- They do not prove tradability.

### Portfolio Metrics

Default MVP portfolio:

- monthly or fixture-defined rebalance.
- long-only top-k.
- equal weight.
- configurable transaction cost.
- benchmark comparison.

Metrics:

- cumulative return.
- annualized return.
- volatility.
- Sharpe ratio.
- Sortino ratio.
- max drawdown.
- Calmar ratio.
- benchmark-relative return.
- turnover.
- average holding period.
- concentration.
- transaction cost sensitivity.

Optional:

- dollar-neutral long/short spread.
- sector-neutral ranking.
- factor-adjusted return.

### Value-Investing Diagnostics

Metrics:

- 3/6/12-month bucketed forward returns.
- upside/downside ratio.
- downside capture.
- drawdown recovery.
- valuation multiple change where data exists.
- earnings revision alignment where data exists.
- thesis hit rate for human-labeled fixture examples.

These are useful because the model is aimed at long-term value investing, not short-term trading.

## Baselines

Baselines are mandatory for interpretability.

### MVP Baselines

- random ranking.
- momentum ranking.
- simple value ranking if fundamentals are available.
- prompt-only base model.
- explanation-only baseline.

### Baseline Rules

- Baselines must use the same universe.
- Baselines must use the same dates.
- Baselines must use the same cost assumptions.
- Baselines must use the same label generation.

### Purpose

Baselines answer:

- Did training improve reasoning format?
- Did evidence retrieval improve explanation discipline?
- Is model ranking better than random?
- Is model ranking better than simple public factors?

## Leakage Controls

### Core Rules

- Time is a first-class join key.
- No evidence after `as_of_time`.
- No future returns in prompts.
- Freeze reasoning outputs before computing labels.
- Track all source timestamps and availability timestamps.

### Automated Checks

- evidence timestamp check.
- price input window check.
- label window check.
- retrieval query replay.
- future-data fixture test.
- train/eval overlap report.

### Leakage Report

Fields:

- `checked_outputs`
- `temporal_violations`
- `future_evidence_ids`
- `label_join_violations`
- `ambiguous_records`
- `status`
- `policy`

Policy:

- hard fail for future evidence in evaluation.
- warn for ambiguous source availability.
- block headline backtest if leakage status is invalid.

## Reporting

### Summary Report

The human-readable report should include:

- disclaimer.
- model version.
- data snapshot.
- universe.
- date range.
- horizons.
- reasoning metrics.
- backtesting metrics.
- baselines comparison.
- leakage status.
- limitations.
- artifact paths.

### Machine-Readable Reports

All metrics should also be written as JSON.

Purpose:

- compare runs.
- feed dashboards.
- drive alarms.
- support registry summaries.

## CLI And API

### CLI

Commands:

- `mega-trading eval reasoning --outputs <path>`
- `mega-trading eval backtest --outputs <path> --prices <path> --horizons 3m,6m,12m`
- `mega-trading eval leakage --outputs <path>`
- `mega-trading eval all --config configs/eval/demo.yaml`
- `mega-trading eval compare --runs run_a,run_b`

### API

Optional endpoints:

- `POST /eval/runs`
- `GET /eval/runs/{eval_run_id}`
- `GET /eval/runs/{eval_run_id}/summary`
- `GET /eval/runs/{eval_run_id}/artifacts`

## Metrics And Alarms

### Reasoning Alarms

- schema pass rate below threshold.
- citation resolvability below threshold.
- unsupported claim rate above threshold.
- temporal violation detected.
- low-confidence rate spikes unexpectedly.

### Backtest Alarms

- missing price coverage above threshold.
- benchmark data missing.
- turnover above configured limit.
- backtest leakage detected.

### Regression Alarms

- rank IC decreases versus previous run.
- citation accuracy decreases versus previous run.
- eval runtime regresses.
- artifact generation fails.

## MVP Scope

### Must Have

- schema compliance eval.
- citation resolvability eval.
- temporal correctness eval.
- 3/6/12-month forward return labels.
- rank IC or bucketed forward returns.
- simple long-only top-k portfolio.
- random baseline.
- one finance baseline such as momentum or value.
- JSON report and markdown summary.
- leakage report.

### Should Have

- prompt-only versus model-grounded comparison.
- price-only versus multi-stream comparison.
- transaction cost sensitivity.
- precision@k.
- drawdown and turnover.
- FinanceBench-style evidence overlap.

### Can Defer

- full factor model.
- institutional transaction cost model.
- survivorship-bias-free full universe.
- human reviewer UI.
- advanced entailment model.

## Scaling Path

### Evaluation Scale

- larger universe.
- walk-forward evaluation.
- sector and market-cap splits.
- regime splits.
- bootstrap confidence intervals.
- factor-adjusted performance.

### Reasoning Scale

- claim-level entailment.
- human audit workflow.
- adversarial evidence suites.
- citation quality model.

### Backtesting Scale

- realistic corporate actions.
- capacity and liquidity modeling.
- shorting costs.
- portfolio constraints.
- risk model integration.

## Open Questions

- Should 6-month or 12-month be the headline horizon?
- Should the default portfolio be long-only top-k or long/short spread?
- How should simple value baseline be computed when fundamentals are sparse?
- Should citation support use deterministic rules only in MVP?
- Should FinanceBench-style data be part of evaluation, retrieval, or both?
- Should backtest reports be generated for every training run or only selected model versions?

## Success Criteria

The Evaluation and Backtesting module is successful if:

- model outputs are evaluated separately for reasoning quality and investment signal diagnostics.
- every backtest is time-aware and leakage-checked.
- baselines make results interpretable.
- reports are honest about limitations.
- metrics can compare model, prompt, data, and training-stage changes.
- reviewers understand that this is a rigorous research evaluation system, not a claim of production alpha.
