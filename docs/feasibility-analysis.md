# Feasibility Analysis

## Executive Summary

MarketFM Forge is feasible as a 72-hour technical submission if the goal is scoped correctly.

The realistic short-term goal is not to train a profitable investment model. The realistic goal is to build a credible, end-to-end prototype of a finance foundation model lab for long-term investment research:

- Public financial documents and market data become versioned, traceable training corpora.
- A small reasoning-oriented model is trained or adapted through CPT/DAPT, SFT, and preference tuning.
- The model outputs evidence-grounded investment theses instead of opaque predictions.
- The evaluation stack combines reasoning quality, long-horizon backtesting metrics, and infrastructure health metrics.
- Every run is reproducible, observable, containerized, and deployable through Kubernetes/IaC, with Modal providing the GPU training path.

The long-term ambition is a value-investing reasoning foundation model. The 72-hour artifact should prove the infrastructure contracts, training loop, evaluation discipline, and scaling path.

## What Is Feasible In 72 Hours

### Feasible

- Build a local-first data plane that consumes small public datasets and fixture samples.
- Pull or ingest SEC metadata, fundamentals, historical prices, and selected financial reasoning datasets.
- Create a versioned corpus with source IDs, timestamps, document provenance, and leakage controls.
- Tokenize and pack text into language-model training shards.
- Run a tiny from-scratch GPT training smoke test to prove the foundation-model training stack.
- LoRA-tune a small open causal LM for evidence-grounded investment thesis generation.
- Add a lightweight DPO/preference step that rewards cited, cautious, temporally valid reasoning.
- Produce 3/6/12-month forward-return evaluation for model-generated rankings or buckets.
- Emit metrics, alarms, lineage manifests, checkpoint/resume validation, and an end-to-end ops report.

### Not Feasible To Claim

- Stable alpha from public data in 72 hours.
- A true frontier finance foundation model.
- A production-grade portfolio manager.
- A real-time trading system.
- Outperformance without careful long-horizon, bias-controlled evaluation.

The README and project documentation should explicitly state this boundary.

## Public Data Sources

### Core Data Sources

These are the highest-value sources for the MVP.

- **SEC EDGAR APIs**
  - Use for company submissions, 10-K/10-Q filing metadata, and XBRL `companyfacts`.
  - Provides authoritative fundamentals such as revenue, net income, assets, liabilities, cash flow, debt, and shares.
  - Public API requires no key, but requires a proper `User-Agent`.
  - Main risk: XBRL tags vary by company and time. Limit the ticker universe for the demo.

- **FinanceBench**
  - Open-source sample includes finance questions, answers, evidence strings, filing document references, and justifications.
  - Best use: evidence-grounded reasoning evaluation and SFT examples.
  - Important because it directly tests whether the model can answer financial questions using cited SEC filing evidence.
  - Related paper: [FinanceBench](https://arxiv.org/abs/2311.11944).

- **FinQA, ConvFinQA, and TAT-QA**
  - Best use: financial numerical reasoning and table/text QA.
  - These datasets test whether the model can reason over financial reports rather than only classify sentiment.
  - Related papers: [FinQA](https://aclanthology.org/2021.emnlp-main.300/), [ConvFinQA](https://aclanthology.org/2022.emnlp-main.421/), [TAT-QA](https://nextplusplus.github.io/TAT-QA/).

- **Historical prices**
  - Use `yfinance` or Stooq for adjusted daily prices.
  - Best use: forward-return labels, ranking evaluation, drawdown, volatility, and simple portfolio metrics.
  - Production caveat: public price APIs are not institutional-grade. They are acceptable for a take-home prototype if disclosed.

- **Fama-French factor data**
  - Use Kenneth French's data library through `pandas-datareader`.
  - Best use: benchmark/factor context and optional alpha/beta analysis.
  - This helps avoid evaluating the model only against naive absolute returns.

### Useful Optional Sources

- **Hugging Face SEC 10-K datasets**
  - `JanosAudran/financial-reports-sec` includes 10-K sections and market-reaction labels.
  - Useful for corpus building and lightweight return-linked labels.

- **Earnings call datasets**
  - `RudrakshNanavaty/earnings-call-data` includes S&P 500 earnings calls, 8-K press materials, price anchors, fundamentals, and post-earnings labels.
  - `lamini/earnings-calls-qa` provides earnings-call-derived Q&A data.
  - Useful for SFT and evidence retrieval.

- **Financial instruction datasets**
  - `gbharti/finance-alpaca`, FinGPT sentiment datasets, Financial PhraseBank, and related instruction datasets.
  - Useful as auxiliary SFT data, but should not become the core differentiator.

- **Continuously updated data**
  - GDELT, company RSS feeds, SEC latest filings, and news RSS.
  - Useful to demonstrate continuous ingestion and freshness alarms.
  - For a long-term value model, these feeds update the research corpus; they are not used for low-latency trading.

## Related Work And Positioning

### BloombergGPT

BloombergGPT shows that finance foundation models benefit from large domain-specific corpora. It trained a 50B model on hundreds of billions of financial and general tokens. This validates the direction but is not reproducible in a 72-hour project.

Project implication: MarketFM Forge should focus on the training factory and data contracts that would make such a system possible at smaller scale.

Reference: [BloombergGPT](https://arxiv.org/abs/2303.17564).

### FinGPT

FinGPT emphasizes a data-centric finance LLM approach, real-time financial data, and LoRA/QLoRA adaptation. It is the closest public ecosystem reference.

Project implication: do not clone FinGPT. Differentiate by focusing on evidence-grounded long-term investment reasoning, lineage, backtesting discipline, and infrastructure operability.

Reference: [FinGPT](https://openreview.net/forum?id=YPNirrH0f9).

### PIXIU / FinMA

PIXIU provides financial instruction tuning data and a benchmark suite. It shows that financial instruction tuning is useful, but also that financial LLMs still struggle with numerical reasoning, generalization, and prediction tasks.

Project implication: use public financial instruction datasets as supporting data, but evaluate reasoning and leakage carefully.

Reference: [PIXIU](https://arxiv.org/abs/2306.05443).

### FinanceBench

FinanceBench is especially relevant because it evaluates financial QA against SEC filing evidence. Reported results show that even strong LLM + retrieval systems can fail on financial QA, which validates the need for a disciplined evidence-grounded reasoning stack.

Project implication: FinanceBench-style evidence fields are a strong target for SFT/evaluation.

Reference: [FinanceBench](https://arxiv.org/abs/2311.11944).

### DPO

DPO is a practical preference optimization method that avoids the complexity of PPO-style RLHF. It is suitable for a 72-hour project because preference pairs can be generated from objective rules: cited evidence beats unsupported claims, temporally valid answers beat leaked answers, and calibrated uncertainty beats overconfident speculation.

Reference: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).

## System Design

### High-Level Architecture

MarketFM Forge should have three planes:

- **Data plane**
  - Ingests offline and continuously updated market research data.
  - Stores raw, normalized, corpus, and shard artifacts.
  - Preserves source provenance, timestamps, quality signals, and replay cursors.

- **Training plane**
  - Converts corpora into tokenized and packed training shards.
  - Runs CPT/DAPT, SFT, and DPO/preference training stages.
  - Supports checkpointing, resume, metrics, and artifact lineage.
  - Uses Modal for GPU execution and local CPU runs for smoke tests.

- **Evaluation and operations plane**
  - Runs reasoning metrics, backtesting metrics, and infrastructure metrics.
  - Emits ops reports and alarms.
  - Validates time-aware leakage constraints.
  - Provides Kubernetes and IaC deployment definitions.

### Storage Layout

Use an S3-compatible object layout that also works locally or with MinIO:

- `bronze/`
  - Raw fetched data.
  - Preserve exact source payloads and metadata.

- `silver/`
  - Normalized documents, fundamentals, prices, and events.
  - Include schemas, entity IDs, ticker mapping, and quality scores.

- `corpus/`
  - Foundation-model text corpora.
  - Include source references, `as_of_time`, document type, ticker, fiscal period, and mixture labels.

- `shards/`
  - Tokenized and packed training shards.
  - Include tokenizer config, sequence length, content hashes, and shard manifests.

- `runs/`
  - Training run configs, checkpoints, metrics, and resume state.

- `evals/`
  - Reasoning evals, backtest reports, leakage checks, and ops reports.

- `registry/`
  - Model artifacts, adapter weights, lineage manifests, and evaluation summaries.

### Lineage Contract

Every generated artifact should carry enough metadata to answer:

- What source documents and market data were used?
- What timestamps and `as_of_time` boundaries applied?
- What corpus mixture and tokenizer config produced this shard?
- What model checkpoint consumed the shard?
- What evaluation run scored the model?
- Could this run be reproduced from public data and config?

### Time-Aware Design

Long-term investing evaluation is easy to corrupt with future leakage. The system should enforce:

- `as_of_time` on every example.
- Retrieval restricted to documents available before `as_of_time`.
- Price windows split into historical input windows and future evaluation windows.
- Filing dates and accepted dates stored separately when possible.
- Backtest labels generated only after model outputs are frozen.

## Model Design

### Model Role

The model is a long-term investment reasoning model, not a real-time trading predictor.

It should answer questions like:

- Is this company attractive for long-term ownership as of this date?
- What is the investment thesis?
- What evidence supports or weakens the thesis?
- What are the major risks and uncertainties?
- How does valuation compare with fundamentals and history?
- What should be tracked next?

### Output Contract

The model should produce structured output:

- `investment_view`: rating, ranking score, or expected return bucket.
- `horizon`: 3-month, 6-month, 12-month, or multi-year.
- `confidence`: calibrated confidence or uncertainty.
- `thesis`: concise long-term investment thesis.
- `business_quality`: moat, growth, margins, capital efficiency, balance sheet, or management quality.
- `valuation`: valuation argument based on multiples, fundamentals, or historical comparison.
- `catalysts`: events or developments that could change the thesis.
- `risks`: downside risks and what evidence would invalidate the thesis.
- `evidence`: cited source IDs, timestamps, snippets, and object paths or URLs.
- `as_of_time`: timestamp boundary for all evidence.
- `lineage`: model version, data mixture, shard IDs, prompt version, and eval run ID.

### Training Stages

#### CPT / DAPT

Purpose: adapt the model to the language of filings, earnings calls, financial statements, and investment research.

Feasible demo:

- Train a tiny GPT from scratch on a small finance corpus to prove the training stack.
- Continue-pretrain a small open causal LM with LoRA on a curated finance corpus.
- Track held-out perplexity and training throughput.

#### SFT

Purpose: teach the model to produce structured, evidence-grounded investment reasoning.

Data sources:

- FinanceBench examples converted to evidence-grounded QA.
- FinQA/TAT-QA examples converted to financial reasoning instructions.
- Earnings-call QA examples.
- Synthetic instruction examples generated from SEC facts and evidence snippets.

Target behavior:

- Cite evidence.
- Respect `as_of_time`.
- Separate thesis, valuation, risks, and uncertainty.
- Output valid JSON or a strict markdown schema.

#### Preference Training / DPO

Purpose: align the model toward reliable investment reasoning.

Preference pair examples:

- Chosen: cites evidence. Rejected: unsupported claim.
- Chosen: says uncertainty is high. Rejected: overconfident answer.
- Chosen: uses only pre-`as_of_time` evidence. Rejected: future leakage.
- Chosen: separates durable fundamentals from short-term noise. Rejected: shallow sentiment.
- Chosen: gives risks and disconfirming evidence. Rejected: one-sided bullish story.

DPO is the recommended short-term method because it is stable and simple compared with PPO-style RL.

## Evaluation Design

### Reasoning Evaluation

Evaluate whether the model reasons faithfully from available evidence:

- Evidence coverage.
- Citation accuracy.
- Faithfulness to retrieved context.
- Temporal correctness.
- JSON/schema compliance.
- Numerical reasoning correctness.
- Uncertainty quality.
- Thesis quality and risk awareness.

### Long-Horizon Backtesting Evaluation

Evaluate whether model-generated scores have any investment signal:

- 3-month, 6-month, and 12-month forward returns.
- Rank IC and IC decay.
- Precision@k for top-ranked ideas.
- Bucketed forward-return analysis.
- Long/short spread.
- Benchmark-relative return.
- Volatility, Sharpe, Sortino, max drawdown, Calmar ratio.
- Turnover and average holding period.
- Transaction cost and slippage sensitivity.
- Sector, market-cap, and regime splits.

The project should report these as research diagnostics, not as proof of tradable alpha.

### Infrastructure Evaluation

Evaluate whether the system behaves like real infrastructure:

- Raw data to corpus time.
- Corpus to shards time.
- Shards to first checkpoint time.
- Checkpoint to evaluation report time.
- Total raw-to-eval wall-clock time.
- Tokens/sec.
- Dataloader wait ratio.
- Sequence packing efficiency.
- Checkpoint save time and resume time.
- Object-store bytes read/written.
- Alerts triggered during failure injection.

## Metrics And Alarms

### Data Alarms

- Stale source feed.
- Duplicate spike.
- Low entity mapping coverage.
- High invalid/quarantined record rate.
- Missing price or fundamentals coverage.
- Future leakage detected.

### Training Alarms

- Dataloader wait ratio above threshold.
- Tokens/sec below target.
- Loss is NaN or stalled.
- Checkpoint save failure.
- Resume validation failure.
- Modal job failure.
- GPU OOM.

### Evaluation Alarms

- Schema compliance regression.
- Citation accuracy regression.
- Temporal correctness failure.
- Backtest leakage detected.
- E2E cycle time regression.

## Recommended MVP

### Universe

Start with 20-50 large-cap U.S. equities. Prefer names with rich SEC filings, fundamentals, and price history. The MVP can include a fixture subset to guarantee deterministic demo behavior.

### Workflow

1. Ingest SEC metadata, company facts, selected filings or fixture filing excerpts, prices, and optional earnings-call snippets.
2. Normalize entities and attach `as_of_time`.
3. Build corpus mixtures for filings, fundamentals summaries, earnings/news snippets, and reasoning QA.
4. Tokenize and pack shards.
5. Run tiny GPT CPT smoke test.
6. Run LoRA SFT on a small open causal LM.
7. Run DPO preference tuning on generated preference pairs.
8. Generate investment thesis outputs for a small evaluation universe.
9. Run reasoning checks and 3/6/12-month forward-return backtest.
10. Emit ops report, artifact lineage, and alert simulation results.

### Baselines

Include simple baselines so results are interpretable:

- Random ranking.
- Momentum ranking.
- Value factor ranking such as low price-to-sales or price-to-book if available.
- Prompt-only base model without SFT.
- SFT model without DPO.

## Main Risks

### Data Quality Risk

Public datasets vary in schema, licensing, and coverage.

Mitigation: use small deterministic fixtures, disclose external dependencies, and isolate live data ingestion from the guaranteed demo path.

### Model Quality Risk

Small models may not produce high-quality investment reasoning.

Mitigation: evaluate concrete behaviors such as schema compliance, citation use, temporal correctness, and held-out reasoning tasks rather than claiming investment alpha.

### Backtest Bias Risk

Financial backtests are vulnerable to survivorship bias, look-ahead bias, and data snooping.

Mitigation: enforce `as_of_time`, disclose universe construction, run walk-forward splits, and label backtests as research diagnostics.

### Scope Risk

CPT, SFT, DPO, data infra, K8s, Modal, observability, and backtesting are too much for a polished full implementation.

Mitigation: make the deterministic local demo excellent, and document the scalable paths clearly. The core repo should prove the contracts, not build every production feature.

## Final Feasibility Verdict

MarketFM Forge is a strong take-home project if positioned as infrastructure for a long-term value investing reasoning model.

The most defensible claim is:

> This repository demonstrates how to build the data, training, evaluation, and operations loop for a finance foundation model that produces evidence-grounded long-term investment reasoning.

The project should not claim:

> This model already produces tradable alpha.

That distinction makes the project technically ambitious, honest, and aligned with Deeter's interest in data infrastructure, training systems, and finance-domain foundation models.
