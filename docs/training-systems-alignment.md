# Training Systems Alignment

## Why This Document Exists

The target role is **AI Infrastructure Engineer (Training Systems)**. The project should therefore be evaluated less like a finance app and more like a training infrastructure artifact.

Mega-Trading should show that the builder can turn data and compute into model capability under real constraints:

- GPU utilization.
- Throughput.
- Memory pressure.
- I/O bottlenecks.
- Checkpoint reliability.
- Training cost.
- Failure recovery.
- Time from idea to evaluated model.
- Learning per unit of compute.

The multi-stream finance model provides the domain. The training system and model architecture are the joint signal: the project should show that data contracts, labels, model inputs, training loops, and metrics were designed together.

## Role Requirements And Project Response

### Build And Optimize Distributed Training Systems

Project response:

- Provide training entry points for multi-stream supervised training plus CPT/DAPT, SFT, and preference support paths.
- Keep training stages config-driven so model requirements can be translated into system configurations.
- Support local CPU smoke tests and Modal GPU jobs through the same artifact contracts.
- Design the training loop so it can later scale to FSDP/DeepSpeed/Ray without rewriting data contracts.

72-hour implementation target:

- One small local training path.
- One Modal GPU training path.
- Shared configs, manifests, metrics, and checkpointing.

Long-term path:

- Multi-GPU and multi-node execution.
- Distributed shard streaming.
- Elastic workers.
- FSDP/DeepSpeed integration.
- Training job queue and experiment scheduler.

### Optimize GPU Utilization, Throughput, And Training Efficiency

Project response:

- Measure `tokens/sec`, `examples/sec`, `step_time`, `dataloader_wait_ratio`, checkpoint save time, and sequence packing efficiency.
- Report whether training is compute-bound, input-bound, or checkpoint-bound.
- Use packed sequences and streaming shards to reduce padding waste and dataloader starvation.
- Track object-store bytes read/written to expose I/O cost.

Core metrics:

- `tokens_per_second`
- `examples_per_second`
- `step_time_p50`
- `step_time_p95`
- `dataloader_wait_ratio`
- `sequence_packing_efficiency`
- `checkpoint_save_seconds`
- `checkpoint_resume_seconds`
- `tokens_per_gpu_second`
- `estimated_cost_per_1m_tokens`

Why this matters:

The JD explicitly values systems that maximize learning per unit of compute, not just systems that run. The project should make training efficiency visible in every run report.

### Translate Model Requirements Into Efficient System Configurations

Project response:

- Represent each training stage as a config:
  - model name and size
  - sequence length
  - precision
  - batch size
  - gradient accumulation
  - LoRA rank
  - checkpoint interval
  - shard mixture
  - Modal GPU type
  - expected memory budget

The system should make trade-offs explicit:

- Longer context improves evidence reasoning but increases memory and step time.
- Larger batch improves utilization but may reduce iteration speed.
- Higher LoRA rank may improve adaptation but increases memory and compute.
- Frequent checkpointing improves recovery but increases overhead.
- Larger corpus mixture improves coverage but slows time-to-first-eval.

72-hour implementation target:

- Include stage configs under `configs/`.
- Emit a run summary that records the selected config and observed efficiency.
- Document one example tuning decision, such as reducing sequence length or increasing packing efficiency to improve time-to-result.

### Improve Training Speed, Cost Efficiency, And Reliability

Project response:

- Optimize for fast local iteration and credible GPU training.
- Separate smoke-test runs from scalable runs.
- Track raw-data-to-eval wall-clock time.
- Validate checkpoint/resume.
- Include failure-injection paths.

Efficiency targets for the prototype:

- `make demo` should complete quickly enough for a reviewer to run.
- First checkpoint should be produced early in the run.
- Evaluation should run automatically after training.
- Ops report should show where time was spent.

Reliability targets:

- Training can resume from checkpoint.
- Run manifests preserve configs and shard IDs.
- Resume validation catches mismatched configs or missing shards.
- Failed records are quarantined instead of silently ignored.

### Debug High-Cost Training Failures

Project response:

- Treat debugging as part of the artifact.
- Provide structured run logs, metrics, and alert rules.
- Include failure-injection demos for common issues.

Failure modes to simulate or document:

- Stale source feed.
- Missing price/fundamental coverage.
- Future leakage detected.
- Slow dataloader.
- NaN loss.
- Checkpoint save failure.
- Resume mismatch.
- Modal job failure.
- GPU OOM.

The project should include runbook entries for:

- How to identify whether training is data-bound or compute-bound.
- How to debug a bad checkpoint.
- How to handle a stale feed.
- How to reduce memory pressure.
- How to reduce time-to-first-eval.

## Training System Design Implications

### Data Loading

The dataloader should be designed as if it will eventually handle large corpora:

- Stream shards from object storage.
- Support deterministic shard ordering for reproducibility.
- Preserve source IDs for auditability.
- Measure dataloader wait time.
- Track padding and packing waste.
- Keep CPU tokenization separate from GPU training.

For the MVP, tokenized local shards are enough. The important design point is that training consumes versioned shards, not ad hoc pandas dataframes.

### Checkpointing

Checkpointing should be a first-class subsystem.

Each checkpoint should include:

- model or adapter weights
- optimizer state when applicable
- scheduler state
- step number
- random seed state
- shard cursor
- config hash
- data mixture version
- metrics history

Resume validation should fail fast if:

- the config hash changed unexpectedly
- required shards are missing
- tokenizer config differs
- model base checkpoint differs
- training stage differs

### Metrics

The system should emit metrics at multiple levels:

- data ingestion metrics
- shard building metrics
- training loop metrics
- checkpoint metrics
- evaluation metrics
- end-to-end cycle metrics

This directly aligns with the role's emphasis on measurable efficiency.

### Alarms

Alarms should be defined as code, not only described in prose.

Example alarm categories:

- Data freshness SLO violated.
- Dataloader wait ratio too high.
- Tokens/sec below threshold.
- Checkpoint failed or too slow.
- Loss is NaN or stalled.
- E2E cycle time regressed.
- Future leakage detected.

### Cost Awareness

Every GPU run should report:

- GPU type.
- Run duration.
- Estimated GPU seconds.
- Tokens processed.
- Tokens per GPU-second.
- Estimated cost per 1M tokens.

This gives the project a training-infra signature rather than a notebook-demo signature.

## Recommended Project Framing

Use this framing in the README and submission:

> Mega-Trading is a foundation model of trading. It demonstrates how offline and continuously updated financial data becomes traceable model training signal, how supervised and explanation stages consume that signal, and how training efficiency, checkpoint reliability, and end-to-end time-to-result are measured.

Avoid this framing:

> This is a finance prediction model that makes profitable investment calls.

The model is the domain workload. The infrastructure is the evaluation signal for the role.

## Concrete 72-Hour Deliverables

The training-system-specific deliverables should be:

- `configs/train/*.yaml` for CPT, SFT, and DPO stages.
- A local training loop that emits training metrics.
- A Modal GPU training entry point.
- Tokenized packed shards with manifests.
- Checkpoint save/resume validation.
- Ops report with training efficiency metrics.
- Alert rules for slow dataloader, low throughput, failed checkpoint, and e2e regression.
- Failure-injection command for at least one training failure and one data failure.
- README section explaining learning per unit of compute.

## Success Criteria

The project is aligned with the AI Infrastructure Engineer role if a reviewer can see:

- The system measures more than model loss.
- The training loop is reproducible from data and config manifests.
- The data pipeline produces shards suitable for efficient training.
- The model stages can be launched locally and on GPU infrastructure.
- The system can recover from checkpointed failure.
- Bottlenecks are visible in metrics and run reports.
- The design has a clear path to distributed GPU training.

That is the strongest signal for this JD.
