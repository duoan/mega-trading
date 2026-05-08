# Training Plane Design

## Purpose

The Training Plane turns versioned financial corpora and tokenized shards into model capability. It is the central module for demonstrating alignment with the AI Infrastructure Engineer (Training Systems) role.

The finance reasoning model is the workload. The Training Plane is the signal:

- Can the system consume data efficiently?
- Can it run multiple training stages reproducibly?
- Can it measure learning per unit of compute?
- Can it recover from failures?
- Can it expose bottlenecks clearly?
- Can it scale from local smoke tests to GPU jobs and eventually distributed training?

## Design Goals

### Model Capability Goals

- Adapt a small causal language model to financial language.
- Teach structured, evidence-grounded long-term investment reasoning.
- Align the model toward cautious, cited, temporally valid outputs.

### Infrastructure Goals

- Use versioned tokenized shards, not ad hoc in-memory datasets.
- Share the same artifact contracts across CPT/DAPT, SFT, and DPO.
- Support local CPU smoke tests and Modal GPU jobs.
- Emit detailed training efficiency metrics.
- Support checkpoint save/resume validation.
- Make bottlenecks visible: data loading, compute, checkpointing, memory, and cost.

### Research Velocity Goals

- Produce a first checkpoint quickly.
- Run small experiments from config changes.
- Compare stages and model variants through manifests.
- Keep the deterministic demo runnable by reviewers.

## Training Stages

### CPT / DAPT

Continual pretraining or domain-adaptive pretraining adapts the model to financial text.

Inputs:

- CPT corpus shards from filings, fundamentals summaries, earnings snippets, and financial QA context.

Outputs:

- CPT checkpoint or LoRA adapter.
- Training metrics.
- Held-out finance perplexity.
- Run manifest.

Metrics:

- train loss.
- eval loss.
- perplexity.
- tokens/sec.
- dataloader wait ratio.
- sequence packing efficiency.
- checkpoint time.

MVP implementation:

- Train a tiny GPT from scratch on local fixture shards.
- Optionally LoRA-continue-pretrain a small open causal LM on Modal.

### SFT

Supervised fine-tuning teaches the model to produce structured investment reasoning.

Inputs:

- SFT instruction shards.
- Evidence packs.
- Target structured JSON or markdown outputs.

Target behavior:

- Form an investment thesis.
- Discuss business quality, valuation, catalysts, risks, and uncertainty.
- Cite source evidence.
- Respect `as_of_time`.
- Output a parseable schema.

Outputs:

- SFT checkpoint or adapter.
- Schema compliance metrics.
- citation inclusion metrics.
- eval examples.

MVP implementation:

- Fine-tune a small open causal LM with LoRA.
- Use a small dataset derived from FinanceBench-style examples, fixture filings, and synthetic thesis examples.

### DPO / Preference Tuning

DPO aligns model behavior without the operational complexity of PPO-style RLHF.

Inputs:

- Preference pair shards.

Preference examples:

- Chosen answer cites evidence; rejected answer makes unsupported claims.
- Chosen answer respects `as_of_time`; rejected answer leaks future information.
- Chosen answer expresses uncertainty; rejected answer overclaims.
- Chosen answer discusses risks; rejected answer gives one-sided bullish reasoning.
- Chosen answer separates durable fundamentals from short-term noise.

Outputs:

- DPO adapter.
- preference loss.
- preference win-rate proxy.
- reasoning eval improvements.

MVP implementation:

- Implement a lightweight DPO training path if time allows.
- If full DPO is too expensive, implement a preference-loss smoke test and document the full path.

## Model Choices

### Tiny GPT From Scratch

Purpose:

- Prove the foundation-model training stack end to end.

Why include it:

- It demonstrates corpus -> tokenization -> packed shards -> language model training -> checkpoint/resume.
- It avoids hiding all training logic behind Hugging Face Trainer.

Expected quality:

- Not expected to reason well.
- Used for infrastructure validation and smoke metrics.

### Small Open Causal LM

Purpose:

- Produce credible reasoning outputs with SFT/DPO.

Candidate models:

- `Qwen2.5-0.5B-Instruct`
- `TinyLlama-1.1B-Chat`
- Another small permissive Hugging Face causal LM.

Selection criteria:

- Can run locally or on small GPU.
- Supports LoRA.
- Reasonable instruction-following baseline.
- License compatible with public demo.
- Stable tokenizer and model loading.

### Adapter Strategy

Use LoRA for the practical adaptation path.

Benefits:

- Lower GPU memory.
- Lower cost.
- Fast iteration.
- Small artifacts.
- Easier comparison across CPT/SFT/DPO stages.

Configurable parameters:

- LoRA rank.
- alpha.
- dropout.
- target modules.
- learning rate.
- batch size.
- gradient accumulation.
- sequence length.

## Config Design

Training should be driven by explicit configs.

Recommended layout:

```text
configs/
  train/
    cpt_tiny_gpt.yaml
    cpt_lora.yaml
    sft_lora.yaml
    dpo_lora.yaml
  modal/
    gpu_small.yaml
    gpu_medium.yaml
  data/
    mixture_demo.yaml
    mixture_financebench.yaml
```

### Common Config Fields

- `run_name`
- `stage`
- `base_model`
- `tokenizer`
- `data_mixture_version`
- `shard_paths`
- `sequence_length`
- `micro_batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `max_steps`
- `eval_interval`
- `checkpoint_interval`
- `precision`
- `seed`
- `output_dir`
- `resume_from`

### Efficiency Config Fields

- `num_workers`
- `prefetch_factor`
- `pin_memory`
- `packing_enabled`
- `target_tokens_per_batch`
- `checkpoint_async`
- `modal_gpu_type`
- `expected_memory_gb`
- `max_cost_usd`

## Data Loading

### Requirements

The dataloader must:

- Read versioned shard manifests.
- Preserve deterministic sample ordering when needed.
- Support streaming reads.
- Track dataloader wait time.
- Track padding and packing efficiency.
- Preserve source IDs for debugging and lineage.
- Fail fast on tokenizer or shard mismatches.

### MVP Dataloader

The MVP can use local files or MinIO-backed shard files.

Supported shard formats:

- JSONL for readability in early implementation.
- Arrow/parquet or binary packed arrays as a scaling path.

### Scaling Dataloader

Future scaling path:

- streaming shards from S3.
- local cache.
- async prefetch.
- memory-mapped packed token arrays.
- distributed shard assignment.
- deterministic resume cursor.

## Checkpointing

Checkpointing is a first-class subsystem.

### Checkpoint Contents

Each checkpoint should include:

- model weights or adapter weights.
- optimizer state.
- scheduler state.
- step number.
- epoch or shard cursor.
- random seed state.
- training config hash.
- tokenizer config hash.
- data mixture version.
- shard manifest IDs.
- metrics history.

### Resume Validation

Resume should fail fast if:

- config hash differs unexpectedly.
- tokenizer differs.
- base model differs.
- training stage differs.
- required shards are missing.
- data mixture version changed.
- checkpoint is incomplete.

### MVP Resume Test

The deterministic demo should include:

1. Train for a few steps.
2. Save checkpoint.
3. Simulate interruption.
4. Resume.
5. Verify step number and config hash.
6. Run a small eval.
7. Record `checkpoint_resume_passed=true` in ops report.

## Metrics

### Training Metrics

- `train_loss`
- `eval_loss`
- `perplexity`
- `learning_rate`
- `grad_norm`
- `tokens_seen`
- `examples_seen`
- `step_time_seconds`
- `tokens_per_second`
- `examples_per_second`

### Efficiency Metrics

- `dataloader_wait_ratio`
- `sequence_packing_efficiency`
- `padding_waste_ratio`
- `checkpoint_save_seconds`
- `checkpoint_resume_seconds`
- `tokens_per_gpu_second`
- `estimated_cost_per_1m_tokens`
- `object_store_read_bytes`
- `object_store_write_bytes`

### Reliability Metrics

- `checkpoint_success_total`
- `checkpoint_failure_total`
- `resume_success_total`
- `resume_failure_total`
- `nan_loss_detected`
- `oom_detected`
- `modal_job_retries`

## Learning Per Unit Of Compute

The project should explicitly report learning per unit of compute.

Prototype definition:

- improvement in held-out finance loss per GPU-second.
- improvement in reasoning eval score per GPU-second.
- improvement in schema/citation metrics per dollar estimate.

Example derived metrics:

- `loss_delta_per_gpu_second`
- `reasoning_score_delta_per_gpu_second`
- `citation_accuracy_delta_per_gpu_second`
- `eval_improvement_per_estimated_dollar`

These numbers do not need to be impressive. They need to be measured and explainable.

## Modal Integration

Modal provides the GPU training path.

### Entry Points

Recommended functions:

- `modal run modal_app.py::cpt`
- `modal run modal_app.py::sft`
- `modal run modal_app.py::dpo`
- `modal run modal_app.py::eval_model`

### Modal Job Responsibilities

- Pull or mount training shards.
- Load config.
- Run training stage.
- Write checkpoints and adapters.
- Write training metrics.
- Write run manifest.
- Return summary.

### Local Fallback

Every Modal job should have a local equivalent:

- `marketfm train cpt --local`
- `marketfm train sft --local`
- `marketfm train dpo --local`

This keeps the project reviewable without Modal credentials.

## CLI And API

### CLI

Commands:

- `marketfm train cpt --config configs/train/cpt_tiny_gpt.yaml`
- `marketfm train sft --config configs/train/sft_lora.yaml`
- `marketfm train dpo --config configs/train/dpo_lora.yaml`
- `marketfm train resume --checkpoint <path>`
- `marketfm train validate-checkpoint --checkpoint <path>`
- `marketfm train profile --run-id <run_id>`

### API

Optional endpoints:

- `POST /train/runs`
- `GET /train/runs/{run_id}`
- `POST /train/runs/{run_id}/cancel`
- `POST /train/resume`
- `GET /train/runs/{run_id}/metrics`

## Failure Modes

### Data Loading Failures

- missing shard.
- corrupt shard.
- tokenizer mismatch.
- slow object store.
- high dataloader wait.

Mitigation:

- shard manifest validation.
- local caching.
- fail-fast config checks.
- dataloader wait alarms.

### Training Failures

- NaN loss.
- OOM.
- exploding gradients.
- stalled loss.
- Modal job timeout.
- dependency mismatch.

Mitigation:

- precision config.
- gradient clipping.
- smaller sequence length.
- smaller batch size.
- retry policy.
- detailed run manifest.

### Checkpoint Failures

- partial checkpoint.
- config mismatch.
- missing optimizer state.
- corrupt adapter weights.

Mitigation:

- atomic checkpoint writes.
- checkpoint manifest.
- checksum validation.
- resume test.

## MVP Scope

### Must Have

- tiny GPT CPT smoke test.
- SFT training path for a small causal LM or lightweight local model.
- DPO/preference path or preference-loss smoke test.
- config-driven training stages.
- local training CLI.
- training metrics JSONL.
- checkpoint save/resume validation.
- run manifests.
- ops report integration.

### Should Have

- Modal GPU entry points.
- LoRA adapter training.
- sequence packing.
- dataloader wait metrics.
- cost estimates.
- simple training profiler.

### Can Defer

- full multi-GPU training.
- FSDP/DeepSpeed implementation.
- sophisticated optimizer sharding.
- production scheduler.
- large-scale hyperparameter sweeps.

## Scaling Path

### Single GPU To Multi-GPU

- add FSDP or DeepSpeed.
- shard model and optimizer states.
- partition data shards across workers.
- aggregate metrics.
- coordinate checkpoints.

### Local Shards To S3 Streaming

- stream shards from S3.
- add local cache.
- prefetch next shard.
- measure object-store latency.
- retry failed reads.

### Manual Runs To Experiment Platform

- job queue.
- experiment registry.
- automatic comparison reports.
- sweeps over data mixtures and training configs.
- cost budgets per run.

### Basic Metrics To Performance Engineering

- detailed step-time breakdown.
- CPU utilization.
- GPU utilization.
- memory utilization.
- network read bandwidth.
- object-store latency.
- profiler traces.

## Open Questions

- Which small causal LM should be the default?
- Should DPO be fully implemented in the MVP or represented as a preference-loss smoke test?
- Should local SFT use Hugging Face Trainer or a custom loop for clearer infra visibility?
- Which metric should headline learning per unit compute?
- Should sequence packing be implemented immediately or after basic training works?
- How much Modal integration should be required for the take-home demo?

## Success Criteria

The Training Plane is successful if:

- A reviewer can run a local training smoke test.
- Training consumes versioned shards, not ad hoc data.
- CPT, SFT, and preference stages share the same artifact contracts.
- Checkpoint/resume is tested and reported.
- Metrics expose throughput, dataloader wait, checkpoint time, and cost estimates.
- The design clearly scales to distributed GPU training.
- The project demonstrates systems thinking about efficient training, not just model fine-tuning.
