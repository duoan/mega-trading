# Training Plane Design

## Purpose

The Training Plane turns versioned order-flow token streams into model capability. The key contract is that data processing, tokenization, model input, training metrics, checkpoints, and evaluation all agree on the same paper-style sequence schema.

## Training Workload

The primary workload is `TradingModel`:

- paper-style event streams with action, side, relative price, price depth, relative size, and interarrival time.
- fitted composite event tokens, one token per event.
- fixed-length autoregressive token blocks.
- Llama 3-style decoder trained with next-token cross entropy.
- evaluation with loss, perplexity, top-k accuracy, and generated event-feature distributions.

## Inputs

Training reads partitioned NumPy token shards:

- `numpy/part-*/tokens.npy`: `int64` arrays with shape `[partition_rows, block_size + 1]`.
- `numpy/part-*/ticker_ids.npy`: `int32` arrays mapping each row to a ticker id.
- `tokens-numpy.json`: dtype, shape, ticker map, and partition paths.

The profile artifact records the stream contract, feature order, sequence counts, block size, event size, vocabulary size, binning method, and tokenizer path.

## Outputs

Each training run writes:

- `runs/<run_id>/metrics.json`
- `runs/<run_id>/checkpoint.pt`
- `manifests/training/<run_id>.json`

Periodic checkpoints can also be written under `runs/<run_id>/checkpoints/step-*.pt` when `training.checkpoint_interval` is set. Resume uses `training.resume_from_checkpoint` and validates the shard path, stream contract, vocabulary size, and block size before loading model and optimizer state.

## Metrics

Training metrics include next-token loss, perplexity, top-1/top-5 token accuracy, validation loss/perplexity, tokens/sec, checkpoint path, and manifest path.

## Distributed Execution

The training loop is built around Hugging Face Accelerate. A normal Python or console-script launch runs single process. `torchrun` or `accelerate launch` turns the same CLI into DDP. Setting `training.distributed_strategy=fsdp` passes an FSDP plugin to Accelerate for sharded training.

Performance-oriented switches are config-driven:

- `training.gradient_accumulation_steps`
- `training.compile` and `training.compile_mode`
- `training.attention_backend`: `auto`, `flash`, `efficient`, or `math`
- `training.precision`: `auto`, `fp32`, or `mixed`

The model already uses PyTorch scaled dot-product attention, so the flash path is selected through CUDA SDPA backends instead of a separate attention implementation.

## Modal

`modal_train.py` provides the cloud launcher. The single-node function runs the existing CLI on one GPU node. The clustered function uses `modal.experimental.clustered(..., rdma=True)` and starts `torch.distributed.run` with the Modal container rank, master address, and one process per GPU.
