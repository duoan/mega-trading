"""Training loop for next-token order-flow modeling."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from accelerate import Accelerator, DataLoaderConfiguration
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mega_trading.checkpoint import load_checkpoint_model_state, normalized_model_state_dict
from mega_trading.config import TrainConfig
from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.dataset import NumpyTickerTimeDataset, cycle_batches, prepared_split_counts, split_totals
from mega_trading.events import STREAM_CONTRACT
from mega_trading.model import TradingModel


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: str
    metrics_path: str
    manifest_path: str


class _NoopProfiler:
    def __enter__(self) -> "_NoopProfiler":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def step(self) -> None:
        return None


class _AsyncMetricLogger:
    """Serialize MLflow and metrics.json writes without blocking the training step."""

    def __init__(self, accelerator: Accelerator, store: LocalObjectStore, metrics_path: Path) -> None:
        self.accelerator = accelerator
        self.store = store
        self.metrics_path = metrics_path
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="metrics-logger")
        self.futures: list[Future[None]] = []

    def __enter__(self) -> "_AsyncMetricLogger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def submit(self, *, metrics: list[dict[str, object]], tracker_payload: dict[str, float], step: int) -> None:
        metrics_snapshot = [dict(row) for row in metrics]
        payload_snapshot = dict(tracker_payload)
        self.futures.append(
            self.executor.submit(self._write, metrics_snapshot=metrics_snapshot, tracker_payload=payload_snapshot, step=step)
        )

    def close(self) -> None:
        try:
            for future in self.futures:
                future.result()
        finally:
            self.executor.shutdown(wait=True)

    def _write(self, *, metrics_snapshot: list[dict[str, object]], tracker_payload: dict[str, float], step: int) -> None:
        self.accelerator.log(tracker_payload, step=step)
        self.store.write_json(self.metrics_path, {"metrics": metrics_snapshot})


class _AsyncCheckpointWriter:
    """Write checkpoint payloads in the background after the training thread snapshots state."""

    def __init__(self, store: LocalObjectStore) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-writer")
        self.futures: list[Future[None]] = []

    def __enter__(self) -> "_AsyncCheckpointWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def submit(self, checkpoint_path: str, payload: dict[str, Any]) -> None:
        target = _checkpoint_target(self.store, checkpoint_path)
        self.futures.append(self.executor.submit(_write_checkpoint_payload, target, payload))

    def close(self) -> None:
        try:
            for future in self.futures:
                future.result()
        finally:
            self.executor.shutdown(wait=True)


@contextmanager
def _nvtx_range(config: TrainConfig, name: str, step: int | None = None) -> Iterator[None]:
    if not config.nvtx_enabled or not torch.cuda.is_available():
        yield
        return
    nvtx = getattr(torch.cuda, "nvtx", None)
    if nvtx is None:
        yield
        return
    label = f"step={step} {name}" if step is not None else name
    nvtx.range_push(label)
    try:
        yield
    finally:
        nvtx.range_pop()


class MuonAdamW(torch.optim.Optimizer):
    """Hybrid Muon/AdamW optimizer for transformer training."""

    def __init__(self, param_groups: list[dict[str, Any]]) -> None:
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            algorithm = str(group["algorithm"])
            if algorithm == "muon":
                self._step_muon_group(group)
            elif algorithm == "adamw":
                self._step_adamw_group(group)
            else:
                raise ValueError(f"unsupported optimizer algorithm: {algorithm}")
        return loss

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        momentum = float(group["momentum"])
        ns_steps = int(group["ns_steps"])
        buckets: dict[tuple[tuple[int, ...], torch.dtype, str], list[tuple[torch.nn.Parameter, torch.Tensor]]] = {}
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients")
            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
            buffer = state["momentum_buffer"]
            buffer.mul_(momentum).add_(parameter.grad)
            # Muon's Nesterov-style update is orthogonalized before applying the weight step.
            update = parameter.grad.add(buffer, alpha=momentum)
            key = (tuple(update.shape), update.dtype, str(update.device))
            buckets.setdefault(key, []).append((parameter, update))
        for bucket in buckets.values():
            updates = torch.stack([update for _, update in bucket])
            orthogonalized = _muon_orthogonalize_batch(updates, ns_steps)
            for (parameter, _), update in zip(bucket, orthogonalized.unbind(dim=0), strict=True):
                parameter.add_(update, alpha=-lr)

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise RuntimeError("MuonAdamW AdamW group does not support sparse gradients")
            grad = parameter.grad
            if grad.is_complex():
                raise RuntimeError("MuonAdamW does not support complex parameters")
            state = self.state[parameter]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            step = int(state["step"])
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            parameter.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))


class Trainer:
    def __init__(self, store: LocalObjectStore, config: TrainConfig) -> None:
        self.store = store
        self.config = config
        self.paths = ArtifactPaths(run_id=config.run_id)

    def train(self, shard_path: str) -> TrainResult:
        torch.manual_seed(self.config.seed)
        profile_path = _profile_path(shard_path)
        profile = self.store.read_json(profile_path)
        if profile.get("stream_contract") != STREAM_CONTRACT:
            raise ValueError(f"{profile_path} is not a valid token profile")
        numpy_metadata = dict(profile.get("numpy_dataset", {}))
        ticker_counts = {str(key): int(value) for key, value in dict(profile["sequence_counts"]).items()}
        split_counts = prepared_split_counts(numpy_metadata, ticker_counts, self.config.validation_fraction)
        split_count_totals = split_totals(split_counts)
        train_count = split_count_totals["train"]
        validation_count = split_count_totals["validation"]
        backtest_count = split_count_totals["backtest"]
        if train_count <= 0:
            raise ValueError("training shard has no train rows")

        device = _resolve_device(self.config.device)
        precision = _resolve_precision(self.config.precision, device)
        accelerator = _accelerator(device, precision, self.config)
        dataset_format = _dataset_format(self.store, profile)
        numpy_array_cache: dict[str, Any] = {}
        train_dataset = _dataset(
            self.store,
            profile,
            shard_path,
            split_counts,
            split="train",
            preload_numpy_arrays=self.config.preload_numpy_arrays,
            array_cache=numpy_array_cache,
        )
        validation_dataset = (
            _dataset(
                self.store,
                profile,
                shard_path,
                split_counts,
                split="validation",
                preload_numpy_arrays=self.config.preload_numpy_arrays,
                array_cache=numpy_array_cache,
            )
            if validation_count
            else None
        )
        dataloader_kwargs = _dataloader_kwargs(self.config, device)
        train_loader = DataLoader(train_dataset, **dataloader_kwargs)
        validation_loader = (
            DataLoader(validation_dataset, **dataloader_kwargs) if validation_dataset else None
        )

        model = TradingModel(
            vocab_size=int(profile["vocab_size"]),
            block_size=int(profile["block_size"]),
            hidden_dim=self.config.hidden_dim,
            layers=self.config.layers,
            attention_heads=self.config.attention_heads,
            kv_heads=self.config.kv_heads,
            intermediate_dim=self.config.intermediate_dim,
            dropout=self.config.dropout,
            rope_theta=self.config.rope_theta,
            norm_eps=self.config.norm_eps,
            attention_backend=self.config.attention_backend,
        )
        model.to(device)
        optimizer = _build_optimizer(model, self.config)
        scheduler = _build_lr_scheduler(optimizer, self.config)
        resume_state = _load_resume_checkpoint(self.store, self.config.resume_from_checkpoint)
        start_step = 1
        metrics: list[dict[str, object]] = []
        if resume_state is not None:
            _validate_resume_checkpoint(resume_state, profile, shard_path)
            load_checkpoint_model_state(model, resume_state)
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            start_step = int(resume_state.get("step", 0)) + 1
            metrics = [dict(row) for row in resume_state.get("metrics", [])]

        _init_trackers(accelerator, self.store, self.config, shard_path, profile, train_count, validation_count, precision)
        _log_model_summary_to_mlflow(
            accelerator=accelerator,
            config=self.config,
            model=model,
            profile=profile,
            device=device,
        )
        model = _maybe_compile_model(model, self.config, device)
        if validation_loader is None:
            model, optimizer, scheduler, train_loader = accelerator.prepare(model, optimizer, scheduler, train_loader)
        else:
            model, optimizer, scheduler, train_loader, validation_loader = accelerator.prepare(
                model,
                optimizer,
                scheduler,
                train_loader,
                validation_loader,
            )
        loss_fn = nn.CrossEntropyLoss()
        iterator = cycle_batches(train_loader)
        metrics_path = self.paths.run("metrics.json")
        progress = _progress_bar(accelerator, self.config, start_step)
        last_step = start_step - 1
        metric_window_tokens = 0
        metric_window_data_wait = 0.0
        metric_window_compute = 0.0
        metric_window_step = 0.0
        metric_window_steps = 0

        profiler = _training_profiler(self.store, self.config, device, accelerator)
        metric_logger = _AsyncMetricLogger(accelerator, self.store, metrics_path)
        checkpoint_writer = _AsyncCheckpointWriter(self.store)
        with profiler as active_profiler, metric_logger, checkpoint_writer:
            for step in range(start_step, self.config.max_steps + 1):
                last_step = step
                step_started = perf_counter()
                with _nvtx_range(self.config, "train/data_wait", step):
                    batch = next(iterator)
                data_wait_elapsed = max(perf_counter() - step_started, 0.0)
                # Separate compiled forward captures per step when CUDA graphs are enabled (e.g. compile_mode reduce-overhead).
                with _nvtx_range(self.config, "train/cudagraph_mark", step):
                    _maybe_cudagraph_mark_step_begin(self.config, device)
                started = perf_counter()
                with accelerator.accumulate(model):
                    with _nvtx_range(self.config, "train/forward", step):
                        with accelerator.autocast(), _attention_kernel_context(self.config.attention_backend, device):
                            logits = model(batch["input_ids"])
                            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
                    with _nvtx_range(self.config, "train/backward", step):
                        accelerator.backward(loss)
                    learning_rate = _current_learning_rate(optimizer)
                    with _nvtx_range(self.config, "train/optimizer", step):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                elapsed = max(perf_counter() - started, 1e-9)
                step_elapsed = max(perf_counter() - step_started, 1e-9)
                metric_window_tokens += int(batch["labels"].numel())
                metric_window_data_wait += data_wait_elapsed
                metric_window_compute += elapsed
                metric_window_step += step_elapsed
                metric_window_steps += 1
                should_record_metrics = _should_record_training_metrics(step, self.config)
                metric_row: dict[str, object] | None = None
                if accelerator.is_main_process and should_record_metrics:
                    with _nvtx_range(self.config, "train/metrics", step):
                        train_loss = float(loss.detach().cpu())
                        step_rates = _metric_window_rates(
                            token_count=int(batch["labels"].numel()),
                            compute_seconds=elapsed,
                            step_seconds=step_elapsed,
                            accelerator_processes=accelerator.num_processes,
                        )
                        window_rates = _metric_window_rates(
                            token_count=metric_window_tokens,
                            compute_seconds=metric_window_compute,
                            step_seconds=metric_window_step,
                            accelerator_processes=accelerator.num_processes,
                        )
                        metric_row = {
                            "step": step,
                            "stage": "training",
                            "optimizer": self.config.optimizer,
                            "lr_schedule": self.config.lr_schedule,
                            "learning_rate": learning_rate,
                            "train_loss": train_loss,
                            "train_perplexity": _perplexity(train_loss),
                            "train_top1_accuracy": _topk_accuracy(logits, batch["labels"], 1),
                            "train_top5_accuracy": _topk_accuracy(logits, batch["labels"], 5),
                            "train_sequence_count": train_count,
                            "validation_sequence_count": validation_count,
                            "backtest_sequence_count": backtest_count,
                            "world_size": accelerator.num_processes,
                            "distributed_strategy": self.config.distributed_strategy,
                            "dataset_format": dataset_format,
                            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                            "compile_enabled": self.config.compile,
                            "attention_backend": self.config.attention_backend,
                            "metric_interval": self.config.metric_interval,
                            "preload_numpy_arrays": self.config.preload_numpy_arrays,
                            "dataloader_num_workers": self.config.dataloader_num_workers,
                            "dataloader_prefetch_factor": self.config.dataloader_prefetch_factor,
                            "dataloader_pin_memory": self.config.dataloader_pin_memory,
                            "dataloader_persistent_workers": self.config.dataloader_persistent_workers,
                            "dataloader_non_blocking": self.config.dataloader_non_blocking,
                            "profiler_enabled": self.config.profiler_enabled,
                            "nvtx_enabled": self.config.nvtx_enabled,
                            "data_wait_seconds": data_wait_elapsed,
                            "train_compute_seconds": elapsed,
                            "train_step_seconds": step_elapsed,
                            "metric_window_data_wait_seconds": metric_window_data_wait,
                            "metric_window_compute_seconds": metric_window_compute,
                            "metric_window_step_seconds": metric_window_step,
                            "metric_window_steps": metric_window_steps,
                            "eval_seconds": 0.0,
                            "checkpoint_seconds": 0.0,
                            "step_tokens_per_second": step_rates["tokens_per_second"],
                            "step_tokens_per_gpu_second": step_rates["tokens_per_gpu_second"],
                            "step_tokens_per_second_e2e": step_rates["tokens_per_second_e2e"],
                            "step_tokens_per_gpu_second_e2e": step_rates["tokens_per_gpu_second_e2e"],
                            "tokens_per_second": window_rates["tokens_per_second"],
                            "tokens_per_gpu_second": window_rates["tokens_per_gpu_second"],
                            "tokens_per_second_e2e": window_rates["tokens_per_second_e2e"],
                            "tokens_per_gpu_second_e2e": window_rates["tokens_per_gpu_second_e2e"],
                            "tokens_per_second_window": window_rates["tokens_per_second"],
                            "tokens_per_gpu_second_window": window_rates["tokens_per_gpu_second"],
                            "tokens_per_second_e2e_window": window_rates["tokens_per_second_e2e"],
                            "tokens_per_gpu_second_e2e_window": window_rates["tokens_per_gpu_second_e2e"],
                        }
                    if validation_loader is not None and (step % self.config.eval_interval == 0 or step == self.config.max_steps):
                        with _nvtx_range(self.config, "train/eval", step):
                            eval_started = perf_counter()
                            metric_row.update(
                                _evaluate(
                                    model,
                                    validation_loader,
                                    accelerator,
                                    loss_fn,
                                    self.config.attention_backend,
                                    device,
                                    self.config.max_eval_batches,
                                )
                            )
                            metric_row["eval_seconds"] = max(perf_counter() - eval_started, 0.0)
                    metrics.append(metric_row)
                    metric_logger.submit(metrics=metrics, tracker_payload=_tracker_metrics(metric_row), step=step)
                    progress.set_postfix(_progress_postfix(metric_row), refresh=False)
                if should_record_metrics:
                    metric_window_tokens = 0
                    metric_window_data_wait = 0.0
                    metric_window_compute = 0.0
                    metric_window_step = 0.0
                    metric_window_steps = 0
                if self.config.checkpoint_interval and step % self.config.checkpoint_interval == 0:
                    with _nvtx_range(self.config, "train/checkpoint", step):
                        checkpoint_started = perf_counter()
                        accelerator.wait_for_everyone()
                        payload = _checkpoint_payload(
                            accelerator,
                            model,
                            optimizer,
                            scheduler,
                            self.config,
                            profile,
                            shard_path,
                            metrics,
                            step,
                        )
                        if accelerator.is_main_process:
                            checkpoint_writer.submit(self.paths.run(f"checkpoints/step-{step:06d}.pt"), payload)
                        accelerator.wait_for_everyone()
                        checkpoint_seconds = max(perf_counter() - checkpoint_started, 0.0)
                    if accelerator.is_main_process and metric_row is not None:
                        metric_row["checkpoint_seconds"] = checkpoint_seconds
                        metric_logger.submit(
                            metrics=metrics,
                            tracker_payload={"train/checkpoint_seconds": checkpoint_seconds},
                            step=step,
                        )
                progress.update(1)
                active_profiler.step()
        progress.close()

        checkpoint_path = self.paths.run("checkpoint.pt")
        manifest_path = self.paths.manifest("training", self.config.run_id)
        _save_checkpoint(
            self.store,
            checkpoint_path,
            accelerator,
            model,
            optimizer,
            scheduler,
            self.config,
            profile,
            shard_path,
            metrics,
            last_step,
        )
        if accelerator.is_main_process:
            self.store.write_manifest(
                manifest_path,
                Manifest(
                    manifest_id=self.config.run_id,
                    artifact_type="training",
                    paths=[shard_path, profile_path, metrics_path, checkpoint_path],
                    metadata={
                        "stage": "training",
                        "run_id": self.config.run_id,
                        "training_backend": "accelerate",
                        "distributed_strategy": self.config.distributed_strategy,
                        "optimizer": self.config.optimizer,
                        "weight_decay": self.config.weight_decay,
                        "lr_schedule": self.config.lr_schedule,
                        "lr_warmup_steps": self.config.lr_warmup_steps,
                        "min_learning_rate": self.config.min_learning_rate,
                        "dataset_format": dataset_format,
                        "world_size": accelerator.num_processes,
                        "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                        "compile_enabled": self.config.compile,
                        "compile_mode": self.config.compile_mode,
                        "attention_backend": self.config.attention_backend,
                        "mixed_precision": precision,
                        "stream_contract": STREAM_CONTRACT,
                        "train_sequence_count": train_count,
                        "validation_sequence_count": validation_count,
                        "backtest_sequence_count": backtest_count,
                        "vocab_size": int(profile["vocab_size"]),
                        "block_size": int(profile["block_size"]),
                        "checkpoint_interval": self.config.checkpoint_interval,
                        "resume_from_checkpoint": self.config.resume_from_checkpoint,
                        "max_eval_batches": self.config.max_eval_batches,
                        "metric_interval": self.config.metric_interval,
                        "preload_numpy_arrays": self.config.preload_numpy_arrays,
                        "profiler_enabled": self.config.profiler_enabled,
                        "nvtx_enabled": self.config.nvtx_enabled,
                        "profiler_trace_dir": str(_profiler_trace_dir(self.store, self.config)),
                    },
                ),
            )
        if self.config.mlflow_enabled:
            accelerator.end_training()
        return TrainResult(checkpoint_path=checkpoint_path, metrics_path=metrics_path, manifest_path=manifest_path)


def _build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    betas = (config.adam_beta1, config.adam_beta2)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            fused=True,
        )
    muon_params: list[torch.nn.Parameter] = []
    muon_names: list[str] = []
    adamw_params: list[torch.nn.Parameter] = []
    adamw_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _use_muon(name, parameter):
            muon_params.append(parameter)
            muon_names.append(name)
        else:
            adamw_params.append(parameter)
            adamw_names.append(name)
    groups: list[dict[str, Any]] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "param_names": muon_names,
                "algorithm": "muon",
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "momentum": config.muon_momentum,
                "ns_steps": config.muon_ns_steps,
            }
        )
    if adamw_params:
        groups.append(
            {
                "params": adamw_params,
                "param_names": adamw_names,
                "algorithm": "adamw",
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "betas": betas,
                "eps": config.adam_eps,
            }
        )
    return MuonAdamW(groups)


def _use_muon(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim < 2:
        return False
    return "embedding" not in name and not name.endswith("output.weight")


def _muon_orthogonalize(update: torch.Tensor, ns_steps: int) -> torch.Tensor:
    return _muon_orthogonalize_batch(update.unsqueeze(0), ns_steps).squeeze(0)


def _muon_orthogonalize_batch(updates: torch.Tensor, ns_steps: int) -> torch.Tensor:
    original_shape = updates.shape[1:]
    matrices = updates.reshape(updates.shape[0], updates.shape[1], -1)
    rows, cols = matrices.shape[-2:]
    transposed = rows > cols
    if transposed:
        matrices = matrices.transpose(1, 2)
    x = matrices.float()
    # Keep the zero-norm guard on device; branching on a CUDA scalar synchronizes every Muon tensor.
    x = x / x.norm(dim=(1, 2), keepdim=True).clamp_min(1e-7)
    for _ in range(ns_steps):
        gram = x @ x.transpose(1, 2)
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    if transposed:
        x = x.transpose(1, 2)
    scale = math.sqrt(max(1.0, rows / cols))
    return x.reshape((updates.shape[0], *original_shape)).to(dtype=updates.dtype) * scale


def _build_lr_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig) -> torch.optim.lr_scheduler.LambdaLR:
    min_lr_ratio = config.min_learning_rate / config.learning_rate

    def multiplier(step_index: int) -> float:
        if config.lr_warmup_steps and step_index < config.lr_warmup_steps:
            return (step_index + 1) / config.lr_warmup_steps
        if config.lr_schedule == "constant":
            return 1.0
        decay_steps = max(config.max_steps - config.lr_warmup_steps - 1, 1)
        progress = min(max((step_index - config.lr_warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    validation_loader: DataLoader,
    accelerator: Accelerator,
    loss_fn: nn.Module,
    attention_backend: str,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    model.eval()
    losses: list[torch.Tensor] = []
    top1: list[float] = []
    top5: list[float] = []
    batches = 0
    for batch in validation_loader:
        if max_batches is not None and batches >= max_batches:
            break
        with accelerator.autocast(), _attention_kernel_context(attention_backend, device):
            logits = model(batch["input_ids"])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
        losses.append(accelerator.gather_for_metrics(loss.detach()).mean().cpu())
        top1.append(_topk_accuracy(logits, batch["labels"], 1))
        top5.append(_topk_accuracy(logits, batch["labels"], 5))
        batches += 1
    model.train()
    validation_loss = float(torch.stack(losses).mean()) if losses else 0.0
    return {
        "validation_loss": validation_loss,
        "validation_perplexity": _perplexity(validation_loss),
        "validation_top1_accuracy": sum(top1) / max(len(top1), 1),
        "validation_top5_accuracy": sum(top5) / max(len(top5), 1),
        "validation_batches": float(batches),
    }


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = logits.detach().topk(min(k, logits.shape[-1]), dim=-1).indices
    hits = topk.eq(labels.unsqueeze(-1)).any(dim=-1)
    return float(hits.float().mean().cpu())


def _perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def _tokens_per_second(batch: dict[str, torch.Tensor], elapsed: float, accelerator: Accelerator) -> float:
    return int(batch["labels"].numel()) * accelerator.num_processes / elapsed


def _metric_window_rates(
    token_count: int,
    compute_seconds: float,
    step_seconds: float,
    accelerator_processes: int,
) -> dict[str, float]:
    total_tokens = int(token_count) * int(accelerator_processes)
    compute_elapsed = max(float(compute_seconds), 1e-9)
    step_elapsed = max(float(step_seconds), 1e-9)
    tokens_per_second = total_tokens / compute_elapsed
    tokens_per_second_e2e = total_tokens / step_elapsed
    return {
        "tokens_per_second": tokens_per_second,
        "tokens_per_gpu_second": tokens_per_second / accelerator_processes,
        "tokens_per_second_e2e": tokens_per_second_e2e,
        "tokens_per_gpu_second_e2e": tokens_per_second_e2e / accelerator_processes,
    }


def _should_record_training_metrics(step: int, config: TrainConfig) -> bool:
    if step == 1 or step == config.max_steps:
        return True
    if step % config.metric_interval == 0:
        return True
    if step % config.eval_interval == 0:
        return True
    return bool(config.checkpoint_interval and step % config.checkpoint_interval == 0)


def _training_profiler(
    store: LocalObjectStore,
    config: TrainConfig,
    device: torch.device,
    accelerator: Accelerator,
) -> Any:
    if not config.profiler_enabled:
        return _NoopProfiler()
    trace_dir = _profiler_trace_dir(store, config)
    if accelerator.is_main_process:
        trace_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    worker_name = f"rank-{accelerator.process_index}"
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=config.profiler_wait_steps,
            warmup=config.profiler_warmup_steps,
            active=config.profiler_active_steps,
            repeat=config.profiler_repeat,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir), worker_name=worker_name),
        record_shapes=config.profiler_record_shapes,
        profile_memory=config.profiler_profile_memory,
        with_stack=config.profiler_with_stack,
    )


def _profiler_trace_dir(store: LocalObjectStore, config: TrainConfig) -> Path:
    trace_dir = config.profiler_trace_dir or f"runs/{config.run_id}/profiler"
    return _checkpoint_target(store, trace_dir)


def _dataset(
    store: LocalObjectStore,
    profile: dict[str, Any],
    shard_path: str,
    split_counts: dict[str, dict[str, int]],
    split: str,
    preload_numpy_arrays: bool = False,
    array_cache: dict[str, Any] | None = None,
) -> torch.utils.data.IterableDataset:
    numpy_metadata = profile.get("numpy_dataset")
    if isinstance(numpy_metadata, dict) and _numpy_dataset_exists(store, numpy_metadata):
        return NumpyTickerTimeDataset(
            store,
            numpy_metadata,
            split_counts,
            split=split,
            preload_numpy_arrays=preload_numpy_arrays,
            array_cache=array_cache,
        )
    raise FileNotFoundError(f"NumPy token dataset is missing for {shard_path}")


def _dataset_format(store: LocalObjectStore, profile: dict[str, Any]) -> str:
    numpy_metadata = profile.get("numpy_dataset")
    if isinstance(numpy_metadata, dict) and _numpy_dataset_exists(store, numpy_metadata):
        if numpy_metadata.get("storage") == "token_stream":
            return "numpy-token-stream"
        return "numpy-partitioned" if bool(numpy_metadata.get("partitioned")) else "numpy"
    raise FileNotFoundError("NumPy token dataset is missing")


def _numpy_dataset_exists(store: LocalObjectStore, numpy_metadata: dict[str, Any]) -> bool:
    if numpy_metadata.get("storage") == "token_stream":
        partitions = numpy_metadata.get("partitions")
        if not isinstance(partitions, list) or not partitions:
            return False
        for partition in partitions:
            if not isinstance(partition, dict):
                return False
            tokens_path = partition.get("tokens_path")
            if not tokens_path or not _checkpoint_target(store, str(tokens_path)).exists():
                return False
        return True
    if bool(numpy_metadata.get("partitioned")):
        partitions = numpy_metadata.get("partitions")
        if not isinstance(partitions, list) or not partitions:
            return False
        for partition in partitions:
            if not isinstance(partition, dict):
                return False
            tokens_path = partition.get("tokens_path")
            ticker_ids_path = partition.get("ticker_ids_path")
            if not tokens_path or not ticker_ids_path:
                return False
            if not _checkpoint_target(store, str(tokens_path)).exists():
                return False
            if not _checkpoint_target(store, str(ticker_ids_path)).exists():
                return False
        return True
    tokens_path = numpy_metadata.get("tokens_path")
    ticker_ids_path = numpy_metadata.get("ticker_ids_path")
    if not tokens_path or not ticker_ids_path:
        return False
    return _checkpoint_target(store, str(tokens_path)).exists() and _checkpoint_target(store, str(ticker_ids_path)).exists()


def _load_resume_checkpoint(store: LocalObjectStore, resume_from_checkpoint: str | None) -> dict[str, Any] | None:
    if resume_from_checkpoint is None:
        return None
    target = _checkpoint_target(store, resume_from_checkpoint)
    if not target.exists():
        raise FileNotFoundError(f"checkpoint not found: {resume_from_checkpoint}")
    return dict(torch.load(target, map_location="cpu", weights_only=False))


def _validate_resume_checkpoint(state: dict[str, Any], profile: dict[str, Any], shard_path: str) -> None:
    if state.get("stage") != "training":
        raise ValueError("resume checkpoint is not a training checkpoint")
    saved_profile = dict(state.get("profile", {}))
    for key in ("stream_contract", "vocab_size", "block_size"):
        if saved_profile.get(key) != profile.get(key):
            raise ValueError(f"resume checkpoint profile mismatch for {key}")
    if state.get("shard_path") != shard_path:
        raise ValueError("resume checkpoint was created for a different shard")
    if "model_state_dict" not in state or "optimizer_state_dict" not in state:
        raise ValueError("resume checkpoint is missing model or optimizer state")


def _save_checkpoint(
    store: LocalObjectStore,
    checkpoint_path: str,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainConfig,
    profile: dict[str, Any],
    shard_path: str,
    metrics: list[dict[str, object]],
    step: int,
) -> None:
    accelerator.wait_for_everyone()
    payload = _checkpoint_payload(
        accelerator,
        model,
        optimizer,
        scheduler,
        config,
        profile,
        shard_path,
        metrics,
        step,
    )
    if accelerator.is_main_process:
        target = _checkpoint_target(store, checkpoint_path)
        _write_checkpoint_payload(target, payload)
    accelerator.wait_for_everyone()


def _checkpoint_payload(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainConfig,
    profile: dict[str, Any],
    shard_path: str,
    metrics: list[dict[str, object]],
    step: int,
) -> dict[str, Any]:
    model_state_dict = normalized_model_state_dict(accelerator.get_state_dict(model))
    return {
        "stage": "training",
        "step": step,
        "model_state_dict": _state_to_cpu(model_state_dict),
        "optimizer_state_dict": _state_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.__dict__,
        "profile": dict(profile),
        "shard_path": shard_path,
        "metrics": [dict(row) for row in metrics],
        "rng_state": torch.get_rng_state(),
    }


def _write_checkpoint_payload(target: Path, payload: dict[str, Any]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def _state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    return value


def _checkpoint_target(store: LocalObjectStore, checkpoint_path: str) -> Path:
    target = Path(checkpoint_path)
    if target.is_absolute():
        return target
    if ".." in target.parts:
        raise ValueError(f"checkpoint path must be relative and safe: {checkpoint_path}")
    return store.root / target


def _profile_path(shard_path: str) -> str:
    return str(shard_path).removesuffix(".npy") + "-profile.json"


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested_device == "mps" and not _mps_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested_device)


def _resolve_precision(requested_precision: str, device: torch.device) -> str:
    if requested_precision == "auto":
        return "mixed" if device.type == "cuda" else "fp32"
    if requested_precision == "mixed" and device.type != "cuda":
        raise RuntimeError("mixed precision is only supported for CUDA training")
    return requested_precision


def _dataloader_kwargs(config: TrainConfig, device: torch.device) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.dataloader_num_workers,
        "pin_memory": device.type == "cuda" and config.dataloader_pin_memory,
    }
    if config.dataloader_num_workers > 0:
        kwargs["prefetch_factor"] = config.dataloader_prefetch_factor
        kwargs["persistent_workers"] = config.dataloader_persistent_workers
    return kwargs


def _maybe_compile_model(model: TradingModel, config: TrainConfig, device: torch.device) -> torch.nn.Module:
    if not config.compile:
        return model
    if device.type == "mps":
        raise RuntimeError("torch.compile is not supported for this training path on MPS")
    return torch.compile(model, mode=config.compile_mode)


def _maybe_cudagraph_mark_step_begin(config: TrainConfig, device: torch.device) -> None:
    """Tell torch.compile when a new training step starts so CUDAGraph outputs are not aliased across steps."""
    if not config.compile or device.type != "cuda":
        return
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if callable(mark):
        mark()


@contextmanager
def _attention_kernel_context(backend: str, device: torch.device) -> Iterator[None]:
    if backend in {"auto", "triton"}:
        yield
        return
    if device.type != "cuda":
        raise RuntimeError(f"{backend} attention backend requires CUDA")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        mapping = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
        }
        with sdpa_kernel(mapping[backend]):
            yield
    except (ImportError, AttributeError):
        flags = {
            "flash": dict(enable_flash=True, enable_mem_efficient=False, enable_math=False),
            "efficient": dict(enable_flash=False, enable_mem_efficient=True, enable_math=False),
            "math": dict(enable_flash=False, enable_mem_efficient=False, enable_math=True),
        }
        with torch.backends.cuda.sdp_kernel(**flags[backend]):
            yield


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _accelerator(device: torch.device, precision: str, config: TrainConfig) -> Accelerator:
    mixed_precision = "bf16" if precision == "mixed" else "no"
    log_with = "mlflow" if config.mlflow_enabled else None
    kwargs: dict[str, Any] = {
        "cpu": device.type == "cpu",
        "mixed_precision": mixed_precision,
        "log_with": log_with,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "dataloader_config": DataLoaderConfiguration(
            non_blocking=device.type == "cuda" and config.dataloader_non_blocking
        ),
    }
    if config.distributed_strategy == "fsdp":
        kwargs["fsdp_plugin"] = _fsdp_plugin()
    return Accelerator(**kwargs)


def _fsdp_plugin() -> Any:
    try:
        from accelerate import FullyShardedDataParallelPlugin
    except ImportError as exc:
        raise RuntimeError("FSDP requires an Accelerate version with FullyShardedDataParallelPlugin") from exc
    return FullyShardedDataParallelPlugin()


def _init_trackers(
    accelerator: Accelerator,
    store: LocalObjectStore,
    config: TrainConfig,
    shard_path: str,
    profile: dict[str, Any],
    train_count: int,
    validation_count: int,
    precision: str,
) -> None:
    if not config.mlflow_enabled:
        return
    tracking_uri = config.mlflow_tracking_uri or f"sqlite:///{(store.root / 'runs' / 'mlflow' / 'mlflow.db').resolve()}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow_artifact_dir = store.root / "runs" / config.run_id / "mlflow-artifacts"
    mlflow_artifact_dir.mkdir(parents=True, exist_ok=True)
    accelerator.init_trackers(
        project_name=config.mlflow_experiment,
        config={
            "run_id": config.run_id,
            "stage": "training",
            "shard_path": shard_path,
            "sequence_count": int(profile["sequence_count"]),
            "train_sequence_count": train_count,
            "validation_sequence_count": validation_count,
            "vocab_size": int(profile["vocab_size"]),
            "block_size": int(profile["block_size"]),
            "hidden_dim": config.hidden_dim,
            "layers": config.layers,
            "attention_heads": config.attention_heads,
            "kv_heads": config.kv_heads or config.attention_heads,
            "intermediate_dim": config.intermediate_dim,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
            "weight_decay": config.weight_decay,
            "lr_schedule": config.lr_schedule,
            "lr_warmup_steps": config.lr_warmup_steps,
            "min_learning_rate": config.min_learning_rate,
            "requested_device": config.device,
            "requested_precision": config.precision,
            "effective_precision": precision,
            "training_backend": "accelerate",
            "distributed_strategy": config.distributed_strategy,
            "dataset_format": _dataset_format(store, profile),
            "world_size": accelerator.num_processes,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "compile_enabled": config.compile,
            "compile_mode": config.compile_mode,
            "attention_backend": config.attention_backend,
            "metric_interval": config.metric_interval,
            "preload_numpy_arrays": config.preload_numpy_arrays,
            "dataloader_num_workers": config.dataloader_num_workers,
            "dataloader_prefetch_factor": config.dataloader_prefetch_factor,
            "dataloader_pin_memory": config.dataloader_pin_memory,
            "dataloader_persistent_workers": config.dataloader_persistent_workers,
            "dataloader_non_blocking": config.dataloader_non_blocking,
            "profiler_enabled": config.profiler_enabled,
            "nvtx_enabled": config.nvtx_enabled,
            "profiler_trace_dir": str(_profiler_trace_dir(store, config)),
            "stream_contract": STREAM_CONTRACT,
        },
        init_kwargs={
            "mlflow": {
                "logging_dir": str(mlflow_artifact_dir),
                "run_name": config.run_id,
                "tags": {
                    "project": "mega-trading",
                    "stage": "training",
                    "model_family": "generative-order-flow",
                    "distributed_strategy": config.distributed_strategy,
                },
            }
        },
    )


def _log_model_summary_to_mlflow(
    *,
    accelerator: Accelerator,
    config: TrainConfig,
    model: torch.nn.Module,
    profile: dict[str, Any],
    device: torch.device,
) -> None:
    if not config.mlflow_enabled or not accelerator.is_main_process:
        return

    import mlflow
    import torchinfo

    unwrapped_model = accelerator.unwrap_model(model)
    was_training = unwrapped_model.training
    unwrapped_model.eval()
    input_data = torch.zeros((1, int(profile["block_size"])), dtype=torch.long, device=device)
    try:
        with torch.no_grad(), _attention_kernel_context(config.attention_backend, device):
            model_summary = torchinfo.summary(
                unwrapped_model,
                input_data=input_data,
                depth=6,
                col_names=("input_size", "output_size", "num_params", "trainable"),
                verbose=0,
            )
        summary_text = str(model_summary)
    except Exception as exc:
        summary_text = f"torchinfo model summary failed: {type(exc).__name__}: {exc}"
    finally:
        unwrapped_model.train(was_training)
    mlflow.log_text(summary_text, artifact_file="model/model_summary.txt")


def _tracker_metrics(row: dict[str, object]) -> dict[str, float]:
    data_wait_seconds = float(row["data_wait_seconds"])
    step_seconds = max(float(row["train_step_seconds"]), 1e-9)
    metrics: dict[str, float] = {
        "train/learning_rate": float(row["learning_rate"]),
        "train/loss": float(row["train_loss"]),
        "train/perplexity": float(row["train_perplexity"]),
        "train/top1_accuracy": float(row["train_top1_accuracy"]),
        "train/top5_accuracy": float(row["train_top5_accuracy"]),
        "train/tokens_per_second": float(row["tokens_per_second"]),
        "train/tokens_per_gpu_second": float(row["tokens_per_gpu_second"]),
        "train/tokens_per_second_e2e": float(row["tokens_per_second_e2e"]),
        "train/tokens_per_gpu_second_e2e": float(row["tokens_per_gpu_second_e2e"]),
        "train/tokens_per_second_window": float(row["tokens_per_second_window"]),
        "train/tokens_per_gpu_second_window": float(row["tokens_per_gpu_second_window"]),
        "train/tokens_per_second_e2e_window": float(row["tokens_per_second_e2e_window"]),
        "train/tokens_per_gpu_second_e2e_window": float(row["tokens_per_gpu_second_e2e_window"]),
        "train/step_tokens_per_second": float(row["step_tokens_per_second"]),
        "train/step_tokens_per_gpu_second": float(row["step_tokens_per_gpu_second"]),
        "train/step_tokens_per_second_e2e": float(row["step_tokens_per_second_e2e"]),
        "train/step_tokens_per_gpu_second_e2e": float(row["step_tokens_per_gpu_second_e2e"]),
        "train/data_wait_seconds": data_wait_seconds,
        "train/dataloader_wait_seconds": data_wait_seconds,
        "train/dataloader_wait_fraction": data_wait_seconds / step_seconds,
        "train/compute_seconds": float(row["train_compute_seconds"]),
        "train/step_seconds": step_seconds,
        "train/metric_window_data_wait_seconds": float(row["metric_window_data_wait_seconds"]),
        "train/metric_window_compute_seconds": float(row["metric_window_compute_seconds"]),
        "train/metric_window_step_seconds": float(row["metric_window_step_seconds"]),
        "train/metric_window_steps": float(row["metric_window_steps"]),
        "train/eval_seconds": float(row.get("eval_seconds", 0.0)),
        "train/checkpoint_seconds": float(row.get("checkpoint_seconds", 0.0)),
    }
    for key in ("validation_loss", "validation_perplexity", "validation_top1_accuracy", "validation_top5_accuracy"):
        if key in row:
            metrics[f"validation/{key.removeprefix('validation_')}"] = float(row[key])
    return metrics


def _progress_bar(accelerator: Accelerator, config: TrainConfig, start_step: int) -> Any:
    return tqdm(
        total=max(config.max_steps - start_step + 1, 0),
        desc=f"train:{config.run_id}",
        disable=not config.progress_bar or not accelerator.is_local_main_process,
        dynamic_ncols=True,
        leave=True,
    )


def _progress_postfix(row: dict[str, object]) -> dict[str, str]:
    return {
        "loss": f"{float(row['train_loss']):.4f}",
        "lr": f"{float(row['learning_rate']):.2e}",
        "ppl": f"{float(row['train_perplexity']):.2f}",
        "top1": f"{float(row['train_top1_accuracy']):.2f}",
        "tok/s": f"{float(row['tokens_per_second']):.1f}",
        "e2e tok/s": f"{float(row['tokens_per_second_e2e']):.1f}",
        "data": f"{float(row['data_wait_seconds']) * 1000.0:.1f}ms",
    }
