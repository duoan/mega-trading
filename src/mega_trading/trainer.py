"""Training loop for next-token order-flow modeling."""

from __future__ import annotations

from contextlib import contextmanager
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from accelerate import Accelerator
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mega_trading.config import TrainConfig
from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.dataset import NumpyTickerTimeDataset, cycle_batches, per_ticker_train_counts
from mega_trading.events import STREAM_CONTRACT
from mega_trading.model import TradingModel


@dataclass(frozen=True)
class TrainResult:
    checkpoint_path: str
    metrics_path: str
    manifest_path: str


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
        train_counts = per_ticker_train_counts(
            {str(key): int(value) for key, value in dict(profile["sequence_counts"]).items()},
            self.config.validation_fraction,
        )
        train_count = sum(train_counts.values())
        validation_count = int(profile["sequence_count"]) - train_count
        if train_count <= 0:
            raise ValueError("training shard has no train rows")

        dataset_format = _dataset_format(self.store, profile)
        train_dataset = _dataset(self.store, profile, shard_path, train_counts, split="train")
        validation_dataset = _dataset(self.store, profile, shard_path, train_counts, split="validation") if validation_count else None
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size)
        validation_loader = (
            DataLoader(validation_dataset, batch_size=self.config.batch_size, shuffle=False) if validation_dataset else None
        )

        device = _resolve_device(self.config.device)
        precision = _resolve_precision(self.config.precision, device)
        accelerator = _accelerator(device, precision, self.config)
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
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        resume_state = _load_resume_checkpoint(self.store, self.config.resume_from_checkpoint)
        start_step = 1
        metrics: list[dict[str, object]] = []
        if resume_state is not None:
            _validate_resume_checkpoint(resume_state, profile, shard_path)
            model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            start_step = int(resume_state.get("step", 0)) + 1
            metrics = [dict(row) for row in resume_state.get("metrics", [])]
        model = _maybe_compile_model(model, self.config, device)
        if validation_loader is None:
            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        else:
            model, optimizer, train_loader, validation_loader = accelerator.prepare(
                model,
                optimizer,
                train_loader,
                validation_loader,
            )

        _init_trackers(accelerator, self.store, self.config, shard_path, profile, train_count, validation_count, precision)
        loss_fn = nn.CrossEntropyLoss()
        iterator = cycle_batches(train_loader)
        metrics_path = self.paths.run("metrics.json")
        progress = _progress_bar(accelerator, self.config, start_step)
        last_step = start_step - 1

        for step in range(start_step, self.config.max_steps + 1):
            last_step = step
            batch = next(iterator)
            started = perf_counter()
            with accelerator.accumulate(model):
                with accelerator.autocast(), _attention_kernel_context(self.config.attention_backend, device):
                    logits = model(batch["input_ids"])
                    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            elapsed = max(perf_counter() - started, 1e-9)
            if accelerator.is_main_process:
                metric_row = {
                    "step": step,
                    "stage": "training",
                    "train_loss": float(loss.detach().cpu()),
                    "train_perplexity": _perplexity(float(loss.detach().cpu())),
                    "train_top1_accuracy": _topk_accuracy(logits, batch["labels"], 1),
                    "train_top5_accuracy": _topk_accuracy(logits, batch["labels"], 5),
                    "train_sequence_count": train_count,
                    "validation_sequence_count": validation_count,
                    "world_size": accelerator.num_processes,
                    "distributed_strategy": self.config.distributed_strategy,
                    "dataset_format": dataset_format,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "compile_enabled": self.config.compile,
                    "attention_backend": self.config.attention_backend,
                    "tokens_per_second": _tokens_per_second(batch, elapsed, accelerator),
                    "tokens_per_gpu_second": _tokens_per_second(batch, elapsed, accelerator) / accelerator.num_processes,
                }
                if validation_loader is not None and (step % self.config.eval_interval == 0 or step == self.config.max_steps):
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
                metrics.append(metric_row)
                accelerator.log(_wandb_metrics(metric_row), step=step)
                self.store.write_json(metrics_path, {"metrics": metrics})
                progress.set_postfix(_progress_postfix(metric_row), refresh=False)
            if self.config.checkpoint_interval and step % self.config.checkpoint_interval == 0:
                _save_checkpoint(
                    self.store,
                    self.paths.run(f"checkpoints/step-{step:06d}.pt"),
                    accelerator,
                    model,
                    optimizer,
                    self.config,
                    profile,
                    shard_path,
                    metrics,
                    step,
                )
            progress.update(1)
        progress.close()

        checkpoint_path = self.paths.run("checkpoint.pt")
        manifest_path = self.paths.manifest("training", self.config.run_id)
        _save_checkpoint(
            self.store,
            checkpoint_path,
            accelerator,
            model,
            optimizer,
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
                        "vocab_size": int(profile["vocab_size"]),
                        "block_size": int(profile["block_size"]),
                        "checkpoint_interval": self.config.checkpoint_interval,
                        "resume_from_checkpoint": self.config.resume_from_checkpoint,
                        "max_eval_batches": self.config.max_eval_batches,
                    },
                ),
            )
        if self.config.wandb_enabled:
            accelerator.end_training()
        return TrainResult(checkpoint_path=checkpoint_path, metrics_path=metrics_path, manifest_path=manifest_path)


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


def _dataset(
    store: LocalObjectStore,
    profile: dict[str, Any],
    shard_path: str,
    train_counts: dict[str, int],
    split: str,
) -> torch.utils.data.IterableDataset:
    numpy_metadata = profile.get("numpy_dataset")
    if isinstance(numpy_metadata, dict) and _numpy_dataset_exists(store, numpy_metadata):
        return NumpyTickerTimeDataset(store, numpy_metadata, train_counts, split=split)
    raise FileNotFoundError(f"NumPy token dataset is missing for {shard_path}")


def _dataset_format(store: LocalObjectStore, profile: dict[str, Any]) -> str:
    numpy_metadata = profile.get("numpy_dataset")
    if isinstance(numpy_metadata, dict) and _numpy_dataset_exists(store, numpy_metadata):
        return "numpy-partitioned" if bool(numpy_metadata.get("partitioned")) else "numpy"
    raise FileNotFoundError("NumPy token dataset is missing")


def _numpy_dataset_exists(store: LocalObjectStore, numpy_metadata: dict[str, Any]) -> bool:
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
    config: TrainConfig,
    profile: dict[str, Any],
    shard_path: str,
    metrics: list[dict[str, object]],
    step: int,
) -> None:
    accelerator.wait_for_everyone()
    model_state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        target = _checkpoint_target(store, checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage": "training",
                "step": step,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.__dict__,
                "profile": profile,
                "shard_path": shard_path,
                "metrics": metrics,
                "rng_state": torch.get_rng_state(),
            },
            target,
        )
    accelerator.wait_for_everyone()


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


def _maybe_compile_model(model: TradingModel, config: TrainConfig, device: torch.device) -> torch.nn.Module:
    if not config.compile:
        return model
    if device.type == "mps":
        raise RuntimeError("torch.compile is not supported for this training path on MPS")
    return torch.compile(model, mode=config.compile_mode)


@contextmanager
def _attention_kernel_context(backend: str, device: torch.device) -> Iterator[None]:
    if backend == "auto":
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
    mixed_precision = "fp16" if precision == "mixed" else "no"
    log_with = "wandb" if config.wandb_enabled else None
    kwargs: dict[str, Any] = {
        "cpu": device.type == "cpu",
        "mixed_precision": mixed_precision,
        "log_with": log_with,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
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
    if not config.wandb_enabled:
        return
    wandb_dir = store.root / "runs" / config.run_id / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    accelerator.init_trackers(
        project_name=config.wandb_project,
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
            "stream_contract": STREAM_CONTRACT,
        },
        init_kwargs={
            "wandb": {
                "name": config.run_id,
                "entity": config.wandb_entity,
                "mode": config.wandb_mode,
                "dir": str(wandb_dir),
                "tags": ["mega-trading", "generative"],
            }
        },
    )


def _wandb_metrics(row: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "train/loss": float(row["train_loss"]),
        "train/perplexity": float(row["train_perplexity"]),
        "train/top1_accuracy": float(row["train_top1_accuracy"]),
        "train/top5_accuracy": float(row["train_top5_accuracy"]),
        "train/tokens_per_second": float(row["tokens_per_second"]),
        "train/tokens_per_gpu_second": float(row["tokens_per_gpu_second"]),
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
        "ppl": f"{float(row['train_perplexity']):.2f}",
        "top1": f"{float(row['train_top1_accuracy']):.2f}",
        "tok/s": f"{float(row['tokens_per_second']):.1f}",
    }
