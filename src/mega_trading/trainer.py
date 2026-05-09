"""Training loop for next-token order-flow modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from accelerate import Accelerator
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mega_trading.config import TrainConfig
from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.dataset import TickerTimeDataset, cycle_batches, per_ticker_train_counts
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

        train_dataset = TickerTimeDataset(self.store, shard_path, train_counts, split="train")
        validation_dataset = TickerTimeDataset(self.store, shard_path, train_counts, split="validation") if validation_count else None
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
        metrics: list[dict[str, object]] = []
        metrics_path = self.paths.run("metrics")
        progress = _progress_bar(accelerator, self.config)

        for step in range(1, self.config.max_steps + 1):
            batch = next(iterator)
            started = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                logits = model(batch["input_ids"])
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
            accelerator.backward(loss)
            optimizer.step()
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
                    "tokens_per_second": int(batch["labels"].numel()) / elapsed,
                }
                if validation_loader is not None and (step % self.config.eval_interval == 0 or step == self.config.max_steps):
                    metric_row.update(_evaluate(model, validation_loader, accelerator, loss_fn))
                metrics.append(metric_row)
                accelerator.log(_wandb_metrics(metric_row), step=step)
                self.store.write_jsonl(metrics_path, metrics)
                progress.set_postfix(_progress_postfix(metric_row), refresh=False)
            progress.update(1)
        progress.close()

        checkpoint_path = self.paths.run("checkpoint.pt")
        checkpoint_target = self.store.root / checkpoint_path
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = self.paths.manifest("training", self.config.run_id)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save(
                {
                    "stage": "training",
                    "model_state_dict": accelerator.get_state_dict(model),
                    "config": self.config.__dict__,
                    "profile": profile,
                },
                checkpoint_target,
            )
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
                        "stream_contract": STREAM_CONTRACT,
                        "train_sequence_count": train_count,
                        "validation_sequence_count": validation_count,
                        "vocab_size": int(profile["vocab_size"]),
                        "block_size": int(profile["block_size"]),
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
) -> dict[str, float]:
    model.eval()
    losses: list[torch.Tensor] = []
    top1: list[float] = []
    top5: list[float] = []
    for batch in validation_loader:
        with accelerator.autocast():
            logits = model(batch["input_ids"])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
        losses.append(accelerator.gather_for_metrics(loss.detach()).mean().cpu())
        top1.append(_topk_accuracy(logits, batch["labels"], 1))
        top5.append(_topk_accuracy(logits, batch["labels"], 5))
    model.train()
    validation_loss = float(torch.stack(losses).mean()) if losses else 0.0
    return {
        "validation_loss": validation_loss,
        "validation_perplexity": _perplexity(validation_loss),
        "validation_top1_accuracy": sum(top1) / max(len(top1), 1),
        "validation_top5_accuracy": sum(top5) / max(len(top5), 1),
    }


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = logits.detach().topk(min(k, logits.shape[-1]), dim=-1).indices
    hits = topk.eq(labels.unsqueeze(-1)).any(dim=-1)
    return float(hits.float().mean().cpu())


def _perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def _profile_path(shard_path: str) -> str:
    return str(shard_path).removesuffix(".jsonl") + "-profile.json"


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


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _accelerator(device: torch.device, precision: str, config: TrainConfig) -> Accelerator:
    mixed_precision = "fp16" if precision == "mixed" else "no"
    log_with = "wandb" if config.wandb_enabled else None
    return Accelerator(cpu=device.type == "cpu", mixed_precision=mixed_precision, log_with=log_with)


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
    }
    for key in ("validation_loss", "validation_perplexity", "validation_top1_accuracy", "validation_top5_accuracy"):
        if key in row:
            metrics[f"validation/{key.removeprefix('validation_')}"] = float(row[key])
    return metrics


def _progress_bar(accelerator: Accelerator, config: TrainConfig) -> Any:
    return tqdm(
        total=config.max_steps,
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
