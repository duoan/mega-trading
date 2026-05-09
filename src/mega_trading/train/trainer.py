"""Training loop for TradingFoundationModel."""

from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from mega_trading.core.registry import ModelRegistry, model_version_id_for
from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.train.config import TradingFoundationTrainConfig, TradingFoundationTrainResult
from mega_trading.train.dataset import TradingFoundationIterableDataset
from mega_trading.train.model import TradingFoundationModel


class TradingFoundationTrainer:
    def __init__(self, store: LocalObjectStore, config: TradingFoundationTrainConfig) -> None:
        self.store = store
        self.config = config
        self.paths = ArtifactPaths(run_id=config.run_id)

    def train(self, shard_path: str) -> TradingFoundationTrainResult:
        torch.manual_seed(self.config.seed)
        shard_profile = _profile_shard(self.store, shard_path, self.config)
        if shard_profile["total_rows"] == 0:
            raise ValueError("training shard is empty")
        total_rows = int(shard_profile["total_rows"])
        train_count, validation_count = _time_ordered_counts(total_rows, self.config.validation_fraction)
        sizes = (
            int(shard_profile["market_window_size"]),
            int(shard_profile["news_size"]),
            int(shard_profile["sec_filing_size"]),
            int(shard_profile["earnings_size"]),
            int(shard_profile["macro_size"]),
        )
        train_dataset = TradingFoundationIterableDataset(
            lambda: _iter_row_slice(self.store, shard_path, 0, train_count),
            *sizes,
        )
        validation_dataset = (
            TradingFoundationIterableDataset(
                lambda: _iter_row_slice(self.store, shard_path, train_count, total_rows),
                *sizes,
            )
            if validation_count
            else None
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size)
        validation_loader = (
            DataLoader(validation_dataset, batch_size=self.config.batch_size, shuffle=False) if validation_dataset else None
        )
        iterator = iter(train_loader)
        device = _resolve_device(self.config.device)
        precision = _resolve_precision(self.config.precision, device)
        amp_enabled = precision == "mixed" and device.type == "cuda"
        scaler = _grad_scaler(amp_enabled)
        model = TradingFoundationModel(
            *sizes,
            hidden_dim=self.config.hidden_dim,
            attention_heads=self.config.attention_heads,
            use_market_data=self.config.use_market_data,
            use_news=self.config.use_news,
            use_sec_filings=self.config.use_sec_filings,
            use_earnings=self.config.use_earnings,
            use_macro=self.config.use_macro,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        metrics: list[dict[str, object]] = []
        metrics_path = self.paths.run("metrics")

        for step in range(1, self.config.max_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            started = perf_counter()
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_enabled):
                return_logits, risk_logits = model(batch)
                return_loss = loss_fn(return_logits, batch["return_label"])
                risk_loss = loss_fn(risk_logits, batch["risk_label"])
                loss = return_loss + risk_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            elapsed = max(perf_counter() - started, 1e-9)
            metrics.append(
                {
                    "step": step,
                    "stage": "trading_foundation_model",
                    "loss": float(loss.detach().cpu()),
                    "train_loss": float(loss.detach().cpu()),
                    "return_accuracy": _accuracy(return_logits, batch["return_label"]),
                    "risk_accuracy": _accuracy(risk_logits, batch["risk_label"]),
                    "train_sample_count": train_count,
                    "validation_sample_count": validation_count,
                    "examples_per_second": int(batch["return_label"].shape[0]) / elapsed,
                }
            )
            if validation_loader is not None and (step % self.config.eval_interval == 0 or step == self.config.max_steps):
                metrics[-1].update(_evaluate(model, validation_loader, device, loss_fn, precision))
            self.store.write_jsonl(metrics_path, metrics)

        checkpoint_path = self.paths.run("checkpoint.pt")
        checkpoint_target = self.store.root / checkpoint_path
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        config_hash = self.config.content_hash()
        model_version_id = model_version_id_for(self.config.run_id, config_hash)
        model_version_path = ArtifactPaths().model(model_version_id)
        torch.save(
            {
                "stage": "trading_foundation_model",
                "model_state_dict": model.state_dict(),
                "config": self.config.__dict__,
                "stream_sizes": {
                    "market_window_size": sizes[0],
                    "news_size": sizes[1],
                    "sec_filing_size": sizes[2],
                    "earnings_size": sizes[3],
                    "macro_size": sizes[4],
                },
                "modalities": {
                    "market_data": self.config.use_market_data,
                    "news": self.config.use_news,
                    "sec_filings": self.config.use_sec_filings,
                    "earnings": self.config.use_earnings,
                    "macro": self.config.use_macro,
                },
                "attention_heads": self.config.attention_heads,
                "train_sample_count": train_count,
                "validation_sample_count": validation_count,
                "step": self.config.max_steps,
                "shard_path": shard_path,
                "model_version_id": model_version_id,
                "requested_device": self.config.device,
                "device": device.type,
                "requested_precision": self.config.precision,
                "precision": precision,
            },
            checkpoint_target,
        )
        manifest = Manifest(
            manifest_id=f"{self.config.run_id}-trading-foundation-model",
            artifact_type="training_run",
            paths=[metrics_path, checkpoint_path],
            metadata={
                "stage": "trading_foundation_model",
                "steps": str(self.config.max_steps),
                "shard_path": shard_path,
                "stream_contract": "market_data_news_sec_filings_earnings_macro",
                "attention_heads": str(self.config.attention_heads),
                "train_sample_count": str(train_count),
                "validation_sample_count": str(validation_count),
                "validation_fraction": str(self.config.validation_fraction),
                "eval_interval": str(self.config.eval_interval),
                "use_market_data": str(self.config.use_market_data),
                "use_news": str(self.config.use_news),
                "use_sec_filings": str(self.config.use_sec_filings),
                "use_earnings": str(self.config.use_earnings),
                "use_macro": str(self.config.use_macro),
                "config_hash": config_hash,
                "model_version_id": model_version_id,
                "model_version_path": model_version_path,
                "requested_device": self.config.device,
                "device": device.type,
                "requested_precision": self.config.precision,
                "precision": precision,
            },
        )
        manifest_path = self.paths.manifest("runs", f"{self.config.run_id}-trading-foundation-model")
        self.store.write_manifest(manifest_path, manifest)
        ModelRegistry(self.store).register_training_run(
            run_id=self.config.run_id,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            metrics_path=metrics_path,
            config_hash=config_hash,
            feature_version="market_data_news_sec_filings_earnings_macro-v1",
            label_version="forward_return_risk-v1",
            data_snapshot_version=shard_path,
            metadata={
                "stream_contract": "market_data_news_sec_filings_earnings_macro",
                "attention_heads": self.config.attention_heads,
                "train_sample_count": train_count,
                "validation_sample_count": validation_count,
            },
        )
        return TradingFoundationTrainResult(steps=self.config.max_steps, checkpoint_path=checkpoint_path, manifest_path=manifest_path)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().detach().cpu())


def _evaluate(
    model: TradingFoundationModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    loss_fn: nn.Module,
    precision: str = "fp32",
) -> dict[str, float]:
    model.eval()
    amp_enabled = precision == "mixed" and device.type == "cuda"
    total_examples = 0
    total_loss = 0.0
    return_correct = 0
    risk_correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with _autocast_context(device, amp_enabled):
                return_logits, risk_logits = model(batch)
                return_loss = loss_fn(return_logits, batch["return_label"])
                risk_loss = loss_fn(risk_logits, batch["risk_label"])
            batch_size = int(batch["return_label"].shape[0])
            total_loss += float((return_loss + risk_loss).detach().cpu()) * batch_size
            return_correct += int((return_logits.argmax(dim=-1) == batch["return_label"]).sum().detach().cpu())
            risk_correct += int((risk_logits.argmax(dim=-1) == batch["risk_label"]).sum().detach().cpu())
            total_examples += batch_size
    model.train()
    if total_examples == 0:
        return {}
    return {
        "validation_loss": total_loss / total_examples,
        "validation_return_accuracy": return_correct / total_examples,
        "validation_risk_accuracy": risk_correct / total_examples,
    }


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


def _autocast_context(device: torch.device, enabled: bool) -> Any:
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def _grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _profile_shard(
    store: LocalObjectStore,
    shard_path: str,
    config: TradingFoundationTrainConfig,
) -> dict[str, int]:
    total_rows = 0
    market_window_size = config.market_window_size or 1
    news_size = config.news_size or 1
    sec_filing_size = config.sec_filing_size or 1
    earnings_size = config.earnings_size or 1
    macro_size = config.macro_size or 1
    for row in store.iter_jsonl(shard_path):
        total_rows += 1
        if config.market_window_size is None:
            market_window_size = max(market_window_size, len(row.get("market_returns", [])))
        if config.news_size is None:
            news_size = max(news_size, len(row.get("news_embeddings", [])))
        if config.sec_filing_size is None:
            sec_filing_size = max(sec_filing_size, len(row.get("sec_filing_features", [])))
        if config.earnings_size is None:
            earnings_size = max(earnings_size, len(row.get("earnings_features", [])))
        if config.macro_size is None:
            macro_size = max(macro_size, len(row.get("macro_features", [])))
    return {
        "total_rows": total_rows,
        "market_window_size": max(1, market_window_size),
        "news_size": max(1, news_size),
        "sec_filing_size": max(1, sec_filing_size),
        "earnings_size": max(1, earnings_size),
        "macro_size": max(1, macro_size),
    }


def _time_ordered_counts(total_rows: int, validation_fraction: float) -> tuple[int, int]:
    if validation_fraction == 0.0 or total_rows < 2:
        return total_rows, 0
    validation_count = max(1, int(total_rows * validation_fraction))
    validation_count = min(validation_count, total_rows - 1)
    return total_rows - validation_count, validation_count


def _iter_row_slice(
    store: LocalObjectStore,
    shard_path: str,
    start_index: int,
    stop_index: int,
):
    for index, row in enumerate(store.iter_jsonl(shard_path)):
        if index < start_index:
            continue
        if index >= stop_index:
            break
        yield row
