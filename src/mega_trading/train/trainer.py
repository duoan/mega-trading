"""Training loop for TradingFoundationModel."""

from __future__ import annotations

from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.train.config import TradingFoundationTrainConfig, TradingFoundationTrainResult
from mega_trading.train.dataset import TradingFoundationDataset, infer_stream_sizes
from mega_trading.train.model import TradingFoundationModel


class TradingFoundationTrainer:
    def __init__(self, store: LocalObjectStore, config: TradingFoundationTrainConfig) -> None:
        self.store = store
        self.config = config
        self.paths = ArtifactPaths(run_id=config.run_id)

    def train(self, shard_path: str) -> TradingFoundationTrainResult:
        torch.manual_seed(self.config.seed)
        rows = self.store.read_jsonl(shard_path)
        if not rows:
            raise ValueError("training shard is empty")
        train_rows, validation_rows = _time_ordered_split(rows, self.config.validation_fraction)
        sizes = infer_stream_sizes(
            rows,
            self.config.price_window_size,
            self.config.fundamental_size,
            self.config.evidence_size,
        )
        train_dataset = TradingFoundationDataset(train_rows, *sizes)
        validation_dataset = TradingFoundationDataset(validation_rows, *sizes) if validation_rows else None
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        validation_loader = (
            DataLoader(validation_dataset, batch_size=self.config.batch_size, shuffle=False) if validation_dataset else None
        )
        iterator = iter(train_loader)
        device = torch.device(self.config.device)
        model = TradingFoundationModel(
            *sizes,
            hidden_dim=self.config.hidden_dim,
            attention_heads=self.config.attention_heads,
            use_price=self.config.use_price,
            use_fundamentals=self.config.use_fundamentals,
            use_evidence=self.config.use_evidence,
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
            return_logits, risk_logits = model(batch)
            return_loss = loss_fn(return_logits, batch["return_label"])
            risk_loss = loss_fn(risk_logits, batch["risk_label"])
            loss = return_loss + risk_loss
            loss.backward()
            optimizer.step()
            elapsed = max(perf_counter() - started, 1e-9)
            metrics.append(
                {
                    "step": step,
                    "stage": "trading_foundation_model",
                    "loss": float(loss.detach().cpu()),
                    "train_loss": float(loss.detach().cpu()),
                    "return_accuracy": _accuracy(return_logits, batch["return_label"]),
                    "risk_accuracy": _accuracy(risk_logits, batch["risk_label"]),
                    "train_sample_count": len(train_rows),
                    "validation_sample_count": len(validation_rows),
                    "examples_per_second": int(batch["return_label"].shape[0]) / elapsed,
                }
            )
            if validation_loader and (step % self.config.eval_interval == 0 or step == self.config.max_steps):
                metrics[-1].update(_evaluate(model, validation_loader, device, loss_fn))
            self.store.write_jsonl(metrics_path, metrics)

        checkpoint_path = self.paths.run("checkpoint.pt")
        checkpoint_target = self.store.root / checkpoint_path
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage": "trading_foundation_model",
                "model_state_dict": model.state_dict(),
                "config": self.config.__dict__,
                "stream_sizes": {
                    "price_window_size": sizes[0],
                    "fundamental_size": sizes[1],
                    "evidence_size": sizes[2],
                },
                "modalities": {
                    "price": self.config.use_price,
                    "fundamentals": self.config.use_fundamentals,
                    "evidence": self.config.use_evidence,
                },
                "attention_heads": self.config.attention_heads,
                "train_sample_count": len(train_rows),
                "validation_sample_count": len(validation_rows),
                "step": self.config.max_steps,
                "shard_path": shard_path,
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
                "stream_contract": "price_fundamental_text",
                "attention_heads": str(self.config.attention_heads),
                "train_sample_count": str(len(train_rows)),
                "validation_sample_count": str(len(validation_rows)),
                "validation_fraction": str(self.config.validation_fraction),
                "eval_interval": str(self.config.eval_interval),
                "use_price": str(self.config.use_price),
                "use_fundamentals": str(self.config.use_fundamentals),
                "use_evidence": str(self.config.use_evidence),
                "config_hash": self.config.content_hash(),
                "device": self.config.device,
            },
        )
        manifest_path = self.paths.manifest("runs", f"{self.config.run_id}-trading-foundation-model")
        self.store.write_manifest(manifest_path, manifest)
        return TradingFoundationTrainResult(steps=self.config.max_steps, checkpoint_path=checkpoint_path, manifest_path=manifest_path)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().detach().cpu())


def _evaluate(
    model: TradingFoundationModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_examples = 0
    total_loss = 0.0
    return_correct = 0
    risk_correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            return_logits, risk_logits = model(batch)
            batch_size = int(batch["return_label"].shape[0])
            return_loss = loss_fn(return_logits, batch["return_label"])
            risk_loss = loss_fn(risk_logits, batch["risk_label"])
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


def _time_ordered_split(
    rows: list[dict[str, object]],
    validation_fraction: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("as_of_time", "")),
            str(row.get("ticker", "")),
            str(row.get("sample_id", "")),
        ),
    )
    if validation_fraction == 0.0 or len(ordered_rows) < 2:
        return ordered_rows, []
    validation_count = max(1, int(len(ordered_rows) * validation_fraction))
    validation_count = min(validation_count, len(ordered_rows) - 1)
    split_index = len(ordered_rows) - validation_count
    return ordered_rows[:split_index], ordered_rows[split_index:]
