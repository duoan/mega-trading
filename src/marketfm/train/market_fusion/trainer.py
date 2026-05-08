"""Training loop for the MarketFusion model."""

from __future__ import annotations

from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from marketfm.core.schemas import Manifest
from marketfm.core.store import ArtifactPaths, LocalObjectStore
from marketfm.train.market_fusion.config import MarketFusionTrainConfig, MarketFusionTrainResult
from marketfm.train.market_fusion.dataset import MarketFusionDataset, infer_stream_sizes
from marketfm.train.market_fusion.model import MarketFusionModel


class MarketFusionTrainer:
    def __init__(self, store: LocalObjectStore, config: MarketFusionTrainConfig) -> None:
        self.store = store
        self.config = config
        self.paths = ArtifactPaths(run_id=config.run_id)

    def train(self, shard_path: str) -> MarketFusionTrainResult:
        torch.manual_seed(self.config.seed)
        rows = self.store.read_jsonl(shard_path)
        sizes = infer_stream_sizes(
            rows,
            self.config.price_window_size,
            self.config.fundamental_size,
            self.config.evidence_size,
        )
        dataset = MarketFusionDataset(rows, *sizes)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        iterator = iter(loader)
        device = torch.device(self.config.device)
        model = MarketFusionModel(*sizes, hidden_dim=self.config.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        metrics: list[dict[str, object]] = []
        metrics_path = self.paths.run("metrics")

        for step in range(1, self.config.max_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
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
                    "stage": "market_fusion",
                    "loss": float(loss.detach().cpu()),
                    "return_accuracy": _accuracy(return_logits, batch["return_label"]),
                    "risk_accuracy": _accuracy(risk_logits, batch["risk_label"]),
                    "examples_per_second": int(batch["return_label"].shape[0]) / elapsed,
                }
            )
            self.store.write_jsonl(metrics_path, metrics)

        checkpoint_path = self.paths.run("checkpoint.pt")
        checkpoint_target = self.store.root / checkpoint_path
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage": "market_fusion",
                "model_state_dict": model.state_dict(),
                "config": self.config.__dict__,
                "stream_sizes": {
                    "price_window_size": sizes[0],
                    "fundamental_size": sizes[1],
                    "evidence_size": sizes[2],
                },
                "step": self.config.max_steps,
                "shard_path": shard_path,
            },
            checkpoint_target,
        )
        manifest = Manifest(
            manifest_id=f"{self.config.run_id}-market-fusion",
            artifact_type="training_run",
            paths=[metrics_path, checkpoint_path],
            metadata={
                "stage": "market_fusion",
                "steps": str(self.config.max_steps),
                "shard_path": shard_path,
                "stream_contract": "price_fundamental_text",
                "config_hash": self.config.content_hash(),
                "device": self.config.device,
            },
        )
        manifest_path = self.paths.manifest("runs", f"{self.config.run_id}-market-fusion")
        self.store.write_manifest(manifest_path, manifest)
        return MarketFusionTrainResult(steps=self.config.max_steps, checkpoint_path=checkpoint_path, manifest_path=manifest_path)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().detach().cpu())
