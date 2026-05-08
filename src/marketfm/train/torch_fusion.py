"""PyTorch fusion model for multi-stream market samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from marketfm.core.hashing import stable_hash
from marketfm.core.schemas import Manifest
from marketfm.core.store import ArtifactPaths, LocalObjectStore
from marketfm.train.fusion import RETURN_LABELS, RISK_LABELS


RETURN_TO_ID = {label: index for index, label in enumerate(RETURN_LABELS)}
RISK_TO_ID = {label: index for index, label in enumerate(RISK_LABELS)}


@dataclass(frozen=True)
class TorchFusionTrainConfig:
    run_id: str
    max_steps: int
    hidden_dim: int = 32
    batch_size: int = 8
    learning_rate: float = 1e-3
    price_window_size: int | None = None
    fundamental_size: int | None = None
    evidence_size: int | None = None
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def content_hash(self) -> str:
        return stable_hash(
            {
                "run_id": self.run_id,
                "max_steps": self.max_steps,
                "hidden_dim": self.hidden_dim,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "price_window_size": self.price_window_size,
                "fundamental_size": self.fundamental_size,
                "evidence_size": self.evidence_size,
                "seed": self.seed,
                "device": self.device,
            }
        )


@dataclass(frozen=True)
class TorchFusionTrainResult:
    steps: int
    checkpoint_path: str
    manifest_path: str


class TorchFusionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, object]],
        price_window_size: int,
        fundamental_size: int,
        evidence_size: int,
    ) -> None:
        self.rows = rows
        self.price_window_size = price_window_size
        self.fundamental_size = fundamental_size
        self.evidence_size = evidence_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        price = _price_tensor(
            _float_list(row.get("price_returns", [])),
            _float_list(row.get("price_levels", [])),
            self.price_window_size,
        )
        fundamentals = _fixed_vector(_float_list(row.get("fundamental_values", [])), self.fundamental_size, scale=1_000_000_000.0)
        evidence = _fixed_vector(_float_list(row.get("evidence_token_ids", [])), self.evidence_size, scale=10_000.0)
        return {
            "price": price,
            "fundamentals": fundamentals,
            "evidence": evidence,
            "return_label": torch.tensor(RETURN_TO_ID[str(row["return_label"])], dtype=torch.long),
            "risk_label": torch.tensor(RISK_TO_ID[str(row["risk_label"])], dtype=torch.long),
        }


class TorchFusionModel(nn.Module):
    """Small cross-attention fusion model over price, fundamentals, and evidence streams."""

    def __init__(self, price_window_size: int, fundamental_size: int, evidence_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.price_proj = nn.Linear(2, hidden_dim)
        self.fundamental_proj = nn.Linear(1, hidden_dim)
        self.evidence_proj = nn.Linear(1, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.return_head = nn.Linear(hidden_dim, len(RETURN_LABELS))
        self.risk_head = nn.Linear(hidden_dim, len(RISK_LABELS))
        self.price_window_size = price_window_size
        self.fundamental_size = fundamental_size
        self.evidence_size = evidence_size

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        price_tokens = self.price_proj(batch["price"])
        fundamental_tokens = self.fundamental_proj(batch["fundamentals"].unsqueeze(-1))
        evidence_tokens = self.evidence_proj(batch["evidence"].unsqueeze(-1))
        context_tokens = torch.cat([fundamental_tokens, evidence_tokens], dim=1)
        attended_price, _ = self.cross_attention(price_tokens, context_tokens, context_tokens)
        price_repr = self.norm(price_tokens + attended_price).mean(dim=1)
        fundamental_repr = fundamental_tokens.mean(dim=1)
        evidence_repr = evidence_tokens.mean(dim=1)
        fused = self.fusion(torch.cat([price_repr, fundamental_repr, evidence_repr], dim=1))
        return self.return_head(fused), self.risk_head(fused)


class TorchFusionTrainer:
    def __init__(self, store: LocalObjectStore, config: TorchFusionTrainConfig) -> None:
        self.store = store
        self.config = config
        self.paths = ArtifactPaths(run_id=config.run_id)

    def train(self, shard_path: str) -> TorchFusionTrainResult:
        torch.manual_seed(self.config.seed)
        rows = self.store.read_jsonl(shard_path)
        sizes = _stream_sizes(rows, self.config)
        dataset = TorchFusionDataset(rows, *sizes)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        iterator = iter(loader)
        device = torch.device(self.config.device)
        model = TorchFusionModel(*sizes, hidden_dim=self.config.hidden_dim).to(device)
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
                    "stage": "torch_fusion",
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
                "stage": "torch_fusion",
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
            manifest_id=f"{self.config.run_id}-torch-fusion",
            artifact_type="training_run",
            paths=[metrics_path, checkpoint_path],
            metadata={
                "stage": "torch_fusion",
                "steps": str(self.config.max_steps),
                "shard_path": shard_path,
                "stream_contract": "price_fundamental_text",
                "config_hash": self.config.content_hash(),
                "device": self.config.device,
            },
        )
        manifest_path = self.paths.manifest("runs", f"{self.config.run_id}-torch-fusion")
        self.store.write_manifest(manifest_path, manifest)
        return TorchFusionTrainResult(steps=self.config.max_steps, checkpoint_path=checkpoint_path, manifest_path=manifest_path)


def _stream_sizes(rows: list[dict[str, object]], config: TorchFusionTrainConfig) -> tuple[int, int, int]:
    price_window_size = config.price_window_size or max(1, max((len(row.get("price_returns", [])) for row in rows), default=1))
    fundamental_size = config.fundamental_size or max(1, max((len(row.get("fundamental_values", [])) for row in rows), default=1))
    evidence_size = config.evidence_size or max(1, max((len(row.get("evidence_token_ids", [])) for row in rows), default=1))
    return price_window_size, fundamental_size, evidence_size


def _price_tensor(returns: list[float], levels: list[float], window_size: int) -> torch.Tensor:
    fixed_returns = _pad_or_truncate(returns, window_size)
    fixed_levels = _pad_or_truncate(levels, window_size)
    return torch.tensor(list(zip(fixed_returns, fixed_levels)), dtype=torch.float32)


def _fixed_vector(values: list[float], size: int, scale: float) -> torch.Tensor:
    return torch.tensor([value / scale for value in _pad_or_truncate(values, size)], dtype=torch.float32)


def _pad_or_truncate(values: list[float], size: int) -> list[float]:
    values = values[-size:]
    return [0.0] * (size - len(values)) + values


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().detach().cpu())
