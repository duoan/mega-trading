"""Chronological backtest evaluation on held-out token streams."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from mega_trading.checkpoint import load_checkpoint_model_state
from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.dataset import NumpyTickerTimeDataset, prepared_split_counts, split_totals
from mega_trading.eval import _distribution_l1, _generate_depths, _stylized_facts
from mega_trading.events import STREAM_CONTRACT
from mega_trading.model import TradingModel
from mega_trading.tokenizer import MarketEventTokenizer


@dataclass(frozen=True)
class BacktestResult:
    report_path: str


def run_backtest(
    store: LocalObjectStore,
    run_id: str,
    mixture_name: str = "public",
    max_batches: int = 128,
    rollouts: int = 256,
    generated_tokens: int = 1024,
    device: str = "auto",
) -> BacktestResult:
    """Evaluate a checkpoint on the prepared chronological backtest split."""
    profile = store.read_json(f"datasets/mixture={mixture_name}/tokens-profile.json")
    if profile.get("stream_contract") != STREAM_CONTRACT:
        raise ValueError("profile is not a valid token stream contract")
    numpy_metadata = dict(profile.get("numpy_dataset", {}))
    ticker_counts = {str(key): int(value) for key, value in dict(profile["sequence_counts"]).items()}
    split_counts = prepared_split_counts(numpy_metadata, ticker_counts, validation_fraction=0.0)
    split_count_totals = split_totals(split_counts)
    if split_count_totals["backtest"] <= 0:
        raise ValueError("prepared dataset has no backtest rows")

    tokenizer = MarketEventTokenizer.from_dict(store.read_json(str(profile["tokenizer_path"])))
    checkpoint = torch.load(store.root / f"runs/{run_id}/checkpoint.pt", map_location="cpu")
    model = _model_from_checkpoint(profile, checkpoint)
    load_checkpoint_model_state(model, checkpoint)
    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    model.eval()

    dataset = NumpyTickerTimeDataset(store, numpy_metadata, split_counts, split="backtest")
    batch_size = min(int(checkpoint["config"].get("batch_size", 8)), 8)
    metrics = _score_backtest(model, DataLoader(dataset, batch_size=batch_size), resolved_device, max_batches)
    real_depths = _price_depth_values_from_backtest(dataset, tokenizer, limit_rows=rollouts)
    generated_depths = _generate_depths(model, tokenizer, resolved_device, int(profile["block_size"]), rollouts, generated_tokens)
    report = {
        "stage": "backtest",
        "run_id": run_id,
        "mixture": mixture_name,
        "paper_alignment": (
            "TradeFM evaluates generated order-flow rollouts inside a deterministic market simulator. "
            "This public-data MVP reports open-loop chronological backtest loss plus rollout-vs-real "
            "stylized facts on the held-out backtest split."
        ),
        "split_counts": split_count_totals,
        "max_batches": max_batches,
        **metrics,
        "real_backtest": _stylized_facts(real_depths),
        "generated_rollout": _stylized_facts(generated_depths),
        "price_depth_distribution_l1": _distribution_l1(real_depths, generated_depths),
    }
    report_path = ArtifactPaths().eval(run_id, "backtest")
    store.write_json(report_path, report)
    return BacktestResult(report_path=report_path)


@torch.no_grad()
def _score_backtest(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    total_top1 = 0
    total_top5 = 0
    batches = 0
    for batch in loader:
        if batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        total_loss += float(loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).detach().cpu())
        total_tokens += int(labels.numel())
        total_top1 += _topk_hits(logits, labels, 1)
        total_top5 += _topk_hits(logits, labels, 5)
        batches += 1
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "backtest_batches": float(batches),
        "backtest_tokens": float(total_tokens),
        "backtest_loss": mean_loss,
        "backtest_perplexity": math.exp(min(mean_loss, 20.0)),
        "backtest_top1_accuracy": total_top1 / max(total_tokens, 1),
        "backtest_top5_accuracy": total_top5 / max(total_tokens, 1),
    }


def _price_depth_values_from_backtest(
    dataset: NumpyTickerTimeDataset,
    tokenizer: MarketEventTokenizer,
    limit_rows: int,
) -> list[float]:
    values: list[float] = []
    for rows_seen, example in enumerate(dataset):
        if rows_seen >= limit_rows:
            break
        for token in example["labels"].tolist():
            value = tokenizer.price_depth_value(int(token))
            if value is not None:
                values.append(value)
    return values


def _topk_hits(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    topk = logits.detach().topk(min(k, logits.shape[-1]), dim=-1).indices
    hits = topk.eq(labels.unsqueeze(-1)).any(dim=-1)
    return int(hits.sum().detach().cpu())


def _model_from_checkpoint(profile: dict[str, Any], checkpoint: dict[str, Any]) -> TradingModel:
    config = dict(checkpoint["config"])
    return TradingModel(
        vocab_size=int(profile["vocab_size"]),
        block_size=int(profile["block_size"]),
        hidden_dim=int(config["hidden_dim"]),
        layers=int(config["layers"]),
        attention_heads=int(config["attention_heads"]),
        kv_heads=_optional_int(config.get("kv_heads")),
        intermediate_dim=_optional_int(config.get("intermediate_dim")),
        dropout=float(config["dropout"]),
        rope_theta=float(config.get("rope_theta", 500_000.0)),
        norm_eps=float(config.get("norm_eps", 1e-5)),
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested_device)
