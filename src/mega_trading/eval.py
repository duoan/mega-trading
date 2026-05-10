"""Stylized-fact evaluation for token streams."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from mega_trading.core.store import ArtifactPaths, LocalObjectStore
from mega_trading.model import TradingModel
from mega_trading.tokenizer import BOS_TOKEN, MarketEventTokenizer


@dataclass(frozen=True)
class EvalResult:
    report_path: str


def run_eval(
    store: LocalObjectStore,
    run_id: str,
    mixture_name: str = "public",
    rollouts: int = 8,
    generated_tokens: int = 128,
    device: str = "auto",
) -> EvalResult:
    shard_path = f"datasets/mixture={mixture_name}/tokens.npy"
    profile = store.read_json(f"datasets/mixture={mixture_name}/tokens-profile.json")
    tokenizer = MarketEventTokenizer.from_dict(store.read_json(str(profile["tokenizer_path"])))
    checkpoint = torch.load(store.root / f"runs/{run_id}/checkpoint.pt", map_location="cpu")
    model = TradingModel(
        vocab_size=int(profile["vocab_size"]),
        block_size=int(profile["block_size"]),
        hidden_dim=int(checkpoint["config"]["hidden_dim"]),
        layers=int(checkpoint["config"]["layers"]),
        attention_heads=int(checkpoint["config"]["attention_heads"]),
        kv_heads=_optional_int(checkpoint["config"].get("kv_heads")),
        intermediate_dim=_optional_int(checkpoint["config"].get("intermediate_dim")),
        dropout=float(checkpoint["config"]["dropout"]),
        rope_theta=float(checkpoint["config"].get("rope_theta", 500_000.0)),
        norm_eps=float(checkpoint["config"].get("norm_eps", 1e-5)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    resolved_device = _resolve_device(device)
    model.to(resolved_device)

    real_depths = _price_depth_values_from_rows(store, shard_path, tokenizer, limit=rollouts, numpy_metadata=profile.get("numpy_dataset"))
    generated_depths = _generate_depths(model, tokenizer, resolved_device, int(profile["block_size"]), rollouts, generated_tokens)
    report = {
        "stage": "eval",
        "run_id": run_id,
        "mixture": mixture_name,
        "feature": "price_depth_bps",
        "real": _stylized_facts(real_depths),
        "generated": _stylized_facts(generated_depths),
        "distribution_l1": _distribution_l1(real_depths, generated_depths),
    }
    report_path = ArtifactPaths().eval(run_id, "report")
    store.write_json(report_path, report)
    return EvalResult(report_path=report_path)


def _generate_depths(
    model: TradingModel,
    tokenizer: MarketEventTokenizer,
    device: torch.device,
    block_size: int,
    rollouts: int,
    generated_tokens: int,
) -> list[float]:
    depths: list[float] = []
    seeds = torch.full((rollouts, 1), BOS_TOKEN, dtype=torch.long, device=device)
    tokens = model.generate(seeds, max_new_tokens=generated_tokens)
    for row in tokens.cpu().tolist():
        for token in row[: block_size + 1]:
            value = tokenizer.price_depth_value(int(token))
            if value is not None:
                depths.append(value)
    return depths


def _price_depth_values_from_rows(
    store: LocalObjectStore,
    shard_path: str,
    tokenizer: MarketEventTokenizer,
    limit: int,
    numpy_metadata: object = None,
) -> list[float]:
    values: list[float] = []
    if isinstance(numpy_metadata, dict) and numpy_metadata.get("storage") == "token_stream":
        rows_seen = 0
        sequence_length = int(numpy_metadata["sequence_length"])
        stride = int(numpy_metadata["stride"])
        for partition in numpy_metadata.get("partitions", []):
            tokens = np.load(store.root / str(partition["tokens_path"]), mmap_mode="r")
            for index in range(int(partition["sequence_count"])):
                if rows_seen >= limit:
                    return values
                rows_seen += 1
                offset = index * stride
                for token in tokens[offset : offset + sequence_length]:
                    value = tokenizer.price_depth_value(int(token))
                    if value is not None:
                        values.append(value)
        return values
    if isinstance(numpy_metadata, dict) and bool(numpy_metadata.get("partitioned")):
        rows_seen = 0
        for partition in numpy_metadata.get("partitions", []):
            tokens = np.load(store.root / str(partition["tokens_path"]), mmap_mode="r")
            for row in tokens:
                if rows_seen >= limit:
                    return values
                rows_seen += 1
                for token in row:
                    value = tokenizer.price_depth_value(int(token))
                    if value is not None:
                        values.append(value)
        return values
    numpy_path = store.root / shard_path
    if numpy_path.exists():
        tokens = np.load(numpy_path, mmap_mode="r")
        for row in tokens[:limit]:
            for token in row:
                value = tokenizer.price_depth_value(int(token))
                if value is not None:
                    values.append(value)
        return values
    raise FileNotFoundError(f"NumPy token dataset is missing for {shard_path}")


def _stylized_facts(returns: list[float]) -> dict[str, float]:
    if not returns:
        return {
            "count": 0.0,
            "mean_bps": 0.0,
            "std_bps": 0.0,
            "excess_kurtosis": 0.0,
            "extreme_fraction": 0.0,
            "return_autocorrelation_lag1": 0.0,
            "abs_return_autocorrelation_lag1": 0.0,
        }
    mean = sum(returns) / len(returns)
    centered = [value - mean for value in returns]
    variance = sum(value * value for value in centered) / max(len(centered), 1)
    std = math.sqrt(variance)
    fourth = sum(value**4 for value in centered) / max(len(centered), 1)
    kurtosis = fourth / (variance * variance) - 3.0 if variance > 0 else 0.0
    return {
        "count": float(len(returns)),
        "mean_bps": mean,
        "std_bps": std,
        "excess_kurtosis": kurtosis,
        "extreme_fraction": sum(1 for value in returns if abs(value) >= 150.0) / len(returns),
        "return_autocorrelation_lag1": _autocorrelation(returns),
        "abs_return_autocorrelation_lag1": _autocorrelation([abs(value) for value in returns]),
    }


def _autocorrelation(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    left = values[:-1]
    right = values[1:]
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator_left = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    denominator_right = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    denominator = denominator_left * denominator_right
    return numerator / denominator if denominator else 0.0


def _distribution_l1(real: list[float], generated: list[float]) -> float:
    real_counts = Counter(real)
    generated_counts = Counter(generated)
    keys = set(real_counts) | set(generated_counts)
    real_total = max(sum(real_counts.values()), 1)
    generated_total = max(sum(generated_counts.values()), 1)
    return sum(abs(real_counts[key] / real_total - generated_counts[key] / generated_total) for key in keys)


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested_device)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
