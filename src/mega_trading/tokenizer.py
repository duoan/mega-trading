"""Paper-style composite tokenizer for order-flow events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2

_ACTIONS = ("add", "delete")
_SIDES = ("buy", "sell")
_RELATIVE_PRICE_BINS = (-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100)
_PRICE_DEPTH_BINS = (1, 2, 5, 10, 20, 50, 100, 200)
_LOG_SIZE_BINS = (-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4)
_DT_BINS = (0.001, 0.01, 0.1, 1, 5, 30, 60, 300)
_SPECIAL_TOKENS = 3


@dataclass(frozen=True)
class MarketEventTokenizer:
    """Fit bins and map each order-flow event to one joint token."""

    relative_price_edges: tuple[float, ...] = _RELATIVE_PRICE_BINS
    price_depth_edges: tuple[float, ...] = _PRICE_DEPTH_BINS
    log_size_edges: tuple[float, ...] = _LOG_SIZE_BINS
    dt_edges: tuple[float, ...] = _DT_BINS
    binning_method: str = "fixed"
    clip_quantile: float = 0.0

    @classmethod
    def fit(
        cls,
        events: Iterable[dict[str, Any]],
        *,
        relative_price_bins: int = 16,
        price_bins: int = 16,
        size_bins: int = 16,
        time_bins: int = 8,
        method: str = "quantile",
        clip_quantile: float = 0.01,
    ) -> "MarketEventTokenizer":
        relative_price_values: list[float] = []
        price_depth_values: list[float] = []
        log_size_values: list[float] = []
        dt_values: list[float] = []
        for event in events:
            relative_price_values.append(float(event["relative_price_bps"]))
            price_depth_values.append(float(event["price_depth_bps"]))
            log_size_values.append(_log_size(event))
            dt_values.append(float(event["interarrival_seconds"]))
        if not relative_price_values:
            raise ValueError("cannot fit tokenizer without events")
        if method not in {"quantile", "histogram"}:
            raise ValueError("tokenizer method must be one of: quantile, histogram")
        for name, value in {
            "relative_price_bins": relative_price_bins,
            "price_bins": price_bins,
            "size_bins": size_bins,
            "time_bins": time_bins,
        }.items():
            if value < 2:
                raise ValueError(f"{name} must be at least 2")
        if not 0.0 <= clip_quantile < 0.5:
            raise ValueError("clip_quantile must be in [0.0, 0.5)")
        return cls(
            relative_price_edges=_fit_edges(
                relative_price_values,
                relative_price_bins,
                method,
                clip_quantile,
            ),
            price_depth_edges=_fit_edges(
                price_depth_values, price_bins, method, clip_quantile
            ),
            log_size_edges=_fit_edges(log_size_values, size_bins, method, clip_quantile),
            dt_edges=_fit_edges(dt_values, time_bins, method, clip_quantile),
            binning_method=method,
            clip_quantile=clip_quantile,
        )

    @property
    def event_size(self) -> int:
        return 1

    @property
    def bucket_counts(self) -> dict[str, int]:
        return {
            "action": len(_ACTIONS),
            "side": len(_SIDES),
            "relative_price": len(self.relative_price_edges) + 1,
            "price_depth": len(self.price_depth_edges) + 1,
            "size": len(self.log_size_edges) + 1,
            "time": len(self.dt_edges) + 1,
        }

    @property
    def vocab_size(self) -> int:
        size = 1
        for bucket_count in self.bucket_counts.values():
            size *= bucket_count
        return _SPECIAL_TOKENS + size

    def encode_event(self, event: dict[str, object]) -> list[int]:
        values = (
            _ACTIONS.index(str(event["action"])),
            _SIDES.index(str(event["side"])),
            _bucket(float(event["relative_price_bps"]), self.relative_price_edges),
            _bucket(float(event["price_depth_bps"]), self.price_depth_edges),
            _bucket(_log_size(event), self.log_size_edges),
            _bucket(float(event["interarrival_seconds"]), self.dt_edges),
        )
        sizes = tuple(self.bucket_counts.values())
        for value, size in zip(values, sizes):
            if value < 0 or value >= size:
                raise ValueError(f"token value out of range: {value} >= {size}")
        return [_SPECIAL_TOKENS + _encode_mixed_radix(values, sizes)]

    def decode_token(self, token_id: int) -> tuple[int, int, int, int, int, int]:
        if token_id < _SPECIAL_TOKENS or token_id >= self.vocab_size:
            raise ValueError(f"unknown token_id: {token_id}")
        return _decode_mixed_radix(token_id - _SPECIAL_TOKENS, tuple(self.bucket_counts.values()))

    def token_name(self, token_id: int) -> str:
        if token_id == PAD_TOKEN:
            return "special:pad"
        if token_id == BOS_TOKEN:
            return "special:bos"
        if token_id == EOS_TOKEN:
            return "special:eos"
        action_id, side_id, relative_price_id, depth_id, size_id, time_id = self.decode_token(token_id)
        return (
            f"event:action={_ACTIONS[action_id]}:side={_SIDES[side_id]}:"
            f"relative_price={relative_price_id}:price_depth={depth_id}:size={size_id}:time={time_id}"
        )

    def price_depth_value(self, token_id: int) -> float | None:
        if token_id < _SPECIAL_TOKENS:
            return None
        _action_id, _side_id, _relative_price_id, depth_id, _size_id, _time_id = self.decode_token(token_id)
        return _bucket_center(depth_id, self.price_depth_edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "paper-order-flow-composite",
            "special_tokens": {"pad": PAD_TOKEN, "bos": BOS_TOKEN, "eos": EOS_TOKEN},
            "feature_order": ["action", "side", "relative_price", "price_depth", "size", "time"],
            "actions": list(_ACTIONS),
            "sides": list(_SIDES),
            "relative_price_edges": list(self.relative_price_edges),
            "price_depth_edges": list(self.price_depth_edges),
            "log_size_edges": list(self.log_size_edges),
            "dt_edges": list(self.dt_edges),
            "bucket_counts": self.bucket_counts,
            "event_size": self.event_size,
            "vocab_size": self.vocab_size,
            "binning_method": self.binning_method,
            "clip_quantile": self.clip_quantile,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MarketEventTokenizer":
        return cls(
            relative_price_edges=tuple(float(edge) for edge in value["relative_price_edges"]),
            price_depth_edges=tuple(float(edge) for edge in value["price_depth_edges"]),
            log_size_edges=tuple(float(edge) for edge in value["log_size_edges"]),
            dt_edges=tuple(float(edge) for edge in value["dt_edges"]),
            binning_method=str(value.get("binning_method", "fixed")),
            clip_quantile=float(value.get("clip_quantile", 0.0)),
        )


def _log_size(event: dict[str, object]) -> float:
    return math.log(max(float(event["size"]), 1e-12))


def _bucket(value: float, bins: tuple[float, ...]) -> int:
    for index, edge in enumerate(bins):
        if value <= edge:
            return index
    return len(bins)


def _fit_edges(values: list[float], bucket_count: int, method: str, clip_quantile: float) -> tuple[float, ...]:
    if len(values) < bucket_count:
        bucket_count = max(2, len(values))
    clipped = _clip_array(values, clip_quantile)
    if method == "histogram":
        low = float(np.min(clipped))
        high = float(np.max(clipped))
        if low == high:
            return tuple(low + index for index in range(1, bucket_count))
        step = (high - low) / bucket_count
        return tuple(low + step * index for index in range(1, bucket_count))
    quantiles = np.linspace(1.0 / bucket_count, (bucket_count - 1) / bucket_count, bucket_count - 1)
    return _unique_edges(float(value) for value in np.quantile(clipped, quantiles))


def _clip_array(values: list[float], clip_quantile: float) -> np.ndarray:
    if not values:
        raise ValueError("cannot fit tokenizer bins without values")
    array = np.asarray(values, dtype=np.float64)
    if clip_quantile == 0.0:
        return array
    low, high = np.quantile(array, [clip_quantile, 1.0 - clip_quantile])
    return np.clip(array, low, high)


def _clip(values: list[float], clip_quantile: float) -> list[float]:
    if not values:
        raise ValueError("cannot fit tokenizer bins without values")
    if clip_quantile == 0.0:
        return list(values)
    low = _quantile(values, clip_quantile)
    high = _quantile(values, 1.0 - clip_quantile)
    return [min(max(value, low), high) for value in values]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _unique_edges(edges: Iterable[float]) -> tuple[float, ...]:
    unique: list[float] = []
    for edge in edges:
        if not unique or edge > unique[-1]:
            unique.append(float(edge))
    return tuple(unique)


def _encode_mixed_radix(values: tuple[int, ...], sizes: tuple[int, ...]) -> int:
    token = 0
    for value, size in zip(values, sizes):
        token = token * size + value
    return token


def _decode_mixed_radix(value: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    decoded = []
    for size in reversed(sizes):
        decoded.append(value % size)
        value //= size
    return tuple(reversed(decoded))


def _bucket_center(bucket: int, edges: tuple[float, ...]) -> float:
    if not edges:
        return 0.0
    if bucket == 0:
        lower = edges[0] - (edges[1] - edges[0] if len(edges) > 1 else 1.0)
        upper = edges[0]
    elif bucket >= len(edges):
        lower = edges[-1]
        upper = edges[-1] + (edges[-1] - edges[-2] if len(edges) > 1 else 1.0)
    else:
        lower = edges[bucket - 1]
        upper = edges[bucket]
    return (lower + upper) / 2.0
