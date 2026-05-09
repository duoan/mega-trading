"""Universal proxy tokenizer for market events."""

from __future__ import annotations

from dataclasses import dataclass


PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2

_RETURN_BINS = (-300, -150, -75, -25, 0, 25, 75, 150, 300)
_RANGE_BINS = (10, 25, 50, 100, 200, 400, 800)
_GAP_BINS = (-200, -100, -50, -10, 0, 10, 50, 100, 200)
_VOLUME_BINS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
_DT_BINS = (1, 2, 3, 5, 10)


@dataclass(frozen=True)
class MarketEventTokenizer:
    """Map scale-normalized proxy events to a discrete token sequence."""

    token_groups: tuple[tuple[str, int], ...] = (
        ("side", 3),
        ("return", len(_RETURN_BINS) + 1),
        ("range", len(_RANGE_BINS) + 1),
        ("gap", len(_GAP_BINS) + 1),
        ("volume", len(_VOLUME_BINS) + 1),
        ("dt", len(_DT_BINS) + 1),
        ("weekday", 7),
    )

    @property
    def event_size(self) -> int:
        return len(self.token_groups)

    @property
    def vocab_size(self) -> int:
        return 3 + sum(size for _name, size in self.token_groups)

    def encode_event(self, event: dict[str, object]) -> list[int]:
        side_id = {"sell": 0, "flat": 1, "buy": 2}[str(event["side"])]
        values = (
            side_id,
            _bucket(float(event["return_bps"]), _RETURN_BINS),
            _bucket(float(event["range_bps"]), _RANGE_BINS),
            _bucket(float(event["gap_bps"]), _GAP_BINS),
            _bucket(float(event["log_volume_ratio"]), _VOLUME_BINS),
            _bucket(float(event["dt_days"]), _DT_BINS),
            int(event["weekday"]),
        )
        tokens: list[int] = []
        offset = 3
        for value, (_name, size) in zip(values, self.token_groups):
            if value < 0 or value >= size:
                raise ValueError(f"token value out of range: {value} >= {size}")
            tokens.append(offset + value)
            offset += size
        return tokens

    def token_name(self, token_id: int) -> str:
        if token_id == PAD_TOKEN:
            return "special:pad"
        if token_id == BOS_TOKEN:
            return "special:bos"
        if token_id == EOS_TOKEN:
            return "special:eos"
        offset = 3
        for name, size in self.token_groups:
            if offset <= token_id < offset + size:
                return f"{name}:{token_id - offset}"
            offset += size
        raise ValueError(f"unknown token_id: {token_id}")

    def return_bucket_value(self, token_id: int) -> float | None:
        name = self.token_name(token_id)
        if not name.startswith("return:"):
            return None
        bucket = int(name.split(":", 1)[1])
        edges = (-450, *_RETURN_BINS, 450)
        return (edges[bucket] + edges[bucket + 1]) / 2.0


def _bucket(value: float, bins: tuple[float, ...]) -> int:
    for index, edge in enumerate(bins):
        if value <= edge:
            return index
    return len(bins)
