"""Deterministic paper-style order-flow fixtures for smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

from mega_trading.core.schemas import OrderFlowEventRecord


@dataclass(frozen=True)
class FixtureBundle:
    order_flow: list[OrderFlowEventRecord]
    bad_records: list[dict[str, str]]


def load_fixture_bundle() -> FixtureBundle:
    """Return small deterministic order-flow sequences for tests and demos."""

    events: list[OrderFlowEventRecord] = []
    for ticker_index, ticker in enumerate(("ACME", "NOVA")):
        for index in range(12):
            side = "buy" if (index + ticker_index) % 2 == 0 else "sell"
            action = "add" if index % 3 else "delete"
            timestamp = f"2024-01-02T14:{30 + index:02d}:00Z"
            depth = 2.5 + (index % 5) * 1.5 + ticker_index
            relative_price = depth if side == "buy" else -depth
            size = 100.0 + index * 12.0 + ticker_index * 25.0
            events.append(
                OrderFlowEventRecord(
                    event_id=f"fixture-{ticker}-{index:04d}",
                    ticker=ticker,
                    timestamp=timestamp,
                    date="2024-01-02",
                    action=action,
                    side=side,
                    midprice=100.0 + ticker_index * 10.0 + index * 0.01,
                    relative_price_bps=relative_price,
                    price_depth_bps=depth,
                    size=size / 100.0,
                    interarrival_seconds=60.0,
                    provider="fixture",
                    source_ids=[f"fixture:{ticker}:{index:04d}"],
                    midprice_return_bps=(1.0 if side == "buy" else -1.0) * depth,
                )
            )

    return FixtureBundle(order_flow=events, bad_records=[{"record_id": "bad-order-flow", "reason": "invalid_event"}])
