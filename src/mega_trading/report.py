"""Self-contained HTML reports for training and backtest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mega_trading.backtest import _model_from_checkpoint
from mega_trading.checkpoint import load_checkpoint_model_state
from mega_trading.core.store import ArtifactNotFoundError, LocalObjectStore
from mega_trading.data.public.binance import iter_trade_fields_from_archive
from mega_trading.tokenizer import MarketEventTokenizer


@dataclass(frozen=True)
class ReportResult:
    report_path: str


MAX_CHART_TICKERS = 3
MAX_CANDLE_TRADES = 30_000
MAX_CANDLES = 48
FORECAST_TOKENS = 48
FORECAST_CONTEXT_TOKENS = 128


def run_report(
    store: LocalObjectStore,
    run_id: str,
    mixture_name: str = "public",
    output_path: str | None = None,
) -> ReportResult:
    """Render a local HTML dashboard from prepared dataset, training, and backtest artifacts."""
    profile = store.read_json(f"datasets/mixture={mixture_name}/tokens-profile.json")
    metrics = _load_metrics(store, run_id)
    backtest = _read_optional_json(store, f"evals/{run_id}/backtest.json")
    eval_report = _read_optional_json(store, f"evals/{run_id}/report.json")
    manifest = _read_optional_json(store, f"manifests/training/{run_id}.json")

    report_path = output_path or f"reports/{run_id}/backtest.html"
    target = _artifact_target(store, report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _render_html(
            store=store,
            run_id=run_id,
            mixture_name=mixture_name,
            profile=profile,
            metrics=metrics,
            backtest=backtest,
            eval_report=eval_report,
            manifest=manifest,
        ),
        encoding="utf-8",
    )
    return ReportResult(report_path=report_path)


def _load_metrics(store: LocalObjectStore, run_id: str) -> list[dict[str, Any]]:
    metrics_path = f"runs/{run_id}/metrics.json"
    try:
        value = store.read_json(metrics_path)
    except ArtifactNotFoundError:
        return []
    return [dict(row) for row in value.get("metrics", []) if isinstance(row, dict)]


def _read_optional_json(store: LocalObjectStore, path: str) -> dict[str, Any] | None:
    try:
        return store.read_json(path)
    except ArtifactNotFoundError:
        return None


def _render_html(
    *,
    store: LocalObjectStore,
    run_id: str,
    mixture_name: str,
    profile: dict[str, Any],
    metrics: list[dict[str, Any]],
    backtest: dict[str, Any] | None,
    eval_report: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
) -> str:
    split_totals = _split_totals(profile)
    checkpoint_path = store.root / "runs" / run_id / "checkpoint.pt"
    mlflow_path = store.root / "runs" / "mlflow" / "mlflow.db"
    latest = metrics[-1] if metrics else {}
    cards = [
        ("Run", run_id),
        ("Mixture", mixture_name),
        ("Checkpoint", "present" if checkpoint_path.exists() else "missing"),
        ("Steps", _format_number(latest.get("step", 0))),
        ("Train loss", _format_float(latest.get("train_loss"))),
        ("Validation loss", _format_float(latest.get("validation_loss"))),
        ("Backtest loss", _format_float(backtest.get("backtest_loss") if backtest else None)),
        ("Backtest ppl", _format_float(backtest.get("backtest_perplexity") if backtest else None)),
    ]
    sections = [
        _summary_cards(cards),
        _artifact_status(store, run_id, checkpoint_path, mlflow_path, backtest),
        _split_chart(split_totals),
        _training_chart(metrics),
        _ticker_forecast_section(store, profile, run_id),
        _backtest_section(backtest),
        _stylized_fact_section(backtest, eval_report),
        _config_section(profile, manifest),
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mega-Trading Backtest Report - {escape(run_id)}</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #11182b;
      --panel2: #151f36;
      --text: #edf2ff;
      --muted: #a7b0c3;
      --line: #263451;
      --green: #74d680;
      --blue: #73a7ff;
      --orange: #ffb86b;
      --red: #ff7b86;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
    header {{ margin-bottom: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    h2 {{ margin: 0 0 16px; font-size: 20px; }}
    p {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
    .card, section {{
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
    }}
    section {{ margin-top: 18px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ margin-top: 8px; font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }}
    .two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    code {{ color: #c7d7ff; background: rgba(115, 167, 255, .12); padding: 2px 6px; border-radius: 6px; }}
    .warn {{ color: var(--orange); }}
    .ok {{ color: var(--green); }}
    svg {{ width: 100%; height: auto; display: block; }}
    .legend {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="label">Mega-Trading</div>
    <h1>Backtest Dashboard</h1>
    <p>Self-contained report generated from local training, MLflow, dataset split, and backtest artifacts.</p>
  </header>
  {"".join(sections)}
</main>
</body>
</html>
"""


def _summary_cards(cards: list[tuple[str, str]]) -> str:
    return '<div class="grid">' + "".join(
        f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>'
        for label, value in cards
    ) + "</div>"


def _artifact_status(
    store: LocalObjectStore,
    run_id: str,
    checkpoint_path: Path,
    mlflow_path: Path,
    backtest: dict[str, Any] | None,
) -> str:
    rows = [
        ("Checkpoint", checkpoint_path, checkpoint_path.exists()),
        ("Metrics", store.root / "runs" / run_id / "metrics.json", (store.root / "runs" / run_id / "metrics.json").exists()),
        ("MLflow DB", mlflow_path, mlflow_path.exists()),
        ("Backtest JSON", store.root / "evals" / run_id / "backtest.json", backtest is not None),
    ]
    body = "".join(
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td><code>{escape(str(path))}</code></td>"
        f"<td class=\"{'ok' if exists else 'warn'}\">{'present' if exists else 'missing'}</td>"
        "</tr>"
        for name, path, exists in rows
    )
    hint = "" if backtest is not None else "<p class=\"warn\">Backtest metrics are missing. Run <code>mega-trading backtest --run-id {}</code> before regenerating this report.</p>".format(escape(run_id))
    return f"<section><h2>Artifacts</h2><table><tbody>{body}</tbody></table>{hint}</section>"


def _split_totals(profile: dict[str, Any]) -> dict[str, int]:
    numpy_dataset = profile.get("numpy_dataset", {})
    splits = numpy_dataset.get("splits", {}) if isinstance(numpy_dataset, dict) else {}
    totals = splits.get("totals", {}) if isinstance(splits, dict) else {}
    return {
        "train": int(totals.get("train", 0)),
        "validation": int(totals.get("validation", 0)),
        "backtest": int(totals.get("backtest", 0)),
    }


def _split_chart(totals: dict[str, int]) -> str:
    values = [("train", totals["train"], "#74d680"), ("validation", totals["validation"], "#73a7ff"), ("backtest", totals["backtest"], "#ffb86b")]
    max_value = max((value for _, value, _ in values), default=1) or 1
    bars = []
    labels = []
    for index, (name, value, color) in enumerate(values):
        height = 180 * value / max_value
        x = 90 + index * 170
        y = 220 - height
        bars.append(f'<rect x="{x}" y="{y:.2f}" width="90" height="{height:.2f}" rx="8" fill="{color}"/>')
        labels.append(f'<text x="{x + 45}" y="245" text-anchor="middle" fill="#a7b0c3">{escape(name)}</text>')
        labels.append(f'<text x="{x + 45}" y="{max(y - 10, 20):.2f}" text-anchor="middle" fill="#edf2ff">{_format_number(value)}</text>')
    svg = f'<svg viewBox="0 0 620 270" role="img" aria-label="split counts">{"".join(bars + labels)}</svg>'
    return f"<section><h2>Prepared Split</h2>{svg}<div class=\"legend\">Chronological per-ticker split from prepared dataset metadata.</div></section>"


def _training_chart(metrics: list[dict[str, Any]]) -> str:
    train = [_as_float(row.get("train_loss")) for row in metrics]
    val = [_as_float(row.get("validation_loss")) for row in metrics if row.get("validation_loss") is not None]
    svg = _line_chart(
        [
            ("train loss", "#74d680", train),
            ("validation loss", "#73a7ff", val),
        ],
        title="loss curves",
    )
    return f"<section><h2>Training Curves</h2>{svg}<div class=\"legend\">Green: train loss. Blue: validation loss when evaluated.</div></section>"


def _ticker_forecast_section(store: LocalObjectStore, profile: dict[str, Any], run_id: str) -> str:
    numpy_dataset = profile.get("numpy_dataset", {})
    if not isinstance(numpy_dataset, dict) or numpy_dataset.get("storage") != "token_stream":
        return "<section><h2>Ticker Candles + Forecast</h2><p class=\"warn\">Ticker forecast charts require token-stream Binance data.</p></section>"
    checkpoint_path = store.root / "runs" / run_id / "checkpoint.pt"
    if not checkpoint_path.exists():
        return "<section><h2>Ticker Candles + Forecast</h2><p class=\"warn\">No checkpoint found, so model-implied forecasts cannot be drawn.</p></section>"
    charts = _ticker_forecast_charts(store, profile, run_id)
    if not charts:
        return "<section><h2>Ticker Candles + Forecast</h2><p class=\"warn\">No raw Binance archives matched the prepared token streams.</p></section>"
    return (
        "<section><h2>Ticker Candles + Forecast</h2>"
        "<p>Green/red candles show sampled real Binance trades from a held-out archive. Blue line is the model-implied price path from generated relative-price tokens seeded on the same ticker stream.</p>"
        + "".join(charts)
        + "</section>"
    )


def _ticker_forecast_charts(store: LocalObjectStore, profile: dict[str, Any], run_id: str) -> list[str]:
    try:
        tokenizer = MarketEventTokenizer.from_dict(store.read_json(str(profile["tokenizer_path"])))
        checkpoint = torch.load(store.root / "runs" / run_id / "checkpoint.pt", map_location="cpu", weights_only=False)
        model = _model_from_checkpoint(profile, checkpoint)
        load_checkpoint_model_state(model, checkpoint)
        model.eval()
    except Exception as exc:  # pragma: no cover - defensive for very large or partial checkpoints
        return [f"<p class=\"warn\">Could not load model forecast path: {escape(str(exc))}</p>"]

    charts: list[str] = []
    for ticker in _selected_tickers(profile, MAX_CHART_TICKERS):
        try:
            chart = _ticker_forecast_chart(store, profile, ticker, tokenizer, model)
        except Exception as exc:  # pragma: no cover - report should degrade instead of failing
            chart = f"<div class=\"card\"><h3>{escape(ticker)}</h3><p class=\"warn\">Could not render ticker chart: {escape(str(exc))}</p></div>"
        if chart:
            charts.append(chart)
    return charts


def _selected_tickers(profile: dict[str, Any], limit: int) -> list[str]:
    ticker_to_id = dict(dict(profile.get("numpy_dataset", {})).get("ticker_to_id", {}))
    preferred = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    selected = [ticker for ticker in preferred if ticker in ticker_to_id]
    selected.extend(ticker for ticker in sorted(ticker_to_id) if ticker not in selected)
    return selected[:limit]


def _ticker_forecast_chart(
    store: LocalObjectStore,
    profile: dict[str, Any],
    ticker: str,
    tokenizer: MarketEventTokenizer,
    model: torch.nn.Module,
) -> str:
    partition = _latest_partition_for_ticker(profile, ticker)
    if partition is None:
        return ""
    archive_path = _raw_archive_path(store, ticker, str(partition["partition"]))
    if archive_path is None:
        return ""
    candles = _candles_from_archive(archive_path)
    if not candles:
        return ""
    prediction = _forecast_path_from_partition(store, partition, tokenizer, model, candles[-1]["close"])
    return _candlestick_svg(ticker, str(partition["partition"]), candles, prediction)


def _latest_partition_for_ticker(profile: dict[str, Any], ticker: str) -> dict[str, Any] | None:
    partitions = [
        dict(partition)
        for partition in dict(profile.get("numpy_dataset", {})).get("partitions", [])
        if str(partition.get("ticker")) == ticker and int(partition.get("sequence_count", 0)) > 0
    ]
    if not partitions:
        return None
    return sorted(partitions, key=lambda item: str(item.get("partition", "")))[-1]


def _raw_archive_path(store: LocalObjectStore, ticker: str, partition: str) -> Path | None:
    raw_root = store.root / "stage=01_raw" / "source=binance_trades" / "data" / "spot"
    matches = sorted(raw_root.glob(f"*/trades/{ticker}/{ticker}-trades-{partition}.zip"))
    if matches:
        return matches[-1]
    symbol_roots = sorted(raw_root.glob(f"*/trades/{ticker}"))
    archives = sorted(path for root in symbol_roots for path in root.glob(f"{ticker}-trades-*.zip"))
    return archives[-1] if archives else None


def _candles_from_archive(path: Path) -> list[dict[str, float]]:
    trades: list[tuple[float, float]] = []
    for _trade_id, price, _qty, raw_time, _is_buyer_maker in iter_trade_fields_from_archive(path):
        if price <= 0.0:
            continue
        trades.append((_timestamp_seconds(raw_time), price))
        if len(trades) >= MAX_CANDLE_TRADES:
            break
    if len(trades) < 2:
        return []
    start = trades[0][0]
    end = trades[-1][0]
    interval = max(1.0, (end - start) / MAX_CANDLES)
    buckets: list[dict[str, float]] = []
    current_bucket = -1
    current: dict[str, float] | None = None
    for timestamp, price in trades:
        bucket = int((timestamp - start) // interval)
        if current is None or bucket != current_bucket:
            if current is not None:
                buckets.append(current)
            current_bucket = bucket
            current = {"open": price, "high": price, "low": price, "close": price}
        else:
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
    if current is not None:
        buckets.append(current)
    return buckets[:MAX_CANDLES]


def _timestamp_seconds(value: int) -> float:
    divisor = 1_000_000 if value >= 10**15 else 1_000
    return value / divisor


def _forecast_path_from_partition(
    store: LocalObjectStore,
    partition: dict[str, Any],
    tokenizer: MarketEventTokenizer,
    model: torch.nn.Module,
    start_price: float,
) -> list[float]:
    tokens = np.load(store.root / str(partition["tokens_path"]), mmap_mode="r")
    if int(tokens.shape[0]) < 2:
        return []
    context_length = min(int(tokens.shape[0]), int(getattr(model, "block_size", FORECAST_CONTEXT_TOKENS)), FORECAST_CONTEXT_TOKENS)
    context = torch.from_numpy(np.asarray(tokens[:context_length], dtype=np.int64)).unsqueeze(0)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=FORECAST_TOKENS, top_k=16)[0].detach().cpu().tolist()
    prices = [float(start_price)]
    price = float(start_price)
    for token in generated[context_length:]:
        relative_bps = tokenizer.relative_price_value(int(token))
        if relative_bps is None:
            continue
        price *= math.exp(float(relative_bps) / 10_000.0)
        prices.append(price)
    return prices


def _candlestick_svg(ticker: str, partition: str, candles: list[dict[str, float]], prediction: list[float]) -> str:
    width, height = 860, 300
    padding = 42
    candle_width = max(4.0, (width - 2 * padding) / max(len(candles) + max(len(prediction) - 1, 0), 1) * 0.55)
    values = [value for candle in candles for value in (candle["high"], candle["low"])] + prediction
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        low *= 0.999
        high *= 1.001

    def y(value: float) -> float:
        return height - padding - (height - 2 * padding) * (value - low) / (high - low)

    elements: list[str] = []
    for index, candle in enumerate(candles):
        x = padding + index * (width - 2 * padding) / max(len(candles) + max(len(prediction) - 1, 0), 1)
        color = "#74d680" if candle["close"] >= candle["open"] else "#ff7b86"
        y_open = y(candle["open"])
        y_close = y(candle["close"])
        y_high = y(candle["high"])
        y_low = y(candle["low"])
        elements.append(f'<line x1="{x:.2f}" y1="{y_high:.2f}" x2="{x:.2f}" y2="{y_low:.2f}" stroke="{color}" stroke-width="1.5"/>')
        body_y = min(y_open, y_close)
        body_h = max(abs(y_close - y_open), 1.0)
        elements.append(f'<rect x="{x - candle_width / 2:.2f}" y="{body_y:.2f}" width="{candle_width:.2f}" height="{body_h:.2f}" rx="1.5" fill="{color}"/>')
    if len(prediction) >= 2:
        start_x = padding + (len(candles) - 1) * (width - 2 * padding) / max(len(candles) + len(prediction) - 2, 1)
        step = (width - 2 * padding) / max(len(candles) + len(prediction) - 2, 1)
        points = " ".join(f"{start_x + index * step:.2f},{y(price):.2f}" for index, price in enumerate(prediction))
        elements.append(f'<polyline fill="none" stroke="#73a7ff" stroke-width="3" points="{points}"/>')
    axis = (
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#263451"/>'
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#263451"/>'
        f'<text x="{padding}" y="{padding - 10}" fill="#a7b0c3">{low:.4f} - {high:.4f}</text>'
    )
    return (
        f'<div class="card"><h3>{escape(ticker)} <span class="label">{escape(partition)}</span></h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(ticker)} candles and forecast">{axis}{"".join(elements)}</svg>'
        '<div class="legend">Green/red: real OHLC candles. Blue: generated model-implied price path.</div></div>'
    )


def _backtest_section(backtest: dict[str, Any] | None) -> str:
    if backtest is None:
        return "<section><h2>Backtest Metrics</h2><p class=\"warn\">No backtest report found yet.</p></section>"
    rows = [
        ("Loss", _format_float(backtest.get("backtest_loss"))),
        ("Perplexity", _format_float(backtest.get("backtest_perplexity"))),
        ("Top-1 accuracy", _format_percent(backtest.get("backtest_top1_accuracy"))),
        ("Top-5 accuracy", _format_percent(backtest.get("backtest_top5_accuracy"))),
        ("Scored tokens", _format_number(backtest.get("backtest_tokens", 0))),
        ("Distribution L1", _format_float(backtest.get("price_depth_distribution_l1"))),
    ]
    body = "".join(f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>" for name, value in rows)
    return f"<section><h2>Backtest Metrics</h2><table><tbody>{body}</tbody></table></section>"


def _stylized_fact_section(backtest: dict[str, Any] | None, eval_report: dict[str, Any] | None) -> str:
    source = backtest if backtest is not None else eval_report
    if source is None:
        return "<section><h2>Stylized Facts</h2><p class=\"warn\">No rollout-vs-real report found yet.</p></section>"
    real = dict(source.get("real_backtest") or source.get("real") or {})
    generated = dict(source.get("generated_rollout") or source.get("generated") or {})
    keys = sorted(set(real) | set(generated))
    body = "".join(
        f"<tr><td>{escape(key)}</td><td>{escape(_format_float(real.get(key)))}</td><td>{escape(_format_float(generated.get(key)))}</td></tr>"
        for key in keys
    )
    return f"<section><h2>Stylized Facts</h2><table><thead><tr><th>Metric</th><th>Real</th><th>Generated</th></tr></thead><tbody>{body}</tbody></table></section>"


def _config_section(profile: dict[str, Any], manifest: dict[str, Any] | None) -> str:
    metadata = dict(manifest.get("metadata", {})) if manifest else {}
    rows = [
        ("Vocab size", _format_number(profile.get("vocab_size", 0))),
        ("Block size", _format_number(profile.get("block_size", 0))),
        ("Sequence count", _format_number(profile.get("sequence_count", 0))),
        ("Training backend", str(metadata.get("training_backend", "unknown"))),
        ("Distributed strategy", str(metadata.get("distributed_strategy", "unknown"))),
        ("World size", _format_number(metadata.get("world_size", 0))),
        ("Attention backend", str(metadata.get("attention_backend", "unknown"))),
        ("Compile enabled", str(metadata.get("compile_enabled", "unknown"))),
    ]
    body = "".join(f"<tr><th>{escape(name)}</th><td>{escape(value)}</td></tr>" for name, value in rows)
    return f"<section><h2>Run Configuration</h2><table><tbody>{body}</tbody></table></section>"


def _line_chart(series: list[tuple[str, str, list[float | None]]], title: str) -> str:
    width, height = 760, 260
    padding = 42
    values = [value for _, _, items in series for value in items if value is not None and math.isfinite(value)]
    if not values:
        return f"<p class=\"warn\">No {escape(title)} data found.</p>"
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    longest = max((len(items) for _, _, items in series), default=1)

    def point(index: int, value: float) -> tuple[float, float]:
        x = padding + (width - 2 * padding) * index / max(longest - 1, 1)
        y = height - padding - (height - 2 * padding) * (value - minimum) / (maximum - minimum)
        return x, y

    polylines = []
    legend = []
    for label, color, items in series:
        points = [point(index, float(value)) for index, value in enumerate(items) if value is not None and math.isfinite(value)]
        if len(points) < 2:
            continue
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{encoded}"/>')
        legend.append(f'<span style="color:{color}">{escape(label)}</span>')
    axis = (
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#263451"/>'
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#263451"/>'
        f'<text x="{padding}" y="{padding - 10}" fill="#a7b0c3">{minimum:.3f} - {maximum:.3f}</text>'
    )
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">{axis}{"".join(polylines)}</svg><div class="legend">{" | ".join(legend)}</div>'


def _artifact_target(store: LocalObjectStore, path: str) -> Path:
    target = Path(path)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"artifact path must be relative and safe: {path}")
    return store.root / target


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def _format_percent(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{100.0 * number:.2f}%"


def _format_number(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{int(number):,}"
