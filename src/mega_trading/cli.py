"""Command line interface for Mega-Trading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from mega_trading.core.store import LocalObjectStore
from mega_trading.data.enrich import DataEnricher
from mega_trading.data.ingest import MarketDataIngestRequest, TickerIngestRequest
from mega_trading.data.ingest_config import IngestPipelineConfig, IngestSourceConfig, load_ingest_config
from mega_trading.data.labels import LabelConfig
from mega_trading.data.lance_store import LanceTableStore
from mega_trading.data.public.market import StooqClient, StooqMarketDataIngestor, YahooChartClient, YahooMarketDataIngestor
from mega_trading.data.public.sec import SecClient, SecFilingsIngestor
from mega_trading.data.quality import DataQualityChecker
from mega_trading.data.samples import MultiStreamSampleBuilder
from mega_trading.data.shards import StreamShardBuilder
from mega_trading.eval.backtest import BacktestConfig, run_backtest
from mega_trading.eval.delayed_labels import materialize_delayed_labels
from mega_trading.eval.replay_buffer import ReplayBufferConfig
from mega_trading.eval.replay import run_replay_inference
from mega_trading.serving import FeatureRequest, ModelServer
from mega_trading.train.config import TradingFoundationTrainConfig
from mega_trading.train.online import OnlineAdaptationConfig, run_online_adapter_update
from mega_trading.train.trainer import TradingFoundationTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mega-trading",
        description="Mega-Trading data, model, and training CLI.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the package version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")
    ingest = subparsers.add_parser("ingest", help="run config-driven data ingestion")
    ingest.add_argument("--config", required=True, help="path to ingest TOML config")
    ingest_public = subparsers.add_parser("ingest-public", help="ingest public SEC filings and Yahoo market data")
    ingest_public.add_argument("--tickers", required=True, help="comma-separated ticker symbols")
    ingest_public.add_argument("--start", required=True, help="market data start date YYYY-MM-DD")
    ingest_public.add_argument("--end", required=True, help="market data end date YYYY-MM-DD")
    ingest_public.add_argument("--out", default=".mega-trading/public", help="artifact output directory")
    ingest_public.add_argument("--sec-user-agent", required=True, help="SEC-compliant User-Agent, including contact email")
    train = subparsers.add_parser("train", help="run TradingFoundationModel training from Hydra config")
    train.add_argument(
        "--config-dir",
        default="configs/train",
        help="Hydra config directory for training ablations",
    )
    train.add_argument("--config-name", default="default", help="Hydra config name")
    train.add_argument("overrides", nargs="*", help="Hydra overrides such as training.max_steps=100 model.hidden_dim=64")
    ablate = subparsers.add_parser("ablate", help="run Hydra-defined training ablations")
    ablate.add_argument("--config-dir", default="configs/ablation", help="Hydra config directory for ablation plans")
    ablate.add_argument("--config-name", default="public", help="Hydra ablation config name")
    ablate.add_argument("overrides", nargs="*", help="Hydra overrides for the ablation plan")
    replay = subparsers.add_parser("replay", help="run historical replay inference and write prediction logs")
    replay.add_argument("--data-dir", default=".mega-trading/public", help="artifact data directory")
    replay.add_argument("--mixture", default="public", help="sample/shard mixture name")
    replay.add_argument("--model-version-id", required=True, help="registered model version id")
    replay.add_argument("--replay-id", default="replay", help="prediction replay run id")
    replay.add_argument("--max-predictions", type=int, default=None, help="optional cap for smoke replays")
    replay.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="inference device")
    labels = subparsers.add_parser("materialize-labels", help="materialize delayed labels for prediction logs")
    labels.add_argument("--data-dir", default=".mega-trading/public", help="artifact data directory")
    labels.add_argument("--mixture", default="public", help="sample/shard mixture name")
    labels.add_argument("--prediction-path", required=True, help="prediction log artifact path")
    labels.add_argument("--label-run-id", default="labels", help="delayed label run id")
    backtest = subparsers.add_parser("backtest", help="run deterministic backtest over labeled predictions")
    backtest.add_argument("--data-dir", default=".mega-trading/public", help="artifact data directory")
    backtest.add_argument("--labeled-prediction-path", required=True, help="labeled prediction artifact path")
    backtest.add_argument("--backtest-id", default="backtest", help="backtest run id")
    backtest.add_argument("--fee-bps", type=float, default=1.0, help="round-trip fee in basis points")
    backtest.add_argument("--slippage-bps", type=float, default=1.0, help="slippage in basis points")
    backtest.add_argument("--confidence-threshold", type=float, default=0.0, help="minimum confidence for a trade")
    online = subparsers.add_parser("online-update", help="run online adapter/head update interface")
    online.add_argument("--data-dir", default=".mega-trading/public", help="artifact data directory")
    online.add_argument("--labeled-prediction-path", required=True, help="labeled prediction artifact path")
    online.add_argument("--base-model-version", required=True, help="base model version to adapt")
    online.add_argument("--update-id", default="online-update", help="online update id")
    online.add_argument("--max-samples", type=int, default=128, help="replay buffer sample budget")
    serve = subparsers.add_parser("serve-smoke", help="run serving-compatible local inference smoke test")
    serve.add_argument("--data-dir", default=".mega-trading/public", help="artifact data directory")
    serve.add_argument("--model-version-id", required=True, help="registered model version id")
    serve.add_argument(
        "--shard-path",
        default="stage=05_shards/mixture=public/samples.jsonl",
        help="feature shard path to read one request from",
    )
    serve.add_argument("--sample-id", default=None, help="optional sample id to serve")
    serve.add_argument("--out", default="replay/serve-smoke/response.json", help="output response artifact path")
    serve.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="inference device")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from mega_trading import __version__

        print(__version__)
    elif args.command == "ingest":
        config = load_ingest_config(Path(args.config))
        _run_ingest_config(config)
    elif args.command == "ingest-public":
        tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        config = IngestPipelineConfig(
            output_dir=args.out,
            sec_user_agent=args.sec_user_agent,
            sources=(
                IngestSourceConfig(name="sec_filings", tickers=tuple(tickers)),
                IngestSourceConfig(name="yahoo_market_data", tickers=tuple(tickers), start=args.start, end=args.end),
            ),
        )
        _run_ingest_config(config)
    elif args.command == "train":
        config = load_train_config(Path(args.config_dir), args.config_name, list(args.overrides))
        result = _run_train_config(config)
        print(f"wrote TradingFoundationModel training artifacts to {config.data.data_dir}/runs/{config.run.run_id}")
        print(f"manifest: {result.manifest_path}")
    elif args.command == "ablate":
        config = load_ablation_config(Path(args.config_dir), args.config_name, list(args.overrides))
        report_path = _run_ablation_config(config)
        print(f"wrote ablation summary to {report_path}")
    elif args.command == "replay":
        store = LocalObjectStore(Path(args.data_dir))
        result = run_replay_inference(
            store,
            replay_id=args.replay_id,
            model_version_id=args.model_version_id,
            shard_path=f"stage=05_shards/mixture={args.mixture}/samples.jsonl",
            max_predictions=args.max_predictions,
            device=args.device,
        )
        print(f"wrote replay predictions to {args.data_dir}/{result.prediction_path}")
        print(f"manifest: {result.manifest_path}")
    elif args.command == "materialize-labels":
        store = LocalObjectStore(Path(args.data_dir))
        result = materialize_delayed_labels(
            store,
            label_run_id=args.label_run_id,
            prediction_path=args.prediction_path,
            shard_path=f"stage=05_shards/mixture={args.mixture}/samples.jsonl",
        )
        print(f"wrote delayed labels to {args.data_dir}/{result.label_path}")
        print(f"labeled predictions: {result.labeled_prediction_path}")
        print(f"manifest: {result.manifest_path}")
    elif args.command == "backtest":
        store = LocalObjectStore(Path(args.data_dir))
        result = run_backtest(
            store,
            backtest_id=args.backtest_id,
            labeled_prediction_path=args.labeled_prediction_path,
            config=BacktestConfig(
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                confidence_threshold=args.confidence_threshold,
            ),
        )
        print(f"wrote backtest report to {args.data_dir}/{result.report_path}")
        print(f"manifest: {result.manifest_path}")
    elif args.command == "online-update":
        store = LocalObjectStore(Path(args.data_dir))
        result = run_online_adapter_update(
            store,
            update_id=args.update_id,
            labeled_prediction_path=args.labeled_prediction_path,
            base_model_version=args.base_model_version,
            config=OnlineAdaptationConfig(replay_buffer=ReplayBufferConfig(max_samples=args.max_samples)),
        )
        print(f"wrote online adapter update to {args.data_dir}/{result.adapter_path}")
        print(f"manifest: {result.manifest_path}")
    elif args.command == "serve-smoke":
        store = LocalObjectStore(Path(args.data_dir))
        row = _first_matching_row(store, args.shard_path, args.sample_id)
        response = ModelServer(store, args.model_version_id, device=args.device).predict(FeatureRequest.from_shard_row(row))
        store.write_json(args.out, response.to_dict())
        print(f"wrote serving smoke response to {args.data_dir}/{args.out}")
    return 0


def load_train_config(config_dir: Path, config_name: str, overrides: list[str]) -> DictConfig:
    return _load_hydra_config(config_dir, config_name, overrides)


def load_ablation_config(config_dir: Path, config_name: str, overrides: list[str]) -> DictConfig:
    return _load_hydra_config(config_dir, config_name, overrides)


def _load_hydra_config(config_dir: Path, config_name: str, overrides: list[str]) -> DictConfig:
    config_root = config_dir if config_dir.is_absolute() else Path.cwd() / config_dir
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        config = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(config)
    return config


def _run_train_config(config: DictConfig):
    store = LocalObjectStore(Path(str(config.data.data_dir)))
    shard_path = f"stage=05_shards/mixture={config.data.mixture}/samples.jsonl"
    train_config = TradingFoundationTrainConfig(
        run_id=str(config.run.run_id),
        max_steps=int(config.training.max_steps),
        hidden_dim=int(config.model.hidden_dim),
        attention_heads=int(config.model.attention_heads),
        batch_size=int(config.training.batch_size),
        learning_rate=float(config.training.learning_rate),
        validation_fraction=float(config.training.validation_fraction),
        eval_interval=int(config.training.eval_interval),
        market_window_size=_optional_int(config.model.market_window_size),
        news_size=_optional_int(config.model.news_size),
        sec_filing_size=_optional_int(config.model.sec_filing_size),
        earnings_size=_optional_int(config.model.earnings_size),
        macro_size=_optional_int(config.model.macro_size),
        use_market_data=bool(config.model.use_market_data),
        use_news=bool(config.model.use_news),
        use_sec_filings=bool(config.model.use_sec_filings),
        use_earnings=bool(config.model.use_earnings),
        use_macro=bool(config.model.use_macro),
        seed=int(config.training.seed),
        device=str(config.training.device),
        precision=str(config.training.precision),
    )
    return TradingFoundationTrainer(store, train_config).train(shard_path)


def _run_ablation_config(config: DictConfig) -> str:
    base_overrides = _string_list(config.get("base_overrides"))
    summaries: list[dict[str, Any]] = []
    for run in config.runs:
        run_name = str(run.name)
        overrides = list(base_overrides) + _string_list(run.get("overrides"))
        if not any(override.startswith("run.run_id=") for override in overrides):
            overrides.append(f"run.run_id={run_name}")
        train_config = load_train_config(Path(str(config.train_config_dir)), str(config.train_config_name), overrides)
        result = _run_train_config(train_config)
        summaries.append(_ablation_summary(run_name, train_config, result.manifest_path))

    report = {
        "ablation_id": str(config.get("ablation_id", "default")),
        "train_config": str(config.train_config_name),
        "runs": summaries,
    }
    report_path = Path(str(config.report_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return str(report_path)


def _ablation_summary(run_name: str, config: DictConfig, manifest_path: str) -> dict[str, Any]:
    store = LocalObjectStore(Path(str(config.data.data_dir)))
    run_id = str(config.run.run_id)
    metrics_path = f"runs/{run_id}/metrics.jsonl"
    shard_path = f"stage=05_shards/mixture={config.data.mixture}/samples.jsonl"
    metrics = store.read_jsonl(metrics_path)
    samples = store.read_jsonl(shard_path)
    final_metrics = dict(metrics[-1]) if metrics else {}
    manifest = store.read_manifest(manifest_path)
    return {
        "name": run_name,
        "run_id": run_id,
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
        "checkpoint_path": f"runs/{run_id}/checkpoint.pt",
        "config_hash": manifest.metadata["config_hash"],
        "sample_count": len(samples),
        "return_label_distribution": _distribution(samples, "return_label"),
        "risk_label_distribution": _distribution(samples, "risk_label"),
        "modalities": {
            "market_data": bool(config.model.use_market_data),
            "news": bool(config.model.use_news),
            "sec_filings": bool(config.model.use_sec_filings),
            "earnings": bool(config.model.use_earnings),
            "macro": bool(config.model.use_macro),
        },
        "training": {
            "max_steps": int(config.training.max_steps),
            "batch_size": int(config.training.batch_size),
            "learning_rate": float(config.training.learning_rate),
            "validation_fraction": float(config.training.validation_fraction),
            "eval_interval": int(config.training.eval_interval),
            "device": str(config.training.device),
            "precision": str(config.training.precision),
        },
        "model": {
            "hidden_dim": int(config.model.hidden_dim),
            "attention_heads": int(config.model.attention_heads),
        },
        "final_metrics": final_metrics,
    }


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    return [str(item) for item in value]


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(field, ""))
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _first_matching_row(store: LocalObjectStore, path: str, sample_id: str | None) -> dict[str, Any]:
    for row in store.iter_jsonl(path):
        if sample_id is None or str(row.get("sample_id")) == sample_id:
            return row
    raise ValueError(f"sample_id not found in {path}: {sample_id}")


def _run_ingest_config(config: IngestPipelineConfig) -> None:
    store = LocalObjectStore(Path(config.output_dir))
    table_store = LanceTableStore(Path(config.output_dir) / "lancedb")
    normalization_manifests: list[str] = []
    for source in config.enabled_sources():
        if source.name == "sec_filings":
            if not config.sec_user_agent:
                raise ValueError("sec_filings requires ingest.sec_user_agent")
            result = SecFilingsIngestor(store, SecClient(user_agent=config.sec_user_agent), table_store=table_store).ingest(
                TickerIngestRequest(tickers=source.tickers)
            )
        elif source.name == "yahoo_market_data":
            result = YahooMarketDataIngestor(store, YahooChartClient(), table_store=table_store).ingest(
                MarketDataIngestRequest(tickers=source.tickers, start=str(source.start), end=str(source.end))
            )
        elif source.name == "stooq_market_data":
            result = StooqMarketDataIngestor(store, StooqClient(), table_store=table_store).ingest(
                MarketDataIngestRequest(tickers=source.tickers, start=str(source.start), end=str(source.end))
            )
        else:
            raise ValueError(f"unsupported ingest source: {source.name}")
        normalization_manifests.append(result.normalization_manifest_path)
    quality_passed = True
    if config.quality_enabled:
        quality_result = DataQualityChecker(store).run(
            normalization_manifests,
            run_id="configured-ingest",
            fail_on_error=config.quality_fail_on_error,
        )
        quality_passed = quality_result.passed
    if config.enrichment_enabled and quality_passed:
        DataEnricher(store).run(run_id="configured-ingest")
    if config.training_data_enabled and quality_passed:
        MultiStreamSampleBuilder(
            store,
            LabelConfig(
                input_window_observations=config.training_input_window_observations,
                horizon_observations=config.training_horizon_observations,
                return_threshold=config.training_return_threshold,
            ),
            num_workers=config.training_workers,
        ).build(mixture_name=config.training_mixture_name, run_id="configured-ingest")
        StreamShardBuilder(store, num_workers=config.training_workers).build(config.training_mixture_name)
    print(f"wrote configured ingest artifacts to {config.output_dir}")
