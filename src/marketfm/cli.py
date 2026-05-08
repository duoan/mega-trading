"""Command line interface for MarketFM Forge."""

from __future__ import annotations

import argparse
from pathlib import Path

from marketfm.core.store import LocalObjectStore
from marketfm.data.enrich import DataEnricher
from marketfm.data.corpus import PublicCorpusBuilder
from marketfm.data.ingest import PriceIngestRequest, TickerIngestRequest
from marketfm.data.ingest_config import IngestPipelineConfig, IngestSourceConfig, load_ingest_config
from marketfm.data.labels import LabelConfig
from marketfm.data.lance_store import LanceTableStore
from marketfm.data.public.prices import StooqClient, StooqPriceIngestor, YahooChartClient, YahooPriceIngestor
from marketfm.data.public.sec import SecClient, SecCompanyFactsIngestor
from marketfm.data.quality import DataQualityChecker
from marketfm.data.samples import MultiStreamSampleBuilder
from marketfm.data.tokenize import ShardBuilder, StreamShardBuilder
from marketfm.train.fusion import FusionTrainConfig, TinyFusionTrainer
from marketfm.train.market_fusion import MarketFusionTrainConfig, MarketFusionTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marketfm",
        description="MarketFM Forge data and training infrastructure CLI.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the package version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")
    ingest = subparsers.add_parser("ingest", help="run config-driven data ingestion")
    ingest.add_argument("--config", required=True, help="path to ingest TOML config")
    ingest_public = subparsers.add_parser("ingest-public", help="ingest public SEC fundamentals and Yahoo prices")
    ingest_public.add_argument("--tickers", required=True, help="comma-separated ticker symbols")
    ingest_public.add_argument("--start", required=True, help="price start date YYYY-MM-DD")
    ingest_public.add_argument("--end", required=True, help="price end date YYYY-MM-DD")
    ingest_public.add_argument("--out", default=".marketfm/public", help="artifact output directory")
    ingest_public.add_argument("--sec-user-agent", required=True, help="SEC-compliant User-Agent, including contact email")
    train_fusion = subparsers.add_parser("train-fusion", help="run tiny fusion-model smoke training")
    train_fusion.add_argument("--data-dir", default=".marketfm/public", help="artifact root containing stream shards")
    train_fusion.add_argument("--mixture", default="public", help="mixture name to train from")
    train_fusion.add_argument("--run-id", default="fusion-smoke", help="training run id")
    train_fusion.add_argument("--steps", type=int, default=3, help="number of smoke training steps")
    train_market_fusion = subparsers.add_parser("train-market-fusion", help="run MarketFusion model training")
    train_market_fusion.add_argument("--data-dir", default=".marketfm/public", help="artifact root containing stream shards")
    train_market_fusion.add_argument("--mixture", default="public", help="mixture name to train from")
    train_market_fusion.add_argument("--run-id", default="market-fusion", help="training run id")
    train_market_fusion.add_argument("--steps", type=int, default=10, help="number of training steps")
    train_market_fusion.add_argument("--hidden-dim", type=int, default=32, help="fusion hidden dimension")
    train_market_fusion.add_argument("--batch-size", type=int, default=8, help="training batch size")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from marketfm import __version__

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
                IngestSourceConfig(name="sec_companyfacts", tickers=tuple(tickers)),
                IngestSourceConfig(name="yahoo_prices", tickers=tuple(tickers), start=args.start, end=args.end),
            ),
        )
        _run_ingest_config(config)
    elif args.command == "train-fusion":
        store = LocalObjectStore(Path(args.data_dir))
        shard_path = f"stage=05_shards/mixture={args.mixture}/samples.jsonl"
        TinyFusionTrainer(store, FusionTrainConfig(run_id=args.run_id, max_steps=args.steps)).train(shard_path)
        print(f"wrote fusion training artifacts to {args.data_dir}/runs/{args.run_id}")
    elif args.command == "train-market-fusion":
        store = LocalObjectStore(Path(args.data_dir))
        shard_path = f"stage=05_shards/mixture={args.mixture}/samples.jsonl"
        MarketFusionTrainer(
            store,
            MarketFusionTrainConfig(
                run_id=args.run_id,
                max_steps=args.steps,
                hidden_dim=args.hidden_dim,
                batch_size=args.batch_size,
            ),
        ).train(shard_path)
        print(f"wrote MarketFusion training artifacts to {args.data_dir}/runs/{args.run_id}")
    return 0


def _run_ingest_config(config: IngestPipelineConfig) -> None:
    store = LocalObjectStore(Path(config.output_dir))
    table_store = LanceTableStore(Path(config.output_dir) / "lancedb")
    normalization_manifests: list[str] = []
    for source in config.enabled_sources():
        if source.name == "sec_companyfacts":
            if not config.sec_user_agent:
                raise ValueError("sec_companyfacts requires ingest.sec_user_agent")
            result = SecCompanyFactsIngestor(store, SecClient(user_agent=config.sec_user_agent), table_store=table_store).ingest(
                TickerIngestRequest(tickers=source.tickers)
            )
        elif source.name == "yahoo_prices":
            result = YahooPriceIngestor(store, YahooChartClient(), table_store=table_store).ingest(
                PriceIngestRequest(tickers=source.tickers, start=str(source.start), end=str(source.end))
            )
        elif source.name == "stooq_prices":
            result = StooqPriceIngestor(store, StooqClient(), table_store=table_store).ingest(
                PriceIngestRequest(tickers=source.tickers, start=str(source.start), end=str(source.end))
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
        PublicCorpusBuilder(store).build(mixture_name=config.training_mixture_name)
        MultiStreamSampleBuilder(
            store,
            LabelConfig(
                input_window_observations=config.training_input_window_observations,
                horizon_observations=config.training_horizon_observations,
                return_threshold=config.training_return_threshold,
            ),
        ).build(mixture_name=config.training_mixture_name, run_id="configured-ingest")
        StreamShardBuilder(store).build(config.training_mixture_name)
        ShardBuilder(store, sequence_length=config.training_sequence_length).build(config.training_mixture_name, "cpt")
    print(f"wrote configured ingest artifacts to {config.output_dir}")
