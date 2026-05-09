import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mega_trading.cli import build_parser, load_config, main


class CliTests(unittest.TestCase):
    def test_parser_exposes_mega_trading_program(self) -> None:
        parser = build_parser()

        self.assertEqual(parser.prog, "mega-trading")

    def test_help_smoke(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit) as caught:
            parser.parse_args(["--help"])

        self.assertEqual(caught.exception.code, 0)

    def test_version_command_prints_version(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["--version"])

        self.assertEqual(exit_code, 0)
        self.assertIn("0.1.0", output.getvalue())

    def test_ingest_public_command_writes_normalized_artifacts(self) -> None:
        class FakeSecIngestor:
            def __init__(self, store, client, table_store=None):
                self.store = store

            def ingest(self, request):
                self.store.write_jsonl(
                    "stage=02_normalized/family=entities/source=sec.jsonl",
                    [
                        {
                            "entity_id": "sec-1",
                            "ticker": request.tickers[0],
                            "company_name": "Apple Inc.",
                            "source_ids": ["sec:company_tickers:1"],
                        }
                    ],
                )
                self.store.write_json(
                    "manifests/normalization/sec-companyfacts-normalized.json",
                    {
                        "manifest_id": "sec-companyfacts-normalized",
                        "artifact_type": "normalized",
                        "paths": ["stage=02_normalized/family=entities/source=sec.jsonl"],
                        "metadata": {},
                    },
                )
                from mega_trading.data.ingest import IngestResult

                return IngestResult(
                    raw_manifest_path="manifests/ingest/sec-companyfacts-raw.json",
                    normalization_manifest_path="manifests/normalization/sec-companyfacts-normalized.json",
                    normalized_counts={"entities": 1},
                    quality_summary={"duplicate_records": 0, "quarantined_records": 0},
                )

        class FakeMarketDataIngestor:
            def __init__(self, store, client, table_store=None):
                self.store = store

            def ingest(self, request):
                self.store.write_jsonl(
                    "stage=02_normalized/family=market_data/source=yahoo.jsonl",
                    [
                        {
                            "market_data_id": "yahoo-AAPL-2023-01-03",
                            "ticker": request.tickers[0],
                            "date": request.start,
                            "adjusted_close": 105.0,
                            "provider": "yahoo",
                            "source_ids": ["yahoo-AAPL-2023-01-03"],
                        }
                    ],
                )
                self.store.write_json(
                    "manifests/normalization/yahoo-daily-normalized.json",
                    {
                        "manifest_id": "yahoo-daily-normalized",
                        "artifact_type": "normalized",
                        "paths": ["stage=02_normalized/family=market_data/source=yahoo.jsonl"],
                        "metadata": {},
                    },
                )
                from mega_trading.data.ingest import IngestResult

                return IngestResult(
                    raw_manifest_path="manifests/ingest/yahoo-daily-raw.json",
                    normalization_manifest_path="manifests/normalization/yahoo-daily-normalized.json",
                    normalized_counts={"market_data": 1},
                    quality_summary={"duplicate_records": 0, "quarantined_records": 0},
                )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("mega_trading.cli.SecFilingsIngestor", FakeSecIngestor), patch(
                "mega_trading.cli.YahooMarketDataIngestor", FakeMarketDataIngestor
            ):
                exit_code = main(
                    [
                        "ingest-public",
                        "--tickers",
                        "AAPL",
                        "--start",
                        "2023-01-01",
                        "--end",
                        "2023-01-31",
                        "--out",
                        tmp,
                        "--sec-user-agent",
                        "Mega-Trading test@example.com",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(tmp) / "stage=02_normalized/family=entities/source=sec.jsonl").exists())
            self.assertTrue((Path(tmp) / "stage=02_normalized/family=market_data/source=yahoo.jsonl").exists())

    def test_ingest_command_runs_config_without_building_token_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ingest-demo.toml"
            output_dir = Path(tmp) / "demo"
            config_path.write_text(
                f"""
[ingest]
output_dir = "{output_dir}"

[ingest.quality]
enabled = true
fail_on_error = true

[[ingest.sources]]
name = "fixture"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            exit_code = main(["ingest", "--config", str(config_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "reports/data-readiness.json").exists())
            self.assertFalse((output_dir / "stage=05_shards/mixture=demo/tokens.jsonl").exists())

    def test_config_applies_hydra_overrides(self) -> None:
        config = load_config(
            Path("configs"),
            "default",
            [
                "run.run_id=train-a",
                "training.max_steps=3",
                "model.hidden_dim=16",
                "build.block_size=16",
                "eval.rollouts=2",
            ],
        )

        self.assertEqual(config.run.run_id, "train-a")
        self.assertEqual(config.training.max_steps, 3)
        self.assertEqual(config.model.hidden_dim, 16)
        self.assertEqual(config.build.block_size, 16)
        self.assertEqual(config.eval.rollouts, 2)


if __name__ == "__main__":
    unittest.main()
