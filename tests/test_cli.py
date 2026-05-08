import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from marketfm.cli import build_parser, main


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

    def test_ingest_public_command_writes_artifacts(self) -> None:
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
                self.store.write_jsonl(
                    "stage=02_normalized/family=fundamentals/source=sec.jsonl",
                    [
                        {
                            "fundamental_id": "sec-AAPL-Revenue-2022-12-31-2023-02-01",
                            "entity_id": "sec-1",
                            "ticker": request.tickers[0],
                            "concept": "Revenue",
                            "value": 100.0,
                            "unit": "USD",
                            "period_end": "2022-12-31",
                            "accepted_at": "2023-02-01T00:00:00Z",
                            "as_of_time": "2023-02-01T00:00:00Z",
                            "source_ids": ["sec:f1"],
                        }
                    ],
                )
                self.store.write_json(
                    "manifests/normalization/sec-companyfacts-normalized.json",
                    {
                        "manifest_id": "sec-companyfacts-normalized",
                        "artifact_type": "normalized",
                        "paths": [
                            "stage=02_normalized/family=entities/source=sec.jsonl",
                            "stage=02_normalized/family=fundamentals/source=sec.jsonl",
                        ],
                        "metadata": {},
                    },
                )
                from marketfm.data.ingest import IngestResult

                return IngestResult(
                    raw_manifest_path="manifests/ingest/sec-companyfacts-raw.json",
                    normalization_manifest_path="manifests/normalization/sec-companyfacts-normalized.json",
                    normalized_counts={"entities": 1},
                    quality_summary={"duplicate_records": 0, "quarantined_records": 0},
                )

        class FakePriceIngestor:
            def __init__(self, store, client, table_store=None):
                self.store = store

            def ingest(self, request):
                self.store.write_jsonl(
                    "stage=02_normalized/family=prices/source=stooq.jsonl",
                    [
                        {
                            "price_id": "yahoo-AAPL-2023-01-03",
                            "ticker": request.tickers[0],
                            "date": request.start,
                            "adjusted_close": 105.0,
                            "provider": "yahoo",
                            "source_ids": ["yahoo-AAPL-2023-01-03"],
                        }
                    ],
                )
                self.store.write_jsonl(
                    "stage=02_normalized/family=fundamentals/source=sec.jsonl",
                    [
                        {
                            "fundamental_id": "sec-AAPL-Revenue-2022-12-31-2023-02-01",
                            "entity_id": "sec-1",
                            "ticker": request.tickers[0],
                            "concept": "Revenue",
                            "value": 100.0,
                            "unit": "USD",
                            "period_end": "2022-12-31",
                            "accepted_at": "2023-02-01T00:00:00Z",
                            "as_of_time": "2023-02-01T00:00:00Z",
                            "source_ids": ["sec:f1"],
                        }
                    ],
                )
                self.store.write_json(
                    "manifests/normalization/yahoo-daily-normalized.json",
                    {
                        "manifest_id": "yahoo-daily-normalized",
                        "artifact_type": "normalized",
                        "paths": ["stage=02_normalized/family=prices/source=stooq.jsonl"],
                        "metadata": {},
                    },
                )
                from marketfm.data.ingest import IngestResult

                return IngestResult(
                    raw_manifest_path="manifests/ingest/yahoo-daily-raw.json",
                    normalization_manifest_path="manifests/normalization/yahoo-daily-normalized.json",
                    normalized_counts={"prices": 1},
                    quality_summary={"duplicate_records": 0, "quarantined_records": 0},
                )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("marketfm.cli.SecCompanyFactsIngestor", FakeSecIngestor), patch(
                "marketfm.cli.YahooPriceIngestor", FakePriceIngestor
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
            self.assertTrue((Path(tmp) / "stage=02_normalized/family=prices/source=stooq.jsonl").exists())

    def test_ingest_command_runs_from_config(self) -> None:
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
                from marketfm.data.ingest import IngestResult

                return IngestResult(
                    raw_manifest_path="manifests/ingest/sec-companyfacts-raw.json",
                    normalization_manifest_path="manifests/normalization/sec-companyfacts-normalized.json",
                    normalized_counts={"entities": 1},
                    quality_summary={"duplicate_records": 0, "quarantined_records": 0},
                )

        class FakePriceIngestor:
            def __init__(self, store, client, table_store=None):
                self.store = store

            def ingest(self, request):
                self.store.write_jsonl(
                    "stage=02_normalized/family=prices/source=yahoo.jsonl",
                    [
                        {
                            "price_id": "yahoo-AAPL-2023-01-03",
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
                        "paths": ["stage=02_normalized/family=prices/source=yahoo.jsonl"],
                        "metadata": {},
                    },
                )
                from marketfm.data.ingest import IngestResult

                return IngestResult(
                    raw_manifest_path="manifests/ingest/yahoo-daily-raw.json",
                    normalization_manifest_path="manifests/normalization/yahoo-daily-normalized.json",
                    normalized_counts={"prices": 1},
                    quality_summary={"duplicate_records": 0, "quarantined_records": 0},
                )

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ingest.toml"
            output_dir = Path(tmp) / "out"
            config_path.write_text(
                f"""
[ingest]
output_dir = "{output_dir}"
sec_user_agent = "Mega-Trading test@example.com"

[[ingest.sources]]
name = "sec_companyfacts"
tickers = ["AAPL"]

[[ingest.sources]]
name = "yahoo_prices"
tickers = ["AAPL"]
start = "2023-01-01"
end = "2023-01-31"
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with patch("marketfm.cli.SecCompanyFactsIngestor", FakeSecIngestor), patch(
                "marketfm.cli.YahooPriceIngestor", FakePriceIngestor
            ):
                exit_code = main(["ingest", "--config", str(config_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "stage=02_normalized/family=entities/source=sec.jsonl").exists())
            self.assertTrue((output_dir / "stage=02_normalized/family=prices/source=yahoo.jsonl").exists())
            self.assertTrue((output_dir / "reports/data-readiness.json").exists())
            self.assertTrue((output_dir / "stage=03_enriched/company_snapshots.jsonl").exists())
            self.assertTrue((output_dir / "stage=04_corpus/mixture=public/cpt.jsonl").exists())
            self.assertTrue((output_dir / "stage=04_corpus/mixture=public/sft.jsonl").exists())
            self.assertTrue((output_dir / "stage=04_corpus/mixture=public/samples.jsonl").exists())
            self.assertTrue((output_dir / "stage=05_shards/mixture=public/samples.jsonl").exists())
            self.assertTrue((output_dir / "stage=05_shards/mixture=public/cpt.jsonl").exists())

    def test_train_trading_foundation_model_command_writes_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stream_shard(root)

            exit_code = main(
                [
                    "train-trading-foundation-model",
                    "--data-dir",
                    str(root),
                    "--mixture",
                    "public",
                    "--run-id",
                    "tfm-cli",
                    "--steps",
                    "2",
                    "--hidden-dim",
                    "8",
                    "--batch-size",
                    "2",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "runs/tfm-cli/metrics.jsonl").exists())
            self.assertTrue((root / "runs/tfm-cli/checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()


def _write_stream_shard(root: Path) -> None:
    (root / "stage=05_shards/mixture=public").mkdir(parents=True)
    (root / "stage=05_shards/mixture=public/samples.jsonl").write_text(
        '{"sample_id":"sample-a","ticker":"AAPL","as_of_time":"2024-01-02T00:00:00Z",'
        '"price_returns":[0.0,0.01],"price_levels":[0.0,0.01],"fundamental_values":[100.0],'
        '"evidence_token_ids":[],"return_label":"outperform","risk_label":"low"}\n'
        '{"sample_id":"sample-b","ticker":"MSFT","as_of_time":"2024-01-02T00:00:00Z",'
        '"price_returns":[0.0,-0.01],"price_levels":[0.0,-0.01],"fundamental_values":[50.0],'
        '"evidence_token_ids":[],"return_label":"underperform","risk_label":"high"}\n',
        encoding="utf-8",
    )
