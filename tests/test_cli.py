import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from marketfm.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_parser_exposes_marketfm_program(self) -> None:
        parser = build_parser()

        self.assertEqual(parser.prog, "marketfm")

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
                    "silver/entities/sec.jsonl",
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
                    "silver/fundamentals/sec.jsonl",
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
                        "artifact_type": "silver",
                        "paths": ["silver/entities/sec.jsonl", "silver/fundamentals/sec.jsonl"],
                        "metadata": {},
                    },
                )
                from marketfm.data.ingest import IngestResult

                return IngestResult(
                    bronze_manifest_path="manifests/ingest/sec-companyfacts-bronze.json",
                    normalization_manifest_path="manifests/normalization/sec-companyfacts-normalized.json",
                    normalized_counts={"entities": 1},
                    quality_summary={"duplicate_records": 0, "quarantined_records": 0},
                )

        class FakePriceIngestor:
            def __init__(self, store, client, table_store=None):
                self.store = store

            def ingest(self, request):
                self.store.write_jsonl(
                    "silver/prices/stooq.jsonl",
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
                    "silver/fundamentals/sec.jsonl",
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
                        "artifact_type": "silver",
                        "paths": ["silver/prices/stooq.jsonl"],
                        "metadata": {},
                    },
                )
                from marketfm.data.ingest import IngestResult

                return IngestResult(
                    bronze_manifest_path="manifests/ingest/yahoo-daily-bronze.json",
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
                        "MarketFM test@example.com",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(tmp) / "silver/entities/sec.jsonl").exists())
            self.assertTrue((Path(tmp) / "silver/prices/stooq.jsonl").exists())

    def test_ingest_command_runs_from_config(self) -> None:
        class FakeSecIngestor:
            def __init__(self, store, client, table_store=None):
                self.store = store

            def ingest(self, request):
                self.store.write_jsonl(
                    "silver/entities/sec.jsonl",
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
                        "artifact_type": "silver",
                        "paths": ["silver/entities/sec.jsonl"],
                        "metadata": {},
                    },
                )
                from marketfm.data.ingest import IngestResult

                return IngestResult(
                    bronze_manifest_path="manifests/ingest/sec-companyfacts-bronze.json",
                    normalization_manifest_path="manifests/normalization/sec-companyfacts-normalized.json",
                    normalized_counts={"entities": 1},
                    quality_summary={"duplicate_records": 0, "quarantined_records": 0},
                )

        class FakePriceIngestor:
            def __init__(self, store, client, table_store=None):
                self.store = store

            def ingest(self, request):
                self.store.write_jsonl(
                    "silver/prices/yahoo.jsonl",
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
                        "artifact_type": "silver",
                        "paths": ["silver/prices/yahoo.jsonl"],
                        "metadata": {},
                    },
                )
                from marketfm.data.ingest import IngestResult

                return IngestResult(
                    bronze_manifest_path="manifests/ingest/yahoo-daily-bronze.json",
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
sec_user_agent = "MarketFM test@example.com"

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
            self.assertTrue((output_dir / "silver/entities/sec.jsonl").exists())
            self.assertTrue((output_dir / "silver/prices/yahoo.jsonl").exists())
            self.assertTrue((output_dir / "reports/data-readiness.json").exists())
            self.assertTrue((output_dir / "silver/enriched/company_snapshots.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
