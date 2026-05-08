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
                self.store.write_jsonl("silver/entities/sec.jsonl", [{"ticker": request.tickers[0]}])

        class FakePriceIngestor:
            def __init__(self, store, client, table_store=None):
                self.store = store

            def ingest(self, request):
                self.store.write_jsonl(
                    "silver/prices/stooq.jsonl",
                    [{"ticker": request.tickers[0], "date": request.start, "end": request.end}],
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


if __name__ == "__main__":
    unittest.main()
