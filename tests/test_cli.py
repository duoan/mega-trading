import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

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

    def test_ingest_command_runs_config_without_building_token_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ingest-demo.toml"
            output_dir = Path(tmp) / "demo"
            config_path.write_text(
                f"""
[ingest]
output_dir = "{output_dir}"

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
