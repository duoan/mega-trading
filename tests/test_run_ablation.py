import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_run_ablation():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ablation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_ablation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_ablation = _load_run_ablation()

    def test_plan_commands_expands_data_and_model_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ablation.yaml"
            config_path.write_text(
                """
name: smoke
config_name: default
ingest_config: configs/ingest-demo.toml
data_dir: .mega-trading/ablation-smoke
base_overrides:
  - training.max_steps=1
  - training.batch_size=2
data_sizes:
  - name: tiny
    overrides:
      - build.max_tickers=1
  - name: small
    overrides:
      - build.max_tickers=2
model_sizes:
  - name: micro
    overrides:
      - model.hidden_dim=8
      - model.layers=1
      - model.attention_heads=1
  - name: mini
    overrides:
      - model.hidden_dim=16
      - model.layers=1
      - model.attention_heads=2
backtest:
  max_batches: 1
examples:
  max_sequences: 2
  max_tokens: 4
""".strip()
                + "\n",
                encoding="utf-8",
            )

            commands = self.run_ablation.plan_commands(
                config_path,
                mlflow_tracking_uri="http://127.0.0.1:5000",
            )

            prepare_commands = [command for command in commands if command[:3] == ["uv", "run", "python"]]
            train_commands = [command for command in commands if command[:4] == ["uv", "run", "mega-trading", "train"]]
            backtest_commands = [command for command in commands if command[:4] == ["uv", "run", "mega-trading", "backtest"]]
            examples_commands = [
                command for command in commands if command[:4] == ["uv", "run", "mega-trading", "backtest-examples"]
            ]
            report_commands = [command for command in commands if command[:4] == ["uv", "run", "mega-trading", "report"]]

            self.assertEqual(len(prepare_commands), 2)
            self.assertEqual(len(train_commands), 4)
            self.assertEqual(len(backtest_commands), 4)
            self.assertEqual(len(examples_commands), 4)
            self.assertEqual(len(report_commands), 4)
            self.assertIn("run.run_id=smoke__data_tiny__model_micro", train_commands[0])
            self.assertIn("training.mlflow_tracking_uri=http://127.0.0.1:5000", train_commands[0])
            self.assertIn("+ablation.name=smoke", train_commands[0])
            self.assertIn("+ablation.data_size=tiny", train_commands[0])
            self.assertIn("+ablation.model_size=micro", train_commands[0])
            self.assertIn("data.mixture=smoke-tiny", prepare_commands[0])
            self.assertIn("--max-batches", backtest_commands[0])
            self.assertIn("--max-sequences", examples_commands[0])


if __name__ == "__main__":
    unittest.main()
