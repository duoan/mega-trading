import importlib.util
import sys
import unittest
from pathlib import Path


def _load_run_pipeline():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_pipeline = _load_run_pipeline()

    def test_mlflow_tracking_uri_override_is_training_override(self) -> None:
        self.assertEqual(
            self.run_pipeline._mlflow_overrides("http://127.0.0.1:5000"),
            ["training.mlflow_tracking_uri=http://127.0.0.1:5000"],
        )

    def test_empty_mlflow_tracking_uri_adds_no_override(self) -> None:
        self.assertEqual(self.run_pipeline._mlflow_overrides(None), [])
        self.assertEqual(self.run_pipeline._mlflow_overrides(""), [])


if __name__ == "__main__":
    unittest.main()
