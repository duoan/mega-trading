import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_mlflow_server():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ensure_mlflow_server.py"
    spec = importlib.util.spec_from_file_location("ensure_mlflow_server", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ensure_mlflow_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MlflowServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mlflow_server = _load_mlflow_server()

    def test_ensure_reuses_healthy_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            options = self.mlflow_server.EnsureOptions(
                host="127.0.0.1",
                port=5000,
                backend_store_uri="sqlite:///" + str(Path(tmp) / "mlflow.db"),
                default_artifact_root=Path(tmp) / "artifacts",
                pid_file=Path(tmp) / "mlflow.pid",
                log_file=Path(tmp) / "mlflow.log",
                timeout_seconds=1.0,
            )

            with patch.object(self.mlflow_server, "_server_ready", return_value=True):
                with patch.object(self.mlflow_server.subprocess, "Popen") as popen:
                    self.mlflow_server.ensure_server(options)

        popen.assert_not_called()

    def test_ensure_starts_server_when_not_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = self.mlflow_server.EnsureOptions(
                host="127.0.0.1",
                port=5001,
                backend_store_uri="sqlite:///" + str(root / "mlflow.db"),
                default_artifact_root=root / "artifacts",
                pid_file=root / "mlflow.pid",
                log_file=root / "mlflow.log",
                timeout_seconds=1.0,
            )

            process = type("Process", (), {"pid": 12345})()
            with patch.object(self.mlflow_server, "_server_ready", side_effect=[False, True]):
                with patch.object(self.mlflow_server.subprocess, "Popen", return_value=process) as popen:
                    self.mlflow_server.ensure_server(options)

            command = popen.call_args.args[0]
            self.assertEqual(command[:3], [self.mlflow_server.sys.executable, "-m", "mlflow"])
            self.assertIn("--backend-store-uri", command)
            self.assertIn(options.backend_store_uri, command)
            self.assertIn("--default-artifact-root", command)
            self.assertIn(str(options.default_artifact_root), command)
            self.assertEqual(options.pid_file.read_text(encoding="utf-8").strip(), "12345")

    def test_stop_server_sends_sigterm_for_pid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "mlflow.pid"
            pid_file.write_text("12345\n", encoding="utf-8")

            with patch.object(self.mlflow_server.os, "kill") as kill:
                self.mlflow_server.stop_server(pid_file)

            kill.assert_called_once_with(12345, self.mlflow_server.signal.SIGTERM)
            self.assertFalse(pid_file.exists())


if __name__ == "__main__":
    unittest.main()
