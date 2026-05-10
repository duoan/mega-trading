import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_installer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "install_flash_attn.py"
    spec = importlib.util.spec_from_file_location("install_flash_attn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load install_flash_attn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FlashAttnInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.installer = _load_installer()

    def test_install_command_disables_build_isolation(self) -> None:
        command = self.installer._install_command("flash-attn")

        self.assertIn("flash-attn", command)
        self.assertIn("--no-build-isolation", command)

    def test_cpu_environment_skips_install(self) -> None:
        with patch.object(self.installer, "_flash_attn_installed", return_value=False):
            with patch.object(self.installer, "_cuda_runtime_available", return_value=False):
                with patch.object(self.installer.subprocess, "run") as run:
                    with patch.object(self.installer.sys, "argv", ["install_flash_attn.py"]):
                        self.assertEqual(self.installer.main(), 0)
        run.assert_not_called()

    def test_require_cuda_fails_when_cuda_is_missing(self) -> None:
        with patch.object(self.installer, "_flash_attn_installed", return_value=False):
            with patch.object(self.installer, "_cuda_runtime_available", return_value=False):
                with patch.object(self.installer.sys, "argv", ["install_flash_attn.py", "--require-cuda"]):
                    with self.assertRaisesRegex(RuntimeError, "CUDA not detected"):
                        self.installer.main()

    def test_dry_run_does_not_install(self) -> None:
        with patch.object(self.installer, "_flash_attn_installed", return_value=False):
            with patch.object(self.installer, "_cuda_runtime_available", return_value=True):
                with patch.object(self.installer.subprocess, "run") as run:
                    with patch.object(self.installer.sys, "argv", ["install_flash_attn.py", "--dry-run"]):
                        self.assertEqual(self.installer.main(), 0)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
