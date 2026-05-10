import tempfile
import unittest
from pathlib import Path

from mega_trading.core.schemas import Manifest
from mega_trading.core.store import ArtifactPaths, ArtifactNotFoundError, LocalObjectStore


class StoreTests(unittest.TestCase):
    def test_object_store_round_trip_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))

            store.write_json("artifacts/example.json", {"id": "one"})

            self.assertEqual(store.read_json("artifacts/example.json"), {"id": "one"})

    def test_missing_artifact_fails_with_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))

            with self.assertRaises(ArtifactNotFoundError):
                store.read_json("missing.json")

    def test_manifest_write_read_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))
            manifest = Manifest(
                manifest_id="manifest-1",
                artifact_type="raw",
                paths=["raw/source=fixture/order_flow.jsonl"],
                metadata={"source": "fixture"},
            )

            store.write_manifest("manifests/ingest/manifest-1.json", manifest)
            loaded = store.read_manifest("manifests/ingest/manifest-1.json")

            self.assertEqual(loaded, manifest)
            self.assertEqual(loaded.content_hash(), manifest.content_hash())

    def test_artifact_paths_are_stable(self) -> None:
        paths = ArtifactPaths(run_id="demo")

        self.assertEqual(paths.eval("eval-1", "report"), "evals/eval-1/report.json")
        self.assertEqual(paths.manifest("ingest", "fixture"), "manifests/ingest/fixture.json")
        self.assertEqual(paths.run("metrics"), "runs/demo/metrics.json")
        self.assertEqual(paths.run("checkpoint.json"), "runs/demo/checkpoint.json")


if __name__ == "__main__":
    unittest.main()
