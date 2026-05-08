import tempfile
import unittest
from pathlib import Path

from marketfm.core.schemas import Manifest
from marketfm.core.store import ArtifactPaths, ArtifactNotFoundError, LocalObjectStore


class StoreTests(unittest.TestCase):
    def test_object_store_round_trip_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalObjectStore(Path(tmp))

            store.write_jsonl("stage=01_raw/source=fixture/entities.jsonl", [{"id": "one"}, {"id": "two"}])

            self.assertEqual(
                store.read_jsonl("stage=01_raw/source=fixture/entities.jsonl"),
                [{"id": "one"}, {"id": "two"}],
            )

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
                paths=["stage=01_raw/source=fixture/entities.jsonl"],
                metadata={"source": "fixture"},
            )

            store.write_manifest("manifests/ingest/manifest-1.json", manifest)
            loaded = store.read_manifest("manifests/ingest/manifest-1.json")

            self.assertEqual(loaded, manifest)
            self.assertEqual(loaded.content_hash(), manifest.content_hash())

    def test_artifact_paths_are_stable(self) -> None:
        paths = ArtifactPaths(run_id="demo")

        self.assertEqual(paths.raw("fixture", "entities"), "stage=01_raw/source=fixture/entities.jsonl")
        self.assertEqual(paths.normalized("entities", "fixture"), "stage=02_normalized/family=entities/source=fixture.jsonl")
        self.assertEqual(paths.corpus("demo", "cpt"), "stage=04_corpus/mixture=demo/cpt.jsonl")
        self.assertEqual(paths.shard("demo", "cpt"), "stage=05_shards/mixture=demo/cpt.jsonl")
        self.assertEqual(paths.manifest("ingest", "fixture"), "manifests/ingest/fixture.json")
        self.assertEqual(paths.run("metrics"), "runs/demo/metrics.jsonl")
        self.assertEqual(paths.run("checkpoint.json"), "runs/demo/checkpoint.json")


if __name__ == "__main__":
    unittest.main()
