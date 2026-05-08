import tempfile
import unittest
from pathlib import Path

from marketfm.config import DataMixtureConfig, MixtureSource
from marketfm.corpus import CorpusBuilder
from marketfm.ingest import FixtureIngestor
from marketfm.store import LocalObjectStore
from marketfm.train.sft import SFTExampleFormatter, SFTTrainer, SFTTrainConfig, ThesisOutput


def _prepared_store(tmp: str) -> LocalObjectStore:
    store = LocalObjectStore(Path(tmp))
    FixtureIngestor(store).ingest()
    CorpusBuilder(store).build(DataMixtureConfig("demo", [MixtureSource("qa", 1.0, "sft_instruction")]))
    return store


class SFTTrainerTests(unittest.TestCase):
    def test_sft_examples_format_with_evidence_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)
            examples = SFTExampleFormatter(store).load("corpus/demo/sft.jsonl")

            self.assertEqual(len(examples), 2)
            self.assertIn("Evidence:", examples[0].target)
            self.assertEqual(examples[0].evidence_ids, ["ev-acme-margin"])

    def test_thesis_output_schema_validates_required_fields(self) -> None:
        output = ThesisOutput(
            ticker="ACME",
            as_of_time="2023-02-15T16:30:00Z",
            investment_view="attractive",
            confidence=0.6,
            thesis="ACME has improving recurring revenue.",
            evidence_ids=["ev-acme-margin"],
        )

        self.assertEqual(output.to_dict()["investment_view"], "attractive")

    def test_sft_smoke_run_writes_manifest_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _prepared_store(tmp)

            result = SFTTrainer(store, SFTTrainConfig(run_id="sft-test")).train("corpus/demo/sft.jsonl")
            metrics = store.read_jsonl("runs/sft-test/metrics.jsonl")
            manifest = store.read_manifest(result.manifest_path)

            self.assertEqual(result.examples, 2)
            self.assertEqual(metrics[0]["stage"], "sft")
            self.assertEqual(manifest.metadata["stage"], "sft")
            self.assertEqual(manifest.metadata["examples"], "2")


if __name__ == "__main__":
    unittest.main()
