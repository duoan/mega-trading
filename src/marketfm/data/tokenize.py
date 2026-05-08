"""Deterministic tokenization and shard building."""

from __future__ import annotations

import re
from dataclasses import dataclass

from marketfm.core.hashing import stable_hash
from marketfm.core.schemas import Manifest
from marketfm.core.store import ArtifactPaths, LocalObjectStore


class MalformedCorpusError(ValueError):
    """Raised when a corpus row cannot be tokenized into a shard."""


@dataclass(frozen=True)
class SimpleTokenizer:
    """Small deterministic tokenizer for the fixture demo.

    This intentionally keeps the interface tiny so a Hugging Face tokenizer can
    replace it later without changing shard manifests.
    """

    vocab: dict[str, int]
    name: str = "simple-v1"

    @classmethod
    def fit(cls, texts: list[str]) -> "SimpleTokenizer":
        tokens: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for token in _tokens(text):
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
        vocab = {"<pad>": 0, "<unk>": 1}
        vocab.update({token: index for index, token in enumerate(tokens, start=2)})
        return cls(vocab=vocab)

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in _tokens(text)]

    def decode(self, token_ids: list[int]) -> str:
        inverse = {token_id: token for token, token_id in self.vocab.items()}
        decoded = [inverse.get(token_id, "<unk>") for token_id in token_ids if token_id != 0]
        text = " ".join(decoded)
        return re.sub(r"\s+([.,;:!?])", r"\1", text)

    def vocab_hash(self) -> str:
        return stable_hash(self.vocab)


@dataclass(frozen=True)
class ShardBuildResult:
    shard_path: str
    manifest_path: str
    num_sequences: int
    num_tokens: int
    packing_efficiency: float


class ShardBuilder:
    def __init__(self, store: LocalObjectStore, sequence_length: int) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.store = store
        self.sequence_length = sequence_length
        self.paths = ArtifactPaths()

    def build(self, mixture_name: str, corpus_name: str) -> ShardBuildResult:
        corpus_path = self.paths.corpus(mixture_name, corpus_name)
        rows = self.store.read_jsonl(corpus_path)
        texts = [self._required_text(row) for row in rows]
        tokenizer = SimpleTokenizer.fit(texts)

        packed_rows = self._pack(rows, tokenizer)
        shard_path = f"shards/{mixture_name}/{corpus_name}.jsonl"
        self.store.write_jsonl(shard_path, packed_rows)

        num_tokens = sum(len(row["token_ids"]) for row in packed_rows)
        capacity = max(1, len(packed_rows) * self.sequence_length)
        packing_efficiency = num_tokens / capacity
        manifest = Manifest(
            manifest_id=f"{mixture_name}-{corpus_name}-shards",
            artifact_type="shards",
            paths=[shard_path],
            metadata={
                "corpus_path": corpus_path,
                "tokenizer": tokenizer.name,
                "tokenizer_hash": tokenizer.vocab_hash(),
                "sequence_length": str(self.sequence_length),
                "num_sequences": str(len(packed_rows)),
                "num_tokens": str(num_tokens),
                "packing_efficiency": f"{packing_efficiency:.6f}",
            },
        )
        manifest_path = self.paths.manifest("shards", f"{mixture_name}-{corpus_name}-shards")
        self.store.write_manifest(manifest_path, manifest)

        return ShardBuildResult(
            shard_path=shard_path,
            manifest_path=manifest_path,
            num_sequences=len(packed_rows),
            num_tokens=num_tokens,
            packing_efficiency=packing_efficiency,
        )

    def _pack(self, rows: list[dict[str, object]], tokenizer: SimpleTokenizer) -> list[dict[str, object]]:
        packed: list[dict[str, object]] = []
        current_tokens: list[int] = []
        current_sources: list[str] = []

        for row in rows:
            corpus_id = self._required_string(row, "corpus_id")
            token_ids = tokenizer.encode(self._required_text(row))
            if not token_ids:
                continue
            while token_ids:
                remaining = self.sequence_length - len(current_tokens)
                current_tokens.extend(token_ids[:remaining])
                if corpus_id not in current_sources:
                    current_sources.append(corpus_id)
                token_ids = token_ids[remaining:]
                if len(current_tokens) == self.sequence_length:
                    packed.append(_shard_row(len(packed), current_tokens, current_sources))
                    current_tokens = []
                    current_sources = []

        if current_tokens:
            packed.append(_shard_row(len(packed), current_tokens, current_sources))
        return packed

    def _required_text(self, row: dict[str, object]) -> str:
        return self._required_string(row, "text")

    def _required_string(self, row: dict[str, object], field: str) -> str:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise MalformedCorpusError(f"corpus row missing {field}")
        return value


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())


def _shard_row(index: int, token_ids: list[int], source_corpus_ids: list[str]) -> dict[str, object]:
    return {
        "sequence_id": f"seq-{index:06d}",
        "token_ids": list(token_ids),
        "source_corpus_ids": list(source_corpus_ids),
        "num_tokens": len(token_ids),
    }
