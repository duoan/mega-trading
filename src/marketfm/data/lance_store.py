"""LanceDB table store for normalized data and retrieval-ready artifacts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("RUST_LOG", "error")

import lancedb


@dataclass(frozen=True)
class LanceTables:
    """Stable table names for data-plane artifacts."""

    def silver(self, family: str, source: str) -> str:
        return _table_name("silver", family, source)

    def corpus(self, mixture: str, name: str) -> str:
        return _table_name("corpus", mixture, name)


class LanceTableStore:
    """Small wrapper around LanceDB so callers do not depend on raw APIs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.root)

    def write_table(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if name in self.table_names():
            self.db.drop_table(name)
        self.db.create_table(name, data=rows, mode="create")

    def append_table(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        if name in self.table_names():
            self.db.open_table(name).add(rows)
        else:
            self.write_table(name, rows)

    def read_rows(self, name: str) -> list[dict[str, Any]]:
        return self.db.open_table(name).to_arrow().to_pylist()

    def table_names(self) -> list[str]:
        result = self.db.list_tables()
        if hasattr(result, "tables"):
            return list(result.tables)
        if hasattr(result, "table_names"):
            return list(result.table_names)
        return list(result)


def _table_name(*parts: str) -> str:
    return "_".join(re.sub(r"[^a-zA-Z0-9]+", "_", part).strip("_").lower() for part in parts if part)
