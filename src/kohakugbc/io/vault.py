"""GBC graphs in a KohakuVault database, indexed by job key.

One SQLite file (``kohakuvault.KVault``) holds four tables, every key the job's key as UTF-8:

- ``graphs``: the graph's JSON text (as resolution wrote it, with its ``img_path``),
  zlib-compressed;
- ``stats``: the job's counts, JSON;
- ``failures``: a failed job's ``kind`` / ``node`` / ``message``, JSON (dropped when the key
  later succeeds);
- ``meta``: ``format`` (``FORMAT``) and ``codec`` (``zlib``).

A key holds at most one graph: writing it again replaces it. One process writes a file at a time
(each machine its own file); any number read it. ``GraphVault`` is the sink, ``read_vault`` and
``done_keys`` the readers.
"""

import json
import zlib
from collections.abc import Iterator
from pathlib import Path

from kohakuvault import KVault

FORMAT = "kohakugbc.graph-json/1"
TABLES = ("graphs", "stats", "failures", "meta")


def _open(path: Path, table: str, read_only: bool = False) -> KVault:
    return KVault(str(path), table=table, read_only=read_only)


class GraphVault:
    """The sink: ``write`` keeps each record under its key."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t = {name: _open(self.path, name) for name in TABLES}
        meta = self.t["meta"]
        stored = meta.get(b"format")
        if stored is not None and stored.decode() != FORMAT:
            raise ValueError(f"{self.path}: format {stored.decode()!r}, not {FORMAT!r}")
        meta[b"format"] = FORMAT.encode()
        meta[b"codec"] = b"zlib"
        self.counts = {"ok": 0, "failed": 0}

    def write(self, record) -> None:
        key = record.key.encode()
        if record.ok and record.outputs.get("graph") is not None:
            self.t["graphs"][key] = zlib.compress(record.outputs["graph"].encode(), 6)
            stats = record.outputs.get("stats") or {}
            self.t["stats"][key] = json.dumps(stats, ensure_ascii=False).encode()
            if key in self.t["failures"]:
                del self.t["failures"][key]
            self.counts["ok"] += 1
        else:
            failure = record.failure or {"kind": "empty", "message": "no described record"}
            self.t["failures"][key] = json.dumps(failure, ensure_ascii=False).encode()
            self.counts["failed"] += 1

    def close(self) -> None:
        for vault in self.t.values():
            vault.close()


def read_vault(path: str, keys: list[str] | None = None) -> Iterator[tuple[str, str]]:
    """``(key, graph JSON text)`` of every stored graph, or of ``keys`` (those stored)."""
    vault = _open(Path(path), "graphs", read_only=True)
    try:
        names = (k.encode() for k in keys) if keys is not None else vault.keys()
        for key in names:
            blob = vault.get(key)
            if blob is not None:
                yield key.decode(), zlib.decompress(blob).decode()
    finally:
        vault.close()


def done_keys(paths: list[str]) -> set[str]:
    """The keys holding a graph in any of ``paths`` (missing files hold none)."""
    done: set[str] = set()
    for path in paths:
        if not Path(path).exists():
            continue
        vault = _open(Path(path), "graphs", read_only=True)
        try:
            done.update(k.decode() for k in vault)
        finally:
            vault.close()
    return done
