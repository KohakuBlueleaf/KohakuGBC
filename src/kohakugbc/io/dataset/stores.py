"""The KohakuVault stores of a shard: ``graphs.db`` (key -> ``GbcGraph.to_msgpack`` bytes,
typed msgpack in a zstd frame) and ``meta.db`` (key -> any msgpack-able dict, KohakuVault's
auto-pack).
"""

import os
import sqlite3

from kohakuvault import KVault

from kohakugbc._rs import GbcGraph

GRAPHS = "graphs"
META = "meta"


def encode_graph(graph: "GbcGraph | str | bytes", zstd: int | None = 3) -> bytes:
    """A graph (``GbcGraph``, its JSON text, or msgpack bytes) as stored."""
    if isinstance(graph, str):
        graph = GbcGraph.from_json(graph)
    elif isinstance(graph, bytes):
        graph = GbcGraph.from_msgpack(graph)
    return graph.to_msgpack(zstd=zstd)


def write_store(path: str | os.PathLike, table: str, items: list[tuple[str, object]]) -> None:
    """A finished store: ``items`` written, then the file switched from WAL to a rollback
    journal, so it is one self-contained file (no -wal / -shm; safe to copy, upload, or read
    from a network filesystem)."""
    vault = KVault(str(path), table=table)
    for at in range(0, len(items), 1024):
        vault.put_many(items[at : at + 1024])
    vault.close()
    con = sqlite3.connect(path)
    mode = con.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
    con.close()
    if mode != "delete":
        raise RuntimeError(f"{path}: journal mode {mode}, not delete")


class Store:
    """One finished KohakuVault file of a shard (table ``table``), opened read-only."""

    def __init__(self, path: str | os.PathLike, table: str):
        self.vault = KVault(str(path), table=table, read_only=True)

    def __getitem__(self, key: str):
        return self.vault[key]

    def close(self) -> None:
        self.vault.close()
