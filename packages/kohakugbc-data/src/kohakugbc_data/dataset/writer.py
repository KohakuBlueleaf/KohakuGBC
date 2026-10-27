"""Writing a GBC dataset (layout in ``kohakugbc_data.dataset``)."""

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kohakugbc_data.dataset.images import EXTENSIONS, ImageShardWriter
from kohakugbc_data.dataset.stores import GRAPHS, META, encode_graph, write_store

FORMAT = "kohakugbc-dataset"
VERSION = 1


def shard_name(i: int) -> str:
    return f"shard_{i:05d}"


def write_info(root: Path, shards: list[dict], header: dict, zstd: int | None) -> None:
    info = {
        "format": FORMAT,
        "version": VERSION,
        "samples": sum(s["samples"] for s in shards),
        "shards": shards,
        "images": "kohakuclip-zip",
        "graphs": {"codec": "gbc-msgpack", "zstd": zstd, "table": GRAPHS},
        "meta": {"codec": "kohakuvault-autopack", "table": META},
        "header": header,
    }
    tmp = root / "dataset.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1, ensure_ascii=False)
    os.replace(tmp, root / "dataset.json")


class DatasetWriter:
    """``add`` samples, then ``close``. A shard closes once its image and graph bytes reach
    ``shard_bytes`` or its samples ``per_shard`` (either may be None), whichever comes first.

    ``add(key, image, w, h, codec, graph, meta=None, **row)``: the encoded image with its stored
    size and codec, the graph (``GbcGraph``, JSON text or msgpack), the sample's metadata dict
    and ``row`` keywords as extra manifest columns. Keys are unique (checked within one
    writer). ``header``: dataset-level facts, in ``dataset.json`` and every image index.
    Opening an existing dataset appends shards after its last; ``dataset.json`` is rewritten
    as each shard completes, so it lists only whole shards.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        shard_bytes: int | None = 4 << 30,
        per_shard: int | None = None,
        header: dict | None = None,
        zstd: int | None = 3,
    ):
        if shard_bytes is None and per_shard is None:
            raise ValueError("shard_bytes or per_shard bounds a shard")
        self.root = Path(root)
        self.shard_bytes = shard_bytes
        self.per_shard = per_shard
        self.size = 0
        self.zstd = zstd
        self.shards: list[dict] = []
        self.header = dict(header or {})
        info_path = self.root / "dataset.json"
        if info_path.exists():
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            self.shards = info["shards"]
            self.header = info["header"] | self.header
        (self.root / "shards").mkdir(parents=True, exist_ok=True)
        self.keys: set[str] = set()
        self.images: ImageShardWriter | None = None
        self.graphs: list[tuple[str, bytes]] = []
        self.meta: list[tuple[str, dict]] = []
        self.rows: list[dict] = []

    def _dir(self) -> Path:
        return self.root / "shards" / shard_name(len(self.shards))

    def _close_shard(self) -> None:
        folder = self._dir()
        self.images.close()
        self.images = None
        for name, table, items in (
            ("graphs.db", GRAPHS, self.graphs),
            ("meta.db", META, self.meta),
        ):
            write_store(folder / name, table, items)
        pq.write_table(
            pa.Table.from_pylist(self.rows), folder / "manifest.parquet", compression="zstd"
        )
        self.shards.append({"name": folder.name, "samples": len(self.rows), "bytes": self.size})
        self.graphs, self.meta, self.rows, self.size = [], [], [], 0
        write_info(self.root, self.shards, self.header, self.zstd)

    def add(
        self,
        key: str,
        image: bytes,
        w: int,
        h: int,
        codec: str,
        graph,
        meta: dict | None = None,
        **row,
    ) -> None:
        if key in self.keys:
            raise KeyError(f"duplicate key {key!r}")
        if self.images is None:
            folder = self._dir()
            folder.mkdir(parents=True, exist_ok=False)
            self.images = ImageShardWriter(folder / "images.zip", self.header)
        self.keys.add(key)
        index = self.images.add(f"{key}.{EXTENSIONS[codec]}", image, w, h, codec)
        stored = encode_graph(graph, self.zstd)
        self.graphs.append((key, stored))
        self.meta.append((key, meta or {}))
        self.rows.append(
            {
                "key": key,
                "index": index,
                "w": w,
                "h": h,
                "codec": codec,
                "image_bytes": len(image),
                "graph_bytes": len(stored),
                **row,
            }
        )
        self.size += len(image) + len(stored)
        full = self.shard_bytes is not None and self.size >= self.shard_bytes
        if full or len(self.rows) == self.per_shard:
            self._close_shard()

    def close(self) -> None:
        if self.images is not None:
            self._close_shard()
        elif not (self.root / "dataset.json").exists():
            write_info(self.root, self.shards, self.header, self.zstd)

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is None:
            self.close()
