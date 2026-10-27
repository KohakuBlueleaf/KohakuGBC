"""Reading a GBC dataset (layout in ``kohakugbc_data.dataset``)."""

import io
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from kohakugbc_data._rs import GbcGraph
from kohakugbc_data.dataset.images import ImageShard
from kohakugbc_data.dataset.stores import GRAPHS, META, Store
from kohakugbc_data.dataset.writer import FORMAT, VERSION

MANIFEST_COLUMNS = ["key", "index", "w", "h", "codec"]


@dataclass
class Sample:
    key: str
    shard: str
    w: int
    h: int
    codec: str
    # the encoded bytes, a PIL RGB image (``image="pil"``), or None (``image=None``)
    image: Any
    graph: GbcGraph | None
    meta: dict | None


def read_info(root: str | os.PathLike) -> dict:
    with open(Path(root) / "dataset.json", encoding="utf-8") as f:
        info = json.load(f)
    if info.get("format") != FORMAT or info.get("version") != VERSION:
        raise ValueError(f"{root}: not a {FORMAT} v{VERSION} dataset")
    return info


def split(shards: list[str], rank: int, world: int) -> list[str]:
    """The shards of worker ``rank`` of ``world``: every ``world``-th, from ``rank``."""
    return shards[rank::world]


class GbcDataset:
    """Samples by position (shard order) or by key; a map-style dataset for any loader.

    ``shards``: the shard names to read (all by default; ``split`` hands them out to workers).
    ``image``: "bytes" (as stored), "pil" (decoded, RGB) or None; ``graphs`` / ``meta``: read
    those or leave them None. ``transform(sample)``: what ``ds[i]`` returns instead of the
    ``Sample`` (rendering a graph to a prompt, a meta dict to tags, ...). Files open on first
    use in each process, so a dataset made before DataLoader workers fork is safe to use in them.
    ``ds.manifest``: every read shard's manifest rows as one pyarrow table, ``shard`` added.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        shards: list[str] | None = None,
        image: str | None = "bytes",
        graphs: bool = True,
        meta: bool = True,
        transform: Callable[[Sample], Any] | None = None,
    ):
        if image not in ("bytes", "pil", None):
            raise ValueError(f"image is 'bytes', 'pil' or None, not {image!r}")
        self.root = Path(root)
        self.info = read_info(self.root)
        names = [s["name"] for s in self.info["shards"]]
        unknown = set(shards or ()) - set(names)
        if unknown:
            raise KeyError(f"no shards {sorted(unknown)} in {self.root}")
        self.shards = [n for n in names if shards is None or n in shards]
        self.image_mode, self.graphs, self.meta = image, graphs, meta
        self.transform = transform

        tables = []
        for k, name in enumerate(self.shards):
            t = pq.read_table(self.root / "shards" / name / "manifest.parquet")
            tables.append(t.append_column("shard", pa.array([k] * len(t), pa.int32())))
        self.manifest = pa.concat_tables(tables) if tables else pa.table({})
        cols = self.manifest.select([*MANIFEST_COLUMNS, "shard"]).to_pydict() if tables else {}
        self._keys = cols.get("key", [])
        self._rows = (
            list(zip(*(cols[c] for c in ("shard", "index", "w", "h", "codec")))) if cols else []
        )
        self._at = {k: i for i, k in enumerate(self._keys)}
        self._open: dict = {}
        self._pid = os.getpid()

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: str) -> bool:
        return key in self._at

    def keys(self) -> list[str]:
        return self._keys

    def _handle(self, kind: str, shard: int):
        if os.getpid() != self._pid:
            self._open, self._pid = {}, os.getpid()
        got = self._open.get((kind, shard))
        if got is None:
            folder = self.root / "shards" / self.shards[shard]
            match kind:
                case "images":
                    got = ImageShard(folder / "images.zip")
                case "graphs":
                    got = Store(folder / "graphs.db", GRAPHS)
                case "meta":
                    got = Store(folder / "meta.db", META)
            self._open[(kind, shard)] = got
        return got

    def image(self, key: str) -> bytes:
        shard, index, *_ = self._rows[self._at[key]]
        return self._handle("images", shard).read(index)

    def graph(self, key: str) -> GbcGraph:
        shard = self._rows[self._at[key]][0]
        return GbcGraph.from_msgpack(self._handle("graphs", shard)[key])

    def meta_of(self, key: str) -> dict:
        shard = self._rows[self._at[key]][0]
        return self._handle("meta", shard)[key]

    def sample(self, i: int) -> Sample:
        """Sample ``i`` as read (no ``transform``)."""
        shard, index, w, h, codec = self._rows[i]
        key = self._keys[i]
        image = None
        if self.image_mode is not None:
            image = self._handle("images", shard).read(index)
            if self.image_mode == "pil":
                image = Image.open(io.BytesIO(image)).convert("RGB")
        return Sample(
            key,
            self.shards[shard],
            w,
            h,
            codec,
            image,
            GbcGraph.from_msgpack(self._handle("graphs", shard)[key]) if self.graphs else None,
            self._handle("meta", shard)[key] if self.meta else None,
        )

    def __getitem__(self, i: int | str):
        s = self.sample(self._at[i] if isinstance(i, str) else i)
        return self.transform(s) if self.transform else s

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]

    def close(self) -> None:
        for handle in self._open.values():
            handle.close()
        self._open = {}

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_open": {}}

    def __enter__(self) -> "GbcDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
