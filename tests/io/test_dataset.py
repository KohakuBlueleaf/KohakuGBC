"""The dataset format: shards, the KohakuClip image layout read by an independent zip reader,
graphs and metadata by key, appending, splitting and worker pickling."""

import io
import json
import pickle
import sqlite3
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("kohakuvault")
pytest.importorskip("pyarrow")

from kohakugbc._rs import GbcGraph

from kohakugbc.io.dataset import DatasetWriter, GbcDataset, read_info, split
from kohakugbc.io.dataset.images import MAGIC, RECORD, ImageShard

IMAGES = sorted((Path(__file__).parents[1] / "fixtures" / "traces" / "images").glob("*.webp"))


def graph(i: int, w: int, h: int) -> str:
    box = {"left": 0.0, "top": 0.0, "right": 1.0, "bottom": 1.0, "confidence": None}
    root = {
        "vertex_id": "",
        "bbox": box,
        "label": "image",
        "descs": [{"text": f"picture {i}", "label": "short"}],
        "in_edges": [],
        "out_edges": [],
        "sub_masks": [],
        "super_masks": [],
    }
    return json.dumps({"vertices": [root], "img_path": f"{i}.webp", "img_size": [w, h]})


def samples(start: int, n: int) -> list[dict]:
    out = []
    for i in range(start, start + n):
        data = IMAGES[i % len(IMAGES)].read_bytes()
        out.append(
            {
                "key": str(1000 + i),
                "image": data,
                "w": 10 + i,
                "h": 20 + i,
                "codec": "webp",
                "graph": graph(i, 10 + i, 20 + i),
                "meta": {"id": i, "tags": ["a", f"t{i}"], "score": i / 3},
            }
        )
    return out


def write(root: Path, rows: list[dict], per_shard: int) -> None:
    with DatasetWriter(root, per_shard=per_shard, header={"source": "test"}) as w:
        for r in rows:
            w.add(**r, note=f"row {r['key']}")


def test_shard_bytes(tmp_path):
    rows = samples(0, 5)
    budget = sum(len(r["image"]) for r in rows[:2])
    with DatasetWriter(tmp_path, shard_bytes=budget) as w:
        for r in rows:
            w.add(**r)
    info = read_info(tmp_path)
    m = GbcDataset(tmp_path).manifest.to_pydict()
    sizes = [a + b for a, b in zip(m["image_bytes"], m["graph_bytes"])]
    at = 0
    for s in info["shards"]:
        part = sizes[at : at + s["samples"]]
        assert s["bytes"] == sum(part)
        assert sum(part) >= budget or at + s["samples"] == len(rows)
        assert sum(part[:-1]) < budget
        at += s["samples"]
    assert at == 5 and len(info["shards"]) > 1
    # every store is one file in rollback-journal mode
    for db in tmp_path.glob("shards/*/*.db"):
        con = sqlite3.connect(db)
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        con.close()
    assert not list(tmp_path.glob("shards/*/*-wal")) and not list(tmp_path.glob("shards/*/*-shm"))


def test_round_trip_shards_append(tmp_path):
    first, more = samples(0, 5), samples(5, 3)
    write(tmp_path, first, per_shard=2)
    write(tmp_path, more, per_shard=2)
    rows = first + more

    info = read_info(tmp_path)
    assert [s["samples"] for s in info["shards"]] == [2, 2, 1, 2, 1]
    assert info["samples"] == 8 and info["header"] == {"source": "test"}

    ds = GbcDataset(tmp_path)
    assert len(ds) == 8 and ds.keys() == [r["key"] for r in rows]
    for i, r in enumerate(rows):
        s = ds[i]
        assert (s.key, s.w, s.h, s.codec, s.image) == (r["key"], r["w"], r["h"], "webp", r["image"])
        assert json.loads(s.graph.to_json()) == json.loads(r["graph"])
        assert s.meta == r["meta"]
        assert ds[r["key"]].key == r["key"]
    assert ds.manifest.column("note").to_pylist() == [f"row {r['key']}" for r in rows]


def test_image_shard_layout(tmp_path):
    rows = samples(0, 3)
    write(tmp_path, rows, per_shard=3)
    path = tmp_path / "shards" / "shard_00000" / "images.zip"
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        assert [i.filename for i in infos] == [f"{r['key']}.webp" for r in rows] + ["__index__.bin"]
        assert all(i.compress_type == zipfile.ZIP_STORED for i in infos)
        assert zf.testzip() is None
        index = zf.read("__index__.bin")
    assert index[:8] == MAGIC
    (head_len,) = struct.unpack_from("<Q", index, 8)
    assert json.loads(index[16 : 16 + head_len]) == {"images": 3, "source": "test"}
    at = 16 + head_len + (-(16 + head_len) % 8)
    records = np.frombuffer(index[at:], dtype=RECORD)
    raw = path.read_bytes()
    for rec, r, info in zip(records, rows, infos):
        name_len = struct.unpack_from("<H", raw, info.header_offset + 26)[0]
        assert int(rec["off"]) == info.header_offset + 30 + name_len
        assert raw[int(rec["off"]) : int(rec["off"]) + int(rec["size"])] == r["image"]
        assert (int(rec["w"]), int(rec["h"]), int(rec["codec"]), int(rec["color"])) == (
            r["w"],
            r["h"],
            4,
            0,
        )
    shard = ImageShard(path)
    assert [shard.read(i) for i in range(len(shard))] == [r["image"] for r in rows]
    shard.close()


def test_split_select_transform_pickle(tmp_path):
    rows = samples(0, 5)
    write(tmp_path, rows, per_shard=2)
    names = [s["name"] for s in read_info(tmp_path)["shards"]]
    assert split(names, 0, 2) == ["shard_00000", "shard_00002"]
    assert split(names, 1, 2) == ["shard_00001"]

    ds = GbcDataset(
        tmp_path,
        shards=split(names, 0, 2),
        image=None,
        meta=False,
        transform=lambda s: (s.key, s.image, s.meta, json.loads(s.graph.to_json())["img_path"]),
    )
    assert list(ds) == [
        (r["key"], None, None, f"{i}.webp") for i, r in enumerate(rows) if i != 2 and i != 3
    ]
    clone = pickle.loads(pickle.dumps(GbcDataset(tmp_path, image="pil")))
    want = Image.open(io.BytesIO(rows[3]["image"])).convert("RGB")
    assert clone[3].image.tobytes() == want.tobytes() and clone[3].meta == rows[3]["meta"]
    assert isinstance(clone[3].graph, GbcGraph)
    with pytest.raises(KeyError):
        GbcDataset(tmp_path, shards=["shard_00009"])
