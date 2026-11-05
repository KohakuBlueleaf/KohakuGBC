"""The graph vault and the jsonl source's shard and skip: each graph is stored under its job key
as the exact text the record carried, a failure is kept until the key succeeds, and a machine's
share of the jobs is the same on every run, less the keys its vault holds."""

import json
import sqlite3
import zlib

from togeari import JobRecord

from kohakugbc.io import sinks, sources
from kohakugbc.io.vault import FORMAT, done_keys, read_vault


def record(key: str, ok: bool = True) -> JobRecord:
    graph = json.dumps({"img_path": f"/images/{key}.webp", "vertices": [{"id": key}]})
    outputs = {"graph": graph, "stats": {"vertices": 1}} if ok else {}
    failure = None if ok else {"kind": "ItemError", "node": "chat", "message": "boom"}
    return JobRecord(key, "gbc", ok, outputs, failure)


def test_vault_keeps_each_graph_under_its_key(tmp_path):
    path = tmp_path / "web.db"
    sink = sinks.build({"kind": "vault", "path": str(path)})
    sink.write(record("coyo/1"))
    sink.write(record("laion/2", ok=False))
    sink.write(record("laion/2"))
    sink.write(record("coyo/3", ok=False))
    sink.close()

    # The stored blob is the record's graph text, zlib-compressed, read straight from SQLite
    con = sqlite3.connect(path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    assert {"graphs", "stats", "failures", "meta"} <= tables
    rows = dict(con.execute("select key, value from graphs"))
    assert zlib.decompress(rows[b"coyo/1"]).decode() == record("coyo/1").outputs["graph"]
    meta = dict(con.execute("select key, value from meta"))
    assert meta[b"format"].decode() == FORMAT and meta[b"codec"] == b"zlib"
    failures = {k.decode() for (k,) in con.execute("select key from failures")}
    assert failures == {"coyo/3"}
    con.close()

    # Readers: every graph, chosen keys, the keys done
    got = dict(read_vault(str(path)))
    assert set(got) == {"coyo/1", "laion/2"}
    assert json.loads(got["laion/2"])["img_path"] == "/images/laion/2.webp"
    assert dict(read_vault(str(path), ["laion/2", "missing"])) == {"laion/2": got["laion/2"]}
    assert done_keys([str(path), str(tmp_path / "absent.db")]) == {"coyo/1", "laion/2"}


def test_jsonl_source_shards_then_skips_the_stored_keys(tmp_path):
    jobs = tmp_path / "jobs.jsonl"
    keys = [f"coyo/{i}" for i in range(10)]
    jobs.write_text("".join(json.dumps({"key": k, "image": f"/i/{k}.webp"}) + "\n" for k in keys))
    vault = tmp_path / "card1.db"
    sink = sinks.build({"kind": "vault", "path": str(vault)})
    sink.write(record("coyo/4"))
    sink.close()

    spec = {"kind": "jsonl", "path": str(jobs), "shard": [1, 3], "skip": [str(vault)]}
    got = [item.key for item in sources.build(spec, {})]
    assert got == ["coyo/1", "coyo/7"]
    shares = [
        {item.key for item in sources.build({**spec, "shard": [r, 3], "skip": []}, {})}
        for r in range(3)
    ]
    assert set().union(*shares) == set(keys) and sum(map(len, shares)) == len(keys)
