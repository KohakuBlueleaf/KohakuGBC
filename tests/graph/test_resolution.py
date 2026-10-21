"""Resolution against ml-gbc: the Python reference and the Rust implementation give ml-gbc's
graph, by digest, on the PoC's recorded records and generated record sets
(``scripts/reference/mlgbc_cases.py`` wrote the fixture with ml-gbc itself)."""

import gzip
import json
from pathlib import Path

import pytest

from kohakugbc import graph as g
from kohakugbc.graph import reference as ref

FIXTURE = Path(__file__).parent / "fixtures" / "mlgbc_cases.jsonl.gz"


def load() -> list[dict]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


CASES = load()
GOOD = [c for c in CASES if "error" not in c]
BAD = [c for c in CASES if "error" in c]
IMPLS = {"python": g.python_text, "rust": g.rust_text}


def run(impl, case) -> tuple[str | None, dict]:
    return impl(case["records"], case["img_path"], case.get("img_size"))


def test_fixture_covers_the_cases():
    names = {c["name"].split("/")[0] for c in CASES}
    assert {"generated", "error"} < names and len(names) >= 4, names
    assert len(BAD) >= 5
    # Joins, numbered instances, relations and absent records all occur
    records = [r for c in GOOD for r in c["records"]]
    assert any(len(r["ids"]) > 1 for r in records)
    assert any(r["ids"][0][1] == "relation" for r in records)
    assert any(not r["descs"] for r in records)
    assert any("_" in r["ids"][0][0] and r["ids"][0][0][-1].isdigit() for r in records)


@pytest.mark.parametrize("impl", IMPLS)
def test_equal_to_mlgbc(impl):
    wrong = []
    for case in GOOD:
        text, stats = run(IMPLS[impl], case)
        if g.digest(text) != case["digest"]:
            wrong.append(case["name"])
            continue
        if case["stats"] is not None:
            expected = {k: v for k, v in case["stats"].items() if v is not None}
            assert {k: stats[k] for k in expected} == expected, case["name"]
    assert not wrong, f"{len(wrong)} of {len(GOOD)} differ: {wrong[:10]}"


@pytest.mark.parametrize("impl", IMPLS)
def test_raises_where_mlgbc_raises(impl):
    for case in BAD:
        with pytest.raises(g.ResolutionError):
            run(IMPLS[impl], case)


def test_rust_text_parses_to_python_graph():
    # The two texts differ in number spelling only; their canonical forms are equal
    case = max(GOOD, key=lambda c: len(c["records"]))
    py, _ = run(g.python_text, case)
    rs, _ = run(g.rust_text, case)
    assert json.loads(py) == json.loads(rs)


# A check that cannot fail on a broken resolution is not a check: each variant below breaks one
# step, and the fixture must catch it


def _outer_area(inner, outer, threshold):
    area = ref._area(outer["left"], outer["top"], outer["right"], outer["bottom"])
    overlap = ref._area(
        max(inner["left"], outer["left"]),
        max(inner["top"], outer["top"]),
        min(inner["right"], outer["right"]),
        min(inner["bottom"], outer["bottom"]),
    )
    return (overlap / area if area > 0 else 1) > threshold


def _ascii_numbered(records):
    return json.loads(json.dumps(records).replace("_\\u00b2", "_x"))


BROKEN = {
    "overlap over the outer box": ("_mostly_inside", _outer_area),
    # The module's own ``sorted`` shadows the builtin: vertices stay in record order
    "in-edges in record order": ("sorted", lambda xs, key=None: list(xs)),
    "masks with the image box": ("FULL_BOX", [0.0, 0.0, 1.0, 0.999, None]),
}


@pytest.mark.parametrize("name", BROKEN)
def test_fixture_catches_a_broken_step(name, monkeypatch):
    attr, value = BROKEN[name]
    monkeypatch.setattr(ref, attr, value, raising=False)
    wrong = sum(1 for c in GOOD if g.digest(run(g.python_text, c)[0]) != c["digest"])
    assert wrong > 0, f"{name}: no case notices"


def test_fixture_catches_python_isdigit():
    # Ids numbered only by Python's own str.isdigit ("y_²") change the graph when read as ASCII
    hit = [c for c in GOOD if any("_²" in r["ids"][0][0] for r in c["records"])]
    assert hit
    changed = sum(
        1
        for c in hit
        if g.digest(g.python_text(_ascii_numbered(c["records"]), c["img_path"], c["img_size"])[0])
        != c["digest"]
    )
    assert changed > 0
