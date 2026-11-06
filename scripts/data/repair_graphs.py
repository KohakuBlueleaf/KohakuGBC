"""Repair of stored GBC-Nest graphs (the web1m run1 vaults) for packing, and its check.

The shaping stage of GBC-Nest dropped and merged vertices after resolution had derived the
masks, the in-edges and the relation boxes, and left those as they were. ``repair`` derives them
again from what is left, the way resolution (ml-gbc's ``update_edges_and_masks``) derives them:

1. Out-edges to vertices that are gone are removed.
2. A relation vertex ``[a|b|...]`` stays only while every entity it names still reaches a
   vertex by its edge (edge text = the id's part after the last ``_``, as resolution writes it):
   the entity's own vertex, or else the first edge with its text that no other name takes (a
   stem's first instance, or the vertex it merged into, whose moved edge keeps the text). A
   relation that lost a participant goes, with every edge to it.
3. Edges between non-relation vertices that close a cycle (a join during the rounds could make
   a vertex its own ancestor) are removed: a depth-first walk from the image vertex, out-edges
   in order, drops each edge to a vertex on its current path.
4. A relation's box is the union of the boxes of the vertices its names reach (resolution unions
   the first vertex each named id maps to); confidence None.
5. In-edges are the inverse of the out-edges; sub / super masks are recomputed over every vertex
   with the graph's ``mask_inside_threshold`` (inner inside outer when overlap / inner area >
   threshold, an empty inner box counting as inside).

Nothing else changes: entity, composition and image vertices, their boxes, descriptions and the
graph's captions stay as stored.

``check(fixed, stored)`` is a separate implementation of what a sound graph is (it shares no
code with ``repair``) plus the consumer's own test: ``kohakugbc_data``'s ``GbcGraph`` loads the
graph typed and renders ``prompt()``.

    python scripts/data/repair_graphs.py selftest tests/graph/fixtures/mlgbc_cases.jsonl.gz
    python scripts/data/repair_graphs.py check OUT.json VAULT [VAULT ...] [--limit N] [--procs P]

``selftest``: resolution's own graphs come out of ``repair`` unchanged and pass ``check``.
``check``: every graph of the vaults repaired and checked; counts, problems and examples in
OUT.json. In pack code: ``fixed, report = repair(graph)``; ``check(fixed, graph) == []``.
"""

import argparse
import gzip
import json
import multiprocessing as mp
import sys
import time
import zlib
from collections import Counter

from kohakugbc_data._rs import GbcGraph
from kohakuvault import KVault

from kohakugbc.graph.reference import resolve

# ---- repair ----------------------------------------------------------------------------------


def _named(vertex_id: str) -> list[str]:
    return vertex_id[1:-1].split("|")


def _edge_text(entity_id: str) -> str:
    return entity_id.rsplit("_", 1)[-1]


def _reached(v: dict) -> list[str | None]:
    """For each entity the relation names, in id order, the vertex its edge reaches: the entity's
    own vertex when that edge is there, else the first edge with its text not taken by another
    name (a stem's first instance, or the vertex it merged into); None when there is none. A
    name with blank text has no edge and reaches nothing."""
    edges = [(e["text"], e["target"]) for e in v["out_edges"]]
    names = _named(v["vertex_id"])
    out: list[str | None] = [None] * len(names)
    taken: set[int] = set()
    for i, n in enumerate(names):
        for j, (text, target) in enumerate(edges):
            if j not in taken and text == _edge_text(n) and target == n:
                out[i] = target
                taken.add(j)
                break
    for i, n in enumerate(names):
        if out[i] is not None:
            continue
        for j, (text, target) in enumerate(edges):
            if j not in taken and text == _edge_text(n):
                out[i] = target
                taken.add(j)
                break
    return out


def _holds(v: dict) -> bool:
    names = _named(v["vertex_id"])
    reached = _reached(v)
    return bool(v["out_edges"]) and all(
        t is not None for n, t in zip(names, reached) if _edge_text(n).strip()
    )


def _area(left, top, right, bottom) -> float:
    return max(0, right - left) * max(0, bottom - top)


def _inside(inner: dict, outer: dict, threshold: float) -> bool:
    area = _area(inner["left"], inner["top"], inner["right"], inner["bottom"])
    overlap = _area(
        max(inner["left"], outer["left"]),
        max(inner["top"], outer["top"]),
        min(inner["right"], outer["right"]),
        min(inner["bottom"], outer["bottom"]),
    )
    return (overlap / area if area > 0 else 1) > threshold


def _add(ids: list, vertex_id: str) -> None:
    if vertex_id not in ids:
        ids.append(vertex_id)


def _link_and_mask(vertices: list[dict], threshold: float) -> None:
    """Resolution's derivation of in-edges and masks, step for step."""
    by_id = {v["vertex_id"]: v for v in vertices}
    ordered = sorted(vertices, key=lambda v: v["vertex_id"])
    for v in ordered:
        kept = []
        for edge in v["out_edges"]:
            if edge["target"] not in by_id:
                continue
            child = by_id[edge["target"]]
            if edge not in child["in_edges"]:
                child["in_edges"].append(edge)
            kept.append(edge)
        v["out_edges"] = kept
        kept = []
        for edge in v["in_edges"]:
            if edge["source"] not in by_id:
                continue
            parent = by_id[edge["source"]]
            if edge not in parent["out_edges"]:
                parent["out_edges"].append(edge)
            kept.append(edge)
        v["in_edges"] = kept
        v["sub_masks"] = []
        v["super_masks"] = []
        for alt in ordered:
            if alt["vertex_id"] == v["vertex_id"]:
                continue
            if _inside(v["bbox"], alt["bbox"], threshold):
                _add(alt["sub_masks"], v["vertex_id"])
                _add(v["super_masks"], alt["vertex_id"])
            if _inside(alt["bbox"], v["bbox"], threshold):
                _add(v["sub_masks"], alt["vertex_id"])
                _add(alt["super_masks"], v["vertex_id"])


def _break_cycles(vertices: list[dict], by_id: dict) -> int:
    """Edges between non-relation vertices that close a cycle removed: a depth-first walk from
    the image vertex, out-edges in order, drops each edge to a vertex on its current path. The
    number removed."""
    removed = 0
    state: dict[str, int] = {}
    starts = [v["vertex_id"] for v in vertices if v["label"] == "image"]
    starts += [v["vertex_id"] for v in vertices if v["label"] != "image"]
    for s in starts:
        if s in state:
            continue
        state[s] = 1
        stack = [(s, 0)]
        while stack:
            k, i = stack.pop()
            edges = by_id[k]["out_edges"]
            if i >= len(edges):
                state[k] = 2
                continue
            stack.append((k, i + 1))
            t = edges[i]["target"]
            if by_id[t]["label"] == "relation" or by_id[k]["label"] == "relation":
                continue
            if state.get(t) == 1:
                edges.pop(i)
                stack[-1] = (k, i)
                removed += 1
            elif t not in state:
                state[t] = 1
                stack.append((t, 0))
    return removed


def repair(graph: dict) -> tuple[dict, dict]:
    """The repaired graph (a new dict; ``graph`` is not modified) and what changed:
    ``relations`` (ids removed), ``edges`` (out-edges removed), ``masks_before`` /
    ``masks_after`` (mask references), ``dangling_masks`` (references to absent vertices)."""
    vertices = [dict(v) for v in graph["vertices"]]
    ids = {v["vertex_id"] for v in vertices}
    masks_before = sum(len(v["sub_masks"]) + len(v["super_masks"]) for v in vertices)
    dangling = sum(1 for v in vertices for m in v["sub_masks"] + v["super_masks"] if m not in ids)
    edges = 0
    for v in vertices:
        kept = [dict(e) for e in v["out_edges"] if e["target"] in ids]
        edges += len(v["out_edges"]) - len(kept)
        v["out_edges"] = kept

    gone = [v["vertex_id"] for v in vertices if v["label"] == "relation" and not _holds(v)]
    if gone:
        drop = set(gone)
        vertices = [v for v in vertices if v["vertex_id"] not in drop]
        for v in vertices:
            kept = [e for e in v["out_edges"] if e["target"] not in drop]
            edges += len(v["out_edges"]) - len(kept)
            v["out_edges"] = kept

    by_id = {v["vertex_id"]: v for v in vertices}
    cycle_edges = _break_cycles(vertices, by_id)
    edges += cycle_edges
    for v in vertices:
        if v["label"] == "relation":
            # a blank-named entity has no edge; resolution still unions its vertex's box
            reached = [
                t if _edge_text(n).strip() else (n if n in by_id else None)
                for n, t in zip(_named(v["vertex_id"]), _reached(v))
            ]
            boxes = [by_id[t]["bbox"] for t in reached if t is not None]
            v["bbox"] = {
                "left": min(b["left"] for b in boxes),
                "top": min(b["top"] for b in boxes),
                "right": max(b["right"] for b in boxes),
                "bottom": max(b["bottom"] for b in boxes),
                "confidence": None,
            }
        v["in_edges"] = []
    _link_and_mask(vertices, graph["mask_inside_threshold"])
    masks_after = sum(len(v["sub_masks"]) + len(v["super_masks"]) for v in vertices)
    report = {
        "relations": gone,
        "edges": edges,
        "cycle_edges": cycle_edges,
        "masks_before": masks_before,
        "masks_after": masks_after,
        "dangling_masks": dangling,
    }
    return {**graph, "vertices": vertices}, report


# ---- check (independent of repair) -----------------------------------------------------------


def _covered(a: dict, b: dict, t: float) -> bool:
    """``a`` lies in ``b`` by more than ``t`` of its own area (a degenerate ``a`` always does)."""
    w = min(a["right"], b["right"]) - max(a["left"], b["left"])
    h = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    own = max(0.0, a["right"] - a["left"]) * max(0.0, a["bottom"] - a["top"])
    if own <= 0:
        return True
    return max(0.0, w) * max(0.0, h) / own > t


def check(fixed: dict, stored: dict, consumer: bool = True) -> list[str]:
    """What is wrong with ``fixed`` as a sound version of ``stored``; empty when nothing is.
    ``consumer``: also load it with ``GbcGraph`` (typed) and render its prompt."""
    bad = []
    thr = fixed["mask_inside_threshold"]
    vs = {v["vertex_id"]: v for v in fixed["vertices"]}
    if len(vs) != len(fixed["vertices"]):
        bad.append("a vertex id twice")
    old = {v["vertex_id"]: v for v in stored["vertices"]}

    # Only relation vertices may leave; the rest stay as stored
    for k, v in old.items():
        if k not in vs:
            if v["label"] != "relation":
                bad.append(f"{v['label']} {k!r} removed")
            continue
        w = vs[k]
        bad += [f"{k!r}: {f} changed" for f in ("label", "descs") if w[f] != v[f]]
        if v["label"] != "relation" and w["bbox"] != v["bbox"]:
            bad.append(f"{k!r}: box changed")
    if set(vs) - set(old):
        bad.append("vertices added")
    bad += [
        f"graph field {f!r} changed"
        for f in stored
        if f != "vertices" and fixed.get(f) != stored[f]
    ]

    # Edges: targets present, sources right, in-edges the inverse of the out-edges, each once
    outs, ins = set(), []
    for k, v in vs.items():
        for e in v["out_edges"]:
            if e["source"] != k:
                bad.append(f"{k!r}: out-edge with source {e['source']!r}")
            if e["target"] not in vs:
                bad.append(f"{k!r}: out-edge to absent {e['target']!r}")
            outs.add((e["source"], e["target"], e["text"]))
        for e in v["in_edges"]:
            if e["target"] != k:
                bad.append(f"{k!r}: in-edge with target {e['target']!r}")
            ins.append((e["source"], e["target"], e["text"]))
    if outs != set(ins) or len(ins) != len(set(ins)):
        bad.append("in-edges are not the inverse of the out-edges")

    # Every out-edge of the stored graph to a kept target survives, unless it closed a cycle
    # (its target reaches its source in the stored graph, relation vertices aside)
    def reaches(a: str, b: str) -> bool:
        seen, todo = {a}, [a]
        while todo:
            x = todo.pop()
            if x == b:
                return True
            for e in old[x]["out_edges"]:
                y = e["target"]
                if y in old and y not in seen and old[y]["label"] != "relation":
                    seen.add(y)
                    todo.append(y)
        return False

    for k, v in old.items():
        if k not in vs:
            continue
        have = {(e["target"], e["text"]) for e in vs[k]["out_edges"]}
        for e in v["out_edges"]:
            if e["target"] in vs and (e["target"], e["text"]) not in have:
                plain = v["label"] != "relation" and old[e["target"]]["label"] != "relation"
                if not (plain and reaches(e["target"], k)):
                    bad.append(f"{k!r}: edge to {e['target']!r} lost")

    # No cycle among the non-relation vertices (Kahn's order takes every one)
    plain = {k for k, v in vs.items() if v["label"] != "relation"}
    indeg = {k: 0 for k in plain}
    for k in plain:
        for t in {e["target"] for e in vs[k]["out_edges"]} & plain:
            indeg[t] += 1
    order = [k for k, n in indeg.items() if n == 0]
    for k in order:
        for t in {e["target"] for e in vs[k]["out_edges"]} & plain:
            indeg[t] -= 1
            if indeg[t] == 0:
                order.append(t)
    if len(order) != len(plain):
        bad.append("a cycle among the non-relation vertices")

    # Relations: every named participant present by its edge text; box the participants' union
    for k, v in vs.items():
        if v["label"] != "relation":
            continue
        if not (k.startswith("[") and k.endswith("]")):
            bad.append(f"relation id {k!r}")
            continue
        if not v["out_edges"]:
            bad.append(f"{k!r}: no participants")
            continue
        free = list(enumerate(v["out_edges"]))
        names = k[1:-1].split("|")
        got = {}
        # exact vertices first, then the rest in id order, each edge used once
        for exact in (True, False):
            for i, name in enumerate(names):
                if i in got:
                    continue
                want = name.split("_")[-1] if "_" in name else name
                hit = next(
                    (
                        (j, e)
                        for j, e in free
                        if e["text"] == want and (e["target"] == name or not exact)
                    ),
                    None,
                )
                if hit is not None:
                    got[i] = hit[1]["target"]
                    free.remove(hit)
        missing = [
            n
            for i, n in enumerate(names)
            if i not in got and (n.split("_")[-1] if "_" in n else n).strip()
        ]
        if missing:
            bad.append(f"{k!r}: participants {missing} missing")
            continue
        got |= {
            i: n
            for i, n in enumerate(names)
            if not (n.split("_")[-1] if "_" in n else n).strip() and n in vs
        }
        boxes = [vs[t]["bbox"] for t in got.values() if t in vs]
        if boxes:
            union = (
                min(b["left"] for b in boxes),
                min(b["top"] for b in boxes),
                max(b["right"] for b in boxes),
                max(b["bottom"] for b in boxes),
            )
            have = (v["bbox"]["left"], v["bbox"]["top"], v["bbox"]["right"], v["bbox"]["bottom"])
            if have != union:
                bad.append(f"{k!r}: box is not its participants' union")

    # Masks: exactly the containment pairs over the vertices present
    for k, v in vs.items():
        sub = {o for o in vs if o != k and _covered(vs[o]["bbox"], v["bbox"], thr)}
        sup = {o for o in vs if o != k and _covered(v["bbox"], vs[o]["bbox"], thr)}
        if set(v["sub_masks"]) != sub or len(v["sub_masks"]) != len(sub):
            bad.append(f"{k!r}: sub_masks wrong")
        if set(v["super_masks"]) != sup or len(v["super_masks"]) != len(sup):
            bad.append(f"{k!r}: super_masks wrong")

    # The consumer: typed, and the prompt renders
    if not consumer:
        return bad
    try:
        g = GbcGraph.from_json(json.dumps(fixed, ensure_ascii=False))
        if not g.typed:
            bad.append("GbcGraph keeps it raw")
        else:
            g.prompt()
    except Exception as exc:  # noqa: BLE001 - every consumer failure is a finding
        bad.append(f"GbcGraph: {type(exc).__name__}: {exc}")
    return bad


# ---- runs ------------------------------------------------------------------------------------


def _one(item: tuple[str, str, bytes]) -> dict:
    path, key, blob = item
    stored = json.loads(zlib.decompress(blob))
    stored_typed = GbcGraph.from_json(json.dumps(stored, ensure_ascii=False)).typed
    fixed, report = repair(stored)
    problems = check(fixed, stored)
    labels = Counter(v["label"] for v in stored["vertices"])
    return {
        "vault": path,
        "key": key,
        "stored_typed": stored_typed,
        "relations_removed": len(report["relations"]),
        "relations": labels["relation"],
        "edges_removed": report["edges"],
        "cycle_edges": report["cycle_edges"],
        "dangling_masks": report["dangling_masks"],
        "masks_before": report["masks_before"],
        "masks_after": report["masks_after"],
        "changed": fixed != stored,
        "problems": problems,
    }


def _items(paths: list[str], limit: int | None):
    for path in paths:
        vault = KVault(path, table="graphs", read_only=True)
        try:
            for n, key in enumerate(vault.keys()):
                if limit is not None and n >= limit:
                    break
                yield path, key.decode(), vault.get(key)
        finally:
            vault.close()


def run_check(out: str, paths: list[str], limit: int | None, procs: int) -> int:
    t0 = time.time()
    total = Counter()
    per_vault: dict[str, Counter] = {}
    problems = Counter()
    examples: list[dict] = []
    keys: set[str] = set()
    dup = 0
    with mp.Pool(procs) as pool:
        for r in pool.imap_unordered(_one, _items(paths, limit), chunksize=64):
            c = per_vault.setdefault(r["vault"], Counter())
            if r["key"] in keys:
                dup += 1
            keys.add(r["key"])
            for counter in (c, total):
                counter["graphs"] += 1
                counter["stored_raw"] += not r["stored_typed"]
                counter["changed"] += r["changed"]
                counter["relations"] += r["relations"]
                counter["relations_removed"] += r["relations_removed"]
                counter["graphs_relations_removed"] += r["relations_removed"] > 0
                counter["edges_removed"] += r["edges_removed"]
                counter["cycle_edges"] += r["cycle_edges"]
                counter["graphs_cycles"] += r["cycle_edges"] > 0
                counter["dangling_masks"] += r["dangling_masks"]
                counter["graphs_dangling_masks"] += r["dangling_masks"] > 0
                counter["masks_before"] += r["masks_before"]
                counter["masks_after"] += r["masks_after"]
                counter["bad"] += bool(r["problems"])
            for p in r["problems"]:
                problems[p.split(": ", 1)[-1] if p.startswith("'") else p] += 1
            if r["problems"] and len(examples) < 50:
                examples.append(
                    {"vault": r["vault"], "key": r["key"], "problems": r["problems"][:10]}
                )
            if total["graphs"] % 50000 == 0:
                rate = total["graphs"] / (time.time() - t0)
                print(f"{total['graphs']} graphs, {total['bad']} bad, {rate:.0f}/s", flush=True)
    result = {
        "vaults": paths,
        "limit": limit,
        "seconds": round(time.time() - t0, 1),
        "total": dict(total),
        "per_vault": {k: dict(v) for k, v in per_vault.items()},
        "keys": len(keys),
        "keys_in_several_vaults": dup,
        "problems": dict(problems.most_common()),
        "examples": examples,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("seconds", "total", "keys", "keys_in_several_vaults", "problems")
            },
            indent=1,
        )
    )
    return 0 if total["bad"] == 0 else 1


def run_selftest(cases: str) -> int:
    """Resolution's own graphs (ml-gbc-equal by the KohakuGBC tests) come out of ``repair``
    unchanged and pass ``check``."""
    with gzip.open(cases, "rt", encoding="utf-8") as f:
        good = [c for c in map(json.loads, f) if "error" not in c]
    n = changed = bad = empty = 0
    for case in good:
        graph = resolve(case["records"], case["img_path"], case.get("img_size"))
        if graph is None:
            continue
        graph = json.loads(json.dumps(graph))
        fixed, report = repair(graph)
        n += 1
        # Resolution itself writes a relation without participants for a blank entity name;
        # repair removes those, and only those
        hollow = sorted(
            v["vertex_id"]
            for v in graph["vertices"]
            if v["label"] == "relation" and not v["out_edges"]
        )
        if hollow:
            empty += 1
            if sorted(report["relations"]) != hollow:
                changed += 1
                print("removed other than hollow relations:", case["name"], report)
        elif fixed != graph:
            changed += 1
            print("changed:", case["name"], report)
        problems = check(fixed, graph, consumer=False)
        if problems:
            bad += 1
            print("problems:", case["name"], problems[:5])
    print(
        f"{n} resolved graphs: {changed} changed wrongly by repair, {bad} with problems; "
        f"{empty} held relations without participants (removed)"
    )
    return 0 if n and not changed and not bad else 1


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("cases")
    c = sub.add_parser("check")
    c.add_argument("out")
    c.add_argument("vaults", nargs="+")
    c.add_argument("--limit", type=int)
    c.add_argument("--procs", type=int, default=32)
    a = p.parse_args()
    if a.cmd == "selftest":
        return run_selftest(a.cases)
    return run_check(a.out, a.vaults, a.limit, a.procs)


if __name__ == "__main__":
    sys.exit(main())
