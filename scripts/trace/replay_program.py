"""Replay recorded PoC traces through the GBC program under KIR's reference interpreter and
compare every graph with the one the PoC produced.

Per arm and job: whether the graphs are equal (digest of the canonical text, image path by
name), the requests the program made that the trace does not hold (misses) or holds in other
words (mismatches), and for unequal graphs the vertices only one side has and the vertices whose
content differs. Writes ``report.json`` beside the traces and prints a summary.

    python scripts/trace/replay_program.py runs/traces/poc30 [--arms text tags] [--jobs N]
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from kohakugbc.graph import digest
from kohakugbc.io.sources import TraceJobs
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace
from kohakugbc.program.reference import run


def normalized(graph) -> dict | None:
    if graph is None:
        return None
    g = json.loads(graph) if isinstance(graph, str) else json.loads(json.dumps(graph))
    if g.get("img_path"):
        g["img_path"] = Path(g["img_path"]).name
    if g.get("img_size") is not None:
        g["img_size"] = list(g["img_size"])
    return g


def differences(ours: dict | None, theirs: dict | None) -> dict:
    if ours is None or theirs is None:
        return {"graph": [ours is not None, theirs is not None]}
    a = {v["vertex_id"]: v for v in ours["vertices"]}
    b = {v["vertex_id"]: v for v in theirs["vertices"]}
    changed = {}
    for vid in a.keys() & b.keys():
        fields = [k for k in a[vid] if a[vid][k] != b[vid].get(k)]
        if fields:
            changed[vid] = fields
    out = {
        "only_program": sorted(a.keys() - b.keys()),
        "only_poc": sorted(b.keys() - a.keys()),
        "changed": changed,
    }
    if list(a) != list(b) and set(a) == set(b):
        out["order"] = True
    others = [k for k in ours if k != "vertices" and ours[k] != theirs.get(k)]
    if others:
        out["fields"] = others
    return out


def replay_arm(path: Path, jobs: int | None) -> dict:
    trace = Trace(path)
    arm = trace.meta.get("arm", path.name)
    rows = []
    for k, item in enumerate(TraceJobs(path)):
        if jobs is not None and k >= jobs:
            break
        key, inputs, job = item.key, item.inputs, trace.jobs[item.key]
        models = ReplayModels(trace)
        row = {"job": key}
        start = time.time()
        try:
            out = run(inputs, models)
            ours, theirs = normalized(out["graph"]), normalized(job["graph"])
            row["equal"] = digest(ours) == digest(theirs)
            if not row["equal"]:
                row["diff"] = differences(ours, theirs)
        except Exception as exc:  # noqa: BLE001
            row["equal"] = False
            row["error"] = "".join(traceback.format_exception_only(exc)).strip()[:500]
        row["seconds"] = round(time.time() - start, 3)
        row["misses"] = models.misses
        row["mismatches"] = models.mismatches
        row["calls"] = models.calls
        rows.append(row)
    return {"arm": arm, "jobs": rows, "equal": sum(r["equal"] for r in rows), "total": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()
    root = Path(args.traces)
    report = []
    for path in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        if args.arms and path.name not in args.arms:
            continue
        result = replay_arm(path, args.jobs)
        report.append(result)
        misses = sum(len(r["misses"]) for r in result["jobs"])
        mism = sum(len(r["mismatches"]) for r in result["jobs"])
        errors = sum(1 for r in result["jobs"] if "error" in r)
        print(
            f"{result['arm']:<11} equal {result['equal']}/{result['total']}"
            f"  misses {misses}  mismatches {mism}  errors {errors}",
            flush=True,
        )
    (root / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    return 0 if all(r["equal"] == r["total"] for r in report) else 1


if __name__ == "__main__":
    sys.exit(main())
