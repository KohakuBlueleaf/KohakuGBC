"""The GBC machine on real engines over recorded jobs: does every job finish, does every job
conform to KIR's reference interpreter, and how do its graphs compare with the PoC's?

Per arm (``runs/traces/poc30/<arm>``) it runs the trace's jobs on the engines of the given
configuration layers, recording every job, and writes ``<out>/<name>/<arm>/`` (the run report,
``graphs.jsonl``) and one summary row:

- ``jobs``, ``ok``, ``failed`` (with the failure kinds), ``conform``: jobs whose recorded run
  the reference interpreter reproduces;
- ``wall_s``, ``jobs_per_s``;
- graph statistics, ours and the PoC's: vertices, entity / relation / composition vertices,
  edges, captions per graph (means over the arm's graphs).

    python scripts/bench/real_machine.py configs/machine/qwen35_local.py --name qwen35
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from togeari.conformance import conform

from kohakugbc.machine import build, config
from kohakugbc.nodes.specs.trace import Trace


def stats(graphs: list[dict]) -> dict:
    """Mean counts per graph."""

    def mean(f) -> float:
        return round(statistics.mean(f(g) for g in graphs), 2) if graphs else 0.0

    def kind(g: dict, label: str) -> int:
        return sum(v["label"] == label for v in g["vertices"])

    return {
        "graphs": len(graphs),
        "vertices": mean(lambda g: len(g["vertices"])),
        "entities": mean(lambda g: kind(g, "entity")),
        "relations": mean(lambda g: kind(g, "relation")),
        "compositions": mean(lambda g: kind(g, "composition")),
        "edges": mean(lambda g: sum(len(v["out_edges"]) for v in g["vertices"])),
        "captions": mean(lambda g: sum(len(v["descs"]) for v in g["vertices"])),
    }


def run(layers: list[str], arm: str, args) -> dict:
    trace_dir = Path(args.traces) / arm
    out = Path(args.out) / args.name / arm
    overlay = {
        "source": {
            "kind": "trace",
            "path": str(trace_dir),
            "override_controls": args.override_controls,
        },
        "sink": {"kind": "memory"},
        "machine": {"report": str(out / "report"), "record": 1.0},
    }
    machine, source, sink = build(*layers, overlay)
    start = time.time()
    with machine:
        machine.run(source, sink)
        job_program = machine.code.entries["gbc"].region.job
    wall = time.time() - start

    inputs = {i.key: i.inputs for i in source}
    records = sink.records
    failures = Counter((r.failure or {}).get("kind", "?") for r in records if not r.ok)
    conforming = sum(conform(job_program, r, inputs[r.key]).ok for r in records if r.record)
    ok = [r for r in records if r.ok and r.outputs.get("graph") is not None]
    with open(out / "graphs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(r.outputs["graph"] + "\n" for r in ok)
    with open(out / "failures.jsonl", "w", encoding="utf-8") as f:
        f.writelines(
            json.dumps({"key": r.key, **(r.failure or {})}) + "\n" for r in records if not r.ok
        )
    ours = [json.loads(r.outputs["graph"]) for r in ok]
    poc = [job["graph"] for job in Trace(trace_dir).jobs.values()]
    return {
        "name": args.name,
        "arm": arm,
        "jobs": len(records),
        "ok": sum(r.ok for r in records),
        "failed": dict(failures),
        "conform": f"{conforming}/{sum(1 for r in records if r.record)}",
        "wall_s": round(wall, 2),
        "jobs_per_s": round(len(records) / wall, 3),
        "ours": stats(ours),
        "poc": stats(poc),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="configuration layers (the engines)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--arms", nargs="*", default=["text", "query"])
    parser.add_argument("--traces", default="runs/traces/poc30")
    parser.add_argument("--out", default="runs/bench/real")
    parser.add_argument(
        "--override-controls",
        action="store_true",
        help="the layers' controls over the recorded jobs' own (a size preset, for one)",
    )
    args = parser.parse_args()
    config.load(*args.configs)  # fails early on a bad layer
    rows = []
    for arm in args.arms:
        row = run(args.configs, arm, args)
        rows.append(row)
        print(json.dumps(row), flush=True)
    path = Path(args.out) / args.name / "summary.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
