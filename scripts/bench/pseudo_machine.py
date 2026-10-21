"""The GBC machine on pseudo engines: does it keep the VLM lane full, and what does it cost?

Each configuration runs recorded jobs (``runs/traces/poc30/<arm>``, every job ``--repeat``
times) on replay engines with the trace's recorded latencies divided by ``--speed``. Per
configuration it writes the run report (``<out>/<name>/report``) and one summary row:

- ``wall_s``, ``jobs_per_s``; ``bound_s``: the VLM lane's own floor (every recorded VLM latency
  of the run, summed, over the lane's capacity) and ``efficiency`` = bound / wall;
- ``vlm_inflight``: the VLM lane's mean requests in flight over its capacity, from the
  per-second metrics after the first second and before the last job ends;
- ``dispatcher_busy``, detect and embed mean batch sizes, resident jobs (mean);
- ``equal``: jobs whose graph equals the PoC's by digest; ``conform``: recorded jobs that
  conform to KIR's reference interpreter.

    python scripts/bench/pseudo_machine.py --traces runs/traces/poc30 --out runs/bench/pseudo
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from togeari.conformance import conform

from kohakugbc.graph import digest
from kohakugbc.io.sources import TraceJobs
from kohakugbc.machine import build
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace
from kohakugbc.program.reference import run as reference

# name -> (arm, VLM capacity, vlm_waiting, window, detect capacity, other knobs)
CONFIGS = {
    "text_c32_noahead": ("text", 32, 32, 64, 2, {"run_ahead": False}),
    "text_c32_w32": ("text", 32, 32, 64, 2),
    "text_c32_w8": ("text", 32, 8, 64, 2),
    "text_c32_w128": ("text", 32, 128, 64, 2),
    "text_c32_win16": ("text", 32, 32, 16, 2),
    "text_c64_w64": ("text", 64, 64, 64, 2),
    "text_c32_det1": ("text", 32, 32, 64, 1),
    "query_c32_w32": ("query", 32, 32, 64, 2),
    "tags_c32_w32": ("tags", 32, 32, 64, 2),
    "image_text_c32_w32": ("image_text", 32, 32, 64, 2),
    "batch4_c32_w32": ("batch4", 32, 32, 64, 2),
    "leafcap_c32_w32": ("leafcap", 32, 32, 64, 2),
}


def poc_digests(trace: Trace) -> dict[str, str]:
    out = {}
    for key, job in trace.jobs.items():
        g = json.loads(json.dumps(job["graph"]))
        g["img_path"] = Path(g["img_path"]).name
        out[key] = digest(g)
    return out


def ours(text: str) -> str:
    g = json.loads(text)
    g["img_path"] = Path(g["img_path"]).name
    return digest(g)


def metrics(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(
    name: str, args, arm: str, capacity: int, waiting: int, window: int, det: int, knobs=None
) -> dict:
    trace_dir = Path(args.traces) / arm
    report = Path(args.out) / name / "report"
    engines = {
        s: {"entry": "replay", "trace": str(trace_dir), "speed": args.speed, "capacity": capacity}
        for s in ("vlm", "detect", "embed")
    }
    engines["detect"]["capacity"] = det
    engines["embed"]["capacity"] = 2
    layers = {
        "engines": engines,
        "source": {"kind": "trace", "path": str(trace_dir), "repeat": args.repeat},
        "machine": {
            "vlm_waiting": waiting,
            "window": window,
            "report": str(report),
            "record": args.record,
            "knobs": dict(knobs or {}),
        },
    }
    machine, source, sink = build(layers)
    start = time.time()
    with machine:
        machine.run(source, sink)
        job_program = machine.code.entries["gbc"].region.job
    wall = time.time() - start

    trace = Trace(trace_dir)
    expected = poc_digests(trace)
    inputs = {i.key: i.inputs for i in source}
    equal = conforming = recorded = 0
    for rec in sink.records:
        if rec.ok and ours(rec.outputs["graph"]) == expected[rec.key.split("#")[0]]:
            equal += 1
        if rec.record is not None:
            recorded += 1
            conforming += conform(job_program, rec, inputs[rec.key]).ok

    # The recorded latency of every VLM request the program asks (the reference interpreter
    # asks the same requests), per repeat
    models = ReplayModels(trace)
    for item in TraceJobs(trace_dir):
        reference(item.inputs, models)
    vlm_s = sum(trace.vlm[k]["latency"] or 0.0 for k in models.asked)
    vlm_s *= args.repeat / args.speed
    lines = metrics(report / "metrics.jsonl")
    steady = [m for m in lines[1:] if m["jobs"]["resident"] > 0] or lines
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))

    def mean_batch(lane: str) -> float:
        sizes = summary["lanes"][lane]["batch_sizes"]
        n = sum(sizes.values())
        return sum(int(k) * v for k, v in sizes.items()) / n if n else 0.0

    return {
        "name": name,
        "arm": arm,
        "vlm_capacity": capacity,
        "vlm_waiting": waiting,
        "window": window,
        "detect_capacity": det,
        "jobs": len(sink.records),
        "ok": sum(r.ok for r in sink.records),
        "equal": equal,
        "conform": f"{conforming}/{recorded}",
        "wall_s": round(wall, 2),
        "jobs_per_s": round(len(sink.records) / wall, 2),
        "bound_s": round(vlm_s / capacity, 2),
        "efficiency": round(vlm_s / capacity / wall, 3),
        "vlm_inflight": round(
            statistics.mean(m["lanes"]["vlm"]["inflight"] for m in steady) / capacity, 3
        ),
        "vlm_queue": round(statistics.mean(m["lanes"]["vlm"]["queue"] for m in steady), 1),
        "resident": round(statistics.mean(m["jobs"]["resident"] for m in steady), 1),
        "dispatcher_busy": round(summary["dispatcher_busy"], 3),
        "detect_batch": round(mean_batch("detect"), 2),
        "embed_batch": round(mean_batch("embed"), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default="runs/traces/poc30")
    parser.add_argument("--out", default="runs/bench/pseudo")
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--record", type=float, default=0.25)
    parser.add_argument("--configs", nargs="*", default=None)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, spec in CONFIGS.items():
        if args.configs and name not in args.configs:
            continue
        if (out / name).exists():
            print(f"skip {name}: exists", flush=True)
            continue
        row = run(name, args, *spec)
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = out / "summary.jsonl"
    with open(summary, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
