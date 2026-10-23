"""SAM 3 through the ``detect`` slot as GBC builds it (``engines.torch.entries.detector`` on
Togeari's ``TorchEngine``: prep workers, priority streams, batch slots), per ``--impl``.

Each workload's requests are submitted to the slot's unit ``--batch`` payloads per submit, all
at once (the unit's slots and the device runtime queue them), after one warm pass; the wall
time of each pass (submit to the last reply), the device memory peak, and the detections are
written in ``stages.py``'s layout, so ``compare.py`` sets two impls side by side.

    python scripts/bench/detector/engine.py --graphs G --image I --trace T --out OUT \
        --impl native transformers
"""

import argparse
import gc
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
from stages import environment
from workloads import graph_file_requests, trace_requests

from kohakugbc.engines.torch.entries import detector


class Item:
    """The engine's view of one request: its payload, and where its raw result lands."""

    def __init__(self, payload: dict):
        self.data, self.raw = payload, None

    def payload(self) -> dict:
        return self.data

    def finish(self, raw):
        self.raw = raw
        return raw


def one_pass(unit, requests: list, batch: int, device: str) -> tuple[float, list, list]:
    items = [Item({"patch": r["patch"], "request": {"texts": r["texts"]}}) for r in requests]
    left = threading.Semaphore(0)
    errors = []

    def done(_ticket, results):
        errors.extend(r for r in results if type(r).__name__ != "Ok")
        left.release()

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    chunks = [items[i : i + batch] for i in range(0, len(items), batch)]
    for chunk in chunks:
        unit.submit(chunk, done)
    for _ in chunks:
        left.acquire()
    wall = (time.perf_counter() - start) * 1e3
    return wall, [i.raw for i in items], errors


def detections(requests: list, raws: list) -> list[dict]:
    return [
        {
            "request": r["name"],
            "texts": r["texts"],
            "boxes": raw["boxes"],
            "scores": raw["scores"],
            "labels": raw["labels"],
        }
        for r, raw in zip(requests, raws)
    ]


def run_impl(args, impl: str, workloads: dict) -> dict:
    device = args.device
    settings = {
        "model": args.model,
        "impl": impl,
        "devices": [device],
        "dtype": args.dtype,
        "max_batch": max(args.batch),
        "box_source": args.box_source,
        "engine": {"schedule": args.schedule},
    }
    shared: dict = {}
    slot = detector(settings, shared)
    engine = next(iter(shared.values()))
    start = time.perf_counter()
    engine.start()
    unit = slot.units[0]
    unit.start()
    load_s = time.perf_counter() - start
    result = {
        "dtype": args.dtype,
        "box_source": f"{args.box_source} engine {impl}",
        "load_s": load_s,
        "workloads": {},
    }
    for wname, requests in workloads.items():
        for batch in args.batch:
            one_pass(unit, requests, batch, device)  # warm-up: tuning, builds, graphs
            passes = [one_pass(unit, requests, batch, device) for _ in range(args.repeat)]
            walls = [p[0] for p in passes]
            peak = {
                "allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            name = f"{wname}_b{batch}"
            result["workloads"][name] = {
                "requests": len(requests),
                "pairs": sum(len(r["texts"]) for r in requests),
                "wall_ms": statistics.median(walls),
                "wall_ms_passes": walls,
                "errors": [repr(e) for p in passes for e in p[2]][:20],
                "peak": peak,
                "stages": {},
                "calls": [],
                "detections": detections(requests, passes[-1][1]),
            }
            errors = len(result["workloads"][name]["errors"])
            print(
                f"{impl} {name}: {statistics.median(walls):.1f} ms (passes {walls}), "
                f"peak {peak}, errors {errors}",
                flush=True,
            )
    out_dir = Path(args.out) / f"{args.dtype}_{args.box_source}_engine_{impl}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stages.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    unit.stop()
    engine.stop()
    del slot, engine, unit, shared
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--box-source", default="pred")
    parser.add_argument("--impl", nargs="+", default=["native", "transformers"])
    parser.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--schedule", default="co_resident")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    workloads = {
        "w1": graph_file_requests(args.graphs, args.image),
        "w2": trace_requests(args.trace),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    env = environment(args.device) | {"args": vars(args)}
    (Path(args.out) / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    for impl in args.impl:
        run_impl(args, impl, workloads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
