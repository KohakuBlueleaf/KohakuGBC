"""Any ``vlm`` entry measured through Togeari's engine contract on recorded GBC requests, so
engines compare on the same work: the requests the program asks on a trace's jobs (replayed),
submitted in their recorded order with at most ``--concurrency`` in flight.

Writes ``<out>/<name>/``: ``manifest.json`` (entry settings, unit spec, arguments, host),
``requests.jsonl`` (per request: key, schema, submit and answer times, latency, result, reply
tokens, JSON validity, whether the parsed reply equals the recorded one), ``memory.jsonl``
(device memory in use every 0.5 s), ``summary.json`` (wall time, requests/s, reply tokens/s,
latency quantiles, validity, equality, peak memory, the unit's own counters) and, with
``--step-log`` (an entry that takes a ``step_log`` setting), ``steps.jsonl``.

    python scripts/bench/vlm_units.py --trace runs/traces/poc30/text --name qwen35 \
        --engine '{"entry": "qwen35", "model": "...", "memory": 0.8}'

``--save-requests`` writes the captured requests (pickled: a trusted local file) for a run on
another node or OS (``--requests``), so every engine gets the same bytes.
"""

import argparse
import json
import pickle
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

from togeari import Ok, WorkItem
from transformers import AutoTokenizer

from kohakugbc.engines import (
    builtin,  # noqa: F401  (registers every entry)
    placement,
    registry,
)
from kohakugbc.io.sources import TraceJobs
from kohakugbc.nodes.specs.parse import load_json
from kohakugbc.nodes.specs.replay import ReplayModels
from kohakugbc.nodes.specs.trace import Trace
from kohakugbc.program.reference import run as reference


class Capture(ReplayModels):
    def __init__(self, trace):
        super().__init__(trace)
        self.requests: list[tuple[dict, str]] = []

    def chat(self, request: dict) -> str:
        reply = super().chat(request)
        self.requests.append((request, reply))
        return reply


def recorded(trace_dir: Path) -> list[tuple[dict, str]]:
    models = Capture(Trace(trace_dir))
    for item in TraceJobs(trace_dir):
        reference(item.inputs, models)
    return models.requests


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


class MemoryLog(threading.Thread):
    def __init__(self, device: str, path: Path):
        super().__init__(daemon=True)
        self.device, self.path = device, path
        self.stopping = threading.Event()
        self.peak = 0

    def run(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            t0 = time.perf_counter()
            while not self.stopping.wait(0.5):
                mem = placement.memory(self.device)
                if mem is None:
                    return
                used = mem[1] - mem[0]
                self.peak = max(self.peak, used)
                f.write(json.dumps({"t": round(time.perf_counter() - t0, 3), "used": used}) + "\n")


def drive(unit, adapter, requests, concurrency: int, log: list) -> float:
    """Submits every request with at most ``concurrency`` open; returns the wall time."""
    gate = threading.Semaphore(concurrency)
    lock = threading.Lock()
    done_all = threading.Event()
    left = [len(requests)]

    def answer(i, t_submit):
        def done(_, results):
            r = results[0]
            log[i].update(t_submit=t_submit, t_done=time.perf_counter(), result=type(r).__name__)
            if isinstance(r, Ok):
                log[i]["text"] = r.value
            gate.release()
            with lock:
                left[0] -= 1
                if left[0] == 0:
                    done_all.set()

        return done

    t0 = time.perf_counter()
    for i, (request, _) in enumerate(requests):
        gate.acquire()
        while unit.capacity() <= 0:
            time.sleep(0.001)
        item = WorkItem("chat", {"request": request}, adapter)
        unit.submit(item, answer(i, time.perf_counter()))
    done_all.wait()
    return time.perf_counter() - t0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--trace", type=Path, help="capture the requests from this trace")
    p.add_argument("--requests", type=Path, help="or load them (a file --save-requests wrote)")
    p.add_argument("--save-requests", type=Path, help="write the captured requests and exit")
    p.add_argument("--engine", help="the vlm slot's settings: JSON or a JSON file")
    p.add_argument("--name")
    p.add_argument("--out", type=Path, default=Path("runs/bench/vlm"))
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--warmup", type=int, default=16, help="requests run first, not measured")
    p.add_argument("--limit", type=int, default=0, help="measure only the first N requests")
    p.add_argument("--tokenizer", default=None, help="for reply token counts (default: model)")
    p.add_argument("--step-log", action="store_true", help="the unit logs every step")
    a = p.parse_args(argv)

    if a.requests is not None:
        with a.requests.open("rb") as f:
            requests = pickle.load(f)
    else:
        requests = recorded(a.trace)
    if a.save_requests is not None:
        a.save_requests.parent.mkdir(parents=True, exist_ok=True)
        with a.save_requests.open("wb") as f:
            pickle.dump(requests, f)
        print(f"{len(requests)} requests written to {a.save_requests}")
        return 0
    path = Path(a.engine)
    settings = json.loads(path.read_text("utf-8") if path.is_file() else a.engine)
    out = a.out / a.name
    out.mkdir(parents=True, exist_ok=True)
    if a.step_log:
        settings["step_log"] = str(out / "steps.jsonl")
    warm, measured = requests[: a.warmup], requests[a.warmup :]
    if a.limit:
        measured = measured[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.tokenizer or settings["model"])

    slot = registry.build_slot("vlm", settings, {})
    unit = slot.units[0]
    t_start = time.perf_counter()
    unit.start()
    start_s = time.perf_counter() - t_start
    try:
        drive(unit, slot.adapter, warm, a.concurrency, [{} for _ in warm])
        device = getattr(unit, "device", "cuda:0")
        mem = MemoryLog(str(device), out / "memory.jsonl")
        mem.start()
        log = [{"key": r["key"]} for r, _ in measured]
        wall = drive(unit, slot.adapter, measured, a.concurrency, log)
        mem.stopping.set()
        mem.join()
        stats = unit.stats() if hasattr(unit, "stats") else dict(unit.load().extra)
        spec = unit.spec()
    finally:
        unit.stop()

    lat, tokens, valid, equal, ok, schemas = [], 0, 0, 0, 0, 0
    with (out / "requests.jsonl").open("w", encoding="utf-8") as f:
        for row, (request, reply) in zip(log, measured):
            row["latency_s"] = row["t_done"] - row["t_submit"]
            row["schema"] = request["schema"]
            text = row.pop("text", None)
            if text is not None:
                ok += 1
                lat.append(row["latency_s"])
                row["reply_tokens"] = len(tok(text, add_special_tokens=False)["input_ids"])
                tokens += row["reply_tokens"]
                parsed = load_json(text)
                row["json"] = parsed is not None
                row["equal_recorded"] = parsed is not None and parsed == load_json(reply)
                if request["schema"]:
                    schemas += 1
                    valid += row["json"]
                equal += row["equal_recorded"]
            f.write(json.dumps(row) + "\n")
    n = len(measured)
    summary = {
        "name": a.name,
        "requests": n,
        "ok": ok,
        "start_s": round(start_s, 2),
        "wall_s": round(wall, 3),
        "requests_per_s": round(n / wall, 3),
        "reply_tokens_per_s": round(tokens / wall, 1),
        "reply_tokens_mean": round(tokens / max(ok, 1), 1),
        "latency_s": {
            "p50": quantile(lat, 0.5),
            "p95": quantile(lat, 0.95),
            "mean": statistics.fmean(lat) if lat else None,
        },
        "json_valid_with_schema": valid / max(schemas, 1),
        "equal_recorded": equal / max(ok, 1),
        "peak_device_memory_gib": round(mem.peak / 2**30, 2),
        "unit": stats,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    manifest = {
        "argv": sys.argv,
        "args": {k: str(v) for k, v in vars(a).items()},
        "settings": settings,
        "spec": spec,
        "host": {"platform": platform.platform(), "python": sys.version},
        "warmup": len(warm),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), "utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "unit"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
