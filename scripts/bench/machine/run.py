"""One GBC machine over a folder of images: load time and run time apart, graph statistics,
every card and the host CPU sampled each second, and the counters of the vLLM servers it calls
(if any) before and after the run.

Writes ``OUT/summary.json`` (with each card's and the host's mean load over the run),
``OUT/graphs.jsonl``, ``OUT/gpu.csv`` (time, index, memory MiB, power W, SM clock, util %),
``OUT/cpu.csv`` (time, 1-minute load average, busy % of all cores; Linux),
``OUT/vllm_<port>.{before,after}.txt`` and the machine report. Machines on the other cards of
the host share its CPU: the host columns show when they starve this one.

    python scripts/bench/machine/run.py OUT IMAGES PORTS CONFIG [CONFIG ...]

``PORTS``: comma-separated local vLLM server ports to scrape, or ``none``.
"""

import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from kohakugbc.machine import build

SMI = [
    "nvidia-smi",
    "--query-gpu=index,memory.used,power.draw,clocks.sm,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def cpu_times() -> tuple[int, int]:
    """(busy, total) jiffies of all cores since boot."""
    with open("/proc/stat", encoding="ascii") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v) - v[3] - v[4], sum(v)


def load1() -> float:
    with open("/proc/loadavg", encoding="ascii") as f:
        return float(f.read().split()[0])


def sampler(out: Path, stop: threading.Event) -> None:
    host = Path("/proc/stat").exists()
    last = cpu_times() if host else None
    with open(out / "gpu.csv", "w", encoding="utf-8") as g, open(out / "cpu.csv", "w") as c:
        while not stop.is_set():
            smi = subprocess.run(SMI, capture_output=True, text=True).stdout.strip()
            t = f"{time.time():.1f}"
            g.writelines(f"{t},{line}\n" for line in smi.splitlines())
            g.flush()
            if host:
                now = cpu_times()
                busy = 100 * (now[0] - last[0]) / max(1, now[1] - last[1])
                last = now
                c.write(f"{t},{load1():.2f},{busy:.1f}\n")
                c.flush()
            stop.wait(1.0)


def loads(out: Path, window: tuple[float, float]) -> dict:
    """Mean util %, power W and memory MiB (and its peak) per card, and the host's load and busy
    %, over the run window."""
    cards: dict[str, list] = {}
    with open(out / "gpu.csv", encoding="utf-8") as f:
        for row in f:
            t, idx, mem, power, _, util = row.strip().split(",")
            if window[0] <= float(t) <= window[1]:
                cards.setdefault(idx, []).append((float(util), float(power), float(mem)))
    report = {
        f"card{i}": {
            **{
                k: round(statistics.mean(r[j] for r in rows), 1)
                for j, k in enumerate(("util", "power_w", "mem_mib"))
            },
            "mem_max_mib": max(r[2] for r in rows),
        }
        for i, rows in sorted(cards.items())
    }
    with open(out / "cpu.csv", encoding="utf-8") as f:
        host = [
            (float(load), float(busy))
            for t, load, busy in (r.strip().split(",") for r in f)
            if window[0] <= float(t) <= window[1]
        ]
    if host:
        report["host"] = {
            "load1": round(statistics.mean(h[0] for h in host), 1),
            "busy_pct": round(statistics.mean(h[1] for h in host), 1),
            "busy_pct_max": round(max(h[1] for h in host), 1),
        }
    return report


def kinds(graph: dict, label: str) -> int:
    return sum(v["label"] == label for v in graph["vertices"])


def stats(graphs: list[dict]) -> dict:
    def mean(f):
        return round(statistics.mean(f(g) for g in graphs), 2) if graphs else 0.0

    return {
        "graphs": len(graphs),
        "vertices": mean(lambda g: len(g["vertices"])),
        "entities": mean(lambda g: kinds(g, "entity")),
        "relations": mean(lambda g: kinds(g, "relation")),
        "compositions": mean(lambda g: kinds(g, "composition")),
        "captions": mean(lambda g: sum(len(v["descs"]) for v in g["vertices"])),
    }


def scrape(ports: list[int], out: Path, tag: str) -> None:
    for port in ports:
        try:
            text = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=10).text
        except httpx.HTTPError as exc:
            text = repr(exc)
        (out / f"vllm_{port}.{tag}.txt").write_text(text, encoding="utf-8")


def main() -> int:
    out, images, ports, layers = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
    ports = [int(p) for p in ports.split(",") if p.isdigit()]
    out.mkdir(parents=True, exist_ok=True)
    overlay = {
        "source": {"kind": "folder", "path": images},
        "sink": {"kind": "memory"},
        "machine": {"report": str(out / "report")},
    }

    stop = threading.Event()
    watcher = threading.Thread(target=sampler, args=(out, stop), daemon=True)
    watcher.start()
    t0 = time.time()
    machine, source, sink = build(*layers, overlay)
    with machine:
        t1 = time.time()
        scrape(ports, out, "before")
        machine.run(source, sink)
        t2 = time.time()
        scrape(ports, out, "after")
    stop.set()
    watcher.join()

    records = sink.records
    ok = [r for r in records if r.ok and r.outputs.get("graph") is not None]
    with open(out / "graphs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(r.outputs["graph"] + "\n" for r in ok)
    failed = [{"key": r.key, **(r.failure or {})} for r in records if not r.ok]
    summary = {
        "out": str(out),
        "layers": layers,
        "jobs": len(records),
        "ok": len(ok),
        "failed": len(failed),
        "no_graph": sum(1 for r in records if r.ok and r.outputs.get("graph") is None),
        "first_failures": failed[:5],
        "build_load_s": round(t1 - t0, 1),
        "run_s": round(t2 - t1, 1),
        "jobs_per_s": round(len(records) / (t2 - t1), 3),
        "run_window": [t1, t2],
        "load": loads(out, (t1, t2)),
        "graphs": stats([json.loads(r.outputs["graph"]) for r in ok]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
