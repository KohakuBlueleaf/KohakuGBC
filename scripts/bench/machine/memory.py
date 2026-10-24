"""Memory of a running machine and its vLLM servers over time, and the report of a sampled run.

    python scripts/bench/machine/memory.py sample OUT SECONDS PORTS PID [PID ...]
    python scripts/bench/machine/memory.py report OUT

``sample`` follows each PID and every process below it until SIGTERM / SIGINT, each SECONDS:
``OUT/memory.csv`` (time, the root's index among the PIDs, pid, process name, RSS / anonymous /
file-backed / shared / swapped MiB, threads, the process's CUDA memory MiB) and
``OUT/progress.csv`` (time, port, finished requests, prompt tokens) from each server's
``/metrics`` (``PORTS``: comma-separated, or ``none``).

``report`` writes ``OUT/memory.json``: per root and process name, the summed memory at the
start, after the warm-up (the first 10% of finished requests, or of the time without servers), at
the end and its peak, and the growth after the warm-up per hour and per 1000 finished requests
(least-squares lines), for each column.
"""

import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

STATUS = {
    "VmRSS": "rss",
    "RssAnon": "anon",
    "RssFile": "file",
    "RssShmem": "shmem",
    "VmSwap": "swap",
}
COLUMNS = ("rss", "anon", "file", "shmem", "swap", "threads", "gpu")
COUNTERS = ("vllm:request_success_total", "vllm:prompt_tokens_total")
WARMUP = 0.1


def children() -> dict[int, list[int]]:
    """Each live pid's direct children."""
    tree: dict[int, list[int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = Path(entry.path, "stat").read_text(encoding="ascii", errors="replace")
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        tree.setdefault(ppid, []).append(int(entry.name))
    return tree


def descendants(roots: list[int]) -> dict[int, int]:
    """Each process under the roots (the roots too) and the index of its root."""
    tree, seen = children(), {}
    todo = [(pid, i) for i, pid in enumerate(roots)]
    while todo:
        pid, root = todo.pop()
        if pid not in seen:
            seen[pid] = root
            todo += [(child, root) for child in tree.get(pid, [])]
    return seen


def status(pid: int) -> dict | None:
    """Name, threads and the memory columns (MiB) of one process, or None once it is gone."""
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    row = dict.fromkeys(STATUS.values(), 0.0)
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key == "Name":
            row["name"] = value.strip()
        elif key == "Threads":
            row["threads"] = int(value)
        elif key in STATUS:
            row[STATUS[key]] = int(value.split()[0]) / 1024
    return row


def gpu_by_pid() -> dict[int, float]:
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    out: dict[int, float] = {}
    for line in smi.strip().splitlines():
        pid, _, mib = line.partition(",")
        if mib.strip().isdigit():
            out[int(pid)] = out.get(int(pid), 0.0) + float(mib)
    return out


def counters(port: int) -> list[float]:
    try:
        text = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5).text
    except httpx.HTTPError:
        return [float("nan")] * len(COUNTERS)
    sums = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in sums:
            sums[name] += float(line.rsplit(" ", 1)[1])
    return [sums[c] for c in COUNTERS]


def sample(out: Path, seconds: float, ports: list[int], roots: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    with open(out / "memory.csv", "w", newline="") as m, open(out / "progress.csv", "w") as p:
        mw, pw = csv.writer(m), csv.writer(p)
        mw.writerow(("t", "root", "pid", "name", *COLUMNS))
        pw.writerow(("t", "port", *COUNTERS))
        while not stop.is_set():
            t = round(time.time(), 1)
            gpu = gpu_by_pid()
            for pid, root in descendants(roots).items():
                row = status(pid)
                if row is not None:
                    # The /proc memory columns, then threads and the CUDA memory
                    values = [round(row[c], 1) for c in COLUMNS[:-2]]
                    mw.writerow(
                        (t, root, pid, row["name"], *values, row["threads"], gpu.get(pid, 0.0))
                    )
            for port in ports:
                pw.writerow((t, port, *counters(port)))
            m.flush()
            p.flush()
            stop.wait(seconds)


def slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0


def report(out: Path) -> dict:
    """Per root and process name, each column summed over those processes at each sample; growth
    against the finished requests of all servers at that sample and against time."""
    done: dict[float, float] = {}
    with open(out / "progress.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = float(row[COUNTERS[0]])
            if not math.isnan(value):  # a server that did not answer its /metrics
                done[float(row["t"])] = done.get(float(row["t"]), 0.0) + value
    series: dict[str, dict[float, dict]] = {}
    with open(out / "memory.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = f"root{row['root']}:{row['name']}"
            at = series.setdefault(key, {}).setdefault(float(row["t"]), {})
            for c in COLUMNS:
                at[c] = at.get(c, 0.0) + float(row[c])
    total = max(done.values(), default=0.0)
    result = {"requests": total, "warmup_requests": round(WARMUP * total), "names": {}}
    for name, at in sorted(series.items()):
        times = sorted(at)
        if total:
            after = [t for t in times if done.get(t, 0.0) >= WARMUP * total]
        else:
            after = [t for t in times if t >= times[0] + WARMUP * (times[-1] - times[0])]
        entry = {}
        for c in COLUMNS:
            ys = [at[t][c] for t in after]
            col = {
                "start": at[times[0]][c],
                "after_warmup": ys[0] if ys else None,
                "end": at[times[-1]][c],
                "peak": max(at[t][c] for t in times),
                "per_hour": round(slope([t / 3600 for t in after], ys), 2),
            }
            if total:
                col["per_1000_requests"] = round(slope([done[t] / 1000 for t in after], ys), 2)
            entry[c] = col
        entry["samples"] = len(times)
        result["names"][name] = entry
    (out / "memory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    mode, out = sys.argv[1], Path(sys.argv[2])
    if mode == "sample":
        ports = [int(p) for p in sys.argv[4].split(",") if p.isdigit()]
        sample(out, float(sys.argv[3]), ports, [int(p) for p in sys.argv[5:]])
    else:
        print(json.dumps(report(out)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
