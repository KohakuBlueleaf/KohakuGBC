"""Tuned choices: the config of one (kernel, problem key) on one card, timed once and stored.

``select(kernel, device, key, candidates, launch)``: a stored winner that is still a
candidate binds; otherwise the first ``shortlist`` candidates (each config a dict; PyTorch
paths are candidates like any kernel config) are timed and the fastest is stored in
``<cache>/tuning/<card key>/<kernel>.json`` with every timing, failure and working memory.

The metric (``metric(name)`` scopes it; the choice is stored per metric):

- ``loop``: wall time per call of back-to-back calls, synchronized at the end: what a call
  costs when the host issues it (a launch-bound op's host time included);
- ``graph``: device time per call of back-to-back calls replayed from one CUDA graph: what a
  call costs inside a captured graph (no host time). Off CUDA, ``loop``.

- A config raising a backend failure (``errors.is_backend_failure``) is recorded and skipped;
  every config failing raises ``ConfigsFailed``.
- Working-memory bound: configs run in shortlist order, each bounded to ``WORKING_FACTOR`` x
  the leanest working memory measured before it (at least ``WORKING_FLOOR``); one over it is
  recorded and skipped, so a padded PyTorch path never sets the tuning peak.
- ``set_mode("off")`` binds the stored winner, else the first candidate, and times nothing.
"""

import contextlib
import json
import os
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from kohakugbc_models.kernels.errors import ConfigsFailed, is_backend_failure
from kohakugbc_models.kernels.hardware import cache_dir, card_key, synchronize

WORKING_FACTOR = 8.0
WORKING_FLOOR = 256 << 20
METRICS = ("loop", "graph")
_MODE = {"mode": "table"}
_METRIC = threading.local()
_CHOSEN: dict[tuple, dict] = {}
_TABLES: dict[tuple[str, str], "Table"] = {}
_LOCK = threading.Lock()


def set_mode(mode: str) -> None:
    if mode not in ("table", "off"):
        raise ValueError(f"tuning mode {mode!r}: table or off")
    _MODE["mode"] = mode


def current_metric() -> str:
    return getattr(_METRIC, "name", "loop")


@contextlib.contextmanager
def metric(name: str):
    if name not in METRICS:
        raise ValueError(f"tuning metric {name!r}: one of {METRICS}")
    saved = current_metric()
    _METRIC.name = name
    try:
        yield
    finally:
        _METRIC.name = saved


def name(config: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(config.items()))


class Table:
    """One kernel's winners on one card: a JSON file merged on every write."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict] = self.read()

    def read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def get(self, key: str) -> dict | None:
        row = self.rows.get(key)
        return None if row is None else row["config"]

    def put(self, key: str, row: dict) -> None:
        with _LOCK:
            self.rows = {**self.read(), **self.rows, key: row}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}")
            tmp.write_text(json.dumps(self.rows, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)


def table(kernel: str, device) -> Table:
    card = card_key(str(torch.device(device)))
    if (card, kernel) not in _TABLES:
        _TABLES[(card, kernel)] = Table(cache_dir() / "tuning" / card / f"{kernel}.json")
    return _TABLES[(card, kernel)]


def loop_ms(fn: Callable[[], object], device, calls: int) -> float:
    """Wall ms per call of ``calls`` back-to-back calls (after two warm-ups)."""
    for _ in range(2):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1e3 / calls


def graph_ms(fn: Callable[[], object], device, calls: int) -> float:
    """Device ms per call of ``calls`` calls captured into one CUDA graph and replayed (the
    best of three replays, after two warm-ups outside the graph)."""
    for _ in range(2):
        fn()
    synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    best = float("inf")
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) / calls)
    del graph
    return best


def best_ms(fn: Callable[[], object], device, repeat: int) -> float:
    """The current metric's ms per call over ``repeat`` calls."""
    if current_metric() == "graph" and torch.device(device).type == "cuda":
        return graph_ms(fn, device, repeat)
    return loop_ms(fn, device, repeat)


def working_bytes(fn: Callable[[], object], device) -> int:
    """Peak bytes ``fn`` allocates above what is allocated when it starts (CUDA only)."""
    if torch.device(device).type != "cuda":
        return 0
    synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    synchronize(device)
    return torch.cuda.max_memory_allocated(device) - base


def select(
    kernel: str,
    device,
    key: str,
    candidates: Callable[[], Sequence[dict]],
    launch: Callable[[dict], Callable[[], object]],
    shortlist: int = 4,
    repeat: int = 10,
) -> dict:
    """The config of problem ``key``: stored, else timed (see the module docstring)."""
    if current_metric() != "loop":
        key = f"{key}@{current_metric()}"
    ident = (str(device), kernel, key)
    if ident in _CHOSEN:
        return _CHOSEN[ident]
    ranked = list(candidates())
    tab = table(kernel, device)
    stored = tab.get(key)
    if stored is not None and stored in ranked:
        _CHOSEN[ident] = stored
        return stored
    if _MODE["mode"] == "off" or len(ranked) == 1:
        _CHOSEN[ident] = ranked[0]
        return ranked[0]
    timings, failures, working = {}, {}, {}
    leanest = None
    for config in ranked[:shortlist]:
        label = name(config)
        try:
            run = launch(config)
            used = working_bytes(run, device)
            bound = None if leanest is None else max(WORKING_FLOOR, WORKING_FACTOR * leanest)
            working[label] = used / 2**20
            if bound is not None and used > bound:
                failures[label] = f"over the working-memory bound ({used >> 20} MiB)"
                continue
            leanest = used if leanest is None else min(leanest, used)
            timings[label] = best_ms(run, device, repeat)
        except BaseException as err:
            if not is_backend_failure(err):
                raise
            failures[label] = f"{type(err).__name__}: {err}"[:2000]
    if not timings:
        raise ConfigsFailed(f"{kernel} {key}: every config failed: {failures}")
    winner = min(
        (c for c in ranked[:shortlist] if name(c) in timings), key=lambda c: timings[name(c)]
    )
    tab.put(
        key, {"config": winner, "timings_ms": timings, "failures": failures, "working_mib": working}
    )
    _CHOSEN[ident] = winner
    return winner
