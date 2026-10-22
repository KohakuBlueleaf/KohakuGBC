"""The engine's op registry, the KohakuMossTrainer way: every op has an eager reference and any
number of backends; an ``Ops`` binds one implementation per (op, problem key) once, by a timed
race among the candidates that run on the device, and later calls only look it up.

Ops (all on packed tokens with per-sequence boundaries ``cu`` and pool slots ``slots``):

``delta(q, k, v, a, b, a_log, dt_bias, state, slots, cu, max_len)``
    The gated delta rule of each sequence's new tokens from its state in ``state[slots]``;
    writes the final state back in place and returns the outputs.
``conv(x, weight, state, slots, cu)``
    The DeltaNet's causal depthwise convolution (silu) continuing from ``state[slots]`` (the
    last ``width - 1`` inputs), which it advances in place.
``attention(q, k, v, cache_k, cache_v, pages, ctx, cu, write, max_len, scale)``
    Writes the new keys/values at ``write`` (flat page rows), then causal attention of each new
    token over its sequence's first ``ctx`` cached tokens through its page table ``pages``.

Candidates are bound per key (decode, short extend, prefill by length bucket); the race runs
on scratch copies of what an op changes in place.
"""

import time
from collections.abc import Callable

import torch

REGISTRY: dict[str, dict[str, Callable]] = {}
# Backends whose library or compiler is missing here, with why
UNAVAILABLE: dict[tuple[str, str], str] = {}
# Backends that never sync with the host: the only candidates inside a captured graph
GRAPH_SAFE: set[tuple[str, str]] = set()


def register(op: str, backend: str, graph_safe: bool = False):
    def put(fn):
        REGISTRY.setdefault(op, {})[backend] = fn
        if graph_safe:
            GRAPH_SAFE.add((op, backend))
        return fn

    return put


def bucket(n: int) -> int:
    """Powers of two: keys of problems of similar cost."""
    return 1 << max(0, int(n) - 1).bit_length()


def key_of(op: str, args: tuple) -> str:
    match op:
        case "delta":
            cu, max_len = args[9], args[10]
            return f"n{bucket(cu.shape[0] - 1)}-t{bucket(max_len)}"
        case "conv":
            return f"t{args[0].shape[0]}"
        case "attention":
            cu, max_len = args[7], args[9]
            return f"n{bucket(cu.shape[0] - 1)}-t{bucket(max_len)}"
    raise KeyError(op)


def scratch(op: str, args: tuple) -> tuple:
    """``args`` with what ``op`` changes in place replaced by copies the race may change."""
    args = list(args)
    match op:
        case "delta":
            state, slots = args[7], args[8]
            n = slots.shape[0]
            args[7] = state[slots].clone() if n else state[:0].clone()
            args[8] = torch.arange(n, device=slots.device, dtype=slots.dtype)
        case "conv":
            state, slots = args[2], args[3]
            args[2] = state[slots].clone()
            args[3] = torch.arange(slots.shape[0], device=slots.device, dtype=slots.dtype)
        case "attention":
            # The writes put the same rows back: running twice changes nothing
            pass
    return tuple(args)


class Ops:
    """One implementation per (op, key): ``prefer`` names the candidates of each op in order
    (``{"delta": ["fla", "triton"]}``); with ``race`` the fastest of those that run wins, else
    the first that runs. ``choices`` records what was bound and why."""

    def __init__(self, prefer: dict[str, list[str]], race: bool = True, reps: int = 3):
        self.prefer = {op: list(names) for op, names in prefer.items()}
        self.race, self.reps = race, reps
        self.bound: dict[tuple[str, str], Callable] = {}
        self.choices: dict[tuple[str, str], dict] = {}
        # Set while graphs are warmed up and captured: graph-safe candidates, keys of their own
        self.graph = False

    def candidates(self, op: str) -> list[tuple[str, Callable]]:
        impls = REGISTRY.get(op, {})
        names = self.prefer.get(op) or list(impls)
        return [
            (n, impls[n])
            for n in names
            if n in impls
            and (op, n) not in UNAVAILABLE
            and (not self.graph or (op, n) in GRAPH_SAFE)
        ]

    def __call__(self, op: str, *args):
        key = key_of(op, args) + ("-graph" if self.graph else "")
        fn = self.bound.get((op, key))
        if fn is None:
            fn = self._bind(op, key, args)
        return fn(*args)

    def _bind(self, op: str, key: str, args: tuple) -> Callable:
        options = self.candidates(op)
        if not options:
            raise RuntimeError(f"op {op}: no backend runs here ({self.prefer.get(op)})")
        timings, errors = {}, {}
        race = self.race and len(options) > 1
        for name, fn in options:
            try:
                timings[name] = self._time(op, fn, args) if race else 0.0
            except Exception as exc:  # noqa: BLE001
                errors[name] = repr(exc)
                continue
            if not race:
                break
        if not timings:
            raise RuntimeError(f"op {op} [{key}]: every backend failed: {errors}")
        name = min(timings, key=timings.get)
        self.bound[(op, key)] = dict(options)[name]
        self.choices[(op, key)] = {"backend": name, "ms": timings, "errors": errors}
        return self.bound[(op, key)]

    def _time(self, op: str, fn: Callable, args: tuple) -> float:
        trial = scratch(op, args)
        fn(*trial)
        if trial[0].is_cuda:
            torch.cuda.synchronize(trial[0].device)
        best = float("inf")
        for _ in range(self.reps):
            trial = scratch(op, args)
            if trial[0].is_cuda:
                torch.cuda.synchronize(trial[0].device)
            t0 = time.perf_counter()
            fn(*trial)
            if trial[0].is_cuda:
                torch.cuda.synchronize(trial[0].device)
            best = min(best, (time.perf_counter() - t0) * 1e3)
        return best
