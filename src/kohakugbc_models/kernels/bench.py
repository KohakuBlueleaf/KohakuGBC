"""The kernel bench harness: every candidate of one op on one case, speed and accuracy side by
side.

``run_case(op, make_inputs, candidates, device, dtype)``: ``make_inputs(dtype)`` builds the
case's arguments; each candidate (name -> callable with the op's signature: the eager PyTorch
path, torch.compile of it, every kernel config) is timed (best of ``repeat`` after a warm-up,
the L2 flushed before each timed call on CUDA) and its output profiled against the eager
reference run in fp64 on the same values (``numerics.errors.profile``). The eager PyTorch path
in ``dtype`` is the baseline the others are read against; nothing is judged here.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

import torch

from kohakugbc_models.kernels.hardware import card_key, synchronize
from kohakugbc_models.numerics.errors import profile

L2_FLUSH_BYTES = 256 << 20


def outputs(result) -> list[torch.Tensor]:
    return list(result) if isinstance(result, (tuple, list)) else [result]


def timed(fn: Callable[[], object], device, repeat: int) -> float:
    flush = None
    if torch.device(device).type == "cuda":
        flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    fn()
    synchronize(device)
    best = float("inf")
    for _ in range(repeat):
        if flush is not None:
            flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            best = min(best, start.elapsed_time(end))
        else:
            begin = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - begin) * 1e3)
    return best


def ruler(reference: Callable, args: list) -> list[torch.Tensor]:
    """The reference in fp64, one leading-index slice at a time (fp64 attention runs the math
    path, which holds a whole score matrix): arguments sharing the first argument's leading
    size are sliced, the rest passed whole."""
    wide = [
        a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args
    ]
    n = wide[0].shape[0]
    parts = []
    for i in range(n):
        sliced = [
            a[i : i + 1] if isinstance(a, torch.Tensor) and a.dim() and a.shape[0] == n else a
            for a in wide
        ]
        parts.append(outputs(reference(*sliced)))
    return [torch.cat(column) for column in zip(*parts)]


def run_case(
    op: str,
    case: str,
    make_inputs: Callable,
    reference: Callable,
    candidates: dict[str, Callable],
    device,
    dtype: torch.dtype,
    repeat: int = 20,
) -> dict:
    """One row per candidate: ms, and per output the error profile against fp64."""
    args = make_inputs(dtype)
    with torch.inference_mode():
        ruler_out = ruler(reference, args)
        rows = {}
        for name, fn in candidates.items():
            try:
                got = outputs(fn(*args))
                ms = timed(lambda fn=fn: fn(*args), device, repeat)
            except Exception as err:  # noqa: BLE001
                rows[name] = {"error": f"{type(err).__name__}: {err}"[:2000]}
                continue
            rows[name] = {
                "ms": ms,
                "errors": [profile(g, r) for g, r in zip(got, ruler_out)],
            }
    return {
        "op": op,
        "case": case,
        "dtype": str(dtype).replace("torch.", ""),
        "card": card_key(str(torch.device(device))),
        "rows": rows,
    }


def write(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
