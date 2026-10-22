"""The placement plan: every engine on its devices, checked before any weight loads.

Entries say what they will hold on which device (``Need``): a torch model's estimated weights and
peak activations, an in-process VLM's reservation (its share of the device's memory). The plan
adds them per device against the memory free now, less a headroom, and every device that does
not fit is one line of one error. ``devices = "auto"`` is every CUDA device the process sees, or
the CPU without one; the dtype ``"auto"`` is bfloat16 on CUDA devices of compute capability 8 or
above, float16 below it, float32 on the CPU.

``LOADING``: held while a model's weights load. transformers sets torch's process-wide default
dtype for the length of a load, so two units loading at once (they start on threads of their
own) would build parts of each other's models in the wrong dtype.
"""

import threading
from dataclasses import dataclass

try:
    import torch
except ImportError:
    torch = None

GIB = 1024**3
BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
LOADING = threading.Lock()


@dataclass
class Need:
    owner: str
    device: str
    bytes: int
    what: str


def devices(spec) -> list:
    """The placements of ``spec``: ``"auto"``, a device, or a list of devices and device
    tuples (a tuple: one model split across those devices)."""
    match spec:
        case "auto" | None:
            n = torch.cuda.device_count() if torch is not None else 0
            return [f"cuda:{i}" for i in range(n)] if n else ["cpu"]
        case str():
            return [spec]
        case _:
            return [tuple(d) if isinstance(d, list) else d for d in spec]


def flat(placements: list) -> list[str]:
    out = []
    for p in placements:
        for d in p if isinstance(p, tuple) else (p,):
            if d not in out:
                out.append(d)
    return out


def dtype(name: str, placement) -> str:
    """The dtype name for ``placement`` (``"auto"`` by its device)."""
    if name != "auto":
        if name not in BYTES:
            raise ValueError(f"dtype {name!r}: one of auto, {', '.join(BYTES)}")
        return name
    first = placement[0] if isinstance(placement, tuple) else placement
    if not str(first).startswith("cuda"):
        return "float32"
    major, _ = torch.cuda.get_device_capability(torch.device(first))
    return "bfloat16" if major >= 8 else "float16"


def memory(device: str) -> tuple[int, int] | None:
    """(free, total) bytes of a CUDA device; None for the CPU."""
    if torch is None or not str(device).startswith("cuda"):
        return None
    return torch.cuda.mem_get_info(torch.device(device))


def check(needs: list[Need], headroom: float = 0.05) -> None:
    """Raise one ``ValueError`` naming every device whose needs exceed its free memory."""
    by_device: dict[str, list[Need]] = {}
    for need in needs:
        by_device.setdefault(need.device, []).append(need)
    problems = []
    for device, items in by_device.items():
        mem = memory(device)
        if mem is None:
            continue
        free, total = mem
        total_needed = sum(n.bytes for n in items)
        available = free - headroom * total
        if total_needed > available:
            parts = ", ".join(f"{n.owner} {n.what} {n.bytes / GIB:.2f} GiB" for n in items)
            problems.append(
                f"{device}: needs {total_needed / GIB:.2f} GiB, {available / GIB:.2f} GiB free "
                f"(less {headroom:.0%} headroom): {parts}"
            )
    if problems:
        raise ValueError("placement does not fit:\n  " + "\n  ".join(problems))
