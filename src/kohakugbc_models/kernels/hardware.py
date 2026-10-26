"""The device a kernel runs on: its arch name and arch family (the backend priority's keys, the
family also each kernel's tile set) and its card key (the tuning tables' key), and the cache
directory."""

import os
import re
import threading
from functools import cache
from pathlib import Path

import torch

# A host sync in another thread invalidates a CUDA graph capture; re-entrant (a warm-up tunes)
QUIET = threading.RLock()
_CAPTURE: dict[int, torch.cuda.Stream] = {}


def capture_stream(device=None) -> "torch.cuda.Stream":
    """The one stream every CUDA graph capture on ``device`` (the current device by default)
    runs on: the caching allocator reuses a graph pool's freed blocks only on the stream that
    freed them, so captures sharing a pool on fresh streams each would hold the sum of their
    passes. Captures run one at a time (``QUIET``), so one stream per device never overlaps."""
    device = torch.device("cuda") if device is None else torch.device(device)
    index = torch.cuda.current_device() if device.index is None else device.index
    with QUIET:
        if index not in _CAPTURE:
            _CAPTURE[index] = torch.cuda.Stream(index)
        return _CAPTURE[index]


def arch_of(device) -> str:
    """``smXY`` of a CUDA device, ``xpu`` or ``cpu``."""
    device = torch.device(device)
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return f"sm{major}{minor}"
    return device.type


# Arch families sharing one set of kernels: sm10x the datacenter Blackwell cards (B200 sm100,
# B300 sm103; 228 KiB of shared memory per SM, tcgen05), sm12x the consumer ones (RTX 50xx
# sm120, DGX Spark sm121; about 100 KiB)
FAMILIES = {"sm10x": ("sm100", "sm101", "sm103"), "sm12x": ("sm120", "sm121")}


def family_of(arch: str) -> str:
    """The family an arch's kernels come from (``FAMILIES``), else the arch itself."""
    return next((f for f, members in FAMILIES.items() if arch in members), arch)


def family(device) -> str:
    """The kernel family of a device."""
    return family_of(arch_of(device))


@cache
def card_key(device: str) -> str:
    """``<card>-<arch>-<cuda>-<torch>``: tables are valid for one card and toolchain."""
    dev = torch.device(device)
    if dev.type == "cuda":
        name = torch.cuda.get_device_name(dev)
        runtime = f"cu{torch.version.cuda}"
    elif dev.type == "xpu":
        name = torch.xpu.get_device_name(dev)
        runtime = "xpu"
    else:
        name, runtime = "cpu", "cpu"
    name = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return f"{name}-{arch_of(dev)}-{runtime}-torch{torch.__version__.split('+')[0]}"


def cache_dir() -> Path:
    """``KOHAKUGBC_KERNEL_CACHE``, default ``~/.cache/kohakugbc_kernels``."""
    root = os.environ.get("KOHAKUGBC_KERNEL_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "kohakugbc_kernels"


def synchronize(device) -> None:
    """A host synchronization, never during another thread's CUDA graph capture (``QUIET``)."""
    device = torch.device(device)
    if device.type == "cuda":
        with QUIET:
            torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def tiles_for(device, base: list, extra: dict) -> list:
    """A kernel's tile candidates on a device: ``base`` (every family's), plus ``extra[f]`` for
    the device's family ``f`` (sm10x: tiles and stages only its shared memory holds)."""
    return base + extra.get(family(device), [])
