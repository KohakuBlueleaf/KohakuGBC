"""The device a kernel runs on: its arch name (the backend priority's key) and its card key
(the tuning tables' key), and the cache directory."""

import os
import re
from functools import cache
from pathlib import Path

import torch


def arch_of(device) -> str:
    """``smXY`` of a CUDA device, ``xpu`` or ``cpu``."""
    device = torch.device(device)
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return f"sm{major}{minor}"
    return device.type


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
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
