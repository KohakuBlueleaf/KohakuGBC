"""Gluon kernels for sm10x (datacenter Blackwell: tcgen05, TMEM, TMA). The probe accepts an sm10x
device only (the kernels use its tensor memory) and builds and launches one minimal kernel."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from kohakugbc_models.kernels import registry
from kohakugbc_models.kernels.backends import load_tree
from kohakugbc_models.kernels.hardware import family


@gluon.jit
def _probe_kernel(x, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    offs = gl.arange(0, BLOCK, layout=layout)
    gl.store(x + offs, gl.load(x + offs) + 1)


def probe(device: str) -> str | None:
    if family(device) != "sm10x":
        return f"sm10x only ({family(device)})"
    try:
        x = torch.zeros(32, device=device)
        _probe_kernel[(1,)](x, BLOCK=32, num_warps=1)
        if x.sum().item() != 32:
            return "probe kernel computed a wrong result"
    except Exception as err:  # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    return None


registry.register_probe("gluon", probe)
FAILED = load_tree(__name__, __path__)
